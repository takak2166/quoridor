"""Regression: eval self-play win rate must use games_played, not len(latencies)."""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import patch

import pytest

from app.infrastructure.rl.eval_selfplay import run_eval
from quoridor.domain.actions import Action, Move
from quoridor.domain.game import Game
from quoridor.domain.state import Color, QuoridorState, initial_state
from quoridor.rules import PlayResult


def _legacy_move_count_win_rate(wins_a: int, move_inferences: int) -> float:
    """Pre-cc62e99 bug: denominator was len(latencies) (one per AI inference / ply)."""
    return wins_a / move_inferences if move_inferences else 0.0


@dataclass
class _ScriptedGame(Game):
    """Minimal game stub: counts plies without full rule simulation."""

    win_after: int | None = None
    win_color: Color = "black"
    _plies: int = field(default=0, init=False)

    @classmethod
    def from_script(cls, *, win_after: int | None, win_color: Color = "black") -> _ScriptedGame:
        return cls(state=initial_state(), winner=None, win_after=win_after, win_color=win_color)

    def play(self, action: Action) -> PlayResult:
        _ = action
        self._plies += 1
        if self.win_after is not None and self._plies >= self.win_after:
            self.winner = self.win_color
        return PlayResult(state=self.state, winner=self.winner)


class _CountingStubAi:
    def select_move(self, state: QuoridorState, color: Color) -> Action:
        _ = state, color
        return Move(direction="up", to=(4, 0))


def test_run_eval_win_rate_uses_completed_games_not_move_count() -> None:
    """One decisive game (few plies) plus one timeout (many plies) → rates diverge."""
    scripts = [
        _ScriptedGame.from_script(win_after=3, win_color="black"),
        _ScriptedGame.from_script(win_after=None),
    ]

    def next_game() -> _ScriptedGame:
        return scripts.pop(0)

    stub_ai = _CountingStubAi()
    with (
        patch("app.infrastructure.rl.eval_selfplay.Game.from_initial", side_effect=next_game),
        patch(
            "app.infrastructure.rl.eval_selfplay.ai_for_difficulty",
            return_value=stub_ai,
        ),
    ):
        result = run_eval(
            2,
            "normal",
            "easy",
            max_moves=10,
            min_win_rate=0.45,
        )

    assert result["games"] == pytest.approx(2.0)
    assert result["timeouts"] == pytest.approx(1.0)
    assert result["finished"] == pytest.approx(1.0)
    assert result["win_rate"] == pytest.approx(0.5)

    wins_a = 1
    move_inferences = 3 + 10
    legacy = _legacy_move_count_win_rate(wins_a, move_inferences)
    assert legacy == pytest.approx(1 / 13)
    assert legacy < 0.45
    assert result["win_rate"] != pytest.approx(legacy)


def test_run_eval_win_rate_gate_fails_under_legacy_move_denominator() -> None:
    """Same scenario: min_win_rate=0.45 passes fixed logic, legacy denominator would fail."""
    scripts = [
        _ScriptedGame.from_script(win_after=3, win_color="black"),
        _ScriptedGame.from_script(win_after=None),
    ]

    def next_game() -> _ScriptedGame:
        return scripts.pop(0)

    stub_ai = _CountingStubAi()
    with (
        patch("app.infrastructure.rl.eval_selfplay.Game.from_initial", side_effect=next_game),
        patch(
            "app.infrastructure.rl.eval_selfplay.ai_for_difficulty",
            return_value=stub_ai,
        ),
    ):
        result = run_eval(2, "normal", "easy", max_moves=10, min_win_rate=0.45)

    legacy = _legacy_move_count_win_rate(1, 13)
    assert legacy < 0.45
    assert result["win_rate"] >= 0.45


def test_run_eval_timeout_still_counts_toward_games_played() -> None:
    """Timeouts are loop iterations in games_played (not decisive-outcome-only filtering)."""
    scripts = [_ScriptedGame.from_script(win_after=None) for _ in range(3)]

    def next_game() -> _ScriptedGame:
        return scripts.pop(0)

    with (
        patch("app.infrastructure.rl.eval_selfplay.Game.from_initial", side_effect=next_game),
        patch(
            "app.infrastructure.rl.eval_selfplay.ai_for_difficulty",
            return_value=_CountingStubAi(),
        ),
    ):
        result = run_eval(3, "normal", "easy", max_moves=5)

    assert result["games"] == pytest.approx(3.0)
    assert result["timeouts"] == pytest.approx(3.0)
    assert result["finished"] == pytest.approx(0.0)
    assert result["win_rate"] == pytest.approx(0.0)

    legacy = _legacy_move_count_win_rate(0, 3 * 5)
    assert legacy == pytest.approx(0.0)
    assert result["win_rate"] == legacy  # both zero, but games still counted


def test_run_eval_multi_seed_gate_uses_per_game_win_rates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Multi-seed mean gate must aggregate game-based win_rate, not ply fractions."""
    from app.infrastructure.rl import eval_selfplay as mod

    captured: list[int | None] = []

    def fake_run_eval(
        games: int,
        diff_a: str,
        diff_b: str,
        *,
        min_win_rate=None,
        max_p99_ms=None,
        seed=None,
        **kwargs,
    ):
        captured.append(seed)
        # Game-based rates that average to 0.50; legacy ply math would skew lower.
        rates = {97: 0.88, 17: 0.89, 88: 0.89}
        g = float(games)
        return {
            "games": g,
            "win_rate": rates[seed],
            "p95_ms": 400.0,
            "p99_ms": 450.0,
            "timeouts": 0.0,
            "finished": g,
            "repetition_events": 0.0,
        }

    monkeypatch.setattr(mod, "run_eval", fake_run_eval)
    result = mod.run_eval_multi_seed(
        100,
        "normal",
        "easy",
        [97, 17, 88],
        min_win_rate=0.55,
        max_p99_ms=500.0,
    )
    assert captured == [97, 17, 88]
    assert result["win_rate"] >= 0.55
    assert result["win_rate"] == pytest.approx((0.88 + 0.89 + 0.89) / 3)
