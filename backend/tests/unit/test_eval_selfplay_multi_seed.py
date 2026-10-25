from unittest.mock import patch

import pytest

from app.infrastructure.rl.eval_selfplay import _parse_seeds, run_eval_multi_seed


def test_run_eval_multi_seed_averages_win_rate_and_gates() -> None:
    calls: list[int | None] = []

    def fake_run_eval(games, diff_a, diff_b, *, min_win_rate=None, max_p99_ms=None, seed=None, **kwargs):
        calls.append(seed)
        rates = {97: 0.40, 17: 0.50, 88: 0.60}
        return {
            "games": float(games),
            "win_rate": rates[seed],
            "p95_ms": 400.0,
            "p99_ms": 450.0,
            "timeouts": 0.0,
            "finished": float(games),
            "repetition_events": 0.0,
        }

    with patch("app.infrastructure.rl.eval_selfplay.run_eval", side_effect=fake_run_eval):
        result = run_eval_multi_seed(10, "normal", "easy", [97, 17, 88], min_win_rate=0.48)
    assert calls == [97, 17, 88]
    assert result["win_rate"] == pytest.approx(0.50)


def test_run_eval_multi_seed_gate_fails_on_mean() -> None:
    with patch(
        "app.infrastructure.rl.eval_selfplay.run_eval",
        return_value={
            "games": 10.0,
            "win_rate": 0.40,
            "p95_ms": 400.0,
            "p99_ms": 450.0,
            "timeouts": 0.0,
            "finished": 10.0,
            "repetition_events": 0.0,
        },
    ):
        with pytest.raises(SystemExit, match="Mean win rate gate failed"):
            run_eval_multi_seed(10, "normal", "easy", [97], min_win_rate=0.55)


def test_parse_seeds_empty_commas() -> None:
    assert _parse_seeds(",") == []
    assert _parse_seeds("97,17") == [97, 17]


def test_run_eval_multi_seed_uses_per_seed_scoresheet_dir() -> None:
    dirs: list[str | None] = []

    def fake_run_eval(*args, **kwargs):
        d = kwargs.get("scoresheet_dir")
        dirs.append(str(d) if d is not None else None)
        return {
            "games": 1.0,
            "win_rate": 0.5,
            "p95_ms": 1.0,
            "p99_ms": 1.0,
            "timeouts": 0.0,
            "finished": 1.0,
            "repetition_events": 0.0,
        }

    with patch("app.infrastructure.rl.eval_selfplay.run_eval", side_effect=fake_run_eval):
        run_eval_multi_seed(1, "normal", "easy", [97, 17], scoresheet_dir="/tmp/sheets")
    assert dirs == ["/tmp/sheets/seed_97", "/tmp/sheets/seed_17"]


def test_run_eval_multi_seed_stops_after_repetition() -> None:
    calls: list[int] = []

    def fake_run_eval(*args, **kwargs):
        seed = kwargs["seed"]
        calls.append(seed)
        reps = 1.0 if seed == 97 else 0.0
        return {
            "games": 1.0,
            "win_rate": 0.5,
            "p95_ms": 1.0,
            "p99_ms": 1.0,
            "timeouts": 0.0,
            "finished": 1.0,
            "repetition_events": reps,
        }

    with patch("app.infrastructure.rl.eval_selfplay.run_eval", side_effect=fake_run_eval):
        run_eval_multi_seed(
            1,
            "normal",
            "easy",
            [97, 17, 88],
            stop_after_repetition=True,
        )
    assert calls == [97]
