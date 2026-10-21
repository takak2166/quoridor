from unittest.mock import patch

import pytest

from app.infrastructure.rl.eval_selfplay import run_eval_multi_seed


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
