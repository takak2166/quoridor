import pytest

from app.infrastructure.rl.eval_selfplay import run_eval


@pytest.mark.slow
def test_selfplay_smoke() -> None:
    """Short self-play eval for Phase 5 validation gate."""
    result = run_eval(10, "easy", "normal")
    assert 0.0 <= result["win_rate"] <= 1.0
    assert result["p99_ms"] < 2000


@pytest.mark.slow
def test_normal_beats_easy_gate() -> None:
    result = run_eval(50, "normal", "easy", min_win_rate=0.55, max_p99_ms=1000)
    assert result["win_rate"] >= 0.55


@pytest.mark.slow
def test_very_easy_loses_to_easy_and_normal() -> None:
    for opponent in ("easy", "normal"):
        result = run_eval(30, "very_easy", opponent)
        assert result["win_rate"] == 0.0, opponent
