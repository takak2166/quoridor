from __future__ import annotations

from concurrent.futures import Future
from concurrent.futures import TimeoutError as FuturesTimeoutError
from unittest.mock import MagicMock

import pytest

from app.infrastructure.rl.train_ppo import (
    _build_stages,
    _default_opponent_mix,
    _sb3_learn_timesteps,
    _smoke_game_won,
    smoke_win_rate,
)


def test_default_normal_mix_includes_weaker_opponents() -> None:
    mix = _default_opponent_mix("normal")
    assert mix is not None
    names = {name for name, _ in mix}
    assert names == {"normal", "easy", "very_easy"}


def test_white_win_ramp_starts_at_random_and_keeps_wanderers() -> None:
    from app.infrastructure.rl.train_ppo import WHITE_WIN_CURRICULUM, _white_win_opponent_mix

    stages = _build_stages(
        timesteps=100_000,
        curriculum="very_easy,easy,normal",
        opponent="normal",
        weights_raw=None,
        max_wall_candidates=10,
        white_win_ramp=True,
    )
    assert tuple(stage.opponent for stage in stages) == WHITE_WIN_CURRICULUM
    assert stages[0].opponent_mix is None
    assert stages[0].max_wall_candidates is None
    assert stages[0].agent_white_prob == 0.80
    assert stages[-1].agent_white_prob == 0.50
    assert sum(stage.timesteps for stage in stages) == 100_000
    easy_mix = _white_win_opponent_mix("easy")
    assert easy_mix is not None
    assert {name for name, _ in easy_mix} == {"easy", "very_easy", "random"}
    normal_mix = stages[-1].opponent_mix
    assert normal_mix is not None
    assert "random" in {name for name, _ in normal_mix}


def test_agent_white_prob_overrides_ramp_defaults() -> None:
    stages = _build_stages(
        timesteps=10_000,
        curriculum=None,
        opponent="normal",
        weights_raw=None,
        max_wall_candidates=10,
        white_win_ramp=True,
        agent_white_prob=0.9,
    )
    assert all(stage.agent_white_prob == 0.9 for stage in stages)



def test_sb3_learn_timesteps_is_additional_only() -> None:
    """SB3 _setup_learn adds num_timesteps when reset_num_timesteps=False."""
    assert _sb3_learn_timesteps(400_000) == 400_000


def test_sb3_learn_timesteps_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="additional timesteps"):
        _sb3_learn_timesteps(0)


def test_smoke_timeout_counts_as_loss() -> None:
    timed_out: Future[bool] = Future()
    timed_out.set_exception(FuturesTimeoutError())
    assert _smoke_game_won(timed_out, timeout_sec=1.0, opponent="normal") is False


def test_smoke_error_counts_as_loss() -> None:
    failed: Future[bool] = Future()
    failed.set_exception(RuntimeError("boom"))
    assert _smoke_game_won(failed, timeout_sec=1.0, opponent="normal") is False


def test_smoke_win_is_true() -> None:
    won: Future[bool] = Future()
    won.set_result(True)
    assert _smoke_game_won(won, timeout_sec=1.0, opponent="normal") is True


def test_smoke_win_rate_keeps_timeouts_in_denominator(monkeypatch) -> None:
    from app.infrastructure.rl import train_ppo as mod

    class ImmediateFuture:
        def __init__(self, seed: int) -> None:
            self._seed = seed

        def result(self, timeout: float | None = None) -> bool:
            if self._seed == 1:
                raise FuturesTimeoutError()
            return self._seed == 0

    class FakeExecutor:
        def __init__(self, max_workers: int) -> None:
            self.max_workers = max_workers

        def __enter__(self) -> FakeExecutor:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def submit(self, fn: object, *args: object, **kwargs: object) -> ImmediateFuture:
            return ImmediateFuture(int(kwargs["seed"]))

    monkeypatch.setattr(mod, "ThreadPoolExecutor", FakeExecutor)
    model = MagicMock()
    model.policy.training = True
    rate = smoke_win_rate(
        model,
        "normal",
        games=4,
        gamma=0.99,
        potential_scale=8.0,
        max_wall_candidates=10,
        opening_wall_free_plies=2,
        timeout_sec=1.0,
        workers=4,
    )
    # seed 0 win, seed 1 timeout, seeds 2-3 losses → 1/4, not 1/3
    assert rate == 0.25
    model.policy.set_training_mode.assert_any_call(False)
    model.policy.set_training_mode.assert_called_with(True)
