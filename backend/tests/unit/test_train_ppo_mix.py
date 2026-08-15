from app.infrastructure.rl.train_ppo import _build_stages, _default_opponent_mix


def test_default_normal_mix_includes_weaker_opponents() -> None:
    mix = _default_opponent_mix("normal")
    assert mix is not None
    names = {name for name, _ in mix}
    assert names == {"normal", "easy", "very_easy"}


def test_no_opponent_mix_builds_pure_normal_stage() -> None:
    stages = _build_stages(
        timesteps=100_000,
        curriculum=None,
        opponent="normal",
        weights_raw=None,
        max_wall_candidates=10,
        no_opponent_mix=True,
    )
    assert len(stages) == 1
    assert stages[0].opponent == "normal"
    assert stages[0].opponent_mix is None
    assert stages[0].timesteps == 100_000


def test_resume_learn_target_is_current_plus_additional() -> None:
    from app.infrastructure.rl.train_ppo import _absolute_learn_timesteps

    assert _absolute_learn_timesteps(1_212_416, 400_000, reset=False) == 1_612_416
    assert _absolute_learn_timesteps(0, 400_000, reset=True) == 400_000
