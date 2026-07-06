import numpy as np
import pytest
from gymnasium import error as gym_error

from app.infrastructure.rl.env import QuoridorEnv
from quoridor.domain.actions import Move, encode
from quoridor.domain.state import QuoridorState, empty_walls
from quoridor.rules import check_winner


def test_env_step_valid_mask() -> None:
    env = QuoridorEnv()
    obs, info = env.reset(options={"agent_color": "white"})
    assert obs.shape == (135,)
    assert info["action_masks"].shape == (132,)
    assert env._state.current_player == "white"
    legal = np.where(info["action_masks"])[0]
    obs2, reward, terminated, truncated, info2 = env.step(int(legal[0]))
    assert obs2.shape == (135,)
    assert not truncated
    assert info2["action_masks"].shape == (132,)


def test_env_reward_when_agent_wins() -> None:
    env = QuoridorEnv(agent_color="white", opponent="random")
    env.reset(options={"agent_color": "white"})
    env._state = QuoridorState(
        white=(1, 4),
        black=(8, 0),
        white_walls_remaining=10,
        black_walls_remaining=10,
        horizontal_walls=empty_walls(),
        vertical_walls=empty_walls(),
        current_player="white",
    )
    obs, reward, terminated, truncated, info = env.step(encode(Move(direction="up", to=(0, 4))))
    assert obs.shape == (135,)
    assert info["action_masks"].shape == (132,)
    assert reward == 1.0
    assert terminated
    assert not truncated


def test_env_reward_when_opponent_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    env = QuoridorEnv(agent_color="white", opponent="random")
    env.reset(options={"agent_color": "white"})
    monkeypatch.setattr(
        env,
        "_select_opponent_move",
        lambda _legal: Move(direction="up", to=(8, 4)),
    )
    env._state = QuoridorState(
        white=(2, 0),
        black=(7, 4),
        white_walls_remaining=10,
        black_walls_remaining=10,
        horizontal_walls=empty_walls(),
        vertical_walls=empty_walls(),
        current_player="white",
    )
    action = int(np.where(env._mask())[0][0])
    _, reward, terminated, truncated, _ = env.step(action)
    assert reward == -1.0
    assert terminated
    assert not truncated


def test_reset_white_agent_skips_black_turn() -> None:
    env = QuoridorEnv()
    env.reset(options={"agent_color": "white"})
    assert env.agent_color == "white"
    assert env._state.current_player == "white"
    assert check_winner(env._state) is None


def test_reset_black_agent_starts_immediately() -> None:
    env = QuoridorEnv()
    env.reset(options={"agent_color": "black"})
    assert env.agent_color == "black"
    assert env._state.current_player == "black"
    assert check_winner(env._state) is None


def test_reset_randomizes_agent_color() -> None:
    env = QuoridorEnv()
    # Seed once for a fixed RNG sequence; subsequent resets advance deterministically
    # (not independent random trials), so both colors are guaranteed over 64 draws.
    env.reset(seed=12345)
    seen: set[str] = set()
    for _ in range(64):
        env.reset()
        seen.add(env.agent_color)
    assert seen == {"black", "white"}


def test_step_rejects_opponent_turn() -> None:
    env = QuoridorEnv(agent_color="white", opponent="random")
    env.reset(options={"agent_color": "white"})
    env._state = QuoridorState(
        white=(8, 4),
        black=(0, 4),
        white_walls_remaining=10,
        black_walls_remaining=10,
        horizontal_walls=empty_walls(),
        vertical_walls=empty_walls(),
        current_player="black",
    )
    with pytest.raises(gym_error.InvalidAction, match="Not agent's turn"):
        env.step(0)


def test_mask_only_agent_legal_moves() -> None:
    env = QuoridorEnv(agent_color="white", opponent="random")
    env.reset(options={"agent_color": "white"})

    mask = env._mask()
    assert mask.any()
    assert mask.sum() == len(np.where(mask)[0])

    env._state = QuoridorState(
        white=(8, 4),
        black=(0, 4),
        white_walls_remaining=10,
        black_walls_remaining=10,
        horizontal_walls=empty_walls(),
        vertical_walls=empty_walls(),
        current_player="black",
    )
    assert not env._mask().any()


def test_observation_reflects_agent_color() -> None:
    env = QuoridorEnv()
    obs_white, _ = env.reset(options={"agent_color": "white"})
    assert obs_white[134] == 1.0

    obs_black, _ = env.reset(options={"agent_color": "black"})
    assert obs_black[134] == 0.0
