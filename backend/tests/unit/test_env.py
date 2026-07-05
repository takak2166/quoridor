import numpy as np

from app.infrastructure.rl.env import QuoridorEnv
from quoridor.domain.actions import Move, encode
from quoridor.domain.state import QuoridorState, empty_walls


def test_env_step_valid_mask() -> None:
    env = QuoridorEnv()
    obs, info = env.reset()
    assert obs.shape == (135,)
    assert info["action_masks"].shape == (132,)
    legal = np.where(info["action_masks"])[0]
    obs2, reward, terminated, truncated, info2 = env.step(int(legal[0]))
    assert obs2.shape == (135,)
    assert not truncated
    assert info2["action_masks"].shape == (132,)


def test_env_reward_when_agent_wins() -> None:
    env = QuoridorEnv(agent_color="white", opponent="random")
    env.reset()
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


def test_env_reward_when_opponent_wins() -> None:
    env = QuoridorEnv(agent_color="white", opponent="random")
    env.reset()
    env._state = QuoridorState(
        white=(1, 0),
        black=(7, 4),
        white_walls_remaining=10,
        black_walls_remaining=10,
        horizontal_walls=empty_walls(),
        vertical_walls=empty_walls(),
        current_player="black",
    )
    _, reward, terminated, truncated, _ = env.step(encode(Move(direction="up", to=(8, 4))))
    assert reward == -1.0
    assert terminated
    assert not truncated
