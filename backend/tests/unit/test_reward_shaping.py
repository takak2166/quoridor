from quoridor.domain.actions import Move, WallSlot, encode
from quoridor.domain.state import QuoridorState, empty_walls, initial_state
from quoridor.pathfinding import SimpleDistanceCache
from quoridor.rules import apply_action, get_legal_actions

from app.infrastructure.rl.reward_shaping import potential, shaped_step_reward


def test_potential_increases_when_agent_advances() -> None:
    cache = SimpleDistanceCache()
    before = initial_state()
    after = apply_action(before, Move(direction="down", to=(7, 4)))
    phi_before = potential(before, "black", cache)
    phi_after = potential(after, "black", cache)
    assert phi_after > phi_before


def test_potential_zero_when_terminated() -> None:
    cache = SimpleDistanceCache()
    state = QuoridorState(
        white=(1, 4),
        black=(8, 0),
        white_walls_remaining=10,
        black_walls_remaining=10,
        horizontal_walls=empty_walls(),
        vertical_walls=empty_walls(),
        current_player="white",
    )
    assert potential(state, "white", cache, terminated=True) == 0.0


def test_shaped_reward_can_be_negative_for_useless_wall() -> None:
    cache = SimpleDistanceCache()
    state = initial_state()
    legal = get_legal_actions(state, dist_cache=cache)
    wall = next(a for a in legal if isinstance(a, WallSlot))
    after = apply_action(state, wall)
    reward = shaped_step_reward(
        state_before=state,
        state_after=after,
        agent_color="black",
        cache=cache,
        gamma=0.99,
        terminal_reward=0.0,
        terminated=False,
    )
    assert reward < 0.0


def test_shaped_terminal_reward_matches_terminal_only_when_phi_zero() -> None:
    cache = SimpleDistanceCache()
    state = QuoridorState(
        white=(1, 4),
        black=(8, 0),
        white_walls_remaining=10,
        black_walls_remaining=10,
        horizontal_walls=empty_walls(),
        vertical_walls=empty_walls(),
        current_player="white",
    )
    reward = shaped_step_reward(
        state_before=state,
        state_after=state,
        agent_color="white",
        cache=cache,
        gamma=0.99,
        terminal_reward=1.0,
        terminated=True,
    )
    assert reward == 1.0
