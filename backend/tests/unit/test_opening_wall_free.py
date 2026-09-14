from quoridor.domain.actions import Move, WallSlot
from quoridor.domain.state import initial_state
from quoridor.rules import apply_action, get_legal_actions


def test_estimated_opening_plies_start_at_zero() -> None:
    from app.infrastructure.ai.action_mask import estimated_agent_plies

    state = initial_state()
    assert estimated_agent_plies(state, "black") == 0
    assert estimated_agent_plies(state, "white") == 0


def test_filter_opening_walls_on_first_two_plies() -> None:
    from app.infrastructure.ai.action_mask import (
        filter_opening_wall_free,
        legal_actions_for_policy,
    )

    state = initial_state()
    legal = get_legal_actions(state)
    assert any(isinstance(action, WallSlot) for action in legal)

    filtered = filter_opening_wall_free(legal, state, "black", opening_wall_free_plies=2)
    assert filtered
    assert all(isinstance(action, Move) for action in filtered)

    policy_legal = legal_actions_for_policy(
        state,
        None,
        10,
        color="black",
        opening_wall_free_plies=2,
    )
    assert policy_legal
    assert all(isinstance(action, Move) for action in policy_legal)


def test_opening_walls_allowed_after_two_pawn_steps() -> None:
    from app.infrastructure.ai.action_mask import (
        estimated_agent_plies,
        legal_actions_for_policy,
    )

    state = initial_state()
    for _ in range(2):
        forward = next(
            action
            for action in get_legal_actions(state)
            if isinstance(action, Move) and action.to == (state.black[0] + 1, state.black[1])
        )
        state = apply_action(state, forward)
        if state.current_player == "white":
            white_fwd = next(
                action
                for action in get_legal_actions(state)
                if isinstance(action, Move) and action.to == (state.white[0] - 1, state.white[1])
            )
            state = apply_action(state, white_fwd)

    assert estimated_agent_plies(state, "black") >= 2
    legal = legal_actions_for_policy(
        state,
        None,
        10,
        color="black",
        opening_wall_free_plies=2,
    )
    assert any(isinstance(action, WallSlot) for action in legal)
