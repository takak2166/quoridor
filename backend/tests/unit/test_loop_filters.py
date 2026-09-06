from quoridor.domain.actions import Move, WallSlot
from quoridor.domain.state import initial_state
from quoridor.rules import get_legal_actions


def test_filter_repeat_allows_one_retreat() -> None:
    from app.infrastructure.ai.action_mask import filter_repeat_pawn_cells

    legal = [
        Move(direction="up", to=(2, 6)),
        Move(direction="down", to=(2, 4)),
        WallSlot(orientation="horizontal", row=7, col=4),
    ]
    path = [(1, 4), (2, 5), (2, 6)]
    kept = filter_repeat_pawn_cells(legal, path, max_visits=1)
    assert Move(direction="up", to=(2, 6)) in kept
    assert Move(direction="down", to=(2, 4)) in kept


def test_filter_repeat_blocks_second_return() -> None:
    from app.infrastructure.ai.action_mask import filter_repeat_pawn_cells

    legal = [
        Move(direction="up", to=(2, 6)),
        Move(direction="left", to=(2, 5)),
        WallSlot(orientation="horizontal", row=7, col=4),
    ]
    path = [(2, 6), (2, 7), (2, 6)]
    kept = filter_repeat_pawn_cells(legal, path, max_visits=1)
    dests = {action.to for action in kept if isinstance(action, Move)}
    assert (2, 6) not in dests
    assert (2, 5) in dests
    assert any(isinstance(action, WallSlot) for action in kept)


def test_filter_repeat_keeps_legal_if_all_banned() -> None:
    from app.infrastructure.ai.action_mask import filter_repeat_pawn_cells

    only = [Move(direction="up", to=(1, 4))]
    assert filter_repeat_pawn_cells(only, [(1, 4), (1, 4)], max_visits=1) == only


def test_exclude_previous_action_keeps_alternative() -> None:
    from app.infrastructure.ai.action_mask import exclude_previous_action

    previous = Move(direction="up", to=(2, 7))
    other = Move(direction="left", to=(2, 5))
    wall = WallSlot(orientation="horizontal", row=7, col=4)
    kept = exclude_previous_action([previous, other, wall], previous)
    assert previous not in kept
    assert other in kept
    assert wall in kept


def test_env_mask_blocks_third_visit_to_same_cell() -> None:
    from app.infrastructure.ai.action_mask import legal_action_mask_agent_frame
    from app.infrastructure.rl.env import QuoridorEnv
    from quoridor.agent_frame import encode_for_viewer
    from quoridor.domain.state import QuoridorState, empty_walls

    env = QuoridorEnv(
        agent_color="black",
        opponent="random",
        reward_shaping=False,
        randomize_agent_color=False,
        opening_wall_free_plies=0,
        max_wall_candidates=None,
        repeat_pawn_max_visits=1,
    )
    env.reset(options={"agent_color": "black"})
    env._state = QuoridorState(
        white=(8, 4),
        black=(2, 4),
        white_walls_remaining=10,
        black_walls_remaining=10,
        horizontal_walls=empty_walls(),
        vertical_walls=empty_walls(),
        current_player="black",
    )
    env._agent_path = [(0, 4), (1, 4), (2, 4), (1, 4), (2, 4)]
    env._agent_plies_played = 4
    from_pos = env._state.pawn("black")
    legal = get_legal_actions(env._state)
    retreat = next(action for action in legal if isinstance(action, Move) and action.to == (1, 4))
    side = next(action for action in legal if isinstance(action, Move) and action.to == (2, 5))
    mask = env._mask()
    assert not mask[encode_for_viewer(retreat, from_pos, "black")]
    assert mask[encode_for_viewer(side, from_pos, "black")]
    raw = legal_action_mask_agent_frame(legal, "black", from_pos=from_pos)
    assert raw[encode_for_viewer(retreat, from_pos, "black")]


def test_ppo_policy_excludes_repeated_action_from_same_position() -> None:
    from app.infrastructure.ai.action_mask import estimated_agent_plies
    from app.infrastructure.ai.ppo_policy import PPOPolicy
    from quoridor.domain.state import QuoridorState, empty_walls, position_key

    state = QuoridorState(
        white=(8, 4),
        black=(2, 4),
        white_walls_remaining=10,
        black_walls_remaining=10,
        horizontal_walls=empty_walls(),
        vertical_walls=empty_walls(),
        current_player="black",
    )
    assert estimated_agent_plies(state, "black") >= 2
    previous = next(
        action
        for action in get_legal_actions(state)
        if isinstance(action, Move) and action.to == (3, 4)
    )
    other = next(
        action
        for action in get_legal_actions(state)
        if isinstance(action, Move) and action.to == (2, 5)
    )
    policy = PPOPolicy(model_path="/nonexistent/model.zip")
    policy._last_action_by_pos[("black", position_key(state))] = previous
    policy._pawn_path["black"] = [(0, 4), (1, 4), (2, 4)]
    policy._select_count["black"] = 2
    filtered = policy._apply_loop_filters(state, "black", [previous, other])
    assert previous not in filtered
    assert other in filtered


def test_ppo_policy_breaks_stall_with_greedy_race() -> None:
    from app.infrastructure.ai.ppo_policy import PPOPolicy

    state = initial_state()
    wall = WallSlot(orientation="horizontal", row=1, col=3)
    race = next(
        action
        for action in get_legal_actions(state)
        if isinstance(action, Move) and action.to == (1, 4)
    )
    policy = PPOPolicy(model_path="/nonexistent/model.zip")
    policy._select_count["black"] = 40
    policy._pawn_path["black"] = [(0, 4)]
    chosen = policy._break_stall(state, "black", [wall, race], wall)
    assert chosen == race
