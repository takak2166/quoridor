from io import StringIO

from app.infrastructure.rl.selfplay_debug import RepetitionDebugger, format_action
from quoridor.domain.actions import Move
from quoridor.domain.state import initial_state
from quoridor.rules import apply_action


def test_format_action_move() -> None:
    assert format_action(Move(direction="up", to=(3, 4))) == "move up -> (3, 4)"


def test_repetition_debugger_logs_cycle() -> None:
    state = initial_state()
    up = Move(direction="up", to=(1, 4))
    stream = StringIO()
    debugger = RepetitionDebugger(
        1,
        difficulty_a="easy",
        difficulty_b="normal",
        colors=("black", "white"),
        stream=stream,
    )

    debugger.on_before_move(state, 1)
    debugger.on_after_move(state, "black", up, 1)
    s1 = apply_action(state, up)

    debugger.on_before_move(s1, 2)
    debugger.on_after_move(s1, "white", Move(direction="down", to=(7, 4)), 2)

    debugger.on_before_move(state, 3)

    output = stream.getvalue()
    assert "千日手" in output
    assert "eval=" in output
    assert debugger.repetition_events == 1
