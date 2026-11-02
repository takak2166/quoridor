import pytest

from quoridor.domain.actions import Move
from quoridor.domain.game import Game
from quoridor.domain.state import GOAL_ROW, QuoridorState, empty_walls


def test_play_rejects_illegal_move() -> None:
    game = Game.from_initial()
    with pytest.raises(ValueError, match="illegal move"):
        game.play(Move(direction="up", to=(0, 4)))


def test_play_rejects_illegal_wall() -> None:
    from tests.unit.fixtures.plan_fixtures import B4_CASE

    game = Game(state=B4_CASE["state"])
    with pytest.raises(ValueError, match="illegal wall"):
        game.play(B4_CASE["action"])


def test_play_rejects_after_game_over() -> None:
    state = QuoridorState(
        white=(GOAL_ROW["white"], 4),
        black=(0, 4),
        white_walls_remaining=10,
        black_walls_remaining=10,
        horizontal_walls=empty_walls(),
        vertical_walls=empty_walls(),
        current_player="black",
    )
    game = Game(state=state, winner="white")
    with pytest.raises(ValueError, match="game over"):
        game.play(Move(direction="up", to=(1, 4)))
