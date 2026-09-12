from pathlib import Path

from app.infrastructure.rl.dagger_losses import first_divergence, prefix_from_csv, unique_divergences
from app.infrastructure.rl.white_demonstrations import load_teacher_book
from quoridor.domain.actions import Move
from quoridor.domain.state import initial_state, position_key


def test_prefix_from_csv_roundtrip() -> None:
    assert prefix_from_csv("") == ()
    assert prefix_from_csv("M(1, 4),M(7, 4)") == (("M", 1, 4), ("M", 7, 4))


def test_first_divergence_uncovered_on_empty_book() -> None:
    from app.infrastructure.rl.white_demonstrations import TeacherBook

    book = TeacherBook(actions={})
    found = first_divergence("scoresheet=M(1, 4),M(7, 4)", book, "black")
    assert found is not None
    assert found.reason == "uncovered"
    assert found.ply == 1
    assert found.prefix == ()


def test_first_divergence_off_book_after_teacher_opening() -> None:
    from app.infrastructure.rl.white_demonstrations import TeacherBook

    opening = initial_state()
    book = TeacherBook(
        actions={("black", position_key(opening)): Move(direction="up", to=(1, 4))}
    )
    found = first_divergence("scoresheet=M(0, 5),M(7, 4)", book, "black")
    assert found is not None
    assert found.reason == "off_book"
    assert found.prefix == ()


def test_unique_divergences_ranks_by_count() -> None:
    from app.infrastructure.rl.white_demonstrations import TeacherBook

    opening = initial_state()
    book = TeacherBook(
        actions={("black", position_key(opening)): Move(direction="up", to=(1, 4))}
    )
    texts = [
        "scoresheet=M(1, 4),M(7, 4),M(1, 5)",
        "scoresheet=M(1, 4),M(7, 4),M(1, 5)",
        "scoresheet=M(0, 5),M(7, 4)",
    ]
    ranked = unique_divergences(texts, book, "black", limit=2)
    assert ranked[0][1] == 2
    assert ranked[0][0].prefix == (("M", 1, 4), ("M", 7, 4))
    assert ranked[1][1] == 1
    assert ranked[1][0].prefix == ()


def test_teacher_focus_repeats_unique_off_book_only() -> None:
    from app.infrastructure.rl.dagger_losses import teacher_focus_transitions
    from app.infrastructure.rl.white_demonstrations import TeacherBook
    from quoridor.agent_frame import encode_for_viewer

    opening = initial_state()
    teacher = Move(direction="up", to=(1, 4))
    book = TeacherBook(actions={("black", position_key(opening)): teacher})
    texts = [
        "scoresheet=M(0, 5),M(7, 4)",
        "scoresheet=M(0, 5),M(7, 4)",
        "scoresheet=M(1, 4),M(7, 4)",
    ]
    focused = teacher_focus_transitions(texts, book, "black", repeat=4)
    assert len(focused) == 4
    expected = encode_for_viewer(teacher, opening.black, "black")
    assert {item.action for item in focused} == {expected}


def test_load_hard_loss_texts_filters_by_color(tmp_path: Path) -> None:
    from app.infrastructure.rl.dagger_losses import load_hard_loss_texts

    (tmp_path / "game_001_white_loss_49.txt").write_text(
        "tag=eval-white-loss\nwinner=black\nscoresheet=M(1, 4),M(7, 4)\n",
        encoding="utf-8",
    )
    (tmp_path / "game_002_black_loss_64.txt").write_text(
        "tag=eval-black-loss\nwinner=white\nscoresheet=M(1, 4),M(7, 4)\n",
        encoding="utf-8",
    )
    (tmp_path / "notes.txt").write_text("not a loss\n", encoding="utf-8")
    assert len(load_hard_loss_texts(tmp_path, "white")) == 1
    assert len(load_hard_loss_texts(tmp_path, "black")) == 1


def test_policy_wall_candidate_limit_treats_non_positive_as_open() -> None:
    from app.infrastructure.ai.action_mask import policy_wall_candidate_limit

    assert policy_wall_candidate_limit(None) is None
    assert policy_wall_candidate_limit(0) is None
    assert policy_wall_candidate_limit(-1) is None
    assert policy_wall_candidate_limit(10) == 10


def test_policy_wall_cap_zero_keeps_teacher_setup_wall() -> None:
    from app.infrastructure.ai.action_mask import legal_actions_for_policy
    from app.infrastructure.rl.hunt_black_wins import parse_scoresheet, resolve_prefix_action
    from quoridor.domain.actions import WallSlot
    from quoridor.domain.game import Game
    from quoridor.pathfinding import SimpleDistanceCache

    prefix = (
        "M(1, 4),M(7, 4),M(2, 4),M(6, 4),M(3, 4),H(6,3),M(4, 4),"
        "H(6,5),M(5, 4),V(5,5),M(6, 3),H(6,1),M(6, 2)"
    )
    game = Game.from_initial()
    for spec in parse_scoresheet(f"scoresheet={prefix}"):
        game.play(resolve_prefix_action(game.state, spec))
    teacher = WallSlot(orientation="horizontal", row=4, col=0)
    hard = WallSlot(orientation="vertical", row=5, col=0)
    cache = SimpleDistanceCache()
    capped = legal_actions_for_policy(game.state, cache, 10, color="white")
    full = legal_actions_for_policy(game.state, cache, 0, color="white")
    assert hard in capped
    assert teacher not in capped
    assert teacher in full


def test_teacher_book_and_black_load_accept_comma_paths(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures" / "black_win_vs_normal_m14.txt"
    other = tmp_path / "extra"
    other.mkdir()
    (other / "copy.txt").write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")
    book = load_teacher_book(black_source=f"{fixture},{other}")
    assert book.action_for(initial_state(), "black") is not None

    from app.infrastructure.rl.white_demonstrations import load_black_win_transitions

    loaded = load_black_win_transitions(f"{fixture},{other}", upsample_m14=1)
    assert len(loaded) == 32
