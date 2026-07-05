from __future__ import annotations

from quoridor.domain.actions import WallSlot
from quoridor.domain.state import QuoridorState, empty_walls


def build_state(
    *,
    white: tuple[int, int] = (8, 4),
    black: tuple[int, int] = (0, 4),
    current: str = "white",
    h: frozenset[tuple[int, int]] = frozenset(),
    v: frozenset[tuple[int, int]] = frozenset(),
) -> QuoridorState:
    hw = [list(r) for r in empty_walls()]
    vw = [list(r) for r in empty_walls()]
    for row, col in h:
        hw[row][col] = True
    for row, col in v:
        vw[row][col] = True
    return QuoridorState(
        white=white,
        black=black,
        white_walls_remaining=10,
        black_walls_remaining=10,
        horizontal_walls=tuple(tuple(r) for r in hw),
        vertical_walls=tuple(tuple(r) for r in vw),
        current_player=current,  # type: ignore[arg-type]
    )


JUMP_CASES = [
    {
        "id": "J.1-STRAIGHT",
        "state": build_state(white=(5, 4), black=(4, 4), current="white"),
        "direction": "up",
        "expected_destinations": frozenset({(3, 4)}),
    },
    {
        "id": "J.2-DIAG-BOTH",
        "state": build_state(white=(5, 4), black=(4, 4), current="white", h=frozenset({(3, 4)})),
        "direction": "up",
        "expected_destinations": frozenset({(4, 3), (4, 5)}),
    },
    {
        "id": "J.3-DIAG-ONE",
        "state": build_state(
            white=(5, 4),
            black=(4, 4),
            current="white",
            h=frozenset({(3, 4)}),
            v=frozenset({(4, 3)}),
        ),
        "direction": "up",
        "expected_destinations": frozenset({(4, 5)}),
    },
    {
        "id": "J.4-BLOCK",
        "state": build_state(
            white=(5, 4),
            black=(4, 4),
            current="white",
            h=frozenset({(3, 4)}),
            v=frozenset({(4, 4)}),
        ),
        "direction": "up",
        "expected_destinations": frozenset(),
    },
    {
        "id": "J.5-EDGE",
        "state": build_state(white=(5, 0), black=(4, 0), current="white"),
        "direction": "up",
        "expected_destinations": frozenset({(4, 1)}),
    },
    {
        "id": "J.6-APPROACH",
        "state": build_state(white=(5, 4), black=(3, 4), current="black"),
        "direction": "up",
        "expected_destinations": frozenset({(4, 4)}),
    },
]

B1_CASES = [
    {
        "id": "B.1-L",
        "state": build_state(h=frozenset({(0, 2)})),
        "action": WallSlot(orientation="vertical", row=0, col=3),
        "expected_legal": True,
    },
    {
        "id": "B.1-S",
        "state": build_state(h=frozenset({(0, 3)})),
        "action": WallSlot(orientation="horizontal", row=0, col=3),
        "expected_legal": False,
    },
    {
        "id": "B.1-X",
        "state": build_state(h=frozenset({(2, 4)})),
        "action": WallSlot(orientation="vertical", row=2, col=4),
        "expected_legal": False,
    },
    {
        "id": "B.1-LINE",
        "state": build_state(h=frozenset({(3, 1)})),
        "action": WallSlot(orientation="horizontal", row=3, col=2),
        "expected_legal": False,
    },
    {
        "id": "B.1-LINE-GAP",
        "state": build_state(h=frozenset({(3, 1)})),
        "action": WallSlot(orientation="horizontal", row=3, col=3),
        "expected_legal": True,
    },
    {
        "id": "B.1-VLINE",
        "state": build_state(v=frozenset({(5, 1)})),
        "action": WallSlot(orientation="vertical", row=6, col=1),
        "expected_legal": False,
    },
]

B4_CASE = {
    "state": build_state(
        black=(0, 4),
        white=(8, 4),
        current="black",
        h=frozenset({(0, c) for c in range(8)}),
    ),
    "action": WallSlot(orientation="vertical", row=0, col=3),
    "expected_legal": False,
}

# Wall legality with path checks (pawn blocking / reachability).
WALL_PATH_CASES = [
    {
        "id": "W.PATH-NON-ADJ-LEGAL",
        "state": build_state(
            white=(5, 4),
            black=(3, 4),
            current="black",
            h=frozenset({(4, 2), (4, 3), (4, 5), (4, 6)}),
        ),
        "wall": WallSlot(orientation="vertical", row=5, col=2),
        "expected_legal": True,
    },
    {
        "id": "W.PATH-ADJ-LEGAL",
        "state": build_state(white=(4, 4), black=(4, 5), current="white"),
        "wall": WallSlot(orientation="horizontal", row=6, col=3),
        "expected_legal": True,
    },
    {
        "id": "W.PATH-BARRIER-LAST-ILLEGAL",
        "state": build_state(
            white=(8, 4),
            black=(8, 6),
            current="black",
            h=frozenset({(5, c) for c in range(7)}),
        ),
        "wall": WallSlot(orientation="horizontal", row=5, col=7),
        "expected_legal": False,
    },
    {
        "id": "W.PATH-B4-VERTICAL",
        "state": B4_CASE["state"],
        "wall": B4_CASE["action"],
        "expected_legal": False,
    },
    {
        "id": "W.TAK-101-TOP-RIGHT",
        "state": build_state(
            white=(4, 4),
            black=(1, 4),
            current="black",
            h=frozenset({(0, 0), (0, 2), (0, 5)}),
        ),
        "wall": WallSlot(orientation="horizontal", row=0, col=7),
        "expected_legal": True,
    },
]

PF_CASES = [
    {
        "id": "PF.1-OPP-BLOCK",
        "state": build_state(white=(4, 4), black=(4, 5), current="white"),
        "color": "white",
        "expected_distance": 4,
    },
    {
        "id": "PF.2-NO-PATH",
        "state": build_state(
            white=(4, 4),
            black=(4, 6),
            current="white",
            h=frozenset({(0, c) for c in range(8)}),
            v=frozenset({(r, 4) for r in range(8)}),
        ),
        "color": "white",
        "expected_distance": None,
    },
    {
        "id": "PF.3-JUMP-ADJACENT",
        "state": build_state(white=(5, 4), black=(4, 4), current="white"),
        "color": "white",
        "expected_distance": 4,
    },
    {
        "id": "PF.4-GHOST-NON-ADJ",
        "state": build_state(white=(5, 4), black=(3, 4), current="white"),
        "color": "white",
        "expected_distance": 5,
    },
]

WALL_STEP_CASES = [
    {
        "id": "W.1-HORIZONTAL-RIGHT-DOWN",
        "state": build_state(white=(3, 4), black=(0, 4), current="white", h=frozenset({(3, 3)})),
        "direction": "down",
        "expected_destinations": frozenset(),
    },
    {
        "id": "W.1-HORIZONTAL-RIGHT-UP",
        "state": build_state(white=(4, 4), black=(0, 4), current="white", h=frozenset({(3, 3)})),
        "direction": "up",
        "expected_destinations": frozenset(),
    },
]
