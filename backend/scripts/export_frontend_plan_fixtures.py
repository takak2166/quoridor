"""Export plan_fixtures cases to JSON for frontend parity tests."""

from __future__ import annotations

import json
from pathlib import Path

from quoridor.domain.state import QuoridorState
from tests.unit.fixtures.plan_fixtures import B1_CASES, JUMP_CASES, WALL_PATH_CASES


def _state_dto(state: QuoridorState) -> dict:
    return {
        "white": {
            "row": state.white[0],
            "col": state.white[1],
            "walls_remaining": state.white_walls_remaining,
        },
        "black": {
            "row": state.black[0],
            "col": state.black[1],
            "walls_remaining": state.black_walls_remaining,
        },
        "horizontal_walls": [list(row) for row in state.horizontal_walls],
        "vertical_walls": [list(row) for row in state.vertical_walls],
        "current_player": state.current_player,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    out_path = repo_root / "frontend" / "src" / "__fixtures__" / "planFixtures.generated.json"

    jump_cases = [
        {
            "id": case["id"],
            "state": _state_dto(case["state"]),
            "direction": case["direction"],
            "expected": [list(pair) for pair in sorted(case["expected_destinations"])],
        }
        for case in JUMP_CASES
    ]

    wall_cases = [
        {
            "id": case["id"],
            "state": _state_dto(case["state"]),
            "orientation": case["action"].orientation,
            "row": case["action"].row,
            "col": case["action"].col,
            "expectedLegal": case["expected_legal"],
        }
        for case in B1_CASES
    ]

    wall_path_cases = [
        {
            "id": case["id"],
            "state": _state_dto(case["state"]),
            "orientation": case["wall"].orientation,
            "row": case["wall"].row,
            "col": case["wall"].col,
            "expectedLegal": case["expected_legal"],
        }
        for case in WALL_PATH_CASES
    ]

    payload = {
        "jumpCases": jump_cases,
        "wallCases": wall_cases,
        "wallPathCases": wall_path_cases,
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
