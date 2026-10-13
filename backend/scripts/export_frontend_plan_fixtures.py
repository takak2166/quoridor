"""Export plan_fixtures cases to JSON for frontend parity tests."""

from __future__ import annotations

import json
from pathlib import Path

from app.mappers.game_mapper import state_to_dto
from quoridor.domain.state import QuoridorState
from tests.unit.fixtures.plan_fixtures import B1_CASES, JUMP_CASES, WALL_PATH_CASES, WALL_STEP_CASES


def _state_dto(state: QuoridorState) -> dict:
    return state_to_dto(state).model_dump()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    out_path = repo_root / "frontend" / "src" / "__fixtures__" / "planFixtures.generated.json"

    jump_like = [*JUMP_CASES, *WALL_STEP_CASES]
    jump_cases = [
        {
            "id": case["id"],
            "state": _state_dto(case["state"]),
            "direction": case["direction"],
            "expected": [list(pair) for pair in sorted(case["expected_destinations"])],
        }
        for case in jump_like
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
