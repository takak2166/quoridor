"""Dump board state when a side has no legal moves (training fail-fast)."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quoridor.domain.actions import Action, Move, WallSlot
from quoridor.domain.state import Color, QuoridorState
from quoridor.pathfinding import both_reachable, distances
from quoridor.rules import check_winner, get_legal_actions

logger = logging.getLogger(__name__)

DEFAULT_DUMP_DIR = Path(__file__).resolve().parents[3] / "models" / "stuck_dumps"


def _serialize_action(action: Action | None) -> dict[str, Any] | None:
    if action is None:
        return None
    if isinstance(action, Move):
        return {"type": "move", "direction": action.direction, "to": list(action.to) if action.to else None}
    if isinstance(action, WallSlot):
        return {
            "type": "wall",
            "orientation": action.orientation,
            "row": action.row,
            "col": action.col,
        }
    return {"type": "unknown", "repr": repr(action)}


def log_and_dump_stuck(
    state: QuoridorState,
    *,
    stuck_side: str,
    agent_color: Color,
    opponent: str,
    last_agent_action: Action | None,
    agent_plies_played: int,
    opening_wall_free_plies: int,
    max_wall_candidates: int | None,
    dump_dir: Path | None = None,
) -> Path:
    legal = get_legal_actions(state)
    dist_w, dist_b = distances(state, None)
    payload: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"),
        "stuck_side": stuck_side,
        "agent_color": agent_color,
        "opponent": opponent,
        "agent_plies_played": agent_plies_played,
        "opening_wall_free_plies": opening_wall_free_plies,
        "max_wall_candidates": max_wall_candidates,
        "current_player": state.current_player,
        "white": list(state.white),
        "black": list(state.black),
        "white_walls_remaining": state.white_walls_remaining,
        "black_walls_remaining": state.black_walls_remaining,
        "horizontal_walls": [list(row) for row in state.horizontal_walls],
        "vertical_walls": [list(row) for row in state.vertical_walls],
        "both_reachable": both_reachable(state),
        "dist_white": dist_w,
        "dist_black": dist_b,
        "winner": check_winner(state),
        "legal_count_current": len(legal),
        "legal_actions_current": [_serialize_action(a) for a in legal[:64]],
        "last_agent_action": _serialize_action(last_agent_action),
    }

    out_dir = dump_dir or DEFAULT_DUMP_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = payload["timestamp_utc"]
    safe_side = stuck_side.replace(" ", "_")
    path = out_dir / f"stuck_{safe_side}_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2))
    logger.error("Stuck dump written to %s (%s)", path, stuck_side)
    return path
