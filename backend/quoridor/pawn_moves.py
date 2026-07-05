from __future__ import annotations

from quoridor.board_topology import can_step, is_horizontal_wall, is_vertical_wall
from quoridor.domain.actions import Direction, absolute_delta
from quoridor.domain.state import Color, QuoridorState


def _occupied(state: QuoridorState) -> set[tuple[int, int]]:
    return {state.white, state.black}


def _perpendicular(direction: Direction) -> list[Direction]:
    if direction in ("up", "down"):
        return ["left", "right"]
    return ["up", "down"]


def step_destinations_in_direction(
    state: QuoridorState,
    color: Color,
    pos: tuple[int, int],
    direction: Direction,
) -> frozenset[tuple[int, int]]:
    occupied = _occupied(state)
    dr, dc = absolute_delta(color, direction)
    adj = (pos[0] + dr, pos[1] + dc)

    if not (0 <= adj[0] <= 8 and 0 <= adj[1] <= 8):
        return frozenset()

    if not can_step(state, pos, adj):
        return frozenset()

    if adj not in occupied:
        return frozenset({adj})

    jump = (adj[0] + dr, adj[1] + dc)
    allow_straight_jump = adj[1] not in (0, 8)
    if (
        allow_straight_jump
        and 0 <= jump[0] <= 8
        and 0 <= jump[1] <= 8
        and jump not in occupied
        and can_step(state, adj, jump)
    ):
        return frozenset({jump})

    diags: set[tuple[int, int]] = set()
    for perp in _perpendicular(direction):
        pdr, pdc = absolute_delta(color, perp)
        diag = (adj[0] + pdr, adj[1] + pdc)
        if not (0 <= diag[0] <= 8 and 0 <= diag[1] <= 8):
            continue
        if diag in occupied:
            continue
        if can_step(state, adj, diag):
            if (
                diag[1] < adj[1]
                and dr != 0
                and is_horizontal_wall(state, adj[0] + dr, adj[1])
                and (
                    is_vertical_wall(state, adj[0], adj[1])
                    or is_vertical_wall(state, adj[0] - 1, adj[1])
                )
            ):
                continue
            diags.add(diag)
    return frozenset(diags)


def step_destinations_from(
    state: QuoridorState,
    color: Color,
    pos: tuple[int, int],
) -> frozenset[tuple[int, int]]:
    dests: set[tuple[int, int]] = set()
    for direction in ("up", "down", "left", "right"):
        dests |= step_destinations_in_direction(state, color, pos, direction)
    return frozenset(dests)
