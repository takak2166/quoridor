from __future__ import annotations

from collections import deque
from typing import Protocol

from quoridor.board_topology import can_step
from quoridor.domain.state import GOAL_ROW, Color, QuoridorState, state_hash
from quoridor.pawn_moves import step_destinations_from


class DistanceCache(Protocol):
    def get(self, key: int) -> tuple[int | None, int | None] | None: ...
    def set(self, key: int, dist_white: int | None, dist_black: int | None) -> None: ...


class SimpleDistanceCache:
    def __init__(self, max_entries: int = 20_000) -> None:
        self._data: dict[int, tuple[int | None, int | None]] = {}
        self._max_entries = max_entries

    def get(self, key: int) -> tuple[int | None, int | None] | None:
        return self._data.get(key)

    def set(self, key: int, dist_white: int | None, dist_black: int | None) -> None:
        if len(self._data) >= self._max_entries:
            # Keep memory bounded on long-lived processes.
            for stale in list(self._data.keys())[: self._max_entries // 2]:
                self._data.pop(stale, None)
        self._data[key] = (dist_white, dist_black)


def _pawn_adjacent(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def _expand_bfs_neighbors(
    state: QuoridorState,
    color: Color,
    pos: tuple[int, int],
    opponent: tuple[int, int],
    *,
    ghost_opponent: bool,
) -> list[tuple[int, int]]:
    """Neighbors for BFS expansion.

    When ``ghost_opponent`` is True (pawns not adjacent), the opponent cell is traversable.
    Otherwise, jumps are considered from cells adjacent to the opponent pawn.
    """
    if ghost_opponent:
        neighbors: list[tuple[int, int]] = []
        r, c = pos
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nxt = (r + dr, c + dc)
            if not (0 <= nxt[0] <= 8 and 0 <= nxt[1] <= 8):
                continue
            if not can_step(state, pos, nxt):
                continue
            neighbors.append(nxt)
        return neighbors

    if _pawn_adjacent(pos, opponent):
        return list(step_destinations_from(state, color, pos))

    r, c = pos
    neighbors = []
    for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        nxt = (r + dr, c + dc)
        if nxt == opponent:
            continue
        if not (0 <= nxt[0] <= 8 and 0 <= nxt[1] <= 8):
            continue
        if not can_step(state, pos, nxt):
            continue
        neighbors.append(nxt)
    return neighbors


def _bfs_distance(state: QuoridorState, color: Color, *, for_evaluation: bool = True) -> int | None:
    start = state.pawn(color)
    goal_row = GOAL_ROW[color]
    opponent = state.pawn("black" if color == "white" else "white")
    ghost_opponent = for_evaluation and not _pawn_adjacent(start, opponent)
    visited: set[tuple[int, int]] = {start}
    queue: deque[tuple[tuple[int, int], int]] = deque([(start, 0)])

    while queue:
        pos, dist = queue.popleft()
        r, c = pos
        if r == goal_row:
            return dist
        for nxt in _expand_bfs_neighbors(state, color, pos, opponent, ghost_opponent=ghost_opponent):
            if nxt in visited:
                continue
            visited.add(nxt)
            queue.append((nxt, dist + 1))
    return None


def _bfs_distance_map(
    state: QuoridorState,
    color: Color,
    *,
    for_evaluation: bool = True,
) -> dict[tuple[int, int], int]:
    start = state.pawn(color)
    goal_row = GOAL_ROW[color]
    opponent = state.pawn("black" if color == "white" else "white")
    ghost_opponent = for_evaluation and not _pawn_adjacent(start, opponent)
    dist: dict[tuple[int, int], int] = {start: 0}
    queue: deque[tuple[int, int]] = deque([start])

    while queue:
        pos = queue.popleft()
        if pos[0] == goal_row:
            continue
        base = dist[pos]
        for nxt in _expand_bfs_neighbors(state, color, pos, opponent, ghost_opponent=ghost_opponent):
            if nxt in dist:
                continue
            dist[nxt] = base + 1
            queue.append(nxt)
    return dist


def both_reachable_one_pass(state: QuoridorState) -> bool:
    """Check both players can reach their goals with interleaved BFS."""
    opponent = {"white": state.black, "black": state.white}
    goal_row = GOAL_ROW
    found = {"white": False, "black": False}
    visited: dict[Color, set[tuple[int, int]]] = {
        "white": {state.white},
        "black": {state.black},
    }
    queues: dict[Color, deque[tuple[tuple[int, int], int]]] = {
        "white": deque([(state.white, 0)]),
        "black": deque([(state.black, 0)]),
    }

    while queues["white"] or queues["black"]:
        for color in ("white", "black"):
            if found[color] or not queues[color]:
                continue
            pos, dist = queues[color].popleft()
            r, c = pos
            if r == goal_row[color]:
                found[color] = True
                queues[color].clear()
                continue
            for nxt in _expand_bfs_neighbors(
                state,
                color,
                pos,
                opponent[color],
                ghost_opponent=False,
            ):
                if nxt in visited[color]:
                    continue
                visited[color].add(nxt)
                queues[color].append((nxt, dist + 1))
        if found["white"] and found["black"]:
            return True

    return found["white"] and found["black"]


def both_reachable(
    state: QuoridorState,
    cache: DistanceCache | None = None,
) -> bool:
    key = state_hash(state)
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            return cached[0] is not None and cached[1] is not None
    ok = both_reachable_one_pass(state)
    if cache is not None:
        if ok:
            cache.set(key, 0, 0)
        else:
            dw = _bfs_distance(state, "white", for_evaluation=False)
            db = _bfs_distance(state, "black", for_evaluation=False)
            cache.set(key, dw, db)
    return ok


def distances(
    state: QuoridorState,
    cache: DistanceCache | None = None,
) -> tuple[int | None, int | None]:
    key = state_hash(state)
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            return cached
    dw = _bfs_distance(state, "white", for_evaluation=True)
    db = _bfs_distance(state, "black", for_evaluation=True)
    if cache is not None:
        cache.set(key, dw, db)
    return dw, db


def can_reach_goal(state: QuoridorState, color: Color) -> bool:
    return _bfs_distance(state, color, for_evaluation=False) is not None


def distance_map(
    state: QuoridorState,
    color: Color,
    *,
    for_evaluation: bool = True,
) -> dict[tuple[int, int], int]:
    return _bfs_distance_map(state, color, for_evaluation=for_evaluation)
