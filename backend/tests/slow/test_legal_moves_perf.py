import time

import pytest

from quoridor.domain.state import initial_state
from quoridor.rules import get_legal_actions


@pytest.mark.slow
def test_legal_moves_perf_typical() -> None:
    state = initial_state()
    samples: list[float] = []
    for _ in range(100):
        get_legal_actions(state)
    for _ in range(2000):
        start = time.perf_counter()
        get_legal_actions(state)
        samples.append((time.perf_counter() - start) * 1000)
    samples.sort()
    p99 = samples[int(len(samples) * 0.99)]
    assert p99 < 50.0, f"P99 {p99}ms exceeds threshold"
