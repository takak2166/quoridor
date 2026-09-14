from __future__ import annotations

import shutil
from pathlib import Path


def pytest_configure() -> None:
    """Expert is out of v0.2; tests reuse the official Hard zip as its prior."""
    root = Path(__file__).resolve().parents[2]
    hard = root / "models" / "quoridor_ppo_v1.zip"
    expert = root / "models" / "quoridor_ppo_best.zip"
    if hard.is_file() and not expert.is_file():
        shutil.copy2(hard, expert)
