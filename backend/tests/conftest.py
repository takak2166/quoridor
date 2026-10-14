from __future__ import annotations

import shutil
from pathlib import Path


def pytest_configure() -> None:
    """Seed Expert from Hard only when the Expert checkpoint is absent."""
    root = Path(__file__).resolve().parents[2]
    hard = root / "models" / "quoridor_ppo_v1.zip"
    expert = root / "models" / "quoridor_ppo_best.zip"
    if not hard.is_file():
        return
    if not expert.is_file():
        shutil.copy2(hard, expert)
