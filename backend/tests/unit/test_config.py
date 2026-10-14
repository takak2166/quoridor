from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from app.config import _REPO_ROOT_ENV_FILE

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BACKEND_DIR = _REPO_ROOT / "backend"
_ENV_KEY = "QUORIDOR_AI_CONCURRENT_LIMIT"
_ENV_TEST_VALUE = "1876"


def test_repo_root_env_file_is_absolute() -> None:
    assert _REPO_ROOT_ENV_FILE.is_absolute()
    assert _REPO_ROOT_ENV_FILE.name == ".env"
    assert (_REPO_ROOT_ENV_FILE.parent / "Makefile").is_file()


def test_settings_loads_repo_root_env_when_cwd_is_backend() -> None:
    """Regression: ``make dev-backend`` runs from ``backend/``; repo-root ``.env`` must still apply."""
    env_path = _REPO_ROOT / ".env"
    backup = env_path.read_text(encoding="utf-8") if env_path.is_file() else None
    line = f"{_ENV_KEY}={_ENV_TEST_VALUE}\n"
    try:
        if backup is None:
            env_path.write_text(line, encoding="utf-8")
        else:
            env_path.write_text(backup.rstrip() + "\n" + line, encoding="utf-8")

        child_env = os.environ.copy()
        child_env.pop(_ENV_KEY, None)
        child_env.pop("QUORIDOR_", None)

        script = f"""
import os
import sys

os.chdir({str(_BACKEND_DIR)!r})
os.environ.pop({_ENV_KEY!r}, None)
sys.path.insert(0, ".")
from app.config import settings

assert settings.ai_concurrent_limit == {int(_ENV_TEST_VALUE)}, settings.ai_concurrent_limit
"""
        subprocess.run(
            [sys.executable, "-c", script],
            cwd=_BACKEND_DIR,
            env=child_env,
            check=True,
        )
    finally:
        if backup is None:
            env_path.unlink(missing_ok=True)
        else:
            env_path.write_text(backup, encoding="utf-8")
