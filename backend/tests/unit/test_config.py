from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

from app.config import _BACKEND_ENV_FILE, _REPO_ROOT_ENV_FILE

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BACKEND_DIR = _REPO_ROOT / "backend"
_ENV_KEY = "QUORIDOR_AI_CONCURRENT_LIMIT"
_ENV_TEST_VALUE = "1876"


def test_repo_root_env_file_is_absolute() -> None:
    assert _REPO_ROOT_ENV_FILE.is_absolute()
    assert _REPO_ROOT_ENV_FILE.name == ".env"
    assert (_REPO_ROOT_ENV_FILE.parent / "Makefile").is_file()
    assert _BACKEND_ENV_FILE.is_absolute()
    assert _BACKEND_ENV_FILE.parent.name == "backend"


def _run_settings_probe(*, repo_env: Path, backend_env: Path, chdir_backend: bool) -> None:
    child_env = os.environ.copy()
    child_env.pop(_ENV_KEY, None)
    script = textwrap.dedent(
        f"""
        import os
        from pathlib import Path
        from pydantic_settings import BaseSettings, SettingsConfigDict

        class Probe(BaseSettings):
            model_config = SettingsConfigDict(
                env_prefix="QUORIDOR_",
                env_file=({str(backend_env)!r}, {str(repo_env)!r}),
                extra="ignore",
            )
            ai_concurrent_limit: int = 4

        if {chdir_backend!r}:
            os.chdir({str(_BACKEND_DIR)!r})
        os.environ.pop({_ENV_KEY!r}, None)
        assert Probe().ai_concurrent_limit == {int(_ENV_TEST_VALUE)}
        """
    )
    subprocess.run([sys.executable, "-c", script], env=child_env, check=True)


def test_settings_loads_repo_root_env_when_cwd_is_backend(tmp_path: Path) -> None:
    """Regression: ``make dev-backend`` runs from ``backend/``; repo-root ``.env`` must still apply."""
    repo_env = tmp_path / "root.env"
    missing_backend = tmp_path / "missing-backend.env"
    repo_env.write_text(f"{_ENV_KEY}={_ENV_TEST_VALUE}\n", encoding="utf-8")
    _run_settings_probe(repo_env=repo_env, backend_env=missing_backend, chdir_backend=True)


def test_settings_loads_backend_env_when_repo_root_missing(tmp_path: Path) -> None:
    backend_env = tmp_path / "backend.env"
    missing_root = tmp_path / "missing-root.env"
    backend_env.write_text(f"{_ENV_KEY}={_ENV_TEST_VALUE}\n", encoding="utf-8")
    _run_settings_probe(repo_env=missing_root, backend_env=backend_env, chdir_backend=False)
