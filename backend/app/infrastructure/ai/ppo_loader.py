from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_BACKEND_ROOT = Path(__file__).resolve().parents[3]


def resolve_model_path(model_path: str) -> Path:
    path = Path(model_path)
    if path.is_absolute():
        return path
    for base in (_REPO_ROOT, _BACKEND_ROOT, Path.cwd()):
        candidate = base / path
        if candidate.exists():
            return candidate
    return _REPO_ROOT / path


def _dependencies_ok() -> bool:
    try:
        import sb3_contrib  # noqa: F401
    except ImportError:
        return False
    return True


EXPECTED_PPO_OBS_SHAPE: tuple[int, ...] = (135,)


def _validate_ppo_model(model: object) -> None:
    from quoridor.domain.actions import NUM_ACTIONS

    action_space = getattr(model, "action_space", None)
    action_n = getattr(action_space, "n", None)
    if action_n != NUM_ACTIONS:
        raise RuntimeError(
            "PPO action space mismatch: "
            f"expected Discrete({NUM_ACTIONS}), got {action_space!r}"
        )

    obs_space = getattr(model, "observation_space", None)
    shape = getattr(obs_space, "shape", None)
    if shape != EXPECTED_PPO_OBS_SHAPE:
        raise RuntimeError(
            "PPO observation space mismatch: "
            f"expected shape {EXPECTED_PPO_OBS_SHAPE}, got {shape!r}"
        )


class PPOModelStore:
    def __init__(self) -> None:
        self._models: dict[str, object] = {}
        self._missing: set[str] = set()

    def file_present(self, model_path: str) -> bool:
        return resolve_model_path(model_path).exists()

    def dependencies_ok(self) -> bool:
        return _dependencies_ok()

    def is_available(self, model_path: str) -> bool:
        if not self.file_present(model_path):
            return False
        if not self.dependencies_ok():
            return False
        if model_path in self._missing:
            return False
        return True

    def is_loadable(self, model_path: str) -> bool:
        if not self.is_available(model_path):
            return False
        try:
            self.get(model_path)
            return True
        except (FileNotFoundError, RuntimeError, OSError):
            return False

    def is_loaded(self, model_path: str) -> bool:
        return model_path in self._models

    def get(self, model_path: str):
        if model_path in self._models:
            return self._models[model_path]

        resolved = resolve_model_path(model_path)
        if not resolved.exists():
            self._missing.add(model_path)
            raise FileNotFoundError(f"PPO model not found: {resolved}")

        try:
            from sb3_contrib import MaskablePPO
        except ImportError as e:
            self._missing.add(model_path)
            raise RuntimeError(
                "sb3-contrib is required for Hard/Expert AI. Install with: uv sync --extra rl"
            ) from e

        logger.info("Loading PPO model from %s", resolved)
        try:
            model = MaskablePPO.load(str(resolved))
        except (OSError, ValueError) as e:
            self._missing.add(model_path)
            raise RuntimeError(f"Failed to load PPO model from {resolved}") from e

        try:
            _validate_ppo_model(model)
        except RuntimeError:
            self._missing.add(model_path)
            raise

        self._models[model_path] = model
        return model


ppo_model_store = PPOModelStore()
