from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from gymnasium.spaces import Discrete, MultiBinary

from app.infrastructure.ai import ppo_loader
from app.infrastructure.ai.ppo_loader import (
    EXPECTED_PPO_OBS_SHAPE,
    PPOModelStore,
    _validate_ppo_model,
)
from quoridor.domain.actions import NUM_ACTIONS


def test_validate_ppo_model_accepts_current_spaces() -> None:
    model = SimpleNamespace(
        action_space=Discrete(NUM_ACTIONS),
        observation_space=SimpleNamespace(shape=EXPECTED_PPO_OBS_SHAPE),
    )
    _validate_ppo_model(model)


def test_validate_ppo_model_rejects_multibinary_action_space() -> None:
    model = SimpleNamespace(
        action_space=MultiBinary(NUM_ACTIONS),
        observation_space=SimpleNamespace(shape=EXPECTED_PPO_OBS_SHAPE),
    )
    with pytest.raises(RuntimeError, match="action space mismatch"):
        _validate_ppo_model(model)


def test_validate_ppo_model_rejects_wrong_action_count() -> None:
    model = SimpleNamespace(
        action_space=Discrete(132),
        observation_space=SimpleNamespace(shape=EXPECTED_PPO_OBS_SHAPE),
    )
    with pytest.raises(RuntimeError, match="action space mismatch"):
        _validate_ppo_model(model)


def test_validate_ppo_model_rejects_observation_shape() -> None:
    model = SimpleNamespace(
        action_space=Discrete(NUM_ACTIONS),
        observation_space=SimpleNamespace(shape=(128,)),
    )
    with pytest.raises(RuntimeError, match="observation space mismatch"):
        _validate_ppo_model(model)


def test_is_loadable_false_when_load_raises_keyerror(tmp_path: Path, monkeypatch) -> None:
    model_file = tmp_path / "corrupt.zip"
    model_file.write_bytes(b"placeholder")

    from sb3_contrib import MaskablePPO

    def fake_load(_path: str):
        raise KeyError("missing policy payload")

    monkeypatch.setattr(MaskablePPO, "load", fake_load)
    monkeypatch.setattr(ppo_loader, "resolve_model_path", lambda _p: model_file)

    store = PPOModelStore()
    assert store.is_loadable("models/corrupt.zip") is False
    assert "models/corrupt.zip" in store._missing_mtime


def test_is_loadable_false_for_incompatible_zip(tmp_path: Path, monkeypatch) -> None:
    model_file = tmp_path / "bad.zip"
    model_file.write_bytes(b"placeholder")

    def fake_load(_path: str):
        return SimpleNamespace(
            action_space=Discrete(132),
            observation_space=SimpleNamespace(shape=EXPECTED_PPO_OBS_SHAPE),
        )

    from sb3_contrib import MaskablePPO

    monkeypatch.setattr(MaskablePPO, "load", fake_load)
    monkeypatch.setattr(
        ppo_loader,
        "resolve_model_path",
        lambda _p: model_file,
    )

    store = PPOModelStore()
    assert store.is_loadable("models/bad.zip") is False
    assert "models/bad.zip" in store._missing_mtime


def test_is_loadable_retries_after_checkpoint_replaced(tmp_path: Path, monkeypatch) -> None:
    model_file = tmp_path / "model.zip"
    model_file.write_bytes(b"v1")

    from sb3_contrib import MaskablePPO

    def fake_load(_path: str):
        if model_file.read_bytes() == b"v1":
            return SimpleNamespace(
                action_space=Discrete(132),
                observation_space=SimpleNamespace(shape=EXPECTED_PPO_OBS_SHAPE),
            )
        return SimpleNamespace(
            action_space=Discrete(NUM_ACTIONS),
            observation_space=SimpleNamespace(shape=EXPECTED_PPO_OBS_SHAPE),
        )

    monkeypatch.setattr(MaskablePPO, "load", fake_load)
    monkeypatch.setattr(ppo_loader, "resolve_model_path", lambda _p: model_file)

    store = PPOModelStore()
    assert store.is_loadable("models/model.zip") is False
    import os
    import time

    time.sleep(0.02)
    model_file.write_bytes(b"v2")
    os.utime(model_file, None)
    assert store.is_loadable("models/model.zip") is True


def test_health_ready_marks_incompatible_expert_not_loadable(
    tmp_path: Path, monkeypatch
) -> None:
    from fastapi.testclient import TestClient

    from app.config import settings
    from app.infrastructure.ai.ppo_loader import ppo_model_store
    from app.main import create_app

    good = tmp_path / "good.zip"
    bad = tmp_path / "bad.zip"
    good.write_bytes(b"good")
    bad.write_bytes(b"bad")

    def fake_load(path: str):
        if path.endswith("bad.zip"):
            return SimpleNamespace(
                action_space=Discrete(132),
                observation_space=SimpleNamespace(shape=EXPECTED_PPO_OBS_SHAPE),
            )
        return SimpleNamespace(
            action_space=Discrete(NUM_ACTIONS),
            observation_space=SimpleNamespace(shape=EXPECTED_PPO_OBS_SHAPE),
        )

    from sb3_contrib import MaskablePPO

    monkeypatch.setattr(MaskablePPO, "load", fake_load)
    ppo_model_store._models.clear()
    ppo_model_store._missing_mtime.clear()

    monkeypatch.setattr(settings, "model_hard", str(good))
    monkeypatch.setattr(settings, "model_expert", str(bad))

    client = TestClient(create_app())
    resp = client.get("/health/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["models"]["hard"]["loadable"] is True
    assert body["models"]["expert"]["loadable"] is False
    assert body["effective_ai"]["expert"] == "unavailable"
