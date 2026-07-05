def test_health_smoke() -> None:
    from fastapi.testclient import TestClient

    from app.main import create_app

    client = TestClient(create_app())
    assert client.get("/health/live").json()["status"] == "ok"


def test_ready_exposes_degraded_status() -> None:
    from fastapi.testclient import TestClient

    from app.main import create_app

    client = TestClient(create_app())
    resp = client.get("/health/ready")
    assert resp.status_code == 200
    assert resp.json()["status"] in {"ready", "degraded"}


def test_ready_returns_503_when_required_model_missing(monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from app.config import settings
    from app.main import create_app

    monkeypatch.setattr(settings, "require_expert_model_ready", True)
    monkeypatch.setattr(settings, "model_expert", "models/definitely_missing_expert.zip")
    client = TestClient(create_app())
    resp = client.get("/health/ready")
    assert resp.status_code == 503
    assert resp.json()["status"] == "degraded"
