from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.config import settings
from app.infrastructure.ai.factory import model_status
from app.schemas import EffectiveAiStatus, HealthReadyResponse, ModelHealthDetail

router = APIRouter(tags=["health"])


@router.get("/health/live")
def live() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/ready", response_model=HealthReadyResponse)
def ready() -> HealthReadyResponse | JSONResponse:
    raw = model_status()
    models = {
        name: ModelHealthDetail(**detail)  # type: ignore[arg-type]
        for name, detail in raw.items()
        if name != "effective_ai"
    }
    effective = EffectiveAiStatus(**raw["effective_ai"])  # type: ignore[arg-type]
    degraded = effective.hard != "ppo" or effective.expert != "mcts"
    response = HealthReadyResponse(
        status="degraded" if degraded else "ready",
        models=models,
        effective_ai=effective,
    )
    hard_required_not_ready = settings.require_hard_model_ready and effective.hard != "ppo"
    expert_required_not_ready = settings.require_expert_model_ready and effective.expert != "mcts"
    if hard_required_not_ready or expert_required_not_ready:
        return JSONResponse(status_code=503, content=response.model_dump())
    return response
