import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from app.api.health import router as health_router
from app.api.v1.games import router as games_router
from app.config import settings
from app.middleware.logging import LoggingMiddleware
from app.middleware.metrics import MetricsMiddleware, metrics_store
from app.middleware.rate_limit import RateLimitMiddleware

logging.basicConfig(level=logging.INFO)


def create_app() -> FastAPI:
    if settings.env == "production" and "*" in settings.cors_origin_list:
        raise RuntimeError("CORS wildcard not allowed in production")

    app = FastAPI(title="Quoridor API", version="0.1.0")
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health_router)
    app.include_router(games_router, prefix="/api/v1")

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        return metrics_store.prometheus_text()

    return app


app = create_app()
