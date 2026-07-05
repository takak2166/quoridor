import json
import logging
import time
import uuid
from collections.abc import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("quoridor.access")


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable[[Request], Response]) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        start = time.perf_counter()
        response = await call_next(request)
        latency_ms = (time.perf_counter() - start) * 1000
        log_entry = {
            "request_id": request_id,
            "endpoint": f"{request.method} {request.url.path}",
            "latency_ms": round(latency_ms, 2),
            "status_code": response.status_code,
        }
        if hasattr(request.state, "game_id"):
            log_entry["game_id"] = request.state.game_id
        if hasattr(request.state, "difficulty"):
            log_entry["difficulty"] = request.state.difficulty
        if hasattr(request.state, "error_code"):
            log_entry["error"] = {"code": request.state.error_code}
        logger.info(json.dumps(log_entry))
        response.headers["X-Request-ID"] = request_id
        return response
