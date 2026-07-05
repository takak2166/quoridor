import time
from collections import defaultdict
from collections.abc import Callable
from threading import Lock

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.config import settings
from app.schemas import ErrorDetail, ErrorResponse


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: object) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._game_hits: dict[str, list[float]] = defaultdict(list)
        self._move_hits: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()
        self._last_cleanup = 0.0

    def _client_ip(self, request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if settings.trust_forwarded_for and forwarded:
            return forwarded.split(",")[0].strip()
        if request.client:
            return request.client.host
        return "unknown"

    def _check(self, bucket: dict[str, list[float]], key: str, limit: int, window: float = 60.0) -> bool:
        now = time.time()
        with self._lock:
            hits = [t for t in bucket[key] if now - t < window]
            if len(hits) >= limit:
                bucket[key] = hits
                self._cleanup_empty_buckets(now)
                return False
            hits.append(now)
            bucket[key] = hits
            self._cleanup_empty_buckets(now)
            return True

    def _cleanup_empty_buckets(self, now: float, window: float = 60.0) -> None:
        if now - self._last_cleanup < 30:
            return
        self._last_cleanup = now
        for bucket in (self._game_hits, self._move_hits):
            for key in list(bucket.keys()):
                bucket[key] = [t for t in bucket[key] if now - t < window]
            empty_keys = [k for k, v in bucket.items() if not v]
            for key in empty_keys:
                del bucket[key]

    async def dispatch(self, request: Request, call_next: Callable[[Request], Response]) -> Response:
        ip = self._client_ip(request)
        path = request.url.path
        if request.method == "POST" and path == "/api/v1/games":
            if not self._check(self._game_hits, ip, settings.rate_limit_games_per_min):
                return self._rate_limited()
        if request.method == "POST" and path.endswith("/moves"):
            if not self._check(self._move_hits, ip, settings.rate_limit_moves_per_min):
                return self._rate_limited()
        return await call_next(request)

    def _rate_limited(self) -> JSONResponse:
        body = ErrorResponse(
            error=ErrorDetail(code="RATE_LIMITED", message="Too many requests", details=None)
        )
        return JSONResponse(
            status_code=429,
            content={"detail": body.model_dump()},
            headers={"Retry-After": "60"},
        )
