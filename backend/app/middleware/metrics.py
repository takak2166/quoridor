import time
from collections.abc import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.infrastructure.persistence.game_repository import game_repository


class MetricsStore:
    def __init__(self) -> None:
        self.http_request_duration_seconds: list[float] = []
        self.ai_inference_duration_seconds: list[float] = []
        self.ai_failure_total: int = 0
        self.ai_fallback_total: int = 0
        self.ai_rollback_total: int = 0
        self.mcts_sim_count: list[int] = []

    def record_request(self, duration_s: float) -> None:
        self.http_request_duration_seconds.append(duration_s)
        if len(self.http_request_duration_seconds) > 10000:
            self.http_request_duration_seconds = self.http_request_duration_seconds[-5000:]

    def record_ai_inference(self, duration_s: float) -> None:
        self.ai_inference_duration_seconds.append(duration_s)
        if len(self.ai_inference_duration_seconds) > 10000:
            self.ai_inference_duration_seconds = self.ai_inference_duration_seconds[-5000:]

    def record_ai_failure(self) -> None:
        self.ai_failure_total += 1

    def record_ai_fallback(self) -> None:
        self.ai_fallback_total += 1

    def record_ai_rollback(self) -> None:
        self.ai_rollback_total += 1

    def record_mcts_sims(self, count: int) -> None:
        self.mcts_sim_count.append(count)
        if len(self.mcts_sim_count) > 10000:
            self.mcts_sim_count = self.mcts_sim_count[-5000:]

    def _avg(self, samples: list[float]) -> float | None:
        if not samples:
            return None
        return sum(samples) / len(samples)

    def prometheus_text(self) -> str:
        lines = [
            "# HELP active_games Active game sessions",
            "# TYPE active_games gauge",
            f"active_games {game_repository.active_count()}",
            "# HELP ai_failure_total Total AI failures",
            "# TYPE ai_failure_total counter",
            f"ai_failure_total {self.ai_failure_total}",
            "# HELP ai_fallback_total Total AI minimax fallbacks",
            "# TYPE ai_fallback_total counter",
            f"ai_fallback_total {self.ai_fallback_total}",
            "# HELP ai_rollback_total Total game state rollbacks after AI failure",
            "# TYPE ai_rollback_total counter",
            f"ai_rollback_total {self.ai_rollback_total}",
        ]
        http_avg = self._avg(self.http_request_duration_seconds)
        if http_avg is not None:
            lines.extend(
                [
                    "# HELP http_request_duration_seconds_avg Average HTTP request duration",
                    "# TYPE http_request_duration_seconds_avg gauge",
                    f"http_request_duration_seconds_avg {http_avg:.6f}",
                ]
            )
        ai_avg = self._avg(self.ai_inference_duration_seconds)
        if ai_avg is not None:
            lines.extend(
                [
                    "# HELP ai_inference_duration_seconds_avg Average AI inference duration",
                    "# TYPE ai_inference_duration_seconds_avg gauge",
                    f"ai_inference_duration_seconds_avg {ai_avg:.6f}",
                ]
            )
        if self.mcts_sim_count:
            mcts_avg = sum(self.mcts_sim_count) / len(self.mcts_sim_count)
            lines.extend(
                [
                    "# HELP mcts_sim_count_avg Average MCTS simulations per search",
                    "# TYPE mcts_sim_count_avg gauge",
                    f"mcts_sim_count_avg {mcts_avg:.2f}",
                    "# HELP mcts_sim_count_last Last MCTS simulation count",
                    "# TYPE mcts_sim_count_last gauge",
                    f"mcts_sim_count_last {self.mcts_sim_count[-1]}",
                ]
            )
        return "\n".join(lines) + "\n"


metrics_store = MetricsStore()


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable[[Request], Response]) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        metrics_store.record_request(time.perf_counter() - start)
        return response
