"""Periodic evaluation callback for PPO training."""

from __future__ import annotations

import logging

from stable_baselines3.common.callbacks import BaseCallback

from app.infrastructure.rl.eval_selfplay import run_eval
from app.middleware.metrics import metrics_store

logger = logging.getLogger(__name__)


class SelfPlayEvalCallback(BaseCallback):
    """Run lightweight self-play eval every ``eval_freq`` environment steps."""

    def __init__(
        self,
        *,
        eval_freq: int,
        games: int = 4,
        difficulty_a: str = "hard",
        difficulty_b: str = "normal",
        min_win_rate: float | None = None,
    ) -> None:
        super().__init__(verbose=1)
        self.eval_freq = max(1, eval_freq)
        self.games = games
        self.difficulty_a = difficulty_a
        self.difficulty_b = difficulty_b
        self.min_win_rate = min_win_rate

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        fallback_before = metrics_store.ai_fallback_total
        try:
            result = run_eval(
                self.games,
                self.difficulty_a,
                self.difficulty_b,
                seed=int(self.num_timesteps),
            )
        except SystemExit as exc:
            logger.warning("Eval callback gate failed at step %d: %s", self.num_timesteps, exc)
            return True

        if metrics_store.ai_fallback_total != fallback_before:
            logger.warning(
                "Eval callback detected ai_fallback_total increase at step %d",
                self.num_timesteps,
            )

        logger.info(
            "Eval callback step=%d: %s win_rate=%.1f%% black=%.1f%% white=%.1f%% p99=%.0fms",
            self.num_timesteps,
            self.difficulty_a,
            result["win_rate"] * 100,
            result.get("win_rate_as_black", result["win_rate"]) * 100,
            result.get("win_rate_as_white", result["win_rate"]) * 100,
            result["p99_ms"],
        )

        if self.min_win_rate is not None and result["win_rate"] < self.min_win_rate:
            logger.warning(
                "Eval callback below min win rate %.1f%% (got %.1f%%)",
                self.min_win_rate * 100,
                result["win_rate"] * 100,
            )
        return True
