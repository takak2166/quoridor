from __future__ import annotations

from dataclasses import dataclass

from app.config import settings
from app.infrastructure.ai.mcts import mcts_search
from app.infrastructure.ai.minimax import (
    EasyMinimaxPolicy,
    MinimaxConfig,
    NormalMinimaxPolicy,
    VeryEasyMinimaxPolicy,
)
from app.infrastructure.ai.ppo_loader import ppo_model_store
from app.infrastructure.ai.ppo_policy import PPOPolicy
from app.ports.ai_policy import AiPolicy
from quoridor.domain.actions import Action
from quoridor.domain.state import Color, QuoridorState
from quoridor.rules import get_legal_actions

_POLICY_CACHE: dict[str, AiPolicy] = {}


@dataclass(frozen=True)
class PolicySpec:
    name: str
    model_path: str | None = None


def _spec_for_difficulty(difficulty: str) -> PolicySpec:
    mapping = {
        "very_easy": PolicySpec(name="very_easy_minimax"),
        "easy": PolicySpec(name="easy_minimax"),
        "normal": PolicySpec(name="normal_minimax"),
        "hard": PolicySpec(name="hard_ppo", model_path=settings.model_hard),
        "expert": PolicySpec(name="expert_mcts", model_path=settings.model_expert),
    }
    return mapping.get(difficulty, PolicySpec(name="normal_minimax"))


def ai_for_difficulty(difficulty: str) -> AiPolicy:
    spec = _spec_for_difficulty(difficulty)
    if spec.name in _POLICY_CACHE:
        return _POLICY_CACHE[spec.name]

    config = MinimaxConfig(
        time_budget_ms=400,
        max_nodes=1200,
        max_wall_candidates=10,
        two_phase_search=True,
        primary_depth=settings.minimax_depth_normal,
        fallback_depth=2,
    )
    normal_config = MinimaxConfig(
        time_budget_ms=settings.minimax_time_budget_normal_ms,
        max_nodes=settings.minimax_max_nodes_normal,
        max_wall_candidates=10,
        two_phase_search=True,
        primary_depth=settings.minimax_depth_normal,
        fallback_depth=max(2, settings.minimax_depth_normal - 2),
    )
    if spec.name == "very_easy_minimax":
        policy: AiPolicy = VeryEasyMinimaxPolicy(
            config=MinimaxConfig(time_budget_ms=400, max_nodes=500)
        )
    elif spec.name == "easy_minimax":
        policy = EasyMinimaxPolicy(config=config)
    elif spec.name == "normal_minimax":
        policy = NormalMinimaxPolicy(config=normal_config)
    elif spec.name == "hard_ppo":
        policy = HardPolicy(model_path=spec.model_path or settings.model_hard)
    elif spec.name == "expert_mcts":
        policy = ExpertMCTSPolicy(model_path=spec.model_path or settings.model_expert)
    else:
        policy = NormalMinimaxPolicy(config=config)

    _POLICY_CACHE[spec.name] = policy
    return policy


def _model_detail(model_path: str) -> dict[str, bool]:
    return {
        "file_present": ppo_model_store.file_present(model_path),
        "dependencies_ok": ppo_model_store.dependencies_ok(),
        "loadable": ppo_model_store.is_loadable(model_path),
        "loaded": ppo_model_store.is_loaded(model_path),
    }


def model_status() -> dict[str, object]:
    hard = _model_detail(settings.model_hard)
    expert = _model_detail(settings.model_expert)
    hard_available = hard["file_present"] and hard["dependencies_ok"] and hard["loadable"]
    expert_available = expert["file_present"] and expert["dependencies_ok"] and expert["loadable"]
    return {
        "hard": hard,
        "expert": expert,
        "effective_ai": {
            "very_easy": "minimax",
            "easy": "minimax",
            "normal": "minimax",
            "hard": "ppo" if hard_available else "minimax_fallback",
            "expert": "mcts" if expert_available else "unavailable",
        },
    }


@dataclass
class HardPolicy:
    model_path: str

    def __post_init__(self) -> None:
        self._ppo = PPOPolicy(self.model_path)
        self._fallback = NormalMinimaxPolicy(
            config=MinimaxConfig(
                time_budget_ms=900,
                max_nodes=2200,
                max_wall_candidates=10,
                two_phase_search=True,
                primary_depth=4,
                fallback_depth=3,
            )
        )

    def select_move(self, state: QuoridorState, color: Color) -> Action:
        if self._ppo.is_available():
            return self._ppo.select_move(state, color)
        from app.middleware.metrics import metrics_store

        metrics_store.record_ai_fallback()
        return self._fallback.select_move(state, color)

    def action_prior(self, state: QuoridorState, color: Color):
        if self._ppo.is_available():
            return self._ppo.action_prior(state, color)
        return self._fallback.action_prior(state, color)

    def value(self, state: QuoridorState, color: Color) -> float:
        if self._ppo.is_available():
            return self._ppo.value(state, color)
        return self._fallback.value(state, color)


@dataclass
class ExpertMCTSPolicy:
    model_path: str
    budget_ms: int = 450

    def __post_init__(self) -> None:
        self._ppo = PPOPolicy(self.model_path)
        self._fallback = NormalMinimaxPolicy(
            config=MinimaxConfig(
                time_budget_ms=900,
                max_nodes=2200,
                max_wall_candidates=10,
                two_phase_search=True,
                primary_depth=4,
                fallback_depth=3,
            )
        )

    def select_move(self, state: QuoridorState, color: Color) -> Action:
        legal = get_legal_actions(state)
        if not legal:
            raise RuntimeError("no legal moves")
        if len(legal) == 1:
            return legal[0]
        if not self._ppo.is_available():
            from app.middleware.metrics import metrics_store

            metrics_store.record_ai_fallback()
            return self._fallback.select_move(state, color)
        return mcts_search(
            state,
            color,
            prior_fn=self.action_prior,
            value_fn=self.value,
            legal_actions=legal,
            budget_ms=self.budget_ms,
        )

    def action_prior(self, state: QuoridorState, color: Color):
        if self._ppo.is_available():
            return self._ppo.action_prior(state, color)
        return self._fallback.action_prior(state, color)

    def value(self, state: QuoridorState, color: Color) -> float:
        if self._ppo.is_available():
            return self._ppo.value(state, color)
        return self._fallback.value(state, color)
