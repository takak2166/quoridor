from __future__ import annotations

import logging
import random
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from app.infrastructure.ai.action_mask import legal_action_mask_agent_frame, legal_actions_for_policy
from app.infrastructure.rl.action_resolution import resolve_agent_index_to_action
from app.infrastructure.rl.reward_shaping import (
    DEFAULT_REVISIT_ALPHA,
    DEFAULT_REVISIT_DECAY,
    DEFAULT_REVISIT_MAX_AGE,
    revisit_penalty,
    shaped_step_reward,
)
from app.infrastructure.rl.stuck_diagnostic import log_and_dump_stuck
from app.infrastructure.rl.white_demonstrations import is_greedy_race_action
from app.mappers.observation_mapper import to_observation
from quoridor.domain.actions import NUM_ACTIONS, Move, WallSlot, decode, is_move_index
from quoridor.domain.state import Color, initial_state
from quoridor.pathfinding import SimpleDistanceCache
from quoridor.rules import apply_action, check_winner, get_legal_actions

logger = logging.getLogger(__name__)

_SUPPORTED_OPPONENTS = frozenset(
    {"random", "minimax", "very_easy", "easy", "normal", "hard", "expert"}
)


def _as_color(value: object) -> Color:
    """Normalize agent color to a plain str (np.str_ breaks some comparisons / logs)."""
    color = str(value)
    if color not in ("black", "white"):
        raise ValueError(f"Invalid color: {value!r}")
    return color  # type: ignore[return-value]


def _normalize_opponent_mix(
    opponent: str,
    opponent_mix: tuple[tuple[str, float], ...] | None,
) -> tuple[tuple[str, float], ...] | None:
    if opponent_mix is None:
        return None
    if not opponent_mix:
        raise ValueError("opponent_mix must be non-empty when provided")
    names: list[str] = []
    weights: list[float] = []
    for name, weight in opponent_mix:
        if name not in _SUPPORTED_OPPONENTS:
            raise ValueError(f"Unsupported opponent in mix: {name!r}")
        if weight < 0:
            raise ValueError(f"opponent_mix weights must be non-negative, got {weight}")
        names.append(name)
        weights.append(float(weight))
    total = sum(weights)
    if total <= 0:
        raise ValueError("opponent_mix weights must sum to a positive value")
    return tuple((n, w / total) for n, w in zip(names, weights, strict=True))


class QuoridorEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        agent_color: Color = "white",
        opponent: str = "random",
        *,
        randomize_agent_color: bool = True,
        reward_shaping: bool = True,
        shaping_gamma: float = 0.99,
        potential_scale: float = 8.0,
        max_wall_candidates: int | None = 10,
        opening_wall_free_plies: int = 0,
        opponent_mix: tuple[tuple[str, float], ...] | None = None,
        revisit_alpha: float = DEFAULT_REVISIT_ALPHA,
        revisit_decay: float = DEFAULT_REVISIT_DECAY,
        revisit_max_age: int = DEFAULT_REVISIT_MAX_AGE,
        agent_white_prob: float = 0.5,
        imitation_bonus: float = 0.0,
    ) -> None:
        super().__init__()
        self._default_agent_color: Color = _as_color(agent_color)
        self.agent_color: Color = self._default_agent_color
        if opponent not in _SUPPORTED_OPPONENTS:
            raise ValueError(f"Unsupported opponent: {opponent!r}")
        self._default_opponent = opponent
        self.opponent = opponent
        self.opponent_mix = _normalize_opponent_mix(opponent, opponent_mix)
        self.randomize_agent_color = randomize_agent_color
        self.reward_shaping = reward_shaping
        self.shaping_gamma = shaping_gamma
        self.potential_scale = potential_scale
        self.max_wall_candidates = max_wall_candidates
        self.opening_wall_free_plies = max(0, opening_wall_free_plies)
        self.revisit_alpha = revisit_alpha
        self.revisit_decay = revisit_decay
        self.revisit_max_age = max(0, revisit_max_age)
        if not 0.0 <= agent_white_prob <= 1.0:
            raise ValueError(f"agent_white_prob must be in [0, 1], got {agent_white_prob}")
        self.agent_white_prob = float(agent_white_prob)
        self.imitation_bonus = max(0.0, float(imitation_bonus))
        self._agent_plies_played = 0
        self._agent_path: list[tuple[int, int]] = []
        self._last_agent_action: Move | WallSlot | None = None
        self._opponent_policy = None
        self._opponent_policy_name: str | None = None
        self.observation_space = spaces.Box(0.0, 1.0, shape=(135,), dtype=np.float32)
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self._state = initial_state()
        self._cache = SimpleDistanceCache()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}

        if "agent_color" in options:
            self.agent_color = _as_color(options["agent_color"])
        elif self.randomize_agent_color:
            pick_white = float(self.np_random.random()) < self.agent_white_prob
            self.agent_color = _as_color("white" if pick_white else "black")
        else:
            self.agent_color = self._default_agent_color

        if "opponent" in options:
            chosen = str(options["opponent"])
            if chosen not in _SUPPORTED_OPPONENTS:
                raise ValueError(f"Unsupported opponent: {chosen!r}")
            self.opponent = chosen
        elif self.opponent_mix is not None:
            names = [n for n, _ in self.opponent_mix]
            weights = np.array([w for _, w in self.opponent_mix], dtype=np.float64)
            self.opponent = str(self.np_random.choice(names, p=weights))
        else:
            self.opponent = self._default_opponent

        self._opponent_policy = None
        self._opponent_policy_name = None
        self._state = initial_state()
        self._cache = SimpleDistanceCache()
        self._agent_plies_played = 0
        self._last_agent_action = None
        self._agent_path = []
        self._sync_to_agent_turn(restart_on_terminal=True)
        self._assert_playable_or_raise("reset")
        self._agent_path = [self._state.pawn(self.agent_color)]
        return self._obs(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = int(action)

        if not self._is_agent_to_play():
            logger.error(
                "STEP_TURN_MISS action=%s decoded=%s agent=%s turn=%s mask_sum=%s — ending episode",
                action,
                decode(action),
                self.agent_color,
                self._state.current_player,
                int(self._mask().sum()),
            )
            return self._obs(), -1.0, True, False, self._info()

        mask = self._mask()
        if not mask.any():
            logger.error(
                "STEP_EMPTY_MASK action=%s agent=%s turn=%s — ending episode",
                action,
                self.agent_color,
                self._state.current_player,
            )
            return self._obs(), -1.0, True, False, self._info()

        if not mask[action]:
            wall_idxs = [
                int(i)
                for i in np.flatnonzero(mask)
                if not is_move_index(int(i))
            ]
            logger.error(
                "STEP_MASK_REJECT action=%s decoded=%s agent=%s turn=%s "
                "mask_sum=%s wall_idxs=%s — ending episode",
                action,
                decode(action),
                self.agent_color,
                self._state.current_player,
                int(mask.sum()),
                wall_idxs,
            )
            return self._obs(), -1.0, True, False, self._info()

        legal = legal_actions_for_policy(
            self._state,
            self._cache,
            self.max_wall_candidates,
        )
        try:
            move = resolve_agent_index_to_action(
                action,
                legal,
                self.agent_color,
                from_pos=self._state.pawn(self.agent_color),
            )
        except ValueError as exc:
            logger.error("STEP_RESOLVE_FAIL action=%s err=%s — ending episode", action, exc)
            return self._obs(), -1.0, True, False, self._info()

        state_before = self._state.copy()
        self._last_agent_action = move
        revisit = 0.0
        if isinstance(move, Move) and move.to is not None:
            revisit = revisit_penalty(
                move.to,
                self._agent_path,
                alpha=self.revisit_alpha,
                decay=self.revisit_decay,
                max_age=self.revisit_max_age,
            )
            self._agent_path.append(move.to)

        self._state = apply_action(self._state, move)
        self._agent_plies_played += 1
        terminated = check_winner(self._state) is not None

        if not terminated:
            self._sync_to_agent_turn(restart_on_terminal=False)
            terminated = check_winner(self._state) is not None

        terminal_reward = 0.0
        if terminated:
            winner = check_winner(self._state)
            if winner == self.agent_color:
                terminal_reward = 1.0
            elif winner is not None:
                terminal_reward = -1.0

        if self.reward_shaping:
            reward = shaped_step_reward(
                state_before=state_before,
                state_after=self._state,
                agent_color=self.agent_color,
                cache=self._cache,
                gamma=self.shaping_gamma,
                terminal_reward=terminal_reward,
                terminated=terminated,
                potential_scale=self.potential_scale,
            )
        else:
            reward = terminal_reward
        reward += revisit
        if (
            self.imitation_bonus > 0.0
            and self.agent_color == "white"
            and is_greedy_race_action(state_before, "white", move, self._cache)
        ):
            reward += self.imitation_bonus

        if not terminated and not self._is_agent_to_play():
            logger.error(
                "POST_STEP_DESYNC agent=%s turn=%s — ending episode",
                self.agent_color,
                self._state.current_player,
            )
            return self._obs(), -1.0, True, False, self._info()

        return self._obs(), reward, terminated, False, self._info()

    def _is_agent_to_play(self) -> bool:
        return (
            check_winner(self._state) is None
            and self._state.current_player == self.agent_color
        )

    def _sync_to_agent_turn(self, *, restart_on_terminal: bool) -> None:
        """Advance until agent to play.

        Raises RuntimeError if either side has no legal moves (fail-fast; should
        not occur under correct Quoridor rules). Never leaves the env on the
        opponent's turn without a winner when returning normally.
        """
        while True:
            if check_winner(self._state) is not None:
                if restart_on_terminal:
                    self._state = initial_state()
                    self._cache = SimpleDistanceCache()
                    continue
                return

            if self._state.current_player == self.agent_color:
                legal = get_legal_actions(self._state, dist_cache=self._cache)
                if legal:
                    return
                self._dump_stuck(f"agent {self.agent_color}")
                raise RuntimeError(
                    f"Stuck: agent {self.agent_color} has no legal moves"
                )

            opp_legal = get_legal_actions(self._state, dist_cache=self._cache)
            if not opp_legal:
                self._dump_stuck(f"opponent {self._state.current_player}")
                raise RuntimeError(
                    f"Stuck: opponent {self._state.current_player} has no legal moves"
                )

            opp_action = self._select_opponent_move(opp_legal)
            self._state = apply_action(self._state, opp_action)

    def _dump_stuck(self, stuck_side: str) -> None:
        log_and_dump_stuck(
            self._state,
            stuck_side=stuck_side,
            agent_color=self.agent_color,
            opponent=self.opponent,
            last_agent_action=self._last_agent_action,
            agent_plies_played=self._agent_plies_played,
            opening_wall_free_plies=self.opening_wall_free_plies,
            max_wall_candidates=self.max_wall_candidates,
        )

    def _assert_playable_or_raise(self, where: str) -> None:
        if check_winner(self._state) is not None:
            raise RuntimeError(f"{where}: unexpected terminal state after sync")
        if self._state.current_player != self.agent_color:
            raise RuntimeError(
                f"{where}: expected agent turn ({self.agent_color}), "
                f"got {self._state.current_player}"
            )
        if not self._mask().any():
            raise RuntimeError(f"{where}: empty action mask on agent turn")

    def _select_opponent_move(self, opp_legal: list) -> object:
        if self.opponent == "random":
            return random.choice(opp_legal)

        if self._opponent_policy is None or self._opponent_policy_name != self.opponent:
            from app.infrastructure.ai.factory import ai_for_difficulty

            if self.opponent == "minimax":
                self._opponent_policy = ai_for_difficulty("easy")
            elif self.opponent in ("very_easy", "easy", "normal", "hard", "expert"):
                self._opponent_policy = ai_for_difficulty(self.opponent)
            else:
                raise ValueError(f"Unsupported opponent: {self.opponent!r}")
            self._opponent_policy_name = self.opponent

        return self._opponent_policy.select_move(self._state, self._state.current_player)

    def _obs(self) -> np.ndarray:
        return to_observation(self._state, self.agent_color)

    def _mask(self) -> np.ndarray:
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        if self._state.current_player != self.agent_color:
            return mask
        if check_winner(self._state) is not None:
            return mask

        legal = legal_actions_for_policy(
            self._state,
            self._cache,
            self.max_wall_candidates,
        )
        if self._agent_plies_played < self.opening_wall_free_plies:
            legal = [action for action in legal if isinstance(action, Move)]
        return legal_action_mask_agent_frame(
            legal,
            self.agent_color,
            from_pos=self._state.pawn(self.agent_color),
        )

    def _info(self) -> dict[str, Any]:
        return {"action_masks": self._mask()}
