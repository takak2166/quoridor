from __future__ import annotations

import logging
import random
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from app.infrastructure.rl.move_resolution import resolve_ambiguous_move
from app.infrastructure.rl.reward_shaping import DEFAULT_POTENTIAL_SCALE, shaped_step_reward
from app.mappers.observation_mapper import to_observation
from quoridor.agent_frame import action_from_agent_frame, action_to_agent_frame
from quoridor.domain.actions import NUM_ACTIONS, Move, WallSlot, decode, encode
from quoridor.domain.state import Color, initial_state
from quoridor.pathfinding import SimpleDistanceCache
from quoridor.rules import apply_action, check_winner, get_legal_actions

logger = logging.getLogger(__name__)

_SUPPORTED_OPPONENTS = frozenset(
    {"random", "minimax", "very_easy", "easy", "normal", "hard", "expert"}
)


def _as_color(value: object) -> Color:
    color = str(value)
    if color not in ("black", "white"):
        raise ValueError(f"Invalid color: {value!r}")
    return color  # type: ignore[return-value]


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
        potential_scale: float = DEFAULT_POTENTIAL_SCALE,
    ) -> None:
        super().__init__()
        if opponent not in _SUPPORTED_OPPONENTS:
            raise ValueError(f"Unsupported opponent: {opponent!r}")
        self._default_agent_color: Color = _as_color(agent_color)
        self.agent_color: Color = self._default_agent_color
        self.opponent = opponent
        self.randomize_agent_color = randomize_agent_color
        self.reward_shaping = reward_shaping
        self.shaping_gamma = shaping_gamma
        self.potential_scale = potential_scale
        self._opponent_policy = None
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
            self.agent_color = _as_color(self.np_random.choice(["black", "white"]))
        else:
            self.agent_color = self._default_agent_color

        if "opponent" in options:
            opponent = str(options["opponent"])
            if opponent not in _SUPPORTED_OPPONENTS:
                raise ValueError(f"Unsupported opponent: {opponent!r}")
            self.opponent = opponent
            self._opponent_policy = None

        self._opponent_policy = None
        self._state = initial_state()
        self._cache = SimpleDistanceCache()
        self._advance_opponent_until_agent_turn(restart_on_terminal=True)
        return self._obs(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = int(action)

        if self._state.current_player != self.agent_color:
            logger.error(
                "STEP_TURN_MISS action=%s agent=%s turn=%s — ending episode",
                action,
                self.agent_color,
                self._state.current_player,
            )
            return self._obs(), -1.0, True, False, self._info()

        mask = self._mask()
        if not mask.any() or not mask[action]:
            logger.error(
                "STEP_MASK_REJECT action=%s decoded=%s agent=%s mask_sum=%s — ending episode",
                action,
                decode(action),
                self.agent_color,
                int(mask.sum()),
            )
            return self._obs(), -1.0, True, False, self._info()

        framed = decode(action)
        absolute = action_from_agent_frame(framed, self.agent_color)
        if isinstance(absolute, Move):
            move = resolve_ambiguous_move(self._state, absolute.direction, self.np_random)
        elif isinstance(absolute, WallSlot):
            move = absolute
        else:
            logger.error("STEP_UNSUPPORTED action=%s — ending episode", action)
            return self._obs(), -1.0, True, False, self._info()

        state_before = self._state.copy()
        self._state = apply_action(self._state, move)
        terminated = check_winner(self._state) is not None

        if not terminated:
            self._advance_opponent_until_agent_turn()
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

        return self._obs(), reward, terminated, False, self._info()

    def _advance_opponent_until_agent_turn(
        self,
        *,
        restart_on_terminal: bool = False,
    ) -> None:
        while True:
            if check_winner(self._state) is not None:
                if restart_on_terminal:
                    self._state = initial_state()
                    self._cache = SimpleDistanceCache()
                    continue
                return

            if self._state.current_player == self.agent_color:
                return

            opp_legal = get_legal_actions(self._state, dist_cache=self._cache)
            if not opp_legal:
                return

            opp_action = self._select_opponent_move(opp_legal)
            self._state = apply_action(self._state, opp_action)

    def _select_opponent_move(self, opp_legal: list) -> object:
        if self.opponent == "random":
            return random.choice(opp_legal)

        if self._opponent_policy is None:
            from app.infrastructure.ai.factory import ai_for_difficulty

            if self.opponent == "minimax":
                self._opponent_policy = ai_for_difficulty("easy")
            elif self.opponent in ("very_easy", "easy", "normal", "hard", "expert"):
                self._opponent_policy = ai_for_difficulty(self.opponent)
            else:
                raise ValueError(f"Unsupported opponent: {self.opponent!r}")

        return self._opponent_policy.select_move(self._state, self._state.current_player)

    def _obs(self) -> np.ndarray:
        return to_observation(self._state, self.agent_color)

    def _mask(self) -> np.ndarray:
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        if self._state.current_player != self.agent_color:
            return mask

        legal = get_legal_actions(self._state, dist_cache=self._cache)
        for a in legal:
            mask[encode(action_to_agent_frame(a, self.agent_color))] = True
        return mask

    def _info(self) -> dict[str, Any]:
        return {"action_masks": self._mask()}
