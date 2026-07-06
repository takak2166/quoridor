from __future__ import annotations

import random
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from app.infrastructure.rl.move_resolution import resolve_ambiguous_move
from app.mappers.observation_mapper import to_observation
from quoridor.domain.actions import NUM_ACTIONS, Move, decode, encode
from quoridor.domain.state import Color, initial_state
from quoridor.pathfinding import SimpleDistanceCache
from quoridor.rules import apply_action, check_winner, get_legal_actions


class QuoridorEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        agent_color: Color = "white",
        opponent: str = "random",
        *,
        randomize_agent_color: bool = True,
    ) -> None:
        super().__init__()
        self._default_agent_color: Color = agent_color
        self.agent_color: Color = agent_color
        self.opponent = opponent
        self.randomize_agent_color = randomize_agent_color
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
            self.agent_color = options["agent_color"]
        elif self.randomize_agent_color:
            self.agent_color = self.np_random.choice(["black", "white"])
        else:
            self.agent_color = self._default_agent_color

        if "opponent" in options:
            self.opponent = options["opponent"]

        self._opponent_policy = None
        self._state = initial_state()
        self._cache = SimpleDistanceCache()
        self._advance_opponent_until_agent_turn(restart_on_terminal=True)
        return self._obs(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._state.current_player != self.agent_color:
            raise gym.error.InvalidAction("Not agent's turn")

        mask = self._mask()
        if not mask[action]:
            raise gym.error.InvalidAction(f"Invalid action {action}")

        move = decode(action)
        if isinstance(move, Move):
            move = resolve_ambiguous_move(self._state, move.direction, self.np_random)
        self._state = apply_action(self._state, move)
        terminated = check_winner(self._state) is not None

        if not terminated:
            self._advance_opponent_until_agent_turn()
            terminated = check_winner(self._state) is not None

        reward = 0.0
        if terminated:
            winner = check_winner(self._state)
            if winner == self.agent_color:
                reward = 1.0
            elif winner is not None:
                reward = -1.0
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
            mask[encode(a)] = True
        return mask

    def _info(self) -> dict[str, Any]:
        return {"action_masks": self._mask()}
