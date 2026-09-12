#!/usr/bin/env python3
"""Print opening / ply-13 teacher mass for a Hard PPO zip."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from app.config import settings
from app.infrastructure.ai.action_mask import (
    legal_action_mask_agent_frame,
    legal_actions_for_policy,
    policy_wall_candidate_limit,
)
from app.infrastructure.ai.ppo_loader import ppo_model_store
from app.infrastructure.rl.hunt_black_wins import parse_scoresheet, resolve_prefix_action
from app.mappers.observation_mapper import to_observation
from quoridor.agent_frame import encode_for_viewer
from quoridor.domain.actions import Action, Move, WallSlot
from quoridor.domain.game import Game
from quoridor.domain.state import Color, QuoridorState, initial_state
from quoridor.pathfinding import SimpleDistanceCache

ROOT = Path(__file__).resolve().parents[1]


def _probs(model, state: QuoridorState, color: Color) -> tuple[np.ndarray, list[Action], tuple[int, int]]:
    cache = SimpleDistanceCache()
    legal = legal_actions_for_policy(
        state,
        cache,
        policy_wall_candidate_limit(settings.ppo_max_wall_candidates),
        color=color,
        opening_wall_free_plies=2,
    )
    from_pos = state.pawn(color)
    obs = to_observation(state, color)
    mask = legal_action_mask_agent_frame(legal, color, from_pos=from_pos)
    obs_t = torch.as_tensor(obs, device=model.device).unsqueeze(0)
    mask_t = torch.as_tensor(mask, device=model.device).unsqueeze(0)
    with torch.no_grad():
        dist = model.policy.get_distribution(obs_t, action_masks=mask_t)
        probs = dist.distribution.probs.detach().cpu().numpy().reshape(-1)
    return probs, legal, from_pos


def _mass(probs: np.ndarray, from_pos: tuple[int, int], color: Color, target: Action) -> float:
    return float(probs[encode_for_viewer(target, from_pos, color)])


def _replay(text: str, n_plies: int) -> Game:
    game = Game.from_initial()
    for i, spec in enumerate(parse_scoresheet(text)[:n_plies], start=1):
        action = resolve_prefix_action(game.state, spec)
        if action is None:
            raise SystemExit(f"illegal replay at ply {i}: {spec}")
        game.play(action)
    return game


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path")
    args = parser.parse_args()
    path = Path(args.zip_path)
    if not path.is_file():
        raise SystemExit(f"missing zip: {path}")

    model = ppo_model_store.get(str(path))
    opening = initial_state()
    probs, _legal, from_pos = _probs(model, opening, "black")
    m14 = Move(direction="up", to=(1, 4))
    print(f"Black ply0 M(1,4) softmax={_mass(probs, from_pos, 'black', m14):.4f} argmax={int(probs.argmax())}")

    after_black = Game.from_initial()
    after_black.play(m14)
    probs_w, _legal_w, from_w = _probs(model, after_black.state, "white")
    m74 = Move(direction="down", to=(7, 4))
    print(f"White ply1 M(7,4) softmax={_mass(probs_w, from_w, 'white', m74):.4f} argmax={int(probs_w.argmax())}")

    white_sheet = (
        ROOT
        / "artifacts/white_wins_vs_400ms/pawn_first"
        / "white_win_001_white-pawn-second_M_7_4_M_2_4_M_6_4.txt"
    )
    if white_sheet.is_file():
        game = _replay(white_sheet.read_text(encoding="utf-8"), 3)
        assert game.state.current_player == "white"
        m64 = Move(direction="down", to=(6, 4))
        probs2, _legal2, from2 = _probs(model, game.state, "white")
        print(
            f"White ply3 M(6,4) softmax={_mass(probs2, from2, 'white', m64):.4f} "
            f"argmax={int(probs2.argmax())}"
        )

    black_sheet = ROOT / "artifacts/black_wins_vs_400ms/pawn_first/black_win_vs_400ms_pawn_M14_M15_M25.txt"
    if black_sheet.is_file():
        game = _replay(black_sheet.read_text(encoding="utf-8"), 12)
        assert game.state.current_player == "black"
        h74 = WallSlot(orientation="horizontal", row=7, col=4)
        probs_b, _legal_b, from_b = _probs(model, game.state, "black")
        print(
            f"Black ply13 H(7,4) softmax={_mass(probs_b, from_b, 'black', h74):.4f} "
            f"argmax={int(probs_b.argmax())}"
        )

    if white_sheet.is_file():
        game = _replay(white_sheet.read_text(encoding="utf-8"), 13)
        assert game.state.current_player == "white"
        teacher = WallSlot(orientation="horizontal", row=4, col=0)
        hard = WallSlot(orientation="vertical", row=5, col=0)
        probs14, _legal14, from14 = _probs(model, game.state, "white")
        print(
            f"White ply14 H(4,0) softmax={_mass(probs14, from14, 'white', teacher):.4f} "
            f"argmax={int(probs14.argmax())}"
        )
        print(
            f"White ply14 V(5,0) softmax={_mass(probs14, from14, 'white', hard):.4f}"
        )

    loss64 = ROOT / "artifacts/hard_losses_sidebit/game_003_black_loss_64.txt"
    if loss64.is_file():
        game = _replay(loss64.read_text(encoding="utf-8"), 40)
        assert game.state.current_player == "black"
        hold = Move(direction="left", to=(2, 1))
        wall = WallSlot(orientation="vertical", row=4, col=3)
        probs41, _legal41, from41 = _probs(model, game.state, "black")
        print(
            f"Black ply41 M(2,1) softmax={_mass(probs41, from41, 'black', hold):.4f} "
            f"argmax={int(probs41.argmax())}"
        )
        print(
            f"Black ply41 V(4,3) softmax={_mass(probs41, from41, 'black', wall):.4f}"
        )
        game = _replay(loss64.read_text(encoding="utf-8"), 42)
        assert game.state.current_player == "black"
        race = Move(direction="left", to=(2, 0))
        loop = Move(direction="right", to=(2, 2))
        probs43, _legal43, from43 = _probs(model, game.state, "black")
        print(
            f"Black ply43 M(2,0) softmax={_mass(probs43, from43, 'black', race):.4f} "
            f"argmax={int(probs43.argmax())}"
        )
        print(
            f"Black ply43 M(2,2) softmax={_mass(probs43, from43, 'black', loop):.4f}"
        )


if __name__ == "__main__":
    main()
