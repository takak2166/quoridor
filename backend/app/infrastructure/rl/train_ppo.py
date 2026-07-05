"""Train MaskablePPO on Quoridor environment."""

from __future__ import annotations

import argparse
from pathlib import Path

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from app.infrastructure.rl.env import QuoridorEnv


def mask_fn(env: QuoridorEnv) -> list[bool]:
    return env._mask().tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--opponent", type=str, default="normal", choices=["random", "minimax", "easy", "normal"])
    parser.add_argument("--output", type=str, default="models/quoridor_ppo_v1.zip")
    parser.add_argument("--checkpoint-dir", type=str, default="../models/checkpoints")
    parser.add_argument("--checkpoint-freq", type=int, default=10_240)
    args = parser.parse_args()

    def make_env() -> ActionMasker:
        return ActionMasker(QuoridorEnv(opponent=args.opponent), mask_fn)

    env = DummyVecEnv([make_env for _ in range(args.n_envs)])
    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=512,
        batch_size=128,
        tensorboard_log="runs/quoridor",
    )
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        CheckpointCallback(
            save_freq=max(1, args.checkpoint_freq // args.n_envs),
            save_path=str(checkpoint_dir),
            name_prefix="ppo",
        )
    ]
    model.learn(total_timesteps=args.timesteps, callback=callbacks)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    print(f"Saved model to {out}")


if __name__ == "__main__":
    main()
