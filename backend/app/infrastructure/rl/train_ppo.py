"""Train MaskablePPO on Quoridor environment (agent-relative obs/actions)."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from functools import partial
from pathlib import Path

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from app.infrastructure.rl.env import QuoridorEnv
from app.infrastructure.rl.reward_shaping import DEFAULT_POTENTIAL_SCALE

logger = logging.getLogger(__name__)

DEFAULT_CURRICULUM = ("very_easy", "easy", "normal")
DEFAULT_CURRICULUM_WEIGHTS = (0.4, 0.35, 0.25)


def mask_fn(env: QuoridorEnv) -> list[bool]:
    return env._mask().tolist()


def _init_env(
    opponent: str,
    *,
    gamma: float,
    potential_scale: float,
) -> ActionMasker:
    return ActionMasker(
        QuoridorEnv(
            opponent=opponent,
            reward_shaping=True,
            shaping_gamma=gamma,
            potential_scale=potential_scale,
        ),
        mask_fn,
    )


def _parse_weights(raw: str | None, n: int) -> tuple[float, ...]:
    if raw is None:
        if n != len(DEFAULT_CURRICULUM_WEIGHTS):
            raise ValueError(
                f"Default curriculum weights apply to {len(DEFAULT_CURRICULUM)} stages; got {n}"
            )
        return DEFAULT_CURRICULUM_WEIGHTS
    weights = tuple(float(x.strip()) for x in raw.split(","))
    if len(weights) != n:
        raise ValueError(f"Expected {n} curriculum weights, got {len(weights)}")
    total = sum(weights)
    if total <= 0:
        raise ValueError("Curriculum weights must sum to a positive value")
    return tuple(w / total for w in weights)


def _build_stages(
    *,
    timesteps: int,
    curriculum: str | None,
    opponent: str,
    weights_raw: str | None,
) -> list[tuple[str, int]]:
    if curriculum:
        names = tuple(s.strip() for s in curriculum.split(",") if s.strip())
        if not names:
            raise ValueError("Curriculum must list at least one stage")
        weight_values = _parse_weights(weights_raw, len(names))
        stage_steps = [max(1, int(round(timesteps * w))) for w in weight_values]
        delta = timesteps - sum(stage_steps)
        stage_steps[-1] = max(1, stage_steps[-1] + delta)
        return list(zip(names, stage_steps, strict=True))
    return [(opponent, timesteps)]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument(
        "--opponent",
        type=str,
        default="very_easy",
        choices=["random", "minimax", "very_easy", "easy", "normal", "hard", "expert"],
    )
    parser.add_argument(
        "--curriculum",
        type=str,
        default="very_easy,easy,normal",
        help="Comma-separated opponents; empty string disables curriculum",
    )
    parser.add_argument("--curriculum-weights", type=str, default=None)
    parser.add_argument("--potential-scale", type=float, default=DEFAULT_POTENTIAL_SCALE)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--output", type=str, default="../models/agent_frame_scale8/model.zip")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="../models/agent_frame_scale8/checkpoints",
    )
    parser.add_argument("--checkpoint-freq", type=int, default=10_240)
    parser.add_argument("--vec-env", type=str, default="subproc", choices=["dummy", "subproc"])
    parser.add_argument("--tb-log", type=str, default="runs/quoridor")
    args = parser.parse_args()

    curriculum = args.curriculum.strip() or None
    stages = _build_stages(
        timesteps=args.timesteps,
        curriculum=curriculum,
        opponent=args.opponent,
        weights_raw=args.curriculum_weights,
    )

    def make_vec(opponent: str):
        factory = partial(
            _init_env,
            opponent,
            gamma=args.gamma,
            potential_scale=args.potential_scale,
        )
        factories: list[Callable[[], ActionMasker]] = [factory for _ in range(args.n_envs)]
        if args.vec_env == "subproc":
            return SubprocVecEnv(factories)
        return DummyVecEnv(factories)

    model: MaskablePPO | None = None
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for stage_i, (opponent, steps) in enumerate(stages, start=1):
        logger.info(
            "Stage %s/%s: opponent=%s timesteps=%s potential_scale=%s vec_env=%s",
            stage_i,
            len(stages),
            opponent,
            steps,
            args.potential_scale,
            args.vec_env,
        )
        env = make_vec(opponent)
        try:
            if model is None:
                model = MaskablePPO(
                    "MlpPolicy",
                    env,
                    verbose=1,
                    n_steps=512,
                    batch_size=128,
                    gamma=args.gamma,
                    tensorboard_log=args.tb_log,
                )
            else:
                model.set_env(env)

            callbacks = [
                CheckpointCallback(
                    save_freq=max(1, args.checkpoint_freq // args.n_envs),
                    save_path=str(checkpoint_dir),
                    name_prefix=f"ppo_stage{stage_i}_{opponent}",
                )
            ]
            assert model is not None
            model.learn(total_timesteps=steps, callback=callbacks, reset_num_timesteps=False)
            stage_path = checkpoint_dir / f"ppo_stage_{opponent}.zip"
            model.save(str(stage_path))
            logger.info("Saved stage checkpoint to %s", stage_path)
        finally:
            env.close()

    assert model is not None
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    print(f"Saved model to {out}")


if __name__ == "__main__":
    main()
