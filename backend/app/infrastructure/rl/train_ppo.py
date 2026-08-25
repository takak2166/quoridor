"""Advanced MaskablePPO trainer for Quoridor mix/revisit runs."""

from __future__ import annotations

import argparse
import logging
import random
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from app.infrastructure.rl.env import QuoridorEnv
from app.infrastructure.rl.mask_diagnostic import MaskDiagnosticVecEnv
from app.infrastructure.rl.train_notify import notify_training_finished
from app.infrastructure.rl.white_demonstrations import (
    DEFAULT_WHITE_DEMO_EPOCHS,
    DEFAULT_WHITE_DEMO_WINS,
    behavior_clone,
    collect_white_win_transitions,
)
from quoridor.domain.actions import is_move_index

logger = logging.getLogger(__name__)

DEFAULT_CURRICULUM = ("very_easy", "easy", "normal")
DEFAULT_CURRICULUM_WEIGHTS = (0.25, 0.30, 0.45)
WHITE_WIN_CURRICULUM = ("random", "very_easy", "easy", "normal")
WHITE_WIN_CURRICULUM_WEIGHTS = (0.30, 0.25, 0.22, 0.23)
WHITE_WIN_WHITE_PROBS = (0.80, 0.70, 0.60, 0.50)
DEFAULT_WHITE_WIN_IMITATION_BONUS = 0.20
DEFAULT_MIN_MOVE_PROB_MASS = 0.20
DEFAULT_POTENTIAL_SCALE = 8.0
# Agent plies per smoke game. Without a cap, a 2-cycle never sets terminated and
# ThreadPoolExecutor.__exit__ waits forever even after future.result times out.
SMOKE_MAX_AGENT_PLIES = 200

OpponentMix = tuple[tuple[str, float], ...] | None


@dataclass(frozen=True)
class CurriculumStage:
    opponent: str
    timesteps: int
    max_wall_candidates: int | None
    opponent_mix: OpponentMix = None
    agent_white_prob: float = 0.5


def mask_fn(env: QuoridorEnv) -> list[bool]:
    return env._mask().tolist()


def _init_env(
    opponent: str,
    *,
    reward_shaping: bool,
    gamma: float,
    potential_scale: float,
    max_wall_candidates: int | None,
    opening_wall_free_plies: int,
    opponent_mix: OpponentMix,
    revisit_alpha: float,
    revisit_decay: float,
    revisit_max_age: int,
    agent_white_prob: float,
    imitation_bonus: float,
) -> ActionMasker:
    return ActionMasker(
        QuoridorEnv(
            opponent=opponent,
            reward_shaping=reward_shaping,
            shaping_gamma=gamma,
            potential_scale=potential_scale,
            max_wall_candidates=max_wall_candidates,
            opening_wall_free_plies=opening_wall_free_plies,
            opponent_mix=opponent_mix,
            revisit_alpha=revisit_alpha,
            revisit_decay=revisit_decay,
            revisit_max_age=revisit_max_age,
            agent_white_prob=agent_white_prob,
            imitation_bonus=imitation_bonus,
        ),
        mask_fn,
    )


def _env_factories(
    opponent: str,
    *,
    reward_shaping: bool,
    gamma: float,
    potential_scale: float,
    max_wall_candidates: int | None,
    opening_wall_free_plies: int,
    opponent_mix: OpponentMix,
    revisit_alpha: float,
    revisit_decay: float,
    revisit_max_age: int,
    agent_white_prob: float,
    imitation_bonus: float,
    n_envs: int,
) -> list[Callable[[], ActionMasker]]:
    factory = partial(
        _init_env,
        opponent,
        reward_shaping=reward_shaping,
        gamma=gamma,
        potential_scale=potential_scale,
        max_wall_candidates=max_wall_candidates,
        opening_wall_free_plies=opening_wall_free_plies,
        opponent_mix=opponent_mix,
        revisit_alpha=revisit_alpha,
        revisit_decay=revisit_decay,
        revisit_max_age=revisit_max_age,
        agent_white_prob=agent_white_prob,
        imitation_bonus=imitation_bonus,
    )
    return [factory for _ in range(n_envs)]


def build_vec_env(
    opponent: str,
    *,
    reward_shaping: bool = True,
    gamma: float,
    potential_scale: float,
    max_wall_candidates: int | None,
    opening_wall_free_plies: int,
    opponent_mix: OpponentMix,
    revisit_alpha: float,
    revisit_decay: float,
    revisit_max_age: int,
    n_envs: int,
    vec_env: str,
    agent_white_prob: float = 0.5,
    imitation_bonus: float = 0.0,
) -> MaskDiagnosticVecEnv:
    factories = _env_factories(
        opponent,
        reward_shaping=reward_shaping,
        gamma=gamma,
        potential_scale=potential_scale,
        max_wall_candidates=max_wall_candidates,
        opening_wall_free_plies=opening_wall_free_plies,
        opponent_mix=opponent_mix,
        revisit_alpha=revisit_alpha,
        revisit_decay=revisit_decay,
        revisit_max_age=revisit_max_age,
        agent_white_prob=agent_white_prob,
        imitation_bonus=imitation_bonus,
        n_envs=n_envs,
    )
    if vec_env == "subproc":
        base = DummyVecEnv(factories) if n_envs <= 1 else SubprocVecEnv(factories)
    elif vec_env == "dummy":
        base = DummyVecEnv(factories)
    else:
        raise ValueError(f"Unknown vec_env: {vec_env!r}")
    return MaskDiagnosticVecEnv(base)


def _parse_weights(raw: str | None, n: int) -> tuple[float, ...]:
    if raw is None:
        if n != len(DEFAULT_CURRICULUM_WEIGHTS):
            raise ValueError(
                f"Default curriculum weights apply to {len(DEFAULT_CURRICULUM)} stages; got {n}"
            )
        return DEFAULT_CURRICULUM_WEIGHTS

    weights = tuple(float(x.strip()) for x in raw.split(",") if x.strip())
    if len(weights) != n:
        raise ValueError(f"Expected {n} curriculum weights, got {len(weights)}")
    total = sum(weights)
    if total <= 0:
        raise ValueError("Curriculum weights must sum to a positive value")
    return tuple(w / total for w in weights)


def _default_opponent_mix(opponent: str) -> OpponentMix:
    if opponent == "easy":
        return (("easy", 0.7), ("very_easy", 0.3))
    if opponent == "normal":
        return (("normal", 0.6), ("easy", 0.25), ("very_easy", 0.15))
    return None


def _white_win_opponent_mix(opponent: str) -> OpponentMix:
    """Keep wanderers in the mix so second-player racing can still win."""
    if opponent == "very_easy":
        return (("very_easy", 0.5), ("random", 0.5))
    if opponent == "easy":
        return (("easy", 0.4), ("very_easy", 0.3), ("random", 0.3))
    if opponent == "normal":
        return (("normal", 0.35), ("easy", 0.30), ("very_easy", 0.20), ("random", 0.15))
    return None


def _build_stages(
    *,
    timesteps: int,
    curriculum: str | None,
    opponent: str,
    weights_raw: str | None,
    max_wall_candidates: int,
    opponent_mix_overrides: dict[str, OpponentMix] | None = None,
    no_opponent_mix: bool = False,
    white_win_ramp: bool = False,
    agent_white_prob: float | None = None,
) -> list[CurriculumStage]:
    if white_win_ramp:
        names = WHITE_WIN_CURRICULUM
        default_raw = ",".join(str(weight) for weight in WHITE_WIN_CURRICULUM_WEIGHTS)
        weight_values = _parse_weights(weights_raw or default_raw, len(names))
        stage_steps = [max(1, int(round(timesteps * w))) for w in weight_values]
        delta = timesteps - sum(stage_steps)
        stage_steps[-1] = max(1, stage_steps[-1] + delta)
    elif curriculum:
        names = tuple(s.strip() for s in curriculum.split(",") if s.strip())
        if not names:
            raise ValueError("Curriculum must list at least one stage")
        weight_values = _parse_weights(weights_raw, len(names))
        stage_steps = [max(1, int(round(timesteps * w))) for w in weight_values]
        delta = timesteps - sum(stage_steps)
        stage_steps[-1] = max(1, stage_steps[-1] + delta)
    else:
        names = (opponent,)
        stage_steps = [timesteps]

    if agent_white_prob is not None:
        white_probs = (agent_white_prob,) * len(names)
    elif white_win_ramp:
        white_probs = WHITE_WIN_WHITE_PROBS
    else:
        white_probs = (0.5,) * len(names)

    mix_overrides = opponent_mix_overrides or {}
    stages: list[CurriculumStage] = []
    for name, steps, white_prob in zip(names, stage_steps, white_probs, strict=True):
        wall_k = None if name == "random" else max_wall_candidates
        if no_opponent_mix:
            mix = None
        elif name in mix_overrides:
            mix = mix_overrides[name]
        elif white_win_ramp:
            mix = _white_win_opponent_mix(name)
        else:
            mix = _default_opponent_mix(name)
        stages.append(
            CurriculumStage(
                opponent=name,
                timesteps=steps,
                max_wall_candidates=wall_k,
                opponent_mix=mix,
                agent_white_prob=white_prob,
            )
        )
    return stages


def _sb3_learn_timesteps(additional: int) -> int:
    """Value for ``MaskablePPO.learn(total_timesteps=)``.

    Stable-Baselines3 ``_setup_learn`` adds ``self.num_timesteps`` when
    ``reset_num_timesteps=False``. Pass the *additional* budget only; never
    ``current + additional``.
    """
    if additional <= 0:
        raise ValueError("additional timesteps must be positive")
    return additional


def _seed_smoke_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _predict_action(
    model: MaskablePPO,
    obs: np.ndarray,
    mask: np.ndarray,
    *,
    lock: threading.Lock | None = None,
) -> int:
    """Predict a masked action. ``lock`` serializes shared-model smoke workers."""

    def _predict() -> int:
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        if isinstance(action, np.ndarray):
            return int(action.item() if action.ndim == 0 else action[0])
        return int(action)

    if lock is None:
        return _predict()
    with lock:
        return _predict()


def _play_smoke_game(
    model: MaskablePPO,
    opponent: str,
    *,
    gamma: float,
    potential_scale: float,
    max_wall_candidates: int | None,
    opening_wall_free_plies: int,
    seed: int,
    predict_lock: threading.Lock | None = None,
) -> bool:
    env = QuoridorEnv(
        opponent=opponent,
        reward_shaping=False,
        shaping_gamma=gamma,
        potential_scale=potential_scale,
        max_wall_candidates=max_wall_candidates,
        opening_wall_free_plies=opening_wall_free_plies,
        opponent_mix=None,
        revisit_alpha=0.0,
        revisit_decay=0.0,
        revisit_max_age=0,
    )
    try:
        _seed_smoke_rng(seed)
        obs, info = env.reset(seed=seed)
        reward = 0.0
        for _ply in range(SMOKE_MAX_AGENT_PLIES):
            action = _predict_action(
                model,
                obs,
                info["action_masks"],
                lock=predict_lock,
            )
            obs, reward, terminated, _truncated, info = env.step(action)
            if terminated:
                return reward > 0
        logger.warning(
            "Smoke game vs %s hit %d-ply cap (counted as loss)",
            opponent,
            SMOKE_MAX_AGENT_PLIES,
        )
        return False
    finally:
        env.close()


def _smoke_game_won(
    future: Future[bool],
    *,
    timeout_sec: float,
    opponent: str,
) -> bool:
    """True on a finished win; timeout and errors count as losses."""
    try:
        return bool(future.result(timeout=timeout_sec))
    except FuturesTimeoutError:
        logger.warning(
            "Smoke game vs %s timed out after %.0fs (counted as loss)",
            opponent,
            timeout_sec,
        )
        return False
    except Exception:
        logger.exception("Smoke game vs %s failed (counted as loss)", opponent)
        return False


def smoke_win_rate(
    model: MaskablePPO,
    opponent: str,
    *,
    games: int,
    gamma: float,
    potential_scale: float,
    max_wall_candidates: int | None,
    opening_wall_free_plies: int,
    timeout_sec: float,
    workers: int = 4,
) -> float:
    if games <= 0:
        return 0.0

    was_training = bool(model.policy.training)
    model.policy.set_training_mode(False)
    try:
        max_workers = min(workers, games)
        # MaskablePPO.predict is not thread-safe; serialize shared-model calls.
        predict_lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _play_smoke_game,
                    model,
                    opponent,
                    gamma=gamma,
                    potential_scale=potential_scale,
                    max_wall_candidates=max_wall_candidates,
                    opening_wall_free_plies=opening_wall_free_plies,
                    seed=seed,
                    predict_lock=predict_lock,
                )
                for seed in range(games)
            ]
            wins = sum(
                _smoke_game_won(future, timeout_sec=timeout_sec, opponent=opponent)
                for future in futures
            )
    finally:
        model.policy.set_training_mode(was_training)

    return wins / games


def initial_move_probability_mass(
    model: MaskablePPO,
    *,
    gamma: float,
    potential_scale: float,
    max_wall_candidates: int | None,
    opening_wall_free_plies: int,
) -> float:
    env = QuoridorEnv(
        reward_shaping=False,
        shaping_gamma=gamma,
        potential_scale=potential_scale,
        max_wall_candidates=max_wall_candidates,
        opening_wall_free_plies=opening_wall_free_plies,
        opponent_mix=None,
        revisit_alpha=0.0,
        revisit_decay=0.0,
        revisit_max_age=0,
    )
    try:
        env.reset(options={"agent_color": "black"})
        obs = env._obs()
        mask = env._mask()
        obs_tensor = torch.as_tensor(obs, device=model.device).unsqueeze(0)
        mask_tensor = torch.as_tensor(mask, device=model.device).unsqueeze(0)
        with torch.no_grad():
            dist = model.policy.get_distribution(obs_tensor, action_masks=mask_tensor)
            probs = dist.distribution.probs.detach().cpu().numpy().reshape(-1)

        move_mass = 0.0
        for idx in np.where(mask)[0]:
            if is_move_index(int(idx)):
                move_mass += float(probs[int(idx)])
        return move_mass
    finally:
        env.close()


def log_initial_move_probability_mass(
    model: MaskablePPO,
    *,
    gamma: float,
    potential_scale: float,
    max_wall_candidates: int | None,
    opening_wall_free_plies: int,
) -> float:
    move_mass = initial_move_probability_mass(
        model,
        gamma=gamma,
        potential_scale=potential_scale,
        max_wall_candidates=max_wall_candidates,
        opening_wall_free_plies=opening_wall_free_plies,
    )
    logger.info("Initial move probability mass: %.1f%%", move_mass * 100)
    return move_mass


def enforce_must_gate(
    move_mass: float,
    *,
    threshold: float,
    output: Path,
    model: MaskablePPO,
) -> None:
    if move_mass >= threshold:
        return

    failed_path = output.with_name(f"{output.stem}.failed{output.suffix}")
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(failed_path))
    logger.error(
        "Must gate failed: move probability mass %.1f%% < %.1f%%. "
        "Saved debug checkpoint to %s (release zip not written).",
        move_mass * 100,
        threshold * 100,
        failed_path,
    )
    raise SystemExit(1)


def _checkpoint_callback(
    *,
    checkpoint_dir: Path,
    checkpoint_freq: int,
    n_envs: int,
    stage_i: int,
    opponent: str,
) -> list[CheckpointCallback]:
    return [
        CheckpointCallback(
            save_freq=max(1, checkpoint_freq // n_envs),
            save_path=str(checkpoint_dir),
            name_prefix=f"ppo_stage{stage_i}_{opponent}",
        )
    ]


def _log_stage(
    *,
    stage_i: int,
    stage_count: int,
    stage: CurriculumStage,
    timesteps: int,
    opening_wall_free_plies: int,
    potential_scale: float,
    vec_env: str,
    revisit_alpha: float,
    revisit_decay: float,
    revisit_max_age: int,
) -> None:
    logger.info(
        "Stage %d/%d: opponent=%s mix=%s timesteps=%d "
        "max_wall_candidates=%s opening_wall_free_plies=%s potential_scale=%s "
        "vec_env=%s white_prob=%.2f revisit=(alpha=%.3f,decay=%.3f,max_age=%s)",
        stage_i,
        stage_count,
        stage.opponent,
        stage.opponent_mix,
        timesteps,
        stage.max_wall_candidates,
        opening_wall_free_plies,
        potential_scale,
        vec_env,
        stage.agent_white_prob,
        revisit_alpha,
        revisit_decay,
        revisit_max_age,
    )


def _clone_white_win_demos(model: MaskablePPO, *, demo_wins: int, epochs: int) -> None:
    if demo_wins <= 0:
        return
    demos = collect_white_win_transitions(n_wins=demo_wins)
    if not demos:
        logger.warning("White-win BC skipped: no winning demonstrations")
        return
    behavior_clone(model, demos, epochs=epochs)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument(
        "--vec-env",
        type=str,
        default="subproc",
        choices=["subproc", "dummy"],
        help="Vectorized env backend (default: subproc for parallel rollouts)",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="normal",
        choices=["random", "minimax", "very_easy", "easy", "normal", "hard", "expert"],
    )
    parser.add_argument(
        "--curriculum",
        type=str,
        default="very_easy,easy,normal",
        help="Comma-separated opponents; empty string disables curriculum",
    )
    parser.add_argument(
        "--curriculum-weights",
        type=str,
        default=None,
        help="Comma-separated weights. Default for three-stage mix_revisit: 0.25,0.30,0.45",
    )
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--potential-scale", type=float, default=DEFAULT_POTENTIAL_SCALE)
    parser.add_argument("--max-wall-candidates", type=int, default=10)
    parser.add_argument("--opening-wall-free-plies", type=int, default=2)
    parser.add_argument("--revisit-alpha", type=float, default=0.150)
    parser.add_argument("--revisit-decay", type=float, default=0.500)
    parser.add_argument("--revisit-max-age", type=int, default=4)
    parser.add_argument(
        "--min-move-prob-mass",
        type=float,
        default=DEFAULT_MIN_MOVE_PROB_MASS,
        help="Must gate: minimum initial move probability mass before saving release zip",
    )
    parser.add_argument("--smoke-games", type=int, default=8)
    parser.add_argument(
        "--smoke-timeout-sec",
        type=float,
        default=120.0,
        help="Per-game timeout for stage smoke eval (seconds)",
    )
    parser.add_argument("--smoke-workers", type=int, default=4)
    parser.add_argument(
        "--smoke-hard-gate-min-win-rate",
        type=float,
        default=0.10,
        help="Minimum stage smoke win rate against non-random opponents",
    )
    parser.add_argument(
        "--smoke-hard-gate-extend-steps",
        type=int,
        default=100_000,
        help="Extra timesteps to train a stage after smoke hard-gate failure",
    )
    parser.add_argument(
        "--smoke-hard-gate-max-extends",
        type=int,
        default=3,
        help="Maximum number of smoke hard-gate extensions per stage",
    )
    parser.add_argument("--output", type=str, default="../models/mix_revisit/model.zip")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="../models/mix_revisit/checkpoints",
    )
    parser.add_argument("--checkpoint-freq", type=int, default=10_240)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Load MaskablePPO zip and continue curriculum from this checkpoint",
    )
    parser.add_argument(
        "--start-stage",
        type=int,
        default=1,
        help="1-based curriculum stage to start from (skip already-finished stages)",
    )
    parser.add_argument(
        "--no-opponent-mix",
        action="store_true",
        help="Train against the stage opponent only (no weaker-opponent mix)",
    )
    parser.add_argument(
        "--white-win-ramp",
        action="store_true",
        help=(
            "Second-player curriculum: random→very_easy→easy→normal with wanderer "
            "mix, white-biased resets, racing demos, and white imitation bonus"
        ),
    )
    parser.add_argument(
        "--white-demo-wins",
        type=int,
        default=None,
        help="Winning White racing games to clone before PPO (default: 48 with --white-win-ramp)",
    )
    parser.add_argument(
        "--white-demo-epochs",
        type=int,
        default=DEFAULT_WHITE_DEMO_EPOCHS,
        help="Behavior-cloning epochs over White-win demonstrations",
    )
    parser.add_argument(
        "--agent-white-prob",
        type=float,
        default=None,
        help="P(agent is White) on reset; overrides per-stage ramp defaults",
    )
    parser.add_argument(
        "--imitation-bonus",
        type=float,
        default=None,
        help="Extra reward when White matches the greedy racing move (default: 0.2 with --white-win-ramp)",
    )
    parser.add_argument("--tb-log", type=str, default="runs/quoridor")
    parser.add_argument(
        "--webhook-url",
        type=str,
        default=None,
        help="POST completion JSON here (overrides QUORIDOR_TRAIN_WEBHOOK_URL)",
    )
    args = parser.parse_args()

    curriculum = args.curriculum.strip() if args.curriculum is not None else None
    curriculum = curriculum or None
    if args.white_win_ramp:
        if curriculum and curriculum != ",".join(DEFAULT_CURRICULUM):
            logger.warning("--curriculum=%r ignored because --white-win-ramp is set", curriculum)
        curriculum = ",".join(WHITE_WIN_CURRICULUM)
        logger.info(
            "White-win ramp: curriculum=%s weights=%s",
            curriculum,
            args.curriculum_weights or ",".join(str(w) for w in WHITE_WIN_CURRICULUM_WEIGHTS),
        )
    elif curriculum and args.opponent != "normal":
        logger.warning("--opponent=%r ignored because --curriculum is set", args.opponent)

    demo_wins = args.white_demo_wins
    if demo_wins is None:
        demo_wins = DEFAULT_WHITE_DEMO_WINS if args.white_win_ramp else 0
    imitation_bonus = args.imitation_bonus
    if imitation_bonus is None:
        imitation_bonus = DEFAULT_WHITE_WIN_IMITATION_BONUS if args.white_win_ramp else 0.0

    stages = _build_stages(
        timesteps=args.timesteps,
        curriculum=curriculum,
        opponent=args.opponent,
        weights_raw=args.curriculum_weights,
        max_wall_candidates=args.max_wall_candidates,
        no_opponent_mix=args.no_opponent_mix,
        white_win_ramp=args.white_win_ramp,
        agent_white_prob=args.agent_white_prob,
    )

    if not stages:
        raise ValueError("At least one training stage is required")
    if args.start_stage < 1 or args.start_stage > len(stages):
        raise ValueError(
            f"--start-stage must be in 1..{len(stages)}, got {args.start_stage}"
        )
    if args.start_stage > 1:
        logger.info(
            "Skipping stages 1-%d; starting at stage %d/%d (%s, %d steps)",
            args.start_stage - 1,
            args.start_stage,
            len(stages),
            stages[args.start_stage - 1].opponent,
            stages[args.start_stage - 1].timesteps,
        )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model: MaskablePPO | None = None
    current_env: MaskDiagnosticVecEnv | None = None
    started_at = time.monotonic()
    notify_status = "failed"
    notify_error: str | None = None

    try:
        for stage_i, stage in enumerate(stages, start=1):
            if stage_i < args.start_stage:
                continue
            env_kwargs = dict(
                reward_shaping=True,
                gamma=args.gamma,
                potential_scale=args.potential_scale,
                max_wall_candidates=stage.max_wall_candidates,
                opening_wall_free_plies=args.opening_wall_free_plies,
                opponent_mix=stage.opponent_mix,
                revisit_alpha=args.revisit_alpha,
                revisit_decay=args.revisit_decay,
                revisit_max_age=args.revisit_max_age,
                agent_white_prob=stage.agent_white_prob,
                imitation_bonus=imitation_bonus,
            )

            if model is None and demo_wins > 0:
                dummy = build_vec_env(
                    stage.opponent,
                    n_envs=1,
                    vec_env="dummy",
                    **env_kwargs,
                )
                try:
                    if args.resume:
                        resume_path = Path(args.resume)
                        if not resume_path.is_file():
                            raise FileNotFoundError(
                                f"Resume checkpoint not found: {resume_path}"
                            )
                        logger.info("Resuming from %s", resume_path)
                        cloned = MaskablePPO.load(
                            str(resume_path),
                            env=dummy,
                            tensorboard_log=args.tb_log,
                        )
                    else:
                        cloned = MaskablePPO(
                            "MlpPolicy",
                            dummy,
                            verbose=1,
                            n_steps=512,
                            batch_size=128,
                            ent_coef=args.ent_coef,
                            gamma=args.gamma,
                            tensorboard_log=args.tb_log,
                        )
                    _clone_white_win_demos(
                        cloned,
                        demo_wins=demo_wins,
                        epochs=args.white_demo_epochs,
                    )
                    bc_path = checkpoint_dir / "ppo_white_bc.zip"
                    cloned.save(str(bc_path))
                    logger.info("Saved white-win BC checkpoint to %s", bc_path)
                finally:
                    dummy.close()

            env = build_vec_env(
                stage.opponent,
                n_envs=args.n_envs,
                vec_env=args.vec_env,
                **env_kwargs,
            )

            if model is None:
                if demo_wins > 0:
                    bc_path = checkpoint_dir / "ppo_white_bc.zip"
                    logger.info("Loading white-win BC checkpoint into %d envs", args.n_envs)
                    model = MaskablePPO.load(
                        str(bc_path),
                        env=env,
                        tensorboard_log=args.tb_log,
                    )
                elif args.resume:
                    resume_path = Path(args.resume)
                    if not resume_path.is_file():
                        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
                    logger.info("Resuming from %s", resume_path)
                    model = MaskablePPO.load(
                        str(resume_path),
                        env=env,
                        tensorboard_log=args.tb_log,
                    )
                else:
                    model = MaskablePPO(
                        "MlpPolicy",
                        env,
                        verbose=1,
                        n_steps=512,
                        batch_size=128,
                        ent_coef=args.ent_coef,
                        gamma=args.gamma,
                        tensorboard_log=args.tb_log,
                    )
                current_env = env
            else:
                old_env = current_env
                model.set_env(env)
                current_env = env
                if old_env is not None:
                    old_env.close()

            extend_count = 0
            additional_steps = stage.timesteps
            while True:
                reset_num_timesteps = stage_i == 1 and extend_count == 0 and not args.resume
                learn_steps = _sb3_learn_timesteps(additional_steps)
                _log_stage(
                    stage_i=stage_i,
                    stage_count=len(stages),
                    stage=stage,
                    timesteps=additional_steps,
                    opening_wall_free_plies=args.opening_wall_free_plies,
                    potential_scale=args.potential_scale,
                    vec_env=args.vec_env,
                    revisit_alpha=args.revisit_alpha,
                    revisit_decay=args.revisit_decay,
                    revisit_max_age=args.revisit_max_age,
                )
                if not reset_num_timesteps:
                    current = int(model.num_timesteps)
                    logger.info(
                        "Continue learn: current=%d additional=%d sb3_target=%d",
                        current,
                        additional_steps,
                        current + additional_steps,
                    )
                callbacks = _checkpoint_callback(
                    checkpoint_dir=checkpoint_dir,
                    checkpoint_freq=args.checkpoint_freq,
                    n_envs=args.n_envs,
                    stage_i=stage_i,
                    opponent=stage.opponent,
                )
                model.learn(
                    total_timesteps=learn_steps,
                    callback=callbacks,
                    reset_num_timesteps=reset_num_timesteps,
                )

                stage_path = checkpoint_dir / f"ppo_stage{stage_i}_{stage.opponent}.zip"
                model.save(str(stage_path))
                logger.info("Saved stage checkpoint to %s", stage_path)

                if args.smoke_games <= 0:
                    break

                win_rate = smoke_win_rate(
                    model,
                    stage.opponent,
                    games=args.smoke_games,
                    gamma=args.gamma,
                    potential_scale=args.potential_scale,
                    max_wall_candidates=stage.max_wall_candidates,
                    opening_wall_free_plies=args.opening_wall_free_plies,
                    timeout_sec=args.smoke_timeout_sec,
                    workers=args.smoke_workers,
                )
                logger.info(
                    "Smoke eval vs %s (%d games, timeout=%.0fs): win_rate=%.1f%%",
                    stage.opponent,
                    args.smoke_games,
                    args.smoke_timeout_sec,
                    win_rate * 100,
                )

                if (
                    stage.opponent == "random"
                    or win_rate >= args.smoke_hard_gate_min_win_rate
                ):
                    break

                logger.error(
                    "Smoke hard gate failed: vs %s win_rate %.1f%% < %.1f%% (aborting training)",
                    stage.opponent,
                    win_rate * 100,
                    args.smoke_hard_gate_min_win_rate * 100,
                )
                if extend_count >= args.smoke_hard_gate_max_extends:
                    raise SystemExit(1)

                logger.warning(
                    "Smoke hard gate failed for %s; extending stage by %d timesteps and retrying",
                    stage.opponent,
                    args.smoke_hard_gate_extend_steps,
                )
                extend_count += 1
                additional_steps = args.smoke_hard_gate_extend_steps

        assert model is not None
        final_stage = stages[-1]
        move_mass = log_initial_move_probability_mass(
            model,
            gamma=args.gamma,
            potential_scale=args.potential_scale,
            max_wall_candidates=final_stage.max_wall_candidates,
            opening_wall_free_plies=args.opening_wall_free_plies,
        )

        out = Path(args.output)
        enforce_must_gate(
            move_mass,
            threshold=args.min_move_prob_mass,
            output=out,
            model=model,
        )

        out.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(out))
        print(f"Saved model to {out}")
        notify_status = "success"
    except SystemExit as exc:
        code = exc.code
        notify_error = f"SystemExit({code})"
        if code not in (0, None):
            notify_status = "failed"
        else:
            notify_status = "success"
        raise
    except Exception as exc:
        notify_status = "failed"
        notify_error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if current_env is not None:
            current_env.close()
        notify_training_finished(
            webhook_url=args.webhook_url,
            status=notify_status,
            output=args.output,
            curriculum=curriculum,
            timesteps=args.timesteps,
            elapsed_sec=time.monotonic() - started_at,
            error=notify_error,
        )


if __name__ == "__main__":
    main()
