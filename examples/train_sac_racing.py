#!/usr/bin/env python3
"""Train an SAC agent on the drift environment with racing reward (S track).

Requires the ``rl`` optional dependency group::

    pip install -e ".[rl]"
    # or with uv:
    uv pip install -e ".[rl]"

Usage
-----
Train (headless)::

    python examples/train_sac_racing.py

Train with custom hyperparameters::

    python examples/train_sac_racing.py --total-timesteps 500_000 --learning-rate 1e-3

Evaluate a trained agent (with rendering)::

    python examples/train_sac_racing.py --eval --model-path logs/sac_racing/best_model.zip

Resume training from a checkpoint::

    python examples/train_sac_racing.py --resume logs/sac_racing/best_model.zip
"""

from __future__ import annotations

import argparse
import os
from typing import Any

# Register custom environments
import pg_tutorial  # noqa: F401

try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize
except ImportError as exc:
    raise ImportError("stable-baselines3 is required for this example. " 'Install it with:  pip install -e ".[rl]"') from exc


# ---------------------------------------------------------------------------
# Environment defaults
# ---------------------------------------------------------------------------

DEFAULT_ENV_KWARGS: dict[str, Any] = {
    "track_name": "s_track",
    # "reward_mode": "racing",
    "reward_mode": "racingv2",
    # Drift parameters - mild defaults; increase for harder task
    # "lateral_grip": 0.85,
    # "yaw_damping": 0.3,
    # Dynamics
    # "friction": 0.05,
    # "action_noise_std": 0.0,  # SAC adds its own exploration noise
}


# ---------------------------------------------------------------------------
# Custom callback: log racing-specific info to TensorBoard
# ---------------------------------------------------------------------------


class RacingInfoCallback(BaseCallback):
    """Log extra racing metrics (lap count, progress, slip) to TensorBoard."""

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "lap_count" in info:
                self.logger.record("racing/lap_count", info["lap_count"])
            if "best_lap_time" in info:
                self.logger.record("racing/best_lap_time", info["best_lap_time"])
            if "last_lap_time" in info:
                self.logger.record("racing/last_lap_time", info["last_lap_time"])
        return True


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace, env_id: str) -> None:
    """Run SAC training."""
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)

    n_envs = 1

    # -- training env (Monitor is added automatically by make_vec_env) ------
    vec_env = make_vec_env(
        env_id,
        n_envs=n_envs,
        env_kwargs=DEFAULT_ENV_KWARGS,
    )
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # -- eval env -----------------------------------------------------------
    eval_env = make_vec_env(
        env_id,
        n_envs=1,
        env_kwargs=DEFAULT_ENV_KWARGS,
    )
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=max(args.eval_freq // n_envs, 1),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
    )

    # -- SAC agent ----------------------------------------------------------
    if args.resume:
        print(f"Resuming training from {args.resume}")
        model = SAC.load(args.resume, env=vec_env)
        model.learning_rate = args.learning_rate
    else:
        model = SAC(
            "MlpPolicy",
            vec_env,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            gamma=args.gamma,
            tau=args.tau,
            train_freq=1,
            gradient_steps=n_envs,
            ent_coef="auto",
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=1,
            seed=args.seed,
            tensorboard_log=os.path.join(log_dir, "tb"),
        )

    print("=" * 60)
    print("Training SAC on LineFollowerDrift-v0 (racing, s_track)")
    print(f"  Total timesteps : {args.total_timesteps:,}")
    print(f"  Learning rate   : {args.learning_rate}")
    print(f"  Batch size      : {args.batch_size}")
    print(f"  Buffer size     : {args.buffer_size:,}")
    print(f"  Gamma           : {args.gamma}")
    print(f"  Log directory   : {log_dir}")
    print("=" * 60)

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[eval_callback, RacingInfoCallback()],
            log_interval=10,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        pass

    # Save final model and VecNormalize stats
    final_path = os.path.join(log_dir, "final_model")
    model.save(final_path)
    vec_env.save(os.path.join(log_dir, "vec_normalize.pkl"))
    print(f"\nModel saved to {final_path}.zip")
    print(f"Best model saved to {os.path.join(log_dir, 'best_model.zip')}")

    vec_env.close()
    eval_env.close()


# ---------------------------------------------------------------------------
# Evaluation / enjoy
# ---------------------------------------------------------------------------


def evaluate(args: argparse.Namespace, env_id: str) -> None:
    """Load a trained SAC model and run episodes with rendering."""
    model_path: str = args.model_path
    if not model_path:
        model_path = os.path.join(args.log_dir, "best_model.zip")

    vec_normalize_path = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl")

    print(f"Loading model from {model_path}")
    model = SAC.load(model_path)

    # Wrap in VecEnv + VecNormalize to match training normalisation
    eval_kwargs = {**DEFAULT_ENV_KWARGS, "render_mode": "human"}
    vec_env = make_vec_env(env_id, n_envs=1, env_kwargs=eval_kwargs)
    if os.path.isfile(vec_normalize_path):
        print(f"Loading VecNormalize stats from {vec_normalize_path}")
        vec_env = VecNormalize.load(vec_normalize_path, vec_env)
    else:
        print("Warning: VecNormalize stats not found, running without observation normalisation.")
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, training=False)
    vec_env.training = False
    vec_env.norm_reward = False

    for episode in range(args.eval_episodes):
        obs = vec_env.reset()
        total_reward: float = 0.0
        step_count: int = 0
        done = False

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = vec_env.step(action)
            vec_env.render()
            total_reward += float(reward[0])
            step_count += 1
            done = bool(dones[0])

        info = infos[0]
        laps = info.get("lap_count", 0)
        progress = info.get("total_progress", 0.0)
        print(
            f"Episode {episode + 1}/{args.eval_episodes}  |  "
            f"steps: {step_count}  |  reward: {total_reward:.1f}  |  "
            f"laps: {laps}  |  progress: {progress:.0f} segments"
        )

    vec_env.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or evaluate an SAC agent on the drift racing environment (S track).",
    )
    sub = parser.add_subparsers(dest="command")

    # -- default: train -----------------------------------------------------
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate a trained model instead of training.",
    )

    # Training hyper-parameters
    parser.add_argument("--total-timesteps", type=int, default=300_000, help="Total training timesteps (default: 300k).")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="SAC learning rate (default: 3e-4).")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size (default: 256).")
    parser.add_argument("--buffer-size", type=int, default=300_000, help="Replay buffer size (default: 300k).")
    parser.add_argument("--learning-starts", type=int, default=1_000, help="Steps before training begins (default: 1000).")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor (default: 0.99).")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft update coefficient (default: 0.005).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")

    # Evaluation
    parser.add_argument("--eval-freq", type=int, default=10_000, help="Evaluate every N training steps (default: 10k).")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Number of evaluation episodes (default: 5).")
    parser.add_argument("--model-path", type=str, default="", help="Path to a saved model .zip for evaluation.")

    # Logging / checkpointing
    parser.add_argument("--log-dir", type=str, default="logs/sac_racing", help="Directory for logs and saved models.")
    parser.add_argument("--resume", type=str, default="", help="Path to a model .zip to resume training from.")

    # Allow unused sub-commands to pass through
    del sub

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_id = "LineFollowerDrift-v0"
    if args.eval:
        try:
            evaluate(args, env_id)
        except KeyboardInterrupt:
            pass
    else:
        train(args, env_id)


if __name__ == "__main__":
    main()
