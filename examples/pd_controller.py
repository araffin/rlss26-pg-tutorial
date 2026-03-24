#!/usr/bin/env python3
"""Simple PD controller for the LineFollower-v0 environment.

Run with::

    python examples/pd_controller.py
    python examples/pd_controller.py --track oval
    python examples/pd_controller.py --track s_track
    python examples/pd_controller.py --track hairpin
    python examples/pd_controller.py --drift
    python examples/pd_controller.py --drift --grip 0.5 --yaw-damping 0.7

Available tracks: ``oval`` (default), ``s_track``, ``rounded_l``, ``hairpin``.

The controller reads the *lateral error* and *heading error* from the
observation vector and computes a steering correction using proportional
and derivative terms.  The correction is added/subtracted to a base
forward speed to obtain left and right wheel commands.

Tune the gains (``KP_LATERAL``, ``KD_LATERAL``, ``KP_HEADING``,
``KD_HEADING``) and ``BASE_SPEED`` to see how the robot behaviour
changes.
"""

from __future__ import annotations

import argparse
from typing import Any

import gymnasium as gym
import numpy as np

# Register the custom environment
import pg_tutorial  # noqa: F401
from pg_tutorial.envs.tracks import TRACK_BUILDERS

# ---------------------------------------------------------------------------
# Observation indices (must match LineFollowerEnv._get_observation)
# ---------------------------------------------------------------------------
IDX_LATERAL_ERROR: int = 0
IDX_HEADING_ERROR: int = 1
IDX_FORWARD_VELOCITY: int = 2
IDX_ANGULAR_VELOCITY: int = 3

# ---------------------------------------------------------------------------
# PD gains - feel free to experiment!
# ---------------------------------------------------------------------------
KP_LATERAL: float = 0.015
KD_LATERAL: float = 0.005

KP_HEADING: float = 0.8
KD_HEADING: float = 0.1

BASE_SPEED: float = 1.0  # normalised wheel speed in [-1, 1]


def compute_pd_action(
    observation: np.ndarray,
    prev_lateral_error: float,
    prev_heading_error: float,
    dt: float,
) -> tuple[np.ndarray, float, float]:
    """Compute a ``[left_wheel, right_wheel]`` action using PD control.

    Parameters
    ----------
    observation:
        Current environment observation vector.
    prev_lateral_error:
        Lateral error from the previous time-step (for the D term).
    prev_heading_error:
        Heading error from the previous time-step (for the D term).
    dt:
        Simulation time-step used to compute the derivative.

    Returns
    -------
    action:
        Numpy array of shape ``(2,)`` with wheel speed commands in [-1, 1].
    lateral_error:
        The current lateral error (to be stored for the next call).
    heading_error:
        The current heading error (to be stored for the next call).
    """
    lateral_error: float = float(observation[IDX_LATERAL_ERROR])
    heading_error: float = float(observation[IDX_HEADING_ERROR])

    # Derivative (finite difference)
    lateral_error_derivative = (lateral_error - prev_lateral_error) / dt
    heading_error_derivative = (heading_error - prev_heading_error) / dt

    # PD correction - positive correction steers to the right
    lateral_correction = KP_LATERAL * lateral_error + KD_LATERAL * lateral_error_derivative
    heading_correction = KP_HEADING * heading_error + KD_HEADING * heading_error_derivative

    steering = lateral_correction + heading_correction

    # Differential drive: subtract/add steering from the base speed
    left_wheel = BASE_SPEED + steering
    right_wheel = BASE_SPEED - steering

    action = np.array([left_wheel, right_wheel], dtype=np.float32)
    action = np.clip(action, -1.0, 1.0)

    return action, lateral_error, heading_error


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a PD controller on the LineFollower environment.",
    )
    parser.add_argument(
        "--track",
        type=str,
        default="oval",
        choices=list(TRACK_BUILDERS.keys()),
        help="Name of the built-in track to use (default: oval).",
    )
    parser.add_argument(
        "--racing",
        action="store_true",
        help="Use the racing reward variant (LineFollowerRacing-v0).",
    )
    parser.add_argument(
        "--drift",
        action="store_true",
        help="Enable drift / tyre-slip dynamics (LineFollowerDrift-v0).",
    )
    parser.add_argument(
        "--grip",
        type=float,
        default=0.85,
        help="Lateral grip coefficient when drift is enabled (0=ice, 1=perfect; default: 0.85).",
    )
    parser.add_argument(
        "--yaw-damping",
        type=float,
        default=0.3,
        dest="yaw_damping",
        help="Yaw damping when drift is enabled (0=no momentum, ~1=very spinny; default: 0.3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the environment (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    """Run one episode of the PD controller with rendering."""
    args = parse_args()

    # Select the environment id
    if args.racing:
        env_id = "LineFollowerRacing-v0"
    elif args.drift:
        env_id = "LineFollowerDrift-v0"
    else:
        env_id = "LineFollower-v0"

    # Build keyword arguments — drift params are only relevant for the
    # drift-based environments.
    make_kwargs: dict[str, Any] = {
        "render_mode": "human",
        "track_name": args.track,
        # "friction": 0.05,
        # "action_noise_std": 0.05,
        # "reward_mode": "racingv2",
    }
    if args.drift or args.racing:
        make_kwargs["lateral_grip"] = args.grip
        make_kwargs["yaw_damping"] = args.yaw_damping

    env = gym.make(env_id, **make_kwargs)

    observation, _info = env.reset(seed=args.seed)
    env.render()

    prev_lateral_error: float = float(observation[IDX_LATERAL_ERROR])
    prev_heading_error: float = float(observation[IDX_HEADING_ERROR])

    total_reward: float = 0.0
    step_count: int = 0

    terminated: bool = False
    truncated: bool = False

    while not (terminated or truncated):
        action, prev_lateral_error, prev_heading_error = compute_pd_action(
            observation,
            prev_lateral_error,
            prev_heading_error,
            env.unwrapped.dt,  # type: ignore[attr-defined]
        )

        observation, reward, terminated, truncated, _info = env.step(action)
        env.render()

        total_reward += float(reward)
        step_count += 1

    print(f"Episode finished after {step_count} steps  |  total reward: {total_reward:.1f}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
