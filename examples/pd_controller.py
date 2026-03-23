#!/usr/bin/env python3
"""Simple PD controller for the LineFollower-v0 environment.

Run with::

    python examples/pd_controller.py

The controller reads the *lateral error* and *heading error* from the
observation vector and computes a steering correction using proportional
and derivative terms.  The correction is added/subtracted to a base
forward speed to obtain left and right wheel commands.

Tune the gains (``KP_LATERAL``, ``KD_LATERAL``, ``KP_HEADING``,
``KD_HEADING``) and ``BASE_SPEED`` to see how the robot behaviour
changes.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

# Register the custom environment
import pg_tutorial  # noqa: F401

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
KP_LATERAL: float = 0.02
KD_LATERAL: float = 0.005

KP_HEADING: float = 0.8
KD_HEADING: float = 0.1

BASE_SPEED: float = 0.4  # normalised wheel speed in [-1, 1]


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


def main() -> None:
    """Run one episode of the PD controller with rendering."""
    env = gym.make(
        "LineFollower-v0",
        render_mode="human",
        friction=0.05,
        action_noise_std=0.05,
    )

    observation, _ = env.reset(seed=42)
    env.render()

    prev_lateral_error: float = float(observation[IDX_LATERAL_ERROR])
    prev_heading_error: float = float(observation[IDX_HEADING_ERROR])
    dt: float = 0.1  # must match the env default

    total_reward: float = 0.0
    step_count: int = 0

    terminated: bool = False
    truncated: bool = False

    while not (terminated or truncated):
        action, prev_lateral_error, prev_heading_error = compute_pd_action(
            observation,
            prev_lateral_error,
            prev_heading_error,
            dt,
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
