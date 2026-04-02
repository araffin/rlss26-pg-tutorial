#!/usr/bin/env python3
"""Simple PD controller for the LineFollower-v0 environment.

Run with::

    python examples/pd_controller.py --track s_track

Available tracks: ``oval`` (default), ``s_track``, ``rounded_l``, ``hairpin``.
"""

import argparse

import numpy as np

from pg_tutorial.envs import TRACK_BUILDERS
from pg_tutorial.envs.line_follower import LineFollowerEnv

IDX_LATERAL_ERROR: int = 0


def compute_pd_action(
    observation: np.ndarray,
    prev_lateral_error: float,
    dt: float,
    speed: float | None = 0.5,  # normalised wheel speed in [-1, 1]
    kp: float = 0.005,  # proportional gain
    kd: float = 0.005,  # derivative gain
    min_speed: float | None = None,
    max_speed: float | None = None,
) -> tuple[np.ndarray, float]:
    """Compute a ``[left_wheel, right_wheel]`` action using PD control."""
    # Retrieve lateral error (cross track error) from observation
    lateral_error = float(observation[IDX_LATERAL_ERROR])

    # Derivative (finite difference)
    lateral_error_derivative = (lateral_error - prev_lateral_error) / dt

    # PD correction - positive correction steers to the right
    steering = kp * lateral_error + kd * lateral_error_derivative

    # High speed when close to the center line
    if min_speed and max_speed:
        max_error = 30.0
        # Linear interpolation
        speed = min_speed + (1 - abs(lateral_error / max_error)) * (max_speed - min_speed)

    assert speed is not None, "You must pass 'speed' or 'min_speed' and 'max_speed'"

    # Differential drive: subtract/add steering from the base speed
    left_wheel = speed + steering
    right_wheel = speed - steering

    action = np.clip([left_wheel, right_wheel], -1.0, 1.0)

    return action, lateral_error


def evaluate(
    env: LineFollowerEnv,
    kp: float,
    kd: float,
    speed: float | None = None,
    verbose: int = 1,
    min_speed: float | None = None,
    max_speed: float | None = None,
) -> tuple[float, float]:
    env.reset_lap_times()
    observation, _ = env.reset()
    env.render()

    prev_lateral_error = float(observation[IDX_LATERAL_ERROR])

    total_reward = 0.0
    step_count = 0

    done = False
    best_lap_time = float("inf")
    lap_count = 0

    while not done:
        action, prev_lateral_error = compute_action(
            observation,
            prev_lateral_error,
            env.dt,
            kp=kp,
            kd=kd,
            speed=speed,
        )

        observation, reward, terminated, truncated, info = env.step(action)
        env.render()

        if info["lap_count"] > lap_count:
            lap_count = info["lap_count"]
            last_lap_time = info["last_lap_time"]
            best_lap_time = info["best_lap_time"]
            if verbose:
                print(f"{last_lap_time=:.2f}s | {best_lap_time=:.2f}s")

        total_reward += float(reward)
        step_count += 1
        done = terminated or truncated

    if verbose:
        print(f"Episode finished after {step_count} steps  |  {total_reward=:.1f} | {best_lap_time=:.2f}s")

    return best_lap_time, total_reward


def optimize(
    env: LineFollowerEnv,
    speed: float = 0.5,
    pop_std: float = 0.001,
    pop_size: int = 10,
    n_iterations: int = 10,
    min_speed: float | None = None,
    max_speed: float | None = None,
) -> tuple[float, float]:
    # Start with small gains
    initial_kp = initial_kd = 0.0001
    best_gains = np.array([initial_kp, initial_kd])
    best_lap_time = float("inf")

    for iteration in range(1, n_iterations + 1):
        # Sample around the best_gains
        # Negative gains don't make sense
        candidates = np.abs([best_gains + np.random.normal(0.0, pop_std, size=2) for _ in range(pop_size)])

        for candidate in candidates:
            lap_time, _ = evaluate(env, kp=candidate[0], kd=candidate[1], speed=speed, verbose=0)
            if lap_time < best_lap_time:
                best_lap_time = lap_time
                best_gains = candidate
        print(f"{iteration=} | {best_lap_time=:.2f}s | {best_gains=}")

    kp, kd = best_gains[0], best_gains[1]
    return kp, kd


def main() -> None:
    """Run one episode of the PD controller with rendering."""
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
        "--optimize",
        action="store_true",
        help="Optimize the PD gains using black box optimization.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable rendering (fast evaluation).",
    )
    args = parser.parse_args()

    render_mode = None if args.optimize else "human"
    if args.no_render:
        render_mode = None
    max_episode_steps = 500 if args.optimize else 1000
    env = LineFollowerEnv(track_name=args.track, render_mode=render_mode, max_episode_steps=max_episode_steps)

    if args.optimize:
        kp, kd = optimize(env, n_iterations=10, speed=0.5)
        print(f"Optimized gains: {kp=:.5f}, {kd=:.5f}")
    else:
        evaluate(env, kp=0.005, kd=0.005, speed=0.5)
        # evaluate(env, kp=0.00408, kd=0.00734, speed=0.5)

    env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
