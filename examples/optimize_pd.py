#!/usr/bin/env python3
"""Simple PD controller for the LineFollower-v0 environment.

Run with::

    python examples/pd_controller.py --track s_track

Available tracks: "oval`, "s_track", "rounded_l", "hairpin", "custom".
"""

import argparse
from dataclasses import dataclass

import numpy as np

from pg_tutorial.envs import TRACK_BUILDERS
from pg_tutorial.envs.line_follower import LineFollowerEnv

IDX_LATERAL_ERROR: int = 0
IDX_LATERAL_ERROR_DERIVATIVE: int = 2


@dataclass
class BangBangController:
    steering_step: float = 0.3
    speed: float = 0.5
    error_threshold: float = 0.0

    def compute_action(
        self,
        lateral_error: float,
        prev_lateral_error: float,
    ) -> np.ndarray:
        """Compute a ``[left_wheel, right_wheel]`` action using bang-bang control."""

        steering = 0.0
        if abs(lateral_error) > self.error_threshold:
            steering = np.sign(lateral_error) * self.steering_step

        # Differential drive: subtract/add steering from the base speed
        left_wheel = self.speed + steering
        right_wheel = self.speed - steering

        action = np.clip([left_wheel, right_wheel], -1.0, 1.0)

        return action


@dataclass
class PDController:
    kp: float = 0.0  # proportional gain
    kd: float = 0.0  # derivative gain
    dt: float = 0.0
    speed: float = 0.0  # normalised wheel speed in [-1, 1]

    def compute_action(
        self,
        lateral_error: float,
        prev_lateral_error: float,
    ) -> np.ndarray:
        """Compute a ``[left_wheel, right_wheel]`` action using PD control."""
        assert self.dt > 0, f"{self.dt} <=0! Did you forget to set it?"
        # Derivative (finite difference)
        lateral_error_derivative = (lateral_error - prev_lateral_error) / self.dt

        # PD correction - positive correction steers to the right
        steering = self.kp * lateral_error + self.kd * lateral_error_derivative

        # Differential drive: subtract/add steering from the base speed
        left_wheel = self.speed + steering
        right_wheel = self.speed - steering

        action = np.clip([left_wheel, right_wheel], -1.0, 1.0)

        return action


def evaluate(
    env: LineFollowerEnv,
    controller: BangBangController | PDController,
    verbose: int = 1,
) -> tuple[float, float]:
    env.reset_lap_times()
    observation, _ = env.reset()
    env.render()

    # Initialize lateral error
    lateral_error = prev_lateral_error = float(observation[IDX_LATERAL_ERROR])

    total_reward = 0.0
    step_count = 0

    done = False
    best_lap_time = float("inf")
    lap_count = 0
    # Function to compute the action

    while not done:
        action = controller.compute_action(lateral_error, prev_lateral_error)

        observation, reward, terminated, truncated, info = env.step(action)
        prev_lateral_error = lateral_error
        # Retrieve current lateral error (cross track error) from observation
        lateral_error = float(observation[IDX_LATERAL_ERROR])

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
    controller: PDController,
    pop_std: float = 0.001,
    pop_size: int = 10,
    n_iterations: int = 10,
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
            controller.kp = candidate[0]
            controller.kd = candidate[1]
            lap_time, _ = evaluate(env, controller, verbose=0)
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
        default="s_track",
        choices=list(TRACK_BUILDERS.keys()),
        help="Name of the built-in track to use (default: s_track).",
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
    # inertia=0.1 for bang-bang to work
    env = LineFollowerEnv(
        track_name=args.track,
        render_mode=render_mode,
        max_episode_steps=max_episode_steps,
    )

    # Base, oscillating
    controller = PDController(kp=0.00446, kd=0.01050, dt=env.dt, speed=0.5)
    # Optimized
    # controller = PDController(kp=0.00420, kd=0.00758, dt=env.dt, speed=0.5)
    # controller = BangBangController(steering_step=0.01, speed=0.1)

    if args.optimize:
        kp, kd = optimize(env, PDController(speed=0.5, dt=env.dt), n_iterations=10)
        print(f"Optimized gains: {kp=:.5f}, {kd=:.5f}")
    else:
        evaluate(env, controller)

    env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
