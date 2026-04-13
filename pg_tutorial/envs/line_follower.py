"""Line-follower Gymnasium environment with a differential-drive robot.

The robot has two wheels whose speeds are independently controlled.  It must
follow a track defined as a sequence of waypoints.  Observations include the
robot pose, its velocity, and the lateral / heading error to the nearest
track segment.  Reward encourages staying close to the line and moving
forward along it.

Rendering uses *pygame* and works with ``render_mode="human"`` (window) or
``render_mode="rgb_array"`` (off-screen).
"""

from __future__ import annotations

import math
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from pg_tutorial.envs.rendering import (
    COL_BG,
    render_background_grid,
    render_hud,
    render_robot,
    render_track,
)
from pg_tutorial.envs.tracks import TRACK_BUILDERS, fit_track_to_screen

# ---------------------------------------------------------------------------
# Helper: closest point on a line segment
# ---------------------------------------------------------------------------


def _closest_point_on_segment(
    point: NDArray[np.float64],
    seg_start: NDArray[np.float64],
    seg_end: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    """Return the closest point on the segment and the parameter *t* in [0, 1]."""
    direction = seg_end - seg_start
    length_sq = float(np.dot(direction, direction))
    if length_sq < 1e-12:
        return seg_start.copy(), 0.0
    parameter = float(np.dot(point - seg_start, direction)) / length_sq
    parameter = np.clip(parameter, 0.0, 1.0)
    closest = seg_start + parameter * direction
    return closest, float(parameter)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class LineFollowerEnv(gym.Env[NDArray[np.float32], NDArray[np.float32]]):
    """A 2-D differential-drive robot that must follow a line track.

    This base environment uses **no-slip** differential-drive kinematics and
    a **line-following** reward that encourages staying close to the track
    centre while moving forward.

    For drift / tyre-slip dynamics and alternative reward modes see
    :class:`~pg_tutorial.envs.drift.LineFollowerDriftEnv`.

    Parameters
    ----------
    track_waypoints:
        (N, 2) array of waypoints defining the track.  When *None* a
        built-in track selected by *track_name* is used.
    track_name:
        Name of a built-in track when *track_waypoints* is ``None``.
        One of ``"oval"`` (default), ``"s_track"``,
        ``"rounded_l"``, or ``"hairpin"``.
    wheel_base:
        Distance between the two wheels (metres in sim-units).
    wheel_radius:
        Radius of each wheel.
    max_wheel_speed:
        Maximum angular speed of each wheel (rad / s).
    friction:
        Velocity-proportional friction coefficient applied to the wheel
        speeds each step.  0 means no friction, 1 means full stop every
        step.
    inertia:
        First-order lag coefficient in [0, 1) that blends the previous
        wheel speed toward the new target each step.  0 means instant
        response (no inertia), 0.9 means the wheels are very sluggish.
        Formally: ``speed = inertia * prev_speed + (1 - inertia) * target``.
    action_noise_std:
        Standard deviation of zero-mean Gaussian noise added to each wheel
        command *after* clipping.
    dt:
        Simulation time-step (seconds).
    max_episode_steps:
        Episode is truncated after this many steps.
    track_width:
        Half-width of the "on-track" region used for rendering.
    off_track_threshold:
        Lateral distance beyond which the episode terminates.
    render_mode:
        ``"human"`` for a pygame window, ``"rgb_array"`` for off-screen.
    screen_width:
        Width of the rendering surface in pixels.
    screen_height:
        Height of the rendering surface in pixels.
    """

    metadata: dict[str, Any] = {  # noqa: RUF012
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    # ---- construction -----------------------------------------------------

    def __init__(
        self,
        track_waypoints: NDArray[np.float64] | None = None,
        track_name: str = "s_track",
        wheel_base: float = 20.0,
        wheel_radius: float = 5.0,
        max_wheel_speed: float = 150.0,
        friction: float = 0.05,
        inertia: float = 0.95,
        action_noise_std: float = 0.01,
        dt: float = 1 / 30,
        max_episode_steps: int = 100000,  # overwritten by gym registration
        track_width: float = 60.0,
        off_track_threshold: float = 80.0,
        render_mode: str | None = None,
        screen_width: int = 1000,
        screen_height: int = 800,
    ) -> None:
        super().__init__()

        # Track
        if track_waypoints is not None:
            self.track_waypoints: NDArray[np.float64] = np.asarray(
                track_waypoints,
                dtype=np.float64,
            )
        else:
            builder = TRACK_BUILDERS.get(track_name, TRACK_BUILDERS["oval"])
            raw_track = builder()
            self.track_waypoints = fit_track_to_screen(raw_track, screen_width, screen_height)
        self.num_track_segments: int = len(self.track_waypoints)

        # Robot parameters
        self.wheel_base = wheel_base
        self.wheel_radius = wheel_radius
        self.max_wheel_speed = max_wheel_speed
        self.friction = np.clip(friction, 0.0, 1.0)
        self.inertia = float(np.clip(inertia, 0.0, 1.0 - 1e-6))
        self.action_noise_std = max(action_noise_std, 0.0)
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.track_width = track_width
        self.off_track_threshold = off_track_threshold

        # Rendering
        self.render_mode = render_mode
        self.screen_width = screen_width
        self.screen_height = screen_height
        self._screen: Any | None = None  # pygame.Surface
        self._clock: Any | None = None  # pygame.time.Clock
        self._pygame_initialised: bool = False

        # ---- spaces -------------------------------------------------------
        # Action: [left_wheel_speed, right_wheel_speed] in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        # Observation: [lateral_error, heading_error, forward_velocity,
        #               angular_velocity, left_wheel_speed, right_wheel_speed,
        #               curvature, lookahead_lat_2, lookahead_lat_4, lookahead_lat_6]
        obs_high = np.array(
            [
                self.off_track_threshold,  # lateral error
                np.pi,  # heading error
                np.inf,  # lateral error derivative
                np.inf,  # heading error derivative
                self.max_wheel_speed * self.wheel_radius * 2.0,  # fwd vel
                self.max_wheel_speed * self.wheel_radius * 2.0 / self.wheel_base,  # ang vel
                self.max_wheel_speed,  # left wheel
                self.max_wheel_speed,  # right wheel
                np.pi / 10.0,  # curvature (radians per unit)
                self.off_track_threshold,  # lookahead lat 2
                self.off_track_threshold,  # lookahead lat 4
                self.off_track_threshold,  # lookahead lat 6
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-obs_high,
            high=obs_high,
            dtype=np.float32,
        )

        # ---- internal state (set in reset) --------------------------------
        self.robot_x: float = 0.0
        self.robot_y: float = 0.0
        self.robot_theta: float = 0.0
        self.left_wheel_speed: float = 0.0
        self.right_wheel_speed: float = 0.0
        self.current_segment_index: int = 0
        self.step_count: int = 0
        self.forward_speed: float = 0.0
        self.prev_lateral_error = self.prev_heading_error = 0.0

        # Checkpoints: 5 evenly-spaced segment indices around the track.
        # Checkpoint 0 doubles as the start/finish line.
        num_checkpoints = 5
        self.checkpoint_segment_indices: list[int] = [
            round(idx * self.num_track_segments / num_checkpoints) % self.num_track_segments for idx in range(num_checkpoints)
        ]
        self.num_checkpoints: int = num_checkpoints
        # How close (in segment index) the robot must be to a checkpoint
        # for it to count as crossed.
        self._checkpoint_window: int = 4

        # Lap tracking
        self.lap_count: int = 0
        self.lap_start_step: int = 0
        self.current_lap_time: float = 0.0
        self.best_lap_time = float("inf")
        self.last_lap_time = float("inf")
        self._next_checkpoint: int = 1  # start past CP 0 (robot spawns there)

    # ---- reset / step -----------------------------------------------------
    def reset_lap_times(self):
        self.best_lap_time = float("inf")
        self.last_lap_time = float("inf")

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)

        # Place the robot at the first waypoint, heading toward the second
        start = self.track_waypoints[0]
        next_wp = self.track_waypoints[1 % self.num_track_segments]
        direction = next_wp - start

        self.robot_x = float(start[0])
        self.robot_y = float(start[1])
        self.robot_theta = float(np.arctan2(direction[1], direction[0]))
        self.left_wheel_speed = 0.0
        self.right_wheel_speed = 0.0
        self.current_segment_index = 0
        self.step_count = 0
        self.forward_speed = 0.0
        self.prev_lateral_error = self.prev_heading_error = 0.0

        # Lap tracking
        self.lap_count = 0
        self.lap_start_step = 0
        self.current_lap_time = 0.0
        # self.best_lap_time = float("inf")
        self._next_checkpoint = 1  # start past CP 0 (robot spawns there)

        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    def _apply_action(self, action: NDArray[np.float32]) -> tuple[float, float]:
        """Decode action, apply inertia / friction, return (forward_velocity, angular_velocity).

        This is factored out so subclasses can call it before adding their
        own dynamics (e.g. drift).
        """
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        target_left = float(action[0]) * self.max_wheel_speed
        target_right = float(action[1]) * self.max_wheel_speed

        # Add noise
        if self.action_noise_std > 0.0:
            noise = self.np_random.normal(0.0, self.action_noise_std, size=2)
            target_left += float(noise[0])
            target_right += float(noise[1])

        # Inertia: first-order lag blending previous speed toward the target
        target_left = self.inertia * self.left_wheel_speed + (1.0 - self.inertia) * target_left
        target_right = self.inertia * self.right_wheel_speed + (1.0 - self.inertia) * target_right

        # Apply friction (velocity-proportional drag)
        self.left_wheel_speed = (1.0 - self.friction) * target_left
        self.right_wheel_speed = (1.0 - self.friction) * target_right

        # Clip to physical limits
        self.left_wheel_speed = float(np.clip(self.left_wheel_speed, -self.max_wheel_speed, self.max_wheel_speed))
        self.right_wheel_speed = float(np.clip(self.right_wheel_speed, -self.max_wheel_speed, self.max_wheel_speed))

        # Differential-drive forward / angular velocities
        left_linear = self.left_wheel_speed * self.wheel_radius
        right_linear = self.right_wheel_speed * self.wheel_radius
        forward_velocity = (left_linear + right_linear) / 2.0
        angular_velocity = (right_linear - left_linear) / self.wheel_base

        self.forward_speed = forward_velocity
        return forward_velocity, angular_velocity

    def _integrate_kinematics(self, forward_velocity: float, angular_velocity: float) -> None:
        """No-slip differential-drive position integration."""
        if abs(angular_velocity) < 1e-9:
            # Straight-line motion
            self.robot_x += forward_velocity * math.cos(self.robot_theta) * self.dt
            self.robot_y += forward_velocity * math.sin(self.robot_theta) * self.dt
        else:
            # Arc motion (exact integration)
            turn_radius = forward_velocity / angular_velocity
            delta_theta = angular_velocity * self.dt
            self.robot_x += turn_radius * (math.sin(self.robot_theta + delta_theta) - math.sin(self.robot_theta))
            self.robot_y -= turn_radius * (math.cos(self.robot_theta + delta_theta) - math.cos(self.robot_theta))
            self.robot_theta += delta_theta

        # Normalise heading to [-pi, pi]
        self.robot_theta = math.atan2(
            math.sin(self.robot_theta),
            math.cos(self.robot_theta),
        )

    def _update_lap_detection(self) -> None:
        """Advance checkpoint-based lap detection state."""
        cp_seg = self.checkpoint_segment_indices[self._next_checkpoint]
        seg_diff = (self.current_segment_index - cp_seg) % self.num_track_segments
        # seg_diff is in [0, N); values near 0 or near N mean "close"
        near = min(seg_diff, self.num_track_segments - seg_diff)
        if near <= self._checkpoint_window:
            if self._next_checkpoint == 0:
                # Crossed the start/finish line after all other checkpoints.
                lap_time = (self.step_count - self.lap_start_step) * self.dt
                self.lap_count += 1
                if lap_time < self.best_lap_time:
                    self.best_lap_time = lap_time
                self.last_lap_time = lap_time
                self.lap_start_step = self.step_count
            self._next_checkpoint = (self._next_checkpoint + 1) % self.num_checkpoints

        self.current_lap_time = (self.step_count - self.lap_start_step) * self.dt

    def _compute_reward(
        self,
        forward_velocity: float,
        lateral_error: float,
        heading_error: float,
        *,
        going_reverse: bool = False,
    ) -> float:
        """Line-following reward.  Subclasses can override for other modes."""
        if going_reverse:
            return -10.0

        # TODO: try to convert to reward using exp(-x^2/var) instead of cost
        lateral_penalty = -((lateral_error / self.off_track_threshold) ** 2)
        heading_penalty = -((heading_error / np.pi) ** 2)
        forward_reward = max(forward_velocity * math.cos(heading_error), 0.0) * 0.01
        return 1.0 + lateral_penalty + 0.5 * heading_penalty + forward_reward

    def step(
        self,
        action: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], SupportsFloat, bool, bool, dict[str, Any]]:
        self.step_count += 1

        # ---- action decoding + wheel dynamics -----------------------------
        forward_velocity, angular_velocity = self._apply_action(action)

        # ---- position integration -----------------------------------------
        self._integrate_kinematics(forward_velocity, angular_velocity)

        # ---- track information & lap detection ----------------------------
        lateral_error, heading_error, _ = self._compute_track_errors()
        self._update_lap_detection()

        # ---- termination / truncation -------------------------------------
        # Going reverse: facing the wrong way on the track while moving forward
        going_reverse = forward_velocity > 0 and abs(heading_error) > math.pi / 2
        off_track = abs(lateral_error) > self.off_track_threshold
        terminated = off_track or going_reverse
        # ---- reward -------------------------------------------------------
        reward: float = self._compute_reward(
            forward_velocity, lateral_error, heading_error, going_reverse=going_reverse,
        )

        truncated = self.step_count >= self.max_episode_steps

        observation = self._get_observation()
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    # ---- observation / info helpers ---------------------------------------

    def _get_observation(self) -> NDArray[np.float32]:
        lateral_error, heading_error, _ = self._compute_track_errors()
        # Derivative (finite difference)
        lateral_error_derivative = (lateral_error - self.prev_lateral_error) / self.dt
        heading_error_derivative = (heading_error - self.prev_heading_error) / self.dt
        # Update prev errors
        self.prev_lateral_error = lateral_error
        self.prev_heading_error = heading_error

        left_linear = self.left_wheel_speed * self.wheel_radius
        right_linear = self.right_wheel_speed * self.wheel_radius
        forward_velocity = (left_linear + right_linear) / 2.0
        angular_velocity = (right_linear - left_linear) / self.wheel_base
        curvature = self._compute_curvature()
        lookahead_lat = self._get_lookahead_lateral_errors()

        observation = np.array(
            [
                lateral_error,
                heading_error,
                lateral_error_derivative,
                heading_error_derivative,
                forward_velocity,
                angular_velocity,
                self.left_wheel_speed,
                self.right_wheel_speed,
                curvature,
                lookahead_lat[0],
                lookahead_lat[1],
                lookahead_lat[2],
            ],
            dtype=np.float32,
        )
        return observation

    def _get_info(self) -> dict[str, Any]:
        lateral_error, heading_error, closest = self._compute_track_errors()
        return {
            "robot_x": self.robot_x,
            "robot_y": self.robot_y,
            "robot_theta": self.robot_theta,
            "lateral_error": lateral_error,
            "heading_error": heading_error,
            "closest_point": closest,
            "segment_index": self.current_segment_index,
            "forward_speed": self.forward_speed,
            "lap_count": self.lap_count,
            "current_lap_time": self.current_lap_time,
            "best_lap_time": self.best_lap_time,
            "last_lap_time": self.last_lap_time,
            "next_checkpoint": self._next_checkpoint,
            "num_checkpoints": self.num_checkpoints,
            "checkpoint_segment_indices": self.checkpoint_segment_indices,
        }

    # ---- track geometry ---------------------------------------------------

    def _compute_track_errors(
        self,
    ) -> tuple[float, float, NDArray[np.float64]]:
        """Return (signed lateral error, heading error, closest point)."""
        robot_pos = np.array([self.robot_x, self.robot_y], dtype=np.float64)

        # Search a window of segments around the current best to avoid O(N)
        search_half_window = 10
        best_dist_sq = np.inf
        best_closest = self.track_waypoints[0].copy()
        best_seg_idx = self.current_segment_index

        for offset in range(-search_half_window, search_half_window + 1):
            seg_idx = (self.current_segment_index + offset) % self.num_track_segments
            next_idx = (seg_idx + 1) % self.num_track_segments
            seg_start = self.track_waypoints[seg_idx]
            seg_end = self.track_waypoints[next_idx]
            closest, _parameter = _closest_point_on_segment(robot_pos, seg_start, seg_end)
            dist_sq = float(np.sum((robot_pos - closest) ** 2))
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_closest = closest
                best_seg_idx = seg_idx

        # Advance current segment pointer
        self.current_segment_index = best_seg_idx

        # Signed lateral error (positive = robot is to the left of the track)
        seg_start = self.track_waypoints[best_seg_idx]
        seg_end = self.track_waypoints[(best_seg_idx + 1) % self.num_track_segments]
        track_direction = seg_end - seg_start
        track_angle = float(np.arctan2(track_direction[1], track_direction[0]))

        diff = robot_pos - best_closest
        lateral_error = float(
            -diff[0] * math.sin(track_angle) + diff[1] * math.cos(track_angle),
        )

        # Heading error (how far the robot heading deviates from the track)
        heading_error = self.robot_theta - track_angle
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))

        return lateral_error, heading_error, best_closest

    def _compute_curvature(self) -> float:
        """Compute track curvature at current position.

        Curvature is the rate of change of heading angle with respect to distance.
        Higher values indicate sharper turns.
        """
        # Get the current segment and the next segment
        seg_idx = self.current_segment_index
        next_idx = (seg_idx + 1) % self.num_track_segments
        next_next_idx = (seg_idx + 2) % self.num_track_segments

        seg_start = self.track_waypoints[seg_idx]
        seg_end = self.track_waypoints[next_idx]
        seg_end_next = self.track_waypoints[next_next_idx]

        # Compute heading angles of consecutive segments
        track_direction1 = seg_end - seg_start
        track_direction2 = seg_end_next - seg_end

        angle1 = np.arctan2(track_direction1[1], track_direction1[0])
        angle2 = np.arctan2(track_direction2[1], track_direction2[0])

        # Heading change between segments
        delta_angle = float(np.arctan2(np.sin(angle2 - angle1), np.cos(angle2 - angle1)))

        # Distance between segment centers (approximate arc length)
        dist = float(np.linalg.norm(seg_end - seg_start))
        dist += float(np.linalg.norm(seg_end_next - seg_end))
        dist /= 2.0  # average distance

        if dist < 1e-6:
            return 0.0

        # Curvature = d(theta)/ds (radians per unit distance)
        curvature = delta_angle / dist
        return np.clip(curvature, -np.pi / 10.0, np.pi / 10.0)

    def _get_lookahead_lateral_errors(self) -> tuple[float, float, float]:
        """Compute lateral errors at lookahead points.

        These values help the agent anticipate upcoming turns and plan
        optimal racing lines.
        """
        robot_pos = np.array([self.robot_x, self.robot_y], dtype=np.float64)
        offsets = [2, 8, 16]
        lookahead_errors = []

        for offset in offsets:
            if self.num_track_segments <= offset:
                lookahead_errors.append(0.0)
                continue

            # Get the lookahead segment
            lookahead_seg_idx = (self.current_segment_index + offset) % self.num_track_segments
            lookahead_next_idx = (lookahead_seg_idx + 1) % self.num_track_segments

            seg_start = self.track_waypoints[lookahead_seg_idx]
            seg_end = self.track_waypoints[lookahead_next_idx]

            # Find closest point on this lookahead segment to robot
            closest, _ = _closest_point_on_segment(robot_pos, seg_start, seg_end)

            # Compute lateral error relative to lookahead segment
            track_direction = seg_end - seg_start
            track_angle = float(np.arctan2(track_direction[1], track_direction[0]))

            diff = robot_pos - closest
            lat_error = float(
                -diff[0] * np.sin(track_angle) + diff[1] * np.cos(track_angle),
            )
            lookahead_errors.append(np.clip(lat_error, -self.off_track_threshold, self.off_track_threshold))

        return (lookahead_errors[0], lookahead_errors[1], lookahead_errors[2])

    # ---- rendering --------------------------------------------------------

    def render(self) -> NDArray[np.uint8] | None:  # type: ignore[override]
        """Render the environment using pygame."""
        if self.render_mode is None:
            return None

        try:
            import pygame
            import pygame.gfxdraw
        except ImportError as exc:
            raise gym.error.DependencyNotInstalled(
                "pygame is required for rendering. Install it with `pip install pygame`.",
            ) from exc

        if not self._pygame_initialised:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.set_caption("LineFollower")
                self._screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height),
                )
            else:
                self._screen = pygame.Surface(
                    (self.screen_width, self.screen_height),
                )
            self._clock = pygame.time.Clock()
            self._pygame_initialised = True

        assert self._screen is not None
        surface: pygame.Surface = self._screen

        surface.fill(COL_BG)

        # -- background grid ------------------------------------------------
        render_background_grid(surface, pygame, self.screen_width, self.screen_height)

        # -- track ----------------------------------------------------------
        render_track(
            surface,
            pygame,
            self.track_waypoints,
            self.track_width,
            checkpoint_segment_indices=self.checkpoint_segment_indices,
            next_checkpoint=self._next_checkpoint,
        )

        # -- robot ----------------------------------------------------------
        _lat_err, _head_err, closest_pt = self._compute_track_errors()
        self._render_robot(surface, pygame, closest_pt)

        # -- HUD ------------------------------------------------------------
        lateral_error, heading_error, _ = self._compute_track_errors()
        self._render_hud(surface, pygame, lateral_error, heading_error)

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            assert self._clock is not None
            self._clock.tick(self.metadata["render_fps"])
            return None

        # rgb_array
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(surface)),
            axes=(1, 0, 2),
        ).copy()

    def _render_robot(self, surface: Any, pygame_module: Any, closest_pt: NDArray[np.float64]) -> None:
        """Draw the robot.  Subclasses can override to add drift visuals."""
        render_robot(
            surface,
            pygame_module,
            self.robot_x,
            self.robot_y,
            self.robot_theta,
            self.wheel_base,
            closest_pt,
            self.screen_width,
            self.screen_height,
        )

    def _render_hud(self, surface: Any, pygame_module: Any, lateral_error: float, heading_error: float) -> None:
        """Draw the HUD.  Subclasses can override to add extra fields."""
        curvature = self._compute_curvature()
        render_hud(
            surface,
            pygame_module,
            step_count=self.step_count,
            lateral_error=lateral_error,
            heading_error=heading_error,
            left_wheel_speed=self.left_wheel_speed,
            right_wheel_speed=self.right_wheel_speed,
            forward_speed=self.forward_speed,
            lap_count=self.lap_count,
            current_lap_time=self.current_lap_time,
            best_lap_time=self.best_lap_time,
            last_lap_time=self.last_lap_time,
            next_checkpoint=self._next_checkpoint,
            num_checkpoints=self.num_checkpoints,
            curvature=curvature,
        )

    def close(self) -> None:
        if self._pygame_initialised:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self._pygame_initialised = False
            self._screen = None
            self._clock = None
