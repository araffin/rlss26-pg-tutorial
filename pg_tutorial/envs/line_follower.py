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

# ---------------------------------------------------------------------------
# Default track: a figure-eight made of waypoints
# ---------------------------------------------------------------------------


def _make_figure_eight_track(
    center_x: float = 400.0,
    center_y: float = 300.0,
    radius: float = 150.0,
    num_points: int = 200,
) -> NDArray[np.float64]:
    """Return an (N, 2) array of waypoints forming a figure-eight."""
    angles = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
    waypoints_x = center_x + radius * np.sin(angles)
    waypoints_y = center_y + radius * np.sin(angles) * np.cos(angles)
    return np.column_stack([waypoints_x, waypoints_y])


# ---------------------------------------------------------------------------
# Helper: closest point on a line segment
# ---------------------------------------------------------------------------


def _closest_point_on_segment(
    point: NDArray[np.float64],
    seg_start: NDArray[np.float64],
    seg_end: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    """Return the closest point on the segment and the parameter *t* ∈ [0, 1]."""
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

    Parameters
    ----------
    track_waypoints:
        (N, 2) array of waypoints defining the track.  When *None* a default
        figure-eight is used.
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
        wheel_base: float = 20.0,
        wheel_radius: float = 5.0,
        max_wheel_speed: float = 10.0,
        friction: float = 0.05,
        action_noise_std: float = 0.1,
        dt: float = 0.1,
        max_episode_steps: int = 2000,
        track_width: float = 10.0,
        off_track_threshold: float = 80.0,
        render_mode: str | None = None,
        screen_width: int = 800,
        screen_height: int = 600,
    ) -> None:
        super().__init__()

        # Track
        if track_waypoints is not None:
            self.track_waypoints: NDArray[np.float64] = np.asarray(
                track_waypoints,
                dtype=np.float64,
            )
        else:
            self.track_waypoints = _make_figure_eight_track(
                center_x=screen_width / 2.0,
                center_y=screen_height / 2.0,
            )
        self.num_track_segments: int = len(self.track_waypoints)

        # Robot parameters
        self.wheel_base = wheel_base
        self.wheel_radius = wheel_radius
        self.max_wheel_speed = max_wheel_speed
        self.friction = np.clip(friction, 0.0, 1.0)
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
        #               angular_velocity, left_wheel_speed, right_wheel_speed]
        obs_high = np.array(
            [
                self.off_track_threshold,  # lateral error
                np.pi,  # heading error
                self.max_wheel_speed * self.wheel_radius * 2.0,  # fwd vel
                self.max_wheel_speed * self.wheel_radius * 2.0 / self.wheel_base,  # ang vel
                self.max_wheel_speed,  # left wheel
                self.max_wheel_speed,  # right wheel
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

    # ---- reset / step -----------------------------------------------------

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

        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    def step(
        self,
        action: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], SupportsFloat, bool, bool, dict[str, Any]]:
        self.step_count += 1

        # ---- decode action ------------------------------------------------
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        target_left = float(action[0]) * self.max_wheel_speed
        target_right = float(action[1]) * self.max_wheel_speed

        # Add noise
        if self.action_noise_std > 0.0:
            noise = self.np_random.normal(0.0, self.action_noise_std, size=2)
            target_left += float(noise[0])
            target_right += float(noise[1])

        # Apply friction (exponential drag toward zero)
        self.left_wheel_speed = (1.0 - self.friction) * target_left
        self.right_wheel_speed = (1.0 - self.friction) * target_right

        # Clip to physical limits
        self.left_wheel_speed = float(np.clip(self.left_wheel_speed, -self.max_wheel_speed, self.max_wheel_speed))
        self.right_wheel_speed = float(np.clip(self.right_wheel_speed, -self.max_wheel_speed, self.max_wheel_speed))

        # ---- differential-drive kinematics --------------------------------
        left_linear = self.left_wheel_speed * self.wheel_radius
        right_linear = self.right_wheel_speed * self.wheel_radius

        forward_velocity = (left_linear + right_linear) / 2.0
        angular_velocity = (right_linear - left_linear) / self.wheel_base

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

        # Normalise heading to [-π, π]
        self.robot_theta = math.atan2(
            math.sin(self.robot_theta),
            math.cos(self.robot_theta),
        )

        # ---- track information --------------------------------------------
        lateral_error, heading_error, _ = self._compute_track_errors()

        # ---- reward -------------------------------------------------------
        # Penalise lateral and heading error, reward forward progress
        lateral_penalty = -((lateral_error / self.off_track_threshold) ** 2)
        heading_penalty = -((heading_error / np.pi) ** 2)
        forward_reward = max(forward_velocity * math.cos(heading_error), 0.0) * 0.01

        reward: float = 1.0 + lateral_penalty + 0.5 * heading_penalty + forward_reward

        # ---- termination / truncation -------------------------------------
        terminated = abs(lateral_error) > self.off_track_threshold
        truncated = self.step_count >= self.max_episode_steps

        observation = self._get_observation()
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    # ---- observation / info helpers ---------------------------------------

    def _get_observation(self) -> NDArray[np.float32]:
        lateral_error, heading_error, _ = self._compute_track_errors()
        left_linear = self.left_wheel_speed * self.wheel_radius
        right_linear = self.right_wheel_speed * self.wheel_radius
        forward_velocity = (left_linear + right_linear) / 2.0
        angular_velocity = (right_linear - left_linear) / self.wheel_base

        observation = np.array(
            [
                lateral_error,
                heading_error,
                forward_velocity,
                angular_velocity,
                self.left_wheel_speed,
                self.right_wheel_speed,
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
        }

    # ---- track geometry ---------------------------------------------------

    def _compute_track_errors(
        self,
    ) -> tuple[float, float, NDArray[np.float64]]:
        """Return (signed lateral error, heading error, closest point)."""
        robot_pos = np.array([self.robot_x, self.robot_y], dtype=np.float64)

        # Search a window of segments around the current best to avoid O(N)
        search_half_window = 15
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

    # ---- rendering --------------------------------------------------------

    def render(self) -> NDArray[np.uint8] | None:  # type: ignore[override]
        """Render the environment using pygame."""
        if self.render_mode is None:
            return None

        try:
            import pygame
        except ImportError as exc:
            raise gym.error.DependencyNotInstalled(
                "pygame is required for rendering. " "Install it with `pip install pygame`.",
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

        # Colours
        bg_colour = (30, 30, 30)
        track_colour = (80, 80, 80)
        center_line_colour = (200, 200, 50)
        robot_body_colour = (50, 160, 250)
        robot_heading_colour = (250, 80, 80)
        wheel_colour = (200, 200, 200)

        surface.fill(bg_colour)

        # Draw track band (thick polyline)
        track_points = [(int(wp[0]), int(wp[1])) for wp in self.track_waypoints]
        if len(track_points) > 1:
            pygame.draw.lines(
                surface,
                track_colour,
                closed=True,
                points=track_points,
                width=int(self.track_width * 2),
            )
            # Center line
            pygame.draw.lines(
                surface,
                center_line_colour,
                closed=True,
                points=track_points,
                width=2,
            )

        # Draw robot body
        robot_px = int(self.robot_x)
        robot_py = int(self.robot_y)
        body_radius = int(self.wheel_base / 2.0)
        pygame.draw.circle(surface, robot_body_colour, (robot_px, robot_py), body_radius)

        # Heading indicator
        heading_length = body_radius + 6
        heading_end_x = robot_px + int(heading_length * math.cos(self.robot_theta))
        heading_end_y = robot_py + int(heading_length * math.sin(self.robot_theta))
        pygame.draw.line(
            surface,
            robot_heading_colour,
            (robot_px, robot_py),
            (heading_end_x, heading_end_y),
            width=3,
        )

        # Wheel indicators (small rectangles perpendicular to heading)
        perp_x = -math.sin(self.robot_theta)
        perp_y = math.cos(self.robot_theta)
        half_base = self.wheel_base / 2.0
        wheel_half_len = 4
        wheel_half_width = 2

        for side in (-1.0, 1.0):
            wheel_cx = self.robot_x + side * half_base * perp_x
            wheel_cy = self.robot_y + side * half_base * perp_y
            corners = []
            for dx_sign, dy_sign in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
                corner_x = (
                    wheel_cx + dx_sign * wheel_half_len * math.cos(self.robot_theta) + dy_sign * wheel_half_width * perp_x
                )
                corner_y = (
                    wheel_cy + dx_sign * wheel_half_len * math.sin(self.robot_theta) + dy_sign * wheel_half_width * perp_y
                )
                corners.append((int(corner_x), int(corner_y)))
            pygame.draw.polygon(surface, wheel_colour, corners)

        # HUD text
        font = pygame.font.SysFont("monospace", 14)
        lateral_error, heading_error, _ = self._compute_track_errors()
        hud_lines = [
            f"step: {self.step_count}",
            f"lat err: {lateral_error:+.1f}",
            f"head err: {math.degrees(heading_error):+.1f} deg",
            f"wheels L/R: {self.left_wheel_speed:+.2f} / {self.right_wheel_speed:+.2f}",
        ]
        for line_idx, text in enumerate(hud_lines):
            text_surface = font.render(text, True, (220, 220, 220))
            surface.blit(text_surface, (8, 8 + line_idx * 18))

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

    def close(self) -> None:
        if self._pygame_initialised:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self._pygame_initialised = False
            self._screen = None
            self._clock = None
