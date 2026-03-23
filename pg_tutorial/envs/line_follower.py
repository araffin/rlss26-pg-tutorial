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
# Track builders  (all produce unit-scale tracks centred near the origin)
# ---------------------------------------------------------------------------


def _make_oval_track(num_points: int = 300) -> NDArray[np.float64]:
    """Elliptical / racetrack loop (no self-intersection)."""
    angles = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
    waypoints_x = 2.0 * np.cos(angles)
    waypoints_y = 1.0 * np.sin(angles)
    return np.column_stack([waypoints_x, waypoints_y])


def _make_figure_eight_track(num_points: int = 300) -> NDArray[np.float64]:
    """Lemniscate of Bernoulli (figure-eight). *Does* self-intersect."""
    angles = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
    waypoints_x = np.sin(angles)
    waypoints_y = np.sin(angles) * np.cos(angles)
    return np.column_stack([waypoints_x, waypoints_y])


def _cubic_bezier(
    ctrl_0: NDArray[np.float64],
    ctrl_1: NDArray[np.float64],
    ctrl_2: NDArray[np.float64],
    ctrl_3: NDArray[np.float64],
    num_samples: int,
) -> NDArray[np.float64]:
    """Evaluate a cubic Bezier curve at *num_samples* evenly-spaced t values."""
    t_values = np.linspace(0.0, 1.0, num_samples, endpoint=False)
    one_minus_t = 1.0 - t_values
    # B(t) = (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t) t^2 P2 + t^3 P3
    points = (
        np.outer(one_minus_t**3, ctrl_0)
        + np.outer(3.0 * one_minus_t**2 * t_values, ctrl_1)
        + np.outer(3.0 * one_minus_t * t_values**2, ctrl_2)
        + np.outer(t_values**3, ctrl_3)
    )
    return points


def _smooth_closed_curve(
    control_points: NDArray[np.float64],
    points_per_segment: int = 40,
) -> NDArray[np.float64]:
    """Build a smooth closed curve through *control_points* using Catmull-Rom
    to cubic-Bezier conversion.

    Each pair of adjacent control points becomes one cubic Bezier segment
    whose tangents are derived from the neighbouring points, giving C1
    continuity around the whole loop.
    """
    num_ctrl = len(control_points)
    parts: list[NDArray[np.float64]] = []
    for idx in range(num_ctrl):
        prev_idx = (idx - 1) % num_ctrl
        next_idx = (idx + 1) % num_ctrl
        next_next_idx = (idx + 2) % num_ctrl

        p_prev = control_points[prev_idx]
        p_curr = control_points[idx]
        p_next = control_points[next_idx]
        p_next_next = control_points[next_next_idx]

        # Catmull-Rom tangents → cubic Bezier control points
        tangent_curr = (p_next - p_prev) / 6.0
        tangent_next = (p_next_next - p_curr) / 6.0

        ctrl_1 = p_curr + tangent_curr
        ctrl_2 = p_next - tangent_next

        parts.append(_cubic_bezier(p_curr, ctrl_1, ctrl_2, p_next, points_per_segment))

    return np.vstack(parts)


def _make_s_track(num_points: int = 300) -> NDArray[np.float64]:
    """Smooth S-shaped closed loop (no self-intersection).

    Uses Catmull-Rom spline through hand-placed control points that
    trace two mirrored lobes connected by straights.
    """
    control_points = np.array(
        [
            [0.0, -0.8],
            [1.0, -0.8],
            [1.6, -0.4],
            [1.6, 0.0],
            [1.0, 0.4],
            [0.0, 0.4],
            [-0.6, 0.4],
            [-1.0, 0.8],
            [-1.6, 0.8],
            [-2.0, 0.4],
            [-2.0, 0.0],
            [-1.6, -0.4],
            [-1.0, -0.4],
            [-0.6, -0.4],
        ],
        dtype=np.float64,
    )
    pts_per_seg = max(num_points // len(control_points), 4)
    track = _smooth_closed_curve(control_points, pts_per_seg)
    return track[:num_points]


def _make_rounded_l_track(num_points: int = 300) -> NDArray[np.float64]:
    """L-shaped circuit with smooth rounded corners (no self-intersection).

    Uses Catmull-Rom spline through the vertices of an L-shape, producing
    smooth arcs at every corner without manual arc stitching.
    """
    # Vertices of the outer L going clockwise, with extra mid-edge points
    # so the straights stay straight and only corners get rounded.
    control_points = np.array(
        [
            # Bottom edge (left to right)
            [0.0, 0.0],
            [0.8, 0.0],
            [1.6, 0.0],
            # Bottom-right corner
            [2.0, 0.0],
            [2.0, 0.4],
            # Right edge going up (short leg)
            [2.0, 0.6],
            # Top-right corner of the short leg
            [2.0, 1.0],
            [1.6, 1.0],
            # Inner horizontal edge (right to left)
            [1.4, 1.0],
            # Inner corner (concave)
            [1.0, 1.0],
            [1.0, 1.4],
            # Left tall edge going up
            [1.0, 1.6],
            # Top-left corner of the tall leg
            [1.0, 2.0],
            [0.6, 2.0],
            # Top edge (right to left)
            [0.4, 2.0],
            # Top-left corner
            [0.0, 2.0],
            [0.0, 1.6],
            # Left edge going down
            [0.0, 1.0],
            [0.0, 0.4],
        ],
        dtype=np.float64,
    )
    # Centre around the origin
    control_points -= control_points.mean(axis=0)

    pts_per_seg = max(num_points // len(control_points), 4)
    track = _smooth_closed_curve(control_points, pts_per_seg)
    return track[:num_points]


def _make_hairpin_track(num_points: int = 300) -> NDArray[np.float64]:
    """Elongated track with tight hairpin turns at each end."""
    half = num_points // 2
    remainder = num_points - 2 * half
    straight_count = half // 2
    curve_count = half - straight_count

    parts: list[NDArray[np.float64]] = []

    # Bottom straight (left to right)
    sx = np.linspace(-1.5, 1.5, straight_count, endpoint=False)
    parts.append(np.column_stack([sx, -0.5 * np.ones_like(sx)]))

    # Right hairpin (semicircle, centre at (1.5, 0))
    arc_r = np.linspace(-np.pi / 2.0, np.pi / 2.0, curve_count, endpoint=False)
    parts.append(np.column_stack([1.5 + 0.5 * np.cos(arc_r), 0.5 * np.sin(arc_r)]))

    # Top straight (right to left)
    sx2 = np.linspace(1.5, -1.5, straight_count + remainder, endpoint=False)
    parts.append(np.column_stack([sx2, 0.5 * np.ones_like(sx2)]))

    # Left hairpin (semicircle, centre at (-1.5, 0))
    arc_l = np.linspace(np.pi / 2.0, 3.0 * np.pi / 2.0, curve_count, endpoint=False)
    parts.append(np.column_stack([-1.5 + 0.5 * np.cos(arc_l), 0.5 * np.sin(arc_l)]))

    return np.vstack(parts)


# Mapping from name to builder so users can request tracks by string.
TRACK_BUILDERS: dict[str, Any] = {
    "oval": _make_oval_track,
    "figure_eight": _make_figure_eight_track,
    "s_track": _make_s_track,
    "rounded_l": _make_rounded_l_track,
    "hairpin": _make_hairpin_track,
}


def _fit_track_to_screen(
    track: NDArray[np.float64],
    screen_width: int,
    screen_height: int,
    margin_fraction: float = 0.10,
) -> NDArray[np.float64]:
    """Scale and translate *track* so it fills the screen with a margin."""
    min_xy = track.min(axis=0)
    max_xy = track.max(axis=0)
    extent = max_xy - min_xy
    extent = np.where(extent < 1e-6, 1.0, extent)  # avoid division by zero

    margin_x = screen_width * margin_fraction
    margin_y = screen_height * margin_fraction
    available_w = screen_width - 2.0 * margin_x
    available_h = screen_height - 2.0 * margin_y

    scale = min(available_w / extent[0], available_h / extent[1])

    centred = track - (min_xy + max_xy) / 2.0
    centred *= scale
    centred[:, 0] += screen_width / 2.0
    centred[:, 1] += screen_height / 2.0
    return centred


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
# Rendering colour palette (module-level so helpers can share them)
# ---------------------------------------------------------------------------

_COL_BG = (34, 40, 49)
_COL_GRID = (44, 50, 59)
_COL_TRACK_FILL = (57, 62, 70)
_COL_TRACK_EDGE = (78, 85, 95)
_COL_CENTER_LINE = (255, 211, 105)
_COL_ROBOT_BODY = (0, 173, 181)
_COL_ROBOT_OUTLINE = (0, 140, 148)
_COL_HEADING = (238, 238, 238)
_COL_WHEEL = (218, 218, 218)
_COL_WHEEL_OUTLINE = (130, 130, 130)
_COL_CLOSEST_LINE = (255, 211, 105, 100)
_COL_CLOSEST_DOT = (255, 211, 105)
_COL_HUD_TEXT = (200, 200, 200)
_COL_HUD_BG = (34, 40, 49, 180)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class LineFollowerEnv(gym.Env[NDArray[np.float32], NDArray[np.float32]]):
    """A 2-D differential-drive robot that must follow a line track.

    Parameters
    ----------
    track_waypoints:
        (N, 2) array of waypoints defining the track.  When *None* a
        built-in track selected by *track_name* is used.
    track_name:
        Name of a built-in track when *track_waypoints* is ``None``.
        One of ``"oval"`` (default), ``"figure_eight"``, ``"s_track"``,
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
        track_name: str = "oval",
        wheel_base: float = 20.0,
        wheel_radius: float = 5.0,
        max_wheel_speed: float = 10.0,
        friction: float = 0.05,
        inertia: float = 0.2,
        action_noise_std: float = 0.1,
        dt: float = 0.1,
        max_episode_steps: int = 2000,
        track_width: float = 30.0,
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
            builder = TRACK_BUILDERS.get(track_name, _make_oval_track)
            raw_track = builder()
            self.track_waypoints = _fit_track_to_screen(raw_track, screen_width, screen_height)
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

        # Inertia: first-order lag blending previous speed toward the target
        target_left = self.inertia * self.left_wheel_speed + (1.0 - self.inertia) * target_left
        target_right = self.inertia * self.right_wheel_speed + (1.0 - self.inertia) * target_right

        # Apply friction (velocity-proportional drag)
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

    # ---- rendering helpers ------------------------------------------------

    @staticmethod
    def _compute_track_normals(
        waypoints: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return per-waypoint unit normals (pointing left of travel direction).

        At each waypoint the normal is the average of the normals of the two
        adjacent segments, re-normalised.  This gives smooth offset curves
        when the normals are used to build the road-edge polygons.
        """
        num_wp = len(waypoints)
        normals = np.zeros_like(waypoints)
        for idx in range(num_wp):
            prev_idx = (idx - 1) % num_wp
            next_idx = (idx + 1) % num_wp
            tangent = waypoints[next_idx] - waypoints[prev_idx]
            length = np.linalg.norm(tangent)
            if length < 1e-9:
                normals[idx] = np.array([0.0, -1.0])
            else:
                tangent /= length
                normals[idx] = np.array([-tangent[1], tangent[0]])
        return normals

    @staticmethod
    def _build_road_polygon(
        waypoints: NDArray[np.float64],
        normals: NDArray[np.float64],
        half_width: float,
    ) -> list[tuple[int, int]]:
        """Return a closed polygon tracing the left edge forward, then the
        right edge backward.  When filled this gives a smooth road band with
        no scalloping."""
        left_edge = waypoints + normals * half_width
        right_edge = waypoints - normals * half_width
        # Forward along left, then backward along right → closed loop
        polygon: list[tuple[int, int]] = []
        for point in left_edge:
            polygon.append((int(point[0]), int(point[1])))
        for point in right_edge[::-1]:
            polygon.append((int(point[0]), int(point[1])))
        return polygon

    def _render_track(self, surface: Any, pygame_module: Any) -> None:
        """Draw the road polygon and dashed centre line onto *surface*."""
        # -- draw track as filled polygon -----------------------------------
        normals = self._compute_track_normals(self.track_waypoints)
        road_hw = float(self.track_width)
        edge_hw = road_hw + 2.0

        # Outer edge polygon (slightly wider -> acts as border)
        edge_poly = self._build_road_polygon(self.track_waypoints, normals, edge_hw)
        pygame_module.draw.polygon(surface, _COL_TRACK_EDGE, edge_poly)

        # Road-fill polygon
        fill_poly = self._build_road_polygon(self.track_waypoints, normals, road_hw)
        pygame_module.draw.polygon(surface, _COL_TRACK_FILL, fill_poly)

        # -- dashed centre line ---------------------------------------------
        # Pre-compute cumulative arc length along the track so we can place
        # dashes evenly regardless of waypoint spacing.
        num_wp = len(self.track_waypoints)
        dash_on = 12.0
        dash_off = 10.0
        dash_cycle = dash_on + dash_off

        cumulative_len = 0.0
        for seg_idx in range(num_wp):
            next_idx = (seg_idx + 1) % num_wp
            seg_start = self.track_waypoints[seg_idx]
            seg_end = self.track_waypoints[next_idx]
            seg_vec = seg_end - seg_start
            seg_len = float(np.linalg.norm(seg_vec))

            # Walk along this segment drawing dashes
            pos = 0.0
            while pos < seg_len:
                phase = (cumulative_len + pos) % dash_cycle
                if phase < dash_on:
                    # We are inside a visible dash
                    remaining_on = dash_on - phase
                    draw_end = min(pos + remaining_on, seg_len)
                    frac_a = pos / seg_len if seg_len > 0 else 0.0
                    frac_b = draw_end / seg_len if seg_len > 0 else 0.0
                    pt_a = seg_start + frac_a * seg_vec
                    pt_b = seg_start + frac_b * seg_vec
                    pygame_module.draw.line(
                        surface,
                        _COL_CENTER_LINE,
                        (round(pt_a[0]), round(pt_a[1])),
                        (round(pt_b[0]), round(pt_b[1])),
                        2,
                    )
                    pos = draw_end
                else:
                    # We are in a gap - skip ahead
                    remaining_off = dash_cycle - phase
                    pos += remaining_off

            cumulative_len += seg_len

    def _render_robot(self, surface: Any, pygame_module: Any) -> None:
        """Draw closest-point indicator, robot body, wheels, and heading triangle."""
        # -- line from robot to closest point on track ----------------------
        _lat_err, _head_err, closest_pt = self._compute_track_errors()
        # Draw on a temporary surface for alpha blending
        closest_surf = pygame_module.Surface((self.screen_width, self.screen_height), pygame_module.SRCALPHA)
        pygame_module.draw.line(
            closest_surf,
            _COL_CLOSEST_LINE,
            (int(self.robot_x), int(self.robot_y)),
            (int(closest_pt[0]), int(closest_pt[1])),
            2,
        )
        surface.blit(closest_surf, (0, 0))
        # Small dot at the closest point
        pygame_module.gfxdraw.aacircle(surface, int(closest_pt[0]), int(closest_pt[1]), 3, _COL_CLOSEST_DOT)
        pygame_module.gfxdraw.filled_circle(surface, int(closest_pt[0]), int(closest_pt[1]), 3, _COL_CLOSEST_DOT)

        # -- draw robot -----------------------------------------------------
        robot_px = int(self.robot_x)
        robot_py = int(self.robot_y)
        body_radius = max(int(self.wheel_base * 0.8), 6)
        cos_th = math.cos(self.robot_theta)
        sin_th = math.sin(self.robot_theta)
        perp_x = -sin_th
        perp_y = cos_th
        half_base = self.wheel_base / 2.0

        # Wheels (drawn first so the body overlaps them slightly)
        wheel_half_len = max(int(body_radius * 0.5), 3)
        wheel_half_width = max(int(body_radius * 0.9), 2)
        for side in (-1.0, 1.0):
            wheel_cx = self.robot_x + side * half_base * perp_x
            wheel_cy = self.robot_y + side * half_base * perp_y
            corners: list[tuple[int, int]] = []
            for along_sign, across_sign in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
                corner_x = wheel_cx + along_sign * wheel_half_len * cos_th + across_sign * wheel_half_width * perp_x
                corner_y = wheel_cy + along_sign * wheel_half_len * sin_th + across_sign * wheel_half_width * perp_y
                corners.append((int(corner_x), int(corner_y)))
            pygame_module.draw.polygon(surface, _COL_WHEEL, corners)
            pygame_module.draw.polygon(surface, _COL_WHEEL_OUTLINE, corners, 1)

        # Body circle (anti-aliased)
        pygame_module.gfxdraw.aacircle(surface, robot_px, robot_py, body_radius, _COL_ROBOT_OUTLINE)
        pygame_module.gfxdraw.filled_circle(surface, robot_px, robot_py, body_radius, _COL_ROBOT_BODY)
        pygame_module.gfxdraw.aacircle(surface, robot_px, robot_py, body_radius, _COL_ROBOT_OUTLINE)

        # Heading triangle (points in the direction of travel)
        tri_tip_x = self.robot_x + (body_radius + 4) * cos_th
        tri_tip_y = self.robot_y + (body_radius + 4) * sin_th
        tri_left_x = self.robot_x + body_radius * 0.45 * (-cos_th + perp_x)
        tri_left_y = self.robot_y + body_radius * 0.45 * (-sin_th + perp_y)
        tri_right_x = self.robot_x + body_radius * 0.45 * (-cos_th - perp_x)
        tri_right_y = self.robot_y + body_radius * 0.45 * (-sin_th - perp_y)
        heading_tri = [
            (int(tri_tip_x), int(tri_tip_y)),
            (int(tri_left_x), int(tri_left_y)),
            (int(tri_right_x), int(tri_right_y)),
        ]
        pygame_module.draw.polygon(surface, _COL_HEADING, heading_tri)
        pygame_module.draw.aalines(surface, _COL_ROBOT_OUTLINE, True, heading_tri)

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

        surface.fill(_COL_BG)

        # -- subtle background grid -----------------------------------------
        grid_spacing = 40
        for grid_x in range(0, self.screen_width, grid_spacing):
            pygame.draw.line(surface, _COL_GRID, (grid_x, 0), (grid_x, self.screen_height))
        for grid_y in range(0, self.screen_height, grid_spacing):
            pygame.draw.line(surface, _COL_GRID, (0, grid_y), (self.screen_width, grid_y))

        self._render_track(surface, pygame)
        self._render_robot(surface, pygame)

        # -- HUD with semi-transparent background --------------------------
        lateral_error, heading_error, _ = self._compute_track_errors()
        font = pygame.font.SysFont("monospace", 14)
        hud_lines = [
            f"step: {self.step_count}",
            f"lat err: {lateral_error:+.1f}",
            f"head err: {math.degrees(heading_error):+.1f}\u00b0",
            f"wheels L/R: {self.left_wheel_speed:+.2f} / {self.right_wheel_speed:+.2f}",
        ]
        line_height = 18
        hud_padding = 6
        hud_width = max(font.size(line)[0] for line in hud_lines) + 2 * hud_padding
        hud_height = len(hud_lines) * line_height + 2 * hud_padding

        hud_bg = pygame.Surface((hud_width, hud_height), pygame.SRCALPHA)
        hud_bg.fill(_COL_HUD_BG)
        surface.blit(hud_bg, (4, 4))

        for line_idx, text in enumerate(hud_lines):
            text_surface = font.render(text, True, _COL_HUD_TEXT)
            surface.blit(text_surface, (4 + hud_padding, 4 + hud_padding + line_idx * line_height))

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
