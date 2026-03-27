"""Track builders for the line-follower environment.

Each builder produces a closed loop of (N, 2) waypoints in unit-scale
coordinates centred near the origin.  The ``_fit_track_to_screen`` helper
scales and translates any track so it fills a rendering surface.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Bezier / spline helpers
# ---------------------------------------------------------------------------


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

        # Catmull-Rom tangents -> cubic Bezier control points
        tangent_curr = (p_next - p_prev) / 6.0
        tangent_next = (p_next_next - p_curr) / 6.0

        ctrl_1 = p_curr + tangent_curr
        ctrl_2 = p_next - tangent_next

        parts.append(_cubic_bezier(p_curr, ctrl_1, ctrl_2, p_next, points_per_segment))

    return np.vstack(parts)


# ---------------------------------------------------------------------------
# Track builders  (all produce unit-scale tracks centred near the origin)
# ---------------------------------------------------------------------------


def _make_oval_track(num_points: int = 300) -> NDArray[np.float64]:
    """Elliptical / racetrack loop (no self-intersection)."""
    angles = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
    waypoints_x = 2.0 * np.cos(angles)
    waypoints_y = 1.0 * np.sin(angles)
    return np.column_stack([waypoints_x, waypoints_y])


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


def _make_custom_track(num_points: int = 300) -> NDArray[np.float64]:
    """Custom track with user-defined control points."""
    control_points = np.array(
        [
            [1.8723, -0.8700],
            [1.4142, 0.7071],
            [0.0044, -0.4052],
            [-1.4142, 0.7071],
            [-2.0000, 0.0000],
            [-1.4312, -1.0429],
            [0.1000, -1.4805],
            [1.2403, -1.3889],
        ],
        dtype=np.float64,
    )
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


# ---------------------------------------------------------------------------
# Mapping from name to builder so users can request tracks by string.
# ---------------------------------------------------------------------------

TRACK_BUILDERS: dict[str, Any] = {
    "oval": _make_oval_track,
    "s_track": _make_s_track,
    "rounded_l": _make_rounded_l_track,
    "hairpin": _make_hairpin_track,
    "custom": _make_custom_track,
}


# ---------------------------------------------------------------------------
# Scaling helper
# ---------------------------------------------------------------------------


def fit_track_to_screen(
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
