"""Rendering helpers for the line-follower environment.

This module contains the colour palette, track-rendering, robot-rendering,
and HUD-drawing functions used by :class:`LineFollowerEnv`.  Separating
them keeps the main environment file focused on dynamics and Gymnasium API.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Colour palette (shared by all rendering helpers)
# ---------------------------------------------------------------------------

COL_BG = (34, 40, 49)
COL_GRID = (44, 50, 59)
COL_TRACK_FILL = (57, 62, 70)
COL_TRACK_EDGE = (78, 85, 95)
COL_CENTER_LINE = (255, 211, 105)
COL_ROBOT_BODY = (0, 173, 181)
COL_ROBOT_OUTLINE = (0, 140, 148)
COL_HEADING = (238, 238, 238)
COL_WHEEL = (218, 218, 218)
COL_WHEEL_OUTLINE = (130, 130, 130)
COL_CLOSEST_LINE = (255, 211, 105, 100)
COL_CLOSEST_DOT = (255, 211, 105)
COL_HUD_TEXT = (200, 200, 200)
COL_HUD_BG = (34, 40, 49, 180)

# ---------------------------------------------------------------------------
# Centre-line dash width (pixels).  Increase this value to make the dashed
# centre line thicker.
# ---------------------------------------------------------------------------

CENTER_LINE_WIDTH: float = 3.0


# ---------------------------------------------------------------------------
# Track geometry helpers
# ---------------------------------------------------------------------------


def compute_track_normals(
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
        length = float(np.linalg.norm(tangent))
        if length < 1e-9:
            normals[idx] = np.array([0.0, -1.0])
        else:
            tangent /= length
            normals[idx] = np.array([-tangent[1], tangent[0]])
    return normals


def _draw_road_quads(
    surface: Any,
    pygame_module: Any,
    waypoints: NDArray[np.float64],
    normals: NDArray[np.float64],
    half_width: float,
    colour: tuple[int, int, int],
) -> None:
    """Draw the road as one filled quad per segment.

    Unlike a single giant polygon that traces the full left edge forward and
    right edge backward, per-segment quads never self-intersect at tight
    inner curves.  Overlapping neighbouring quads are harmless because
    they share the same fill colour.
    """
    left_edge = waypoints + normals * half_width
    right_edge = waypoints - normals * half_width
    num_wp = len(waypoints)

    for idx in range(num_wp):
        next_idx = (idx + 1) % num_wp
        quad = [
            (int(left_edge[idx][0]), int(left_edge[idx][1])),
            (int(left_edge[next_idx][0]), int(left_edge[next_idx][1])),
            (int(right_edge[next_idx][0]), int(right_edge[next_idx][1])),
            (int(right_edge[idx][0]), int(right_edge[idx][1])),
        ]
        pygame_module.draw.polygon(surface, colour, quad)


def _aa_thick_line(
    surface: Any,
    pygame_module: Any,
    pt_a: NDArray[np.float64],
    pt_b: NDArray[np.float64],
    half_width: float,
    colour: tuple[int, int, int],
) -> None:
    """Draw an anti-aliased thick line segment as a filled polygon.

    The dash is rendered as an oriented rectangle whose long axis runs from
    *pt_a* to *pt_b* and whose short axis has the given *half_width*.  The
    outline is drawn with ``aalines`` for sub-pixel anti-aliasing.
    """
    direction = pt_b - pt_a
    length = float(np.linalg.norm(direction))
    if length < 0.5:
        return
    # Unit normal perpendicular to the dash direction
    normal = np.array([-direction[1], direction[0]]) / length

    offset = normal * half_width
    corners = [
        (pt_a[0] + offset[0], pt_a[1] + offset[1]),
        (pt_b[0] + offset[0], pt_b[1] + offset[1]),
        (pt_b[0] - offset[0], pt_b[1] - offset[1]),
        (pt_a[0] - offset[0], pt_a[1] - offset[1]),
    ]
    int_corners = [(round(cx), round(cy)) for cx, cy in corners]

    # Filled body
    pygame_module.draw.polygon(surface, colour, int_corners)
    # Anti-aliased outline for smooth edges
    pygame_module.draw.aalines(surface, colour, True, corners)


# ---------------------------------------------------------------------------
# Track rendering
# ---------------------------------------------------------------------------


def render_track(
    surface: Any,
    pygame_module: Any,
    track_waypoints: NDArray[np.float64],
    track_width: float,
) -> None:
    """Draw the road surface and anti-aliased dashed centre line.

    The road is drawn as per-segment quads rather than a single polygon, which
    avoids self-intersection artefacts on the inner edge of tight curves and
    at the loop-closing segment.

    The dashed centre line is drawn as oriented rectangle polygons with AA
    outlines, giving a clean look at any configurable width (see
    ``CENTER_LINE_WIDTH``).
    """
    normals = compute_track_normals(track_waypoints)
    road_hw = float(track_width)
    edge_hw = road_hw + 2.0

    # Outer edge (slightly wider -> acts as border)
    _draw_road_quads(surface, pygame_module, track_waypoints, normals, edge_hw, COL_TRACK_EDGE)

    # Road fill
    _draw_road_quads(surface, pygame_module, track_waypoints, normals, road_hw, COL_TRACK_FILL)

    # -- anti-aliased dashed centre line ------------------------------------
    num_wp = len(track_waypoints)
    dash_on = 12.0
    dash_off = 12.0
    dash_cycle = dash_on + dash_off
    line_hw = CENTER_LINE_WIDTH / 2.0

    cumulative_len = 0.0
    for seg_idx in range(num_wp):
        next_idx = (seg_idx + 1) % num_wp
        seg_start = track_waypoints[seg_idx]
        seg_end = track_waypoints[next_idx]
        seg_vec = seg_end - seg_start
        seg_len = float(np.linalg.norm(seg_vec))

        if seg_len < 1e-6:
            cumulative_len += seg_len
            continue

        # Walk along this segment drawing dashes
        pos = 0.0
        while pos < seg_len:
            phase = (cumulative_len + pos) % dash_cycle
            if phase < dash_on:
                # Inside a visible dash
                remaining_on = dash_on - phase
                draw_end = min(pos + remaining_on, seg_len)
                frac_a = pos / seg_len
                frac_b = draw_end / seg_len
                pt_a = seg_start + frac_a * seg_vec
                pt_b = seg_start + frac_b * seg_vec
                _aa_thick_line(surface, pygame_module, pt_a, pt_b, line_hw, COL_CENTER_LINE)
                pos = draw_end
            else:
                # In a gap - skip ahead
                remaining_off = dash_cycle - phase
                pos += remaining_off

        cumulative_len += seg_len


# ---------------------------------------------------------------------------
# Robot rendering
# ---------------------------------------------------------------------------


def render_robot(
    surface: Any,
    pygame_module: Any,
    robot_x: float,
    robot_y: float,
    robot_theta: float,
    wheel_base: float,
    closest_point: NDArray[np.float64],
    screen_width: int,
    screen_height: int,
) -> None:
    """Draw closest-point indicator, robot body, wheels, and heading line."""
    # -- line from robot to closest point on track --------------------------
    closest_surf = pygame_module.Surface((screen_width, screen_height), pygame_module.SRCALPHA)
    pygame_module.draw.line(
        closest_surf,
        COL_CLOSEST_LINE,
        (int(robot_x), int(robot_y)),
        (int(closest_point[0]), int(closest_point[1])),
        2,
    )
    surface.blit(closest_surf, (0, 0))
    # Small dot at the closest point
    pygame_module.gfxdraw.aacircle(
        surface,
        int(closest_point[0]),
        int(closest_point[1]),
        3,
        COL_CLOSEST_DOT,
    )
    pygame_module.gfxdraw.filled_circle(
        surface,
        int(closest_point[0]),
        int(closest_point[1]),
        3,
        COL_CLOSEST_DOT,
    )

    # -- draw robot ---------------------------------------------------------
    robot_px = int(robot_x)
    robot_py = int(robot_y)
    body_radius = max(int(wheel_base * 0.8), 6)
    cos_th = math.cos(robot_theta)
    sin_th = math.sin(robot_theta)
    perp_x = -sin_th
    perp_y = cos_th
    half_base = wheel_base / 2.0

    # Wheels (drawn first so the body overlaps them slightly)
    wheel_half_len = max(int(body_radius * 0.5), 3)
    wheel_half_width = max(int(body_radius * 0.9), 2)
    for side in (-1.0, 1.0):
        wheel_cx = robot_x + side * half_base * perp_x
        wheel_cy = robot_y + side * half_base * perp_y
        corners: list[tuple[int, int]] = []
        for along_sign, across_sign in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
            corner_x = wheel_cx + along_sign * wheel_half_len * cos_th + across_sign * wheel_half_width * perp_x
            corner_y = wheel_cy + along_sign * wheel_half_len * sin_th + across_sign * wheel_half_width * perp_y
            corners.append((int(corner_x), int(corner_y)))
        pygame_module.draw.polygon(surface, COL_WHEEL, corners)
        pygame_module.draw.polygon(surface, COL_WHEEL_OUTLINE, corners, 1)

    # Body circle (anti-aliased)
    pygame_module.gfxdraw.aacircle(surface, robot_px, robot_py, body_radius, COL_ROBOT_OUTLINE)
    pygame_module.gfxdraw.filled_circle(surface, robot_px, robot_py, body_radius, COL_ROBOT_BODY)
    pygame_module.gfxdraw.aacircle(surface, robot_px, robot_py, body_radius, COL_ROBOT_OUTLINE)

    # Heading indicator (simple line from centre in the direction of travel)
    heading_length = body_radius + 6
    heading_end_x = robot_px + int(heading_length * math.cos(robot_theta))
    heading_end_y = robot_py + int(heading_length * math.sin(robot_theta))
    pygame_module.draw.line(
        surface,
        COL_HEADING,
        (robot_px, robot_py),
        (heading_end_x, heading_end_y),
        width=3,
    )


# ---------------------------------------------------------------------------
# HUD rendering
# ---------------------------------------------------------------------------


def render_hud(
    surface: Any,
    pygame_module: Any,
    step_count: int,
    lateral_error: float,
    heading_error: float,
    left_wheel_speed: float,
    right_wheel_speed: float,
) -> None:
    """Draw a semi-transparent HUD overlay in the top-left corner."""
    font = pygame_module.font.SysFont("monospace", 14)
    hud_lines = [
        f"step: {step_count}",
        f"lat err: {lateral_error:+.1f}",
        f"head err: {math.degrees(heading_error):+.1f}\u00b0",
        f"wheels L/R: {left_wheel_speed:+.2f} / {right_wheel_speed:+.2f}",
    ]
    line_height = 18
    hud_padding = 6
    hud_width = max(font.size(line)[0] for line in hud_lines) + 2 * hud_padding
    hud_height = len(hud_lines) * line_height + 2 * hud_padding

    hud_bg = pygame_module.Surface((hud_width, hud_height), pygame_module.SRCALPHA)
    hud_bg.fill(COL_HUD_BG)
    surface.blit(hud_bg, (4, 4))

    for line_idx, text in enumerate(hud_lines):
        text_surface = font.render(text, True, COL_HUD_TEXT)
        surface.blit(
            text_surface,
            (4 + hud_padding, 4 + hud_padding + line_idx * line_height),
        )


# ---------------------------------------------------------------------------
# Background grid
# ---------------------------------------------------------------------------


def render_background_grid(
    surface: Any,
    pygame_module: Any,
    screen_width: int,
    screen_height: int,
    grid_spacing: int = 40,
) -> None:
    """Draw a subtle background grid."""
    for grid_x in range(0, screen_width, grid_spacing):
        pygame_module.draw.line(surface, COL_GRID, (grid_x, 0), (grid_x, screen_height))
    for grid_y in range(0, screen_height, grid_spacing):
        pygame_module.draw.line(surface, COL_GRID, (0, grid_y), (screen_width, grid_y))
