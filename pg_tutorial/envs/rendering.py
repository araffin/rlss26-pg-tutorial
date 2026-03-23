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
COL_ROBOT_BODY = (0, 160, 170)
COL_ROBOT_OUTLINE = (0, 140, 148)
COL_HEADING = (200, 200, 200)
COL_WHEEL = (218, 218, 218)
COL_WHEEL_OUTLINE = (130, 130, 130)
COL_CLOSEST_LINE = (255, 211, 105, 100)
COL_CLOSEST_DOT = (255, 211, 105)
COL_HUD_TEXT = (200, 200, 200)
COL_HUD_LABEL = (140, 140, 140)
COL_HUD_VALUE = (238, 238, 238)
COL_HUD_ACCENT = (255, 211, 105)
COL_HUD_BEST = (0, 200, 140)
COL_HUD_BG = (34, 40, 49, 200)

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
    wheel_half_len = max(body_radius * 0.5, 3.0)
    wheel_half_width = max(body_radius * 0.9, 2.0)
    for side in (-1.0, 1.0):
        wheel_cx = robot_x + side * half_base * perp_x
        wheel_cy = robot_y + side * half_base * perp_y
        # Float corners for anti-aliased outline
        float_corners: list[tuple[float, float]] = []
        for along_sign, across_sign in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
            corner_x = wheel_cx + along_sign * wheel_half_len * cos_th + across_sign * wheel_half_width * perp_x
            corner_y = wheel_cy + along_sign * wheel_half_len * sin_th + across_sign * wheel_half_width * perp_y
            float_corners.append((corner_x, corner_y))
        int_corners = [(round(cx), round(cy)) for cx, cy in float_corners]
        pygame_module.draw.polygon(surface, COL_WHEEL, int_corners)
        pygame_module.draw.aalines(surface, COL_WHEEL_OUTLINE, True, float_corners)

    # Body circle (anti-aliased)
    pygame_module.gfxdraw.aacircle(surface, robot_px, robot_py, body_radius, COL_ROBOT_OUTLINE)
    pygame_module.gfxdraw.filled_circle(surface, robot_px, robot_py, body_radius, COL_ROBOT_BODY)
    pygame_module.gfxdraw.aacircle(surface, robot_px, robot_py, body_radius, COL_ROBOT_OUTLINE)

    # Heading indicator (anti-aliased thick line from centre outward)
    heading_length = body_radius + 6
    heading_start = np.array([robot_x, robot_y], dtype=np.float64)
    heading_end = np.array(
        [
            robot_x + heading_length * math.cos(robot_theta),
            robot_y + heading_length * math.sin(robot_theta),
        ],
        dtype=np.float64,
    )
    _aa_thick_line(surface, pygame_module, heading_start, heading_end, 2, COL_HEADING)


# ---------------------------------------------------------------------------
# HUD rendering
# ---------------------------------------------------------------------------


def _format_time(seconds: float) -> str:
    """Format a time in seconds as ``M:SS.s`` or ``--:--.-`` when infinite."""
    if seconds == float("inf"):
        return "--:--.--"
    minutes = int(seconds) // 60
    remaining = seconds - minutes * 60
    return f"{minutes}:{remaining:05.2f}"


def render_hud(
    surface: Any,
    pygame_module: Any,
    step_count: int,
    lateral_error: float,
    heading_error: float,
    left_wheel_speed: float,
    right_wheel_speed: float,
    forward_speed: float = 0.0,
    lap_count: int = 0,
    current_lap_time: float = 0.0,
    best_lap_time: float = float("inf"),
) -> None:
    """Draw a semi-transparent HUD overlay in the top-left corner.

    The HUD has two sections:

    * **Top** (larger font) — lap counter, speed, current lap time and best
      lap time.
    * **Bottom** (smaller font) — step counter, lateral / heading errors and
      wheel speeds.
    """
    font_big = pygame_module.font.SysFont("monospace", 20, bold=True)
    font_small = pygame_module.font.SysFont("monospace", 13)

    padding = 8
    section_gap = 6  # vertical gap between top and bottom sections

    # -- top section (big font) ---------------------------------------------
    speed_px_per_s = abs(forward_speed)
    top_items: list[tuple[str, str, tuple[int, ...]]] = [
        ("LAP", f"{lap_count}", COL_HUD_ACCENT),
        ("SPEED", f"{speed_px_per_s:.1f}", COL_HUD_VALUE),
        ("TIME", _format_time(current_lap_time), COL_HUD_VALUE),
        (
            "BEST",
            _format_time(best_lap_time),
            COL_HUD_BEST if best_lap_time < float("inf") else COL_HUD_LABEL,
        ),
    ]
    big_line_height = 24

    # -- bottom section (small font) ----------------------------------------
    bottom_lines: list[tuple[str, tuple[int, ...]]] = [
        (f"step: {step_count}", COL_HUD_TEXT),
        (f"lat err: {lateral_error:+.1f}", COL_HUD_TEXT),
        (f"head err: {math.degrees(heading_error):+.1f}\u00b0", COL_HUD_TEXT),
        (
            f"wheels L/R: {left_wheel_speed:+.2f} / {right_wheel_speed:+.2f}",
            COL_HUD_TEXT,
        ),
    ]
    small_line_height = 17

    # -- measure widths to size the background ------------------------------
    top_widths: list[int] = []
    for label, value, _colour in top_items:
        label_w = font_small.size(label)[0]
        value_w = font_big.size(value)[0]
        top_widths.append(label_w + 4 + value_w)

    bottom_widths = [font_small.size(text)[0] for text, _c in bottom_lines]

    hud_width = max(*top_widths, *bottom_widths) + 2 * padding
    top_height = len(top_items) * big_line_height
    bottom_height = len(bottom_lines) * small_line_height
    hud_height = padding + top_height + section_gap + bottom_height + padding

    # -- draw background ----------------------------------------------------
    hud_x = 4
    hud_y = 4
    hud_bg = pygame_module.Surface((hud_width, hud_height), pygame_module.SRCALPHA)
    hud_bg.fill(COL_HUD_BG)
    surface.blit(hud_bg, (hud_x, hud_y))

    # -- draw top section ---------------------------------------------------
    cursor_y = hud_y + padding
    for label, value, colour in top_items:
        label_surf = font_small.render(label, True, COL_HUD_LABEL)
        value_surf = font_big.render(value, True, colour)
        surface.blit(label_surf, (hud_x + padding, cursor_y + 4))
        surface.blit(
            value_surf,
            (hud_x + padding + label_surf.get_width() + 4, cursor_y),
        )
        cursor_y += big_line_height

    # -- draw bottom section ------------------------------------------------
    cursor_y += section_gap
    for text, colour in bottom_lines:
        text_surf = font_small.render(text, True, colour)
        surface.blit(text_surf, (hud_x + padding, cursor_y))
        cursor_y += small_line_height


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
