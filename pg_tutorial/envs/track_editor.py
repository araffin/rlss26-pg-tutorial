"""Minimal GUI to visualize and edit tracks defined using bezier curves."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
from numpy.typing import NDArray

from pg_tutorial.envs.tracks import (
    TRACK_BUILDERS,
    _smooth_closed_curve,
    fit_track_to_screen,
)


class TrackEditor:
    """GUI application for visualizing and editing tracks."""

    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 600
    POINT_RADIUS = 6

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Track Editor")
        self.root.geometry(f"{self.SCREEN_WIDTH + 250}x{self.SCREEN_HEIGHT + 150}")

        # Track state
        self.control_points: NDArray[np.float64] = np.array([])
        self.current_track_name: str = "oval"
        self.selected_point_idx: int | None = None
        self.num_samples: int = 300

        self._setup_ui()
        self._load_track("oval")

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left side - Canvas for track visualization
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(
            canvas_frame,
            width=self.SCREEN_WIDTH,
            height=self.SCREEN_HEIGHT,
            bg="white",
            cursor="crosshair",
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<Button-3>", self._on_canvas_right_click)

        # Right side - Controls
        controls_frame = ttk.Frame(main_frame, width=200)
        controls_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

        # Track selection
        ttk.Label(controls_frame, text="Track Type:").pack(pady=(0, 5))
        self.track_var = tk.StringVar(value="oval")
        self.track_combo = ttk.Combobox(
            controls_frame,
            textvariable=self.track_var,
            values=list(TRACK_BUILDERS.keys()),
            state="readonly",
            width=15,
        )
        self.track_combo.pack(pady=(0, 10))
        self.track_combo.bind("<<ComboboxSelected>>", self._on_track_change)

        # Sample count
        ttk.Label(controls_frame, text="Sample Count:").pack(pady=(0, 5))
        self.sample_var = tk.StringVar(value="300")
        sample_spin = ttk.Spinbox(controls_frame, from_=50, to=1000, textvariable=self.sample_var, width=10)
        sample_spin.pack(pady=(0, 10))
        sample_spin.bind("<<ComboboxSelected>>", self._on_sample_change)

        # Buttons
        ttk.Button(controls_frame, text="Load Track", command=self._load_current_track).pack(pady=5)
        ttk.Button(controls_frame, text="Reset Control Points", command=self._reset_control_points).pack(pady=5)
        ttk.Button(controls_frame, text="Clear Control Points", command=self._clear_control_points).pack(pady=5)

        ttk.Separator(controls_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # Info display
        ttk.Label(controls_frame, text="Info:").pack(pady=(0, 5))
        self.info_text = tk.Text(controls_frame, height=8, width=20, font=("monospace", 9))
        self.info_text.pack(pady=(0, 10))

        # Export buttons
        ttk.Button(controls_frame, text="Export NumPy Array", command=self._export_numpy).pack(pady=5)
        ttk.Button(controls_frame, text="Export Python Code", command=self._export_python).pack(pady=5)

        # Status label
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(controls_frame, textvariable=self.status_var).pack(pady=(10, 0))

    def _load_track(self, track_name: str) -> None:
        """Load a track by name and generate initial control points."""
        self.current_track_name = track_name
        builder = TRACK_BUILDERS.get(track_name)

        if builder is None:
            messagebox.showerror("Error", f"Unknown track type: {track_name}")
            return

        # Generate initial track to extract control points
        builder(self.num_samples)

        # For editable tracks, we'll use a simplified set of control points
        # based on the track type
        self._generate_control_points_for_track(track_name)

        self._draw_track()
        self._update_info()

    def _generate_control_points_for_track(self, track_name: str) -> None:
        """Generate initial control points based on track type."""
        # Default control points for editable tracks
        if track_name == "oval":
            # Ellipse control points
            angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
            self.control_points = np.column_stack(
                [
                    2.0 * np.cos(angles),
                    1.0 * np.sin(angles),
                ]
            )
        elif track_name == "s_track":
            self.control_points = np.array(
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
        elif track_name == "rounded_l":
            self.control_points = np.array(
                [
                    [0.0, 0.0],
                    [0.8, 0.0],
                    [1.6, 0.0],
                    [2.0, 0.0],
                    [2.0, 0.4],
                    [2.0, 0.6],
                    [2.0, 1.0],
                    [1.6, 1.0],
                    [1.4, 1.0],
                    [1.0, 1.0],
                    [1.0, 1.4],
                    [1.0, 1.6],
                    [1.0, 2.0],
                    [0.6, 2.0],
                    [0.4, 2.0],
                    [0.0, 2.0],
                    [0.0, 1.6],
                    [0.0, 1.0],
                    [0.0, 0.4],
                ],
                dtype=np.float64,
            )
            self.control_points -= self.control_points.mean(axis=0)
        elif track_name == "hairpin":
            # Simplified hairpin control points
            self.control_points = np.array(
                [
                    [-1.5, -0.5],
                    [-0.75, -0.5],
                    [0.0, -0.5],
                    [0.75, -0.5],
                    [1.5, -0.5],
                    [1.5, 0.0],
                    [1.5, 0.5],
                    [0.75, 0.5],
                    [0.0, 0.5],
                    [-0.75, 0.5],
                    [-1.5, 0.5],
                    [-1.5, 0.0],
                ],
                dtype=np.float64,
            )
        else:
            # Fallback to oval
            angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
            self.control_points = np.column_stack(
                [
                    2.0 * np.cos(angles),
                    1.0 * np.sin(angles),
                ]
            )

    def _reset_control_points(self) -> None:
        """Reset control points to default for current track."""
        self._generate_control_points_for_track(self.current_track_name)
        self._draw_track()
        self._update_info()

    def _clear_control_points(self) -> None:
        """Clear all control points."""
        self.control_points = np.array([]).reshape(0, 2)
        self.selected_point_idx = None
        self._draw_track()
        self._update_info()

    def _load_current_track(self) -> None:
        """Reload the current track."""
        self._load_track(self.current_track_name)

    def _on_track_change(self, event: tk.Event | None = None) -> None:
        """Handle track type change."""
        new_track = self.track_var.get()
        self._load_track(new_track)

    def _on_sample_change(self, event: tk.Event | None = None) -> None:
        """Handle sample count change."""
        try:
            self.num_samples = int(self.sample_var.get())
            self.num_samples = max(50, min(1000, self.num_samples))
            self.sample_var.set(str(self.num_samples))
        except ValueError:
            self.num_samples = 300
            self.sample_var.set("300")
        self._draw_track()

    def _compute_track_from_control_points(self) -> NDArray[np.float64]:
        """Compute smooth track from control points using Catmull-Rom spline."""
        if len(self.control_points) < 2:
            return np.array([]).reshape(0, 2)

        points_per_segment = max(self.num_samples // len(self.control_points), 4)
        track = _smooth_closed_curve(self.control_points, points_per_segment)
        return track[: self.num_samples]

    def _world_to_screen(self, x: float, y: float) -> tuple[int, int]:
        """Convert world coordinates to screen coordinates."""
        fit_track_to_screen(self.control_points, self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
        # Compute scale from the fitting
        if len(self.control_points) > 0:
            min_xy = self.control_points.min(axis=0)
            max_xy = self.control_points.max(axis=0)
            extent = max_xy - min_xy
            extent = np.where(extent < 1e-6, 1.0, extent)

            margin_x = self.SCREEN_WIDTH * 0.10
            margin_y = self.SCREEN_HEIGHT * 0.10
            available_w = self.SCREEN_WIDTH - 2.0 * margin_x
            available_h = self.SCREEN_HEIGHT - 2.0 * margin_y
            scale = min(available_w / extent[0], available_h / extent[1])

            # Apply same transformation (matching fit_track_to_screen convention)
            centered_x = x - (min_xy[0] + max_xy[0]) / 2.0
            centered_y = y - (min_xy[1] + max_xy[1]) / 2.0
            screen_x = centered_x * scale + self.SCREEN_WIDTH / 2.0
            screen_y = centered_y * scale + self.SCREEN_HEIGHT / 2.0
            return int(screen_x), int(screen_y)
        return int(self.SCREEN_WIDTH / 2), int(self.SCREEN_HEIGHT / 2)

    def _screen_to_world(self, screen_x: int, screen_y: int) -> tuple[float, float]:
        """Convert screen coordinates to world coordinates."""
        if len(self.control_points) == 0:
            return 0.0, 0.0

        min_xy = self.control_points.min(axis=0)
        max_xy = self.control_points.max(axis=0)
        extent = max_xy - min_xy
        extent = np.where(extent < 1e-6, 1.0, extent)

        margin_x = self.SCREEN_WIDTH * 0.10
        margin_y = self.SCREEN_HEIGHT * 0.10
        available_w = self.SCREEN_WIDTH - 2.0 * margin_x
        available_h = self.SCREEN_HEIGHT - 2.0 * margin_y
        scale = min(available_w / extent[0], available_h / extent[1])

        # Reverse transformation (matching fit_track_to_screen convention)
        centered_x = (screen_x - self.SCREEN_WIDTH / 2.0) / scale
        centered_y = (screen_y - self.SCREEN_HEIGHT / 2.0) / scale
        world_x = centered_x + (min_xy[0] + max_xy[0]) / 2.0
        world_y = centered_y + (min_xy[1] + max_xy[1]) / 2.0
        return world_x, world_y

    def _draw_track(self) -> None:
        """Draw the track and control points on the canvas."""
        self.canvas.delete("all")

        if len(self.control_points) < 2:
            self.status_var.set("Add at least 2 control points")
            return

        # Compute smooth track
        track = self._compute_track_from_control_points()

        if len(track) < 2:
            return

        # Draw track line
        for i in range(len(track) - 1):
            x1, y1 = self._world_to_screen(track[i, 0], track[i, 1])
            x2, y2 = self._world_to_screen(track[i + 1, 0], track[i + 1, 1])
            self.canvas.create_line(x1, y1, x2, y2, fill="blue", width=2, tags="track")

        # Draw control points
        for i, point in enumerate(self.control_points):
            x, y = self._world_to_screen(point[0], point[1])
            color = "red" if i == self.selected_point_idx else "green"
            outline = "darkgreen" if i == self.selected_point_idx else "green"

            self.canvas.create_oval(
                x - self.POINT_RADIUS,
                y - self.POINT_RADIUS,
                x + self.POINT_RADIUS,
                y + self.POINT_RADIUS,
                fill=color,
                outline=outline,
                width=2,
                tags=f"point_{i}",
            )

        # Draw connections between control points
        for i in range(len(self.control_points)):
            next_i = (i + 1) % len(self.control_points)
            x1, y1 = self._world_to_screen(self.control_points[i, 0], self.control_points[i, 1])
            x2, y2 = self._world_to_screen(self.control_points[next_i, 0], self.control_points[next_i, 1])
            self.canvas.create_line(x1, y1, x2, y2, fill="gray", width=1, dash=(4, 4), tags="connections")

        self.status_var.set(f"Control points: {len(self.control_points)}")
        self._update_info()

    def _find_nearest_point(self, screen_x: int, screen_y: int) -> int | None:
        """Find the nearest control point to the given screen position."""
        if len(self.control_points) == 0:
            return None

        min_dist = float("inf")
        nearest_idx = None

        for i, point in enumerate(self.control_points):
            x, y = self._world_to_screen(point[0], point[1])
            dist = (x - screen_x) ** 2 + (y - screen_y) ** 2
            if dist < min_dist and dist < (self.POINT_RADIUS * 3) ** 2:
                min_dist = dist
                nearest_idx = i

        return nearest_idx

    def _on_canvas_click(self, event: tk.Event) -> None:
        """Handle left mouse click on canvas."""
        # Check if clicking on existing point
        self.selected_point_idx = self._find_nearest_point(event.x, event.y)

        if self.selected_point_idx is None:
            # Add new control point at click location
            world_x, world_y = self._screen_to_world(event.x, event.y)
            new_point = np.array([[world_x, world_y]])

            if len(self.control_points) == 0:
                self.control_points = new_point
            else:
                self.control_points = np.vstack([self.control_points, new_point])

            self.selected_point_idx = len(self.control_points) - 1

        self._draw_track()

    def _on_canvas_drag(self, event: tk.Event) -> None:
        """Handle dragging a control point."""
        if self.selected_point_idx is not None:
            world_x, world_y = self._screen_to_world(event.x, event.y)
            self.control_points[self.selected_point_idx] = [world_x, world_y]
            self._draw_track()

    def _on_canvas_right_click(self, event: tk.Event) -> None:
        """Handle right mouse click to delete point."""
        point_idx = self._find_nearest_point(event.x, event.y)

        if point_idx is not None and len(self.control_points) > 2:
            self.control_points = np.delete(self.control_points, point_idx, axis=0)
            if self.selected_point_idx == point_idx:
                self.selected_point_idx = None
            elif self.selected_point_idx is not None and self.selected_point_idx > point_idx:
                self.selected_point_idx -= 1
            self._draw_track()

    def _update_info(self) -> None:
        """Update the info text panel."""
        self.info_text.delete("1.0", tk.END)

        info_lines = [
            f"Track: {self.current_track_name}",
            f"Control points: {len(self.control_points)}",
            f"Samples: {self.num_samples}",
            "",
            "Mouse controls:",
            "  Left-click: Add/select point",
            "  Drag: Move selected point",
            "  Right-click: Delete point",
        ]

        if len(self.control_points) > 0:
            min_xy = self.control_points.min(axis=0)
            max_xy = self.control_points.max(axis=0)
            info_lines.extend(
                [
                    "",
                    "Bounds:",
                    f"  X: [{min_xy[0]:.2f}, {max_xy[0]:.2f}]",
                    f"  Y: [{min_xy[1]:.2f}, {max_xy[1]:.2f}]",
                ]
            )

        self.info_text.insert(tk.END, "\n".join(info_lines))

    def _export_numpy(self) -> None:
        """Export track as NumPy array."""
        track = self._compute_track_from_control_points()

        if len(track) == 0:
            messagebox.showwarning("Warning", "No track to export. Add control points first.")
            return

        # Save to file
        from pathlib import Path

        save_path = Path.home() / "track_array.npy"
        np.save(save_path, track)
        messagebox.showinfo("Export", f"Track saved to:\n{save_path}")

    def _export_python(self) -> None:
        """Export track as Python code."""
        if len(self.control_points) == 0:
            messagebox.showwarning("Warning", "No control points to export. Add points first.")
            return

        code = self._generate_track_code()

        from pathlib import Path

        save_path = Path.home() / "custom_track.py"
        with open(save_path, "w") as f:
            f.write(code)

        messagebox.showinfo("Export", f"Track code saved to:\n{save_path}")

    def _generate_track_code(self) -> str:
        """Generate Python code for the current track configuration."""
        points_str = ",\n            ".join(f"[{p[0]:.4f}, {p[1]:.4f}]" for p in self.control_points)

        return f'''"""Custom track generated by Track Editor."""

import numpy as np
from numpy.typing import NDArray

from pg_tutorial.envs.tracks import _smooth_closed_curve


def make_custom_track(num_points: int = 300) -> NDArray[np.float64]:
    """Custom track with user-defined control points."""
    control_points = np.array(
        [
            {points_str},
        ],
        dtype=np.float64,
    )
    pts_per_seg = max(num_points // len(control_points), 4)
    track = _smooth_closed_curve(control_points, pts_per_seg)
    return track[:num_points]


# Add to TRACK_BUILDERS if needed:
# from pg_tutorial.envs.tracks import TRACK_BUILDERS
# TRACK_BUILDERS["custom"] = make_custom_track
'''


def main() -> None:
    """Run the track editor."""
    root = tk.Tk()
    TrackEditor(root)
    root.mainloop()


if __name__ == "__main__":
    main()
