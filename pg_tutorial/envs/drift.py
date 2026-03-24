"""Drift variant of the line-follower environment.

:class:`LineFollowerDriftEnv` extends :class:`LineFollowerEnv` with optional
tyre-slip dynamics and an alternative *racing* reward mode.  The base class
is left clean and focused on no-slip differential-drive kinematics with a
line-following reward.

Drift model
-----------
Each step the wheel commands produce a target forward velocity and yaw rate
(identical to the base class).  Instead of applying the no-slip kinematic
equations directly, the robot tracks a world-frame velocity vector and an
angular velocity with momentum:

1. The world-frame velocity is decomposed into the body's longitudinal
   (forward) and lateral (sideways) components using the **current** heading.
2. The longitudinal component is set to the wheel-commanded forward speed
   (tyres have good forward traction).
3. The lateral component is decayed by ``(1 - lateral_grip)`` each step —
   ``lateral_grip = 1`` kills all sideways motion instantly (no-slip),
   ``lateral_grip = 0`` means zero lateral grip (ice).
4. The velocity is converted back to the world frame using the **same (old)**
   heading.
5. The body's angular velocity blends toward the wheel-commanded yaw rate
   with ``yaw_damping`` controlling how much rotational momentum carries
   over: ``omega = yaw_damping * prev_omega + (1 - yaw_damping) * target``.
6. The heading is then rotated by ``omega * dt``.

Because the heading rotates but the world velocity keeps its old direction,
the *next* step's body-frame decomposition naturally reveals a lateral
component whenever the heading has turned faster than the velocity could
follow — i.e. drift.

Default parameters are deliberately **mild** (``lateral_grip=0.85``,
``yaw_damping=0.3``) so that a simple PD controller can still drive around
the track while experiencing gentle sliding on corners.  Lower the grip or
raise the damping for more aggressive drift.
"""

from __future__ import annotations

import math
from typing import Any, SupportsFloat

import numpy as np
from numpy.typing import NDArray

from pg_tutorial.envs.line_follower import LineFollowerEnv
from pg_tutorial.envs.rendering import render_hud, render_robot


class LineFollowerDriftEnv(LineFollowerEnv):
    """Line-follower with tyre-slip dynamics and optional racing reward.

    This environment inherits all track, rendering, checkpoint and lap-timing
    logic from :class:`LineFollowerEnv` and **adds**:

    * Drift / tyre-slip physics controlled by *lateral_grip* and
      *yaw_damping*.
    * A *reward_mode* switch: ``"line_following"`` (default, same reward as
      the base class) or ``"racing"`` (rewards fast forward progress around
      the track with a lap-completion bonus).
    * Extra info-dict keys: ``drift``, ``slip_angle``, ``lateral_velocity``,
      ``total_progress``, ``reward_mode``.

    Parameters
    ----------
    lateral_grip:
        How quickly lateral (sideways) velocity is killed each step.
        ``1.0`` = perfect grip (no sliding), ``0.0`` = zero grip (ice).
        Default ``0.85`` gives mild drift that a PD controller can handle.
    yaw_damping:
        Fraction of the previous angular velocity that carries over.
        ``0.0`` = instant heading changes, ``~1.0`` = very spinny.
        Default ``0.3`` is gentle; raise toward 0.9 for aggressive drift.
    reward_mode:
        ``"line_following"`` (default) or ``"racing"``.

    All other parameters are forwarded to :class:`LineFollowerEnv`.
    """

    def __init__(
        self,
        *,
        lateral_grip: float = 0.85,
        yaw_damping: float = 0.3,
        reward_mode: str = "line_following",
        # -- forwarded to base --
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # Drift parameters
        self.lateral_grip = float(np.clip(lateral_grip, 0.0, 1.0))
        self.yaw_damping = float(np.clip(yaw_damping, 0.0, 1.0 - 1e-6))

        # Reward mode
        if reward_mode not in ("line_following", "racing", "racingv2"):
            raise ValueError(f"reward_mode must be 'line_following', 'racing' or 'racingv2', got {reward_mode!r}")
        self.reward_mode = reward_mode

        # Drift state (world-frame velocity and body angular velocity)
        self._vel_x: float = 0.0
        self._vel_y: float = 0.0
        self._omega: float = 0.0
        self.slip_angle: float = 0.0
        self.lateral_velocity: float = 0.0

        # Progress tracking (used by the racing reward)
        self._prev_segment_index: int = 0
        self.total_progress: float = 0.0

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)

        # Reset drift state
        self._vel_x = 0.0
        self._vel_y = 0.0
        self._omega = 0.0
        self.slip_angle = 0.0
        self.lateral_velocity = 0.0
        self._prev_segment_index = 0
        self.total_progress = 0.0

        # Augment info with drift keys
        info.update(self._drift_info())
        return obs, info

    # ------------------------------------------------------------------
    # step  (overrides the full step to slot in drift integration)
    # ------------------------------------------------------------------

    def step(
        self,
        action: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], SupportsFloat, bool, bool, dict[str, Any]]:
        self.step_count += 1

        # ---- action decoding + wheel dynamics (shared with base) ----------
        forward_velocity, angular_velocity = self._apply_action(action)

        # ---- drift / tyre-slip integration --------------------------------
        self._integrate_drift(forward_velocity, angular_velocity)

        # Normalise heading to [-pi, pi]
        self.robot_theta = math.atan2(
            math.sin(self.robot_theta),
            math.cos(self.robot_theta),
        )

        # ---- track information & lap detection ----------------------------
        lateral_error, heading_error, _ = self._compute_track_errors()

        # Track segment progress
        seg_advance = self._compute_segment_advance()

        prev_lap_count = self.lap_count
        self._update_lap_detection()

        # ---- reward -------------------------------------------------------
        reward: float = self._compute_reward_drift(
            forward_velocity,
            lateral_error,
            heading_error,
            seg_advance,
            prev_lap_count,
        )

        # ---- termination / truncation -------------------------------------
        # Off-track or going backward
        terminated = abs(lateral_error) > self.off_track_threshold or forward_velocity < 0 or seg_advance < 0
        truncated = self.step_count >= self.max_episode_steps

        observation = self._get_observation()
        info = self._get_info()
        info.update(self._drift_info())
        return observation, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Drift physics
    # ------------------------------------------------------------------

    def _integrate_drift(self, forward_velocity: float, angular_velocity: float) -> None:
        """Tyre-slip position / heading integration.

        See the module docstring for a detailed description of the model.
        """
        cos_th = math.cos(self.robot_theta)
        sin_th = math.sin(self.robot_theta)

        # Decompose world velocity into body frame (old heading)
        body_vx = self._vel_x * cos_th + self._vel_y * sin_th  # longitudinal
        body_vy = -self._vel_x * sin_th + self._vel_y * cos_th  # lateral

        # Longitudinal: track the wheel command directly
        body_vx = forward_velocity

        # Lateral: decay by grip
        body_vy *= 1.0 - self.lateral_grip

        # Back to world frame using the *old* heading
        self._vel_x = body_vx * cos_th - body_vy * sin_th
        self._vel_y = body_vx * sin_th + body_vy * cos_th

        # Angular velocity with momentum
        self._omega = self.yaw_damping * self._omega + (1.0 - self.yaw_damping) * angular_velocity

        # Rotate heading (world velocity keeps its old direction -> drift)
        self.robot_theta += self._omega * self.dt

        # Integrate position
        self.robot_x += self._vel_x * self.dt
        self.robot_y += self._vel_y * self.dt

        # Expose diagnostics in the *new* body frame
        cos_new = math.cos(self.robot_theta)
        sin_new = math.sin(self.robot_theta)
        self.lateral_velocity = -self._vel_x * sin_new + self._vel_y * cos_new

        speed = math.hypot(self._vel_x, self._vel_y)
        if speed > 1e-6:
            vel_angle = math.atan2(self._vel_y, self._vel_x)
            self.slip_angle = math.atan2(
                math.sin(vel_angle - self.robot_theta),
                math.cos(vel_angle - self.robot_theta),
            )
        else:
            self.slip_angle = 0.0

    # ------------------------------------------------------------------
    # Segment-progress tracking
    # ------------------------------------------------------------------

    def _compute_segment_advance(self) -> int:
        """Return how many segments the robot advanced this step (signed)."""
        seg_advance = (self.current_segment_index - self._prev_segment_index) % self.num_track_segments
        if seg_advance > self.num_track_segments // 2:
            seg_advance -= self.num_track_segments
        self.total_progress += seg_advance
        self._prev_segment_index = self.current_segment_index
        return seg_advance

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward_drift(
        self,
        forward_velocity: float,
        lateral_error: float,
        heading_error: float,
        seg_advance: int,
        prev_lap_count: int,
    ) -> float:
        if self.reward_mode == "racing":
            progress_reward = float(seg_advance) / self.num_track_segments
            centering_penalty = -0.1 * (math.fabs(lateral_error) / self.off_track_threshold) ** 2
            lap_bonus = 10.0 if self._next_checkpoint == 1 and self.lap_count > prev_lap_count else 0.0
            return progress_reward + centering_penalty + lap_bonus

        elif self.reward_mode == "racingv2":
            max_speed = self.max_wheel_speed
            # going fast close to the center of lane yeilds best reward
            return (1.0 - (math.fabs(lateral_error) / self.off_track_threshold) ** 2) * (forward_velocity / max_speed)

        # line_following — delegate to the base reward
        return self._compute_reward(forward_velocity, lateral_error, heading_error)

    # ------------------------------------------------------------------
    # Extra info keys
    # ------------------------------------------------------------------

    def _drift_info(self) -> dict[str, Any]:
        return {
            "drift": True,
            "slip_angle": self.slip_angle,
            "lateral_velocity": self.lateral_velocity,
            "total_progress": self.total_progress,
            "reward_mode": self.reward_mode,
        }

    # ------------------------------------------------------------------
    # Rendering overrides
    # ------------------------------------------------------------------

    def _render_robot(self, surface: Any, pygame_module: Any, closest_pt: NDArray[np.float64]) -> None:
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
            vel_x=self._vel_x,
            vel_y=self._vel_y,
            slip_angle=self.slip_angle,
        )

    def _render_hud(
        self,
        surface: Any,
        pygame_module: Any,
        lateral_error: float,
        heading_error: float,
    ) -> None:
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
            next_checkpoint=self._next_checkpoint,
            num_checkpoints=self.num_checkpoints,
            drift=True,
            slip_angle=self.slip_angle,
            lateral_velocity=self.lateral_velocity,
        )
