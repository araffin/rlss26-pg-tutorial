"""Constant-speed variant of the line-follower environment.

:class:`LineFollowerConstantSpeedEnv` inherits from :class:`LineFollowerEnv`
and reduces the action space to a **single scalar** — steering — while
keeping the forward speed fixed.  This matches the interface of a PD
controller: the agent only decides *how much to steer*, not *how fast to go*.

Action mapping
--------------
The agent outputs ``steering`` ∈ [-1, 1].  Internally this is converted to
differential-drive wheel commands::

    left_wheel  = base_speed + steering
    right_wheel = base_speed - steering

before being passed through the usual inertia / friction / noise pipeline
of the base class.  ``base_speed`` is a constructor parameter (default 0.6,
i.e. 60 % of ``max_wheel_speed``).

The observation space is identical to the base environment so that policies
and value functions transfer without modification.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from pg_tutorial.envs.line_follower import LineFollowerEnv


class LineFollowerConstantSpeedEnv(LineFollowerEnv):
    """Line-follower where the agent controls **steering only**.

    The robot travels at a constant base speed and the single-dimensional
    action dictates how that speed is split between the left and right
    wheels (differential steering).

    Parameters
    ----------
    base_speed:
        Normalised forward speed in (0, 1].  ``1.0`` means both wheels run
        at ``max_wheel_speed`` when steering is zero; ``0.6`` (default)
        leaves headroom for the steering signal.
    **kwargs:
        Forwarded to :class:`LineFollowerEnv` (track selection, physics
        parameters, rendering options, …).
    """

    def __init__(
        self,
        *,
        base_speed: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.base_speed = float(np.clip(base_speed, -1.0, 1.0))

        # Override: 1-D action  →  steering ∈ [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Convert steering to left/right wheel speed command
    # ------------------------------------------------------------------

    def _apply_action(self, action: NDArray[np.float32]) -> tuple[float, float]:
        """Convert scalar steering to two-wheel command, then delegate."""
        steering = float(np.clip(action, -1.0, 1.0))

        # left/right wheel speed command
        two_wheel_action = np.array(
            [self.base_speed + steering, self.base_speed - steering],
            dtype=np.float32,
        )
        return super()._apply_action(two_wheel_action)

    def _compute_reward(
        self,
        forward_velocity: float,
        lateral_error: float,
        heading_error: float,
        *,
        going_reverse: bool = False,
    ) -> float:
        """Line-following reward."""
        if going_reverse:
            return -10.0

        # Pure PD cost
        lateral_penalty = -((lateral_error / self.off_track_threshold) ** 2)
        # heading_penalty = -((heading_error / np.pi) ** 2)
        alive_bonus = 1.0  # otherwise the agent learns to determinate early
        return alive_bonus + lateral_penalty  # + 0.5 * heading_penalty
