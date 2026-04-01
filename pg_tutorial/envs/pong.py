"""Pong environment with gymnasium interface using pygame for rendering.

This environment implements a classic Pong game where an RL agent controls
one paddle against a simple AI opponent. The observation space includes
the positions and velocities of both paddles and the ball.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

try:
    import pygame
except ImportError:
    pygame = None  # type: ignore


class PongEnv(gym.Env):
    """Classic Pong game environment for reinforcement learning.

    The agent controls the left paddle in a 2D Pong game. The goal is to
    keep the ball in play and score points by getting the ball past the
    opponent's paddle.

    Observation Space:
        The observation is a 7-dimensional vector containing:
        - Ball position x (normalized to [0, 1])
        - Ball position y (normalized to [0, 1])
        - Ball velocity x (normalized to [-1, 1])
        - Ball velocity y (normalized to [-1, 1])
        - Left paddle position (normalized to [0, 1])
        - Right paddle position (normalized to [0, 1])
        - Left paddle velocity (normalized to [-1, 1])

    Action Space:
        Discrete actions:
        - 0: Move paddle up
        - 1: Move paddle down
        - 2: Stay stationary

    Reward:
        - +1 for scoring a point (ball passes right paddle)
        - -1 for losing a point (ball passes left paddle)
        - Each point ends the current episode
        - Match score persists across resets until a player wins the match

    Parameters
    ----------
    screen_width : int
        Width of the pygame screen in pixels. Default: 800
    screen_height : int
        Height of the pygame screen in pixels. Default: 600
    paddle_height : float
        Height of paddles as fraction of screen height. Default: 0.15
    paddle_width : float
        Width of paddles as fraction of screen width. Default: 0.02
    ball_size : float
        Radius of the ball as fraction of screen height. Default: 0.02
    paddle_speed : float
        Maximum paddle speed as fraction of screen height per second. Default: 0.5
    ball_speed : float
        Initial ball speed as fraction of screen width per second. Default: 0.4
    max_episode_steps : int
        Maximum number of steps per episode. Default: 500
    render_mode : str | None
        Render mode: None (no rendering), 'human' (display window), 'rgb_array' (return RGB array)
    winning_score : int
        Number of points needed to win a match. Default: 10
    result_callback : Callable[[dict[str, Any]], None] | None
        Optional callback invoked when a match ends. Useful for custom SB3 logging.
    """

    metadata = {  # noqa: RUF012
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }

    def __init__(
        self,
        screen_width: int = 800,
        screen_height: int = 600,
        paddle_height: float = 0.15,
        paddle_width: float = 0.02,
        ball_size: float = 0.02,
        paddle_speed: float = 0.8,
        ball_speed: float = 1.0,
        max_episode_steps: int = 2000,
        render_mode: str | None = None,
        winning_score: int = 10,
        result_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        super().__init__()

        # Screen dimensions
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Paddle dimensions (normalized)
        self.paddle_height = paddle_height
        self.paddle_width = paddle_width

        # Ball size (normalized)
        self.ball_size = ball_size

        # Speeds (normalized)
        self.paddle_speed = paddle_speed
        self.ball_speed = ball_speed

        # Maximum episode steps
        self.max_episode_steps = max_episode_steps
        self.winning_score = winning_score
        self.result_callback = result_callback

        # Render mode
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError(f"render_mode must be None, 'human', or 'rgb_array', got {render_mode!r}")
        self.render_mode = render_mode

        # Game state variables
        self.ball_x: float = 0.0
        self.ball_y: float = 0.0
        self.ball_vx: float = 0.0
        self.ball_vy: float = 0.0

        self.left_paddle_y: float = 0.5
        self.right_paddle_y: float = 0.5
        self.left_paddle_v: float = 0.0
        self.right_paddle_v: float = 0.0

        self.step_count: int = 0
        self.score_left: int = 0
        self.score_right: int = 0

        # Pygame surface (initialized on first render)
        self.screen: Any = None
        self.clock: Any = None
        self.font: Any = None

        # Define action space: 0 = up, 1 = down, 2 = stationary
        self.action_space = spaces.Discrete(3)

        # Define observation space: [ball_x, ball_y, ball_vx, ball_vy, left_paddle_y, right_paddle_y, left_paddle_v]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0, -1.0, 0.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        # Reset step counter
        self.step_count = 0

        if self.score_left >= self.winning_score or self.score_right >= self.winning_score:
            self.score_left = 0
            self.score_right = 0

        if self.score_left == 0 and self.score_right == 0:
            self._reset_match()
        else:
            serving_left = self.ball_vx >= 0
            self._reset_serve(serving_left=serving_left)

        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    def step(
        self,
        action: int,
    ) -> tuple[NDArray[np.float32], SupportsFloat, bool, bool, dict[str, Any]]:
        """Take a step in the environment.

        Parameters
        ----------
        action : int
            Action to take: 0 = up, 1 = down, 2 = stationary

        Returns
        -------
        observation : np.ndarray
            Current observation
        reward : float
            Reward for this step
        terminated : bool
            Whether episode terminated
        truncated : bool
            Whether episode truncated
        info : dict
            Additional information
        """
        self.step_count += 1

        # Apply action to left paddle
        self.left_paddle_v = self._action_to_velocity(action)

        # Update opponent AI before moving paddles so it also works without rendering
        self._update_opponent()

        # Update paddle positions
        self.left_paddle_y += self.left_paddle_v * self.dt
        self.right_paddle_y += self.right_paddle_v * self.dt

        # Constrain paddle positions
        self.left_paddle_y = np.clip(self.left_paddle_y, self.paddle_height / 2, 1 - self.paddle_height / 2)
        self.right_paddle_y = np.clip(self.right_paddle_y, self.paddle_height / 2, 1 - self.paddle_height / 2)

        # Update ball position
        self.ball_x += self.ball_vx * self.dt
        self.ball_y += self.ball_vy * self.dt

        # Ball-wall collisions (top and bottom)
        if self.ball_y <= self.ball_size:
            self.ball_y = self.ball_size
            self.ball_vy = abs(self.ball_vy)
        elif self.ball_y >= 1 - self.ball_size:
            self.ball_y = 1 - self.ball_size
            self.ball_vy = -abs(self.ball_vy)

        # Ball-paddle collisions
        self._handle_paddle_collisions()

        # Check for scoring
        reward = 0.0
        terminated = False

        if self.ball_x <= 0:
            # Right side scores (agent loses point)
            self.score_right += 1
            reward = -1.0
            terminated = True
            if self.score_right >= self.winning_score:
                self._notify_match_result()
        elif self.ball_x >= 1:
            # Left side scores (agent wins point)
            self.score_left += 1
            reward = 1.0
            terminated = True
            if self.score_left >= self.winning_score:
                self._notify_match_result()

        # Truncation by step count
        truncated = self.step_count >= self.max_episode_steps

        observation = self._get_observation()
        info = self._get_info()
        return observation, float(reward), terminated, truncated, info

    def _action_to_velocity(self, action: int) -> float:
        """Convert discrete action to paddle velocity."""
        if action == 0:  # Up
            return -self.paddle_speed
        elif action == 1:  # Down
            return self.paddle_speed
        else:  # Stationary
            return 0.0

    def _handle_paddle_collisions(self) -> None:
        """Handle ball-paddle collisions."""
        # Left paddle collision
        paddle_left_x = self.paddle_width / 2
        paddle_right_x = 1 - self.paddle_width / 2

        # Check left paddle
        if (
            self.ball_x - self.ball_size <= paddle_left_x + self.paddle_width / 2
            and self.ball_x + self.ball_size >= paddle_left_x - self.paddle_width / 2
            and abs(self.ball_y - self.left_paddle_y) <= self.paddle_height / 2 + self.ball_size
            and self.ball_vx < 0
        ):
            # Collision with left paddle
            self.ball_x = paddle_left_x + self.paddle_width / 2 + self.ball_size
            self.ball_vx = abs(self.ball_vx)
            # Add some variation based on hit position
            hit_pos = (self.ball_y - self.left_paddle_y) / (self.paddle_height / 2)
            self.ball_vy += hit_pos * 0.1
            self.ball_vy = np.clip(self.ball_vy, -self.ball_speed, self.ball_speed)

        # Check right paddle
        if (
            self.ball_x + self.ball_size >= paddle_right_x - self.paddle_width / 2
            and self.ball_x - self.ball_size <= paddle_right_x + self.paddle_width / 2
            and abs(self.ball_y - self.right_paddle_y) <= self.paddle_height / 2 + self.ball_size
            and self.ball_vx > 0
        ):
            # Collision with right paddle
            self.ball_x = paddle_right_x - self.paddle_width / 2 - self.ball_size
            self.ball_vx = -abs(self.ball_vx)
            # Add some variation based on hit position
            hit_pos = (self.ball_y - self.right_paddle_y) / (self.paddle_height / 2)
            self.ball_vy += hit_pos * 0.1
            self.ball_vy = np.clip(self.ball_vy, -self.ball_speed, self.ball_speed)

    def _update_opponent(self) -> None:
        """Update the opponent paddle with simple ball prediction."""
        max_speed = self.paddle_speed * 0.95
        dead_zone = 0.015

        if self.ball_vx > 0:
            time_to_paddle = (1 - self.paddle_width - self.ball_x) / max(self.ball_vx, 1e-6)
            predicted_y = self.ball_y + self.ball_vy * max(time_to_paddle, 0.0)
            predicted_y = self._reflect_vertical_position(predicted_y)
            target_y = predicted_y
        else:
            target_y = 0.5 + 0.35 * (self.ball_y - 0.5)

        diff = target_y - self.right_paddle_y
        if abs(diff) > dead_zone:
            self.right_paddle_v = float(np.clip(diff * 6.0, -max_speed, max_speed))
        else:
            self.right_paddle_v = 0.0

    def _reflect_vertical_position(self, y: float) -> float:
        """Reflect a y position into the playfield accounting for wall bounces."""
        min_y = self.ball_size
        max_y = 1 - self.ball_size
        span = max_y - min_y

        if span <= 0:
            return 0.5

        shifted = y - min_y
        period = 2 * span
        wrapped = shifted % period
        if wrapped > span:
            wrapped = period - wrapped
        return float(min_y + wrapped)

    def _reset_match(self) -> None:
        """Reset scores and positions for a new match."""
        self.score_left = 0
        self.score_right = 0
        self.left_paddle_y = 0.5
        self.right_paddle_y = 0.5
        self.left_paddle_v = 0.0
        self.right_paddle_v = 0.0
        serving_left = bool(self.np_random.integers(0, 2))
        self._reset_serve(serving_left=serving_left)

    def _reset_serve(self, serving_left: bool) -> None:
        """Reset paddles and ball for the next point."""
        self.ball_x = 0.5
        self.ball_y = float(np.clip(self.np_random.uniform(0.25, 0.75), self.ball_size, 1 - self.ball_size))
        self.left_paddle_y = 0.5
        self.right_paddle_y = 0.5
        self.left_paddle_v = 0.0
        self.right_paddle_v = 0.0

        direction = 1 if serving_left else -1
        launch_angle = float(self.np_random.uniform(-0.6, 0.6))
        self.ball_vx = direction * self.ball_speed
        self.ball_vy = launch_angle * self.ball_speed

    def _notify_match_result(self) -> None:
        """Notify listeners when a match ends."""
        if self.result_callback is None:
            return

        self.result_callback(
            {
                "winner": "left" if self.score_left > self.score_right else "right",
                "score_left": self.score_left,
                "score_right": self.score_right,
                "steps": self.step_count,
            }
        )

    def _get_observation(self) -> NDArray[np.float32]:
        """Get current observation."""
        observation = np.array(
            [
                float(self.ball_x),
                float(self.ball_y),
                float(self.ball_vx / self.ball_speed),  # Normalize
                float(self.ball_vy / self.ball_speed),  # Normalize
                float(self.left_paddle_y),
                float(self.right_paddle_y),
                float(self.left_paddle_v / self.paddle_speed),  # Normalize
            ],
            dtype=np.float32,
        )
        return observation

    def _get_info(self) -> dict[str, Any]:
        """Get additional information."""
        left_won = self.score_left >= self.winning_score
        right_won = self.score_right >= self.winning_score
        match_over = left_won or right_won

        return {
            "score_left": self.score_left,
            "score_right": self.score_right,
            "steps": self.step_count,
            "winning_score": self.winning_score,
            "point_difference": self.score_left - self.score_right,
            "match_over": match_over,
            "winner": "left" if left_won else "right" if right_won else None,
            "sb3_logs": {
                "pong/score_left": float(self.score_left),
                "pong/score_right": float(self.score_right),
                "pong/point_difference": float(self.score_left - self.score_right),
                "pong/match_over": float(match_over),
                "pong/agent_win": float(left_won),
                "pong/opponent_win": float(right_won),
            },
        }

    @property
    def dt(self) -> float:
        """Time step size in seconds."""
        return 1.0 / 60.0

    def render(self) -> np.ndarray | None:  # type: ignore[override]
        """Render the environment.

        Returns
        -------
        rgb_array : np.ndarray | None
            RGB array if render_mode is 'rgb_array', None otherwise
        """
        if pygame is None:
            raise RuntimeError("pygame is not installed. Install it with: pip install pygame")

        if self.render_mode is None:
            return None

        if self.screen is None:
            self._init_pygame()

        # Clear screen
        self.screen.fill((0, 0, 0))

        # Draw center line
        for y in range(0, self.screen_height, 20):
            pygame.draw.rect(
                self.screen,
                (255, 255, 255),
                (self.screen_width // 2 - 2, y, 4, 10),
            )

        # Draw left paddle
        paddle_rect = pygame.Rect(
            int(self.paddle_width / 2 * self.screen_width),
            int((self.left_paddle_y - self.paddle_height / 2) * self.screen_height),
            int(self.paddle_width * self.screen_width),
            int(self.paddle_height * self.screen_height),
        )
        pygame.draw.rect(self.screen, (0, 100, 255), paddle_rect)

        # Draw right paddle
        paddle_rect = pygame.Rect(
            int((1 - self.paddle_width / 2) * self.screen_width - self.paddle_width * self.screen_width),
            int((self.right_paddle_y - self.paddle_height / 2) * self.screen_height),
            int(self.paddle_width * self.screen_width),
            int(self.paddle_height * self.screen_height),
        )
        pygame.draw.rect(self.screen, (255, 100, 0), paddle_rect)

        # Draw ball
        ball_radius = int(self.ball_size * self.screen_height)
        pygame.draw.circle(
            self.screen,
            (255, 255, 255),
            (int(self.ball_x * self.screen_width), int(self.ball_y * self.screen_height)),
            ball_radius,
        )

        # Draw scores
        if self.font is not None:
            score_text = self.font.render(f"{self.score_left} - {self.score_right}", True, (255, 255, 255))
            self.screen.blit(score_text, (self.screen_width // 2 - score_text.get_width() // 2, 20))

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None

        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

        return None

    def _init_pygame(self) -> None:
        """Initialize pygame and create screen."""
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Pong")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 74)

    def close(self) -> None:
        """Close the environment and clean up pygame."""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None
            self.font = None
