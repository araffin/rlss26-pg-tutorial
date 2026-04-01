"""Tests for the Pong environment."""

from __future__ import annotations

import gymnasium as gym
from gymnasium.utils.env_checker import check_env

# Ensure all environments are registered
import pg_tutorial  # noqa: F401
from pg_tutorial.envs.pong import PongEnv


class TestPongEnv:
    """Tests for the Pong environment using Gymnasium's check_env."""

    def test_check_env_default(self) -> None:
        """Gymnasium check_env passes with default Pong environment."""
        env = PongEnv()
        check_env(env, skip_render_check=True)
        env.close()

    def test_check_env_with_rgb_render(self) -> None:
        """check_env passes with rgb_array render mode."""
        env = gym.make("MiniPong-v0", render_mode="rgb_array")
        check_env(env.unwrapped, skip_render_check=False)
        env.close()

    def test_check_env_via_gym_make(self) -> None:
        """check_env passes when creating env via gym.make."""
        env = gym.make("MiniPong-v0")
        check_env(env.unwrapped, skip_render_check=True)
        env.close()
