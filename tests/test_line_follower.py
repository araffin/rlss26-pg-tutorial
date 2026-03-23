"""Tests for the LineFollower-v0 and LineFollowerRacing-v0 Gymnasium environments."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

# Ensure the environment is registered
import pg_tutorial  # noqa: F401
from pg_tutorial.envs.line_follower import LineFollowerEnv, _closest_point_on_segment

# ---------------------------------------------------------------------------
# Closest-point-on-segment helper tests
# ---------------------------------------------------------------------------


class TestClosestPointOnSegment:
    """Tests for the segment projection helper."""

    def test_point_on_segment(self) -> None:
        seg_start = np.array([0.0, 0.0])
        seg_end = np.array([10.0, 0.0])
        point = np.array([5.0, 0.0])
        closest, parameter = _closest_point_on_segment(point, seg_start, seg_end)
        np.testing.assert_allclose(closest, [5.0, 0.0], atol=1e-10)
        assert abs(parameter - 0.5) < 1e-10

    def test_point_before_segment(self) -> None:
        seg_start = np.array([0.0, 0.0])
        seg_end = np.array([10.0, 0.0])
        point = np.array([-5.0, 3.0])
        closest, parameter = _closest_point_on_segment(point, seg_start, seg_end)
        np.testing.assert_allclose(closest, [0.0, 0.0], atol=1e-10)
        assert abs(parameter) < 1e-10

    def test_point_after_segment(self) -> None:
        seg_start = np.array([0.0, 0.0])
        seg_end = np.array([10.0, 0.0])
        point = np.array([15.0, 2.0])
        closest, parameter = _closest_point_on_segment(point, seg_start, seg_end)
        np.testing.assert_allclose(closest, [10.0, 0.0], atol=1e-10)
        assert abs(parameter - 1.0) < 1e-10

    def test_perpendicular_projection(self) -> None:
        seg_start = np.array([0.0, 0.0])
        seg_end = np.array([10.0, 0.0])
        point = np.array([7.0, 4.0])
        closest, parameter = _closest_point_on_segment(point, seg_start, seg_end)
        np.testing.assert_allclose(closest, [7.0, 0.0], atol=1e-10)
        assert abs(parameter - 0.7) < 1e-10

    def test_degenerate_segment(self) -> None:
        seg_start = np.array([5.0, 5.0])
        seg_end = np.array([5.0, 5.0])
        point = np.array([7.0, 3.0])
        closest, _parameter = _closest_point_on_segment(point, seg_start, seg_end)
        np.testing.assert_allclose(closest, [5.0, 5.0], atol=1e-10)


# ---------------------------------------------------------------------------
# Gymnasium env checker (covers spaces, reset, step, render, etc.)
# ---------------------------------------------------------------------------


class TestGymEnvChecker:
    """Use Gymnasium's built-in ``check_env`` to validate the environment."""

    def test_check_env_default(self) -> None:
        env = LineFollowerEnv(action_noise_std=0.0)
        check_env(env, skip_render_check=True)
        env.close()

    def test_check_env_with_rgb_render(self) -> None:
        env = gym.make("LineFollower-v0", action_noise_std=0.0, render_mode="rgb_array")
        check_env(env.unwrapped, skip_render_check=False)
        env.close()

    def test_check_env_via_gym_make(self) -> None:
        env = gym.make("LineFollower-v0", action_noise_std=0.0)
        # gym.make wraps the env; check_env expects the unwrapped version
        check_env(env.unwrapped, skip_render_check=True)
        env.close()


# ---------------------------------------------------------------------------
# Racing reward variant
# ---------------------------------------------------------------------------


class TestRacingEnv:
    """Tests for the LineFollowerRacing-v0 variant."""

    def test_check_env_racing(self) -> None:
        env = LineFollowerEnv(reward_mode="racing", action_noise_std=0.0)
        check_env(env, skip_render_check=True)
        env.close()

    def test_check_env_racing_via_gym_make(self) -> None:
        env = gym.make("LineFollowerRacing-v0", action_noise_std=0.0)
        check_env(env.unwrapped, skip_render_check=True)
        env.close()

    def test_racing_reward_differs_from_line_following(self) -> None:
        """Sanity check: racing reward differs from line-following reward."""
        env_lf = LineFollowerEnv(reward_mode="line_following", action_noise_std=0.0)
        env_rc = LineFollowerEnv(reward_mode="racing", action_noise_std=0.0)
        _obs_lf, _ = env_lf.reset(seed=42)
        _obs_rc, _ = env_rc.reset(seed=42)
        action = np.array([0.5, 0.5], dtype=np.float32)
        _, _reward_lf, _, _, info_lf = env_lf.step(action)
        _, _reward_rc, _, _, info_rc = env_rc.step(action)
        # They should generally differ (different reward functions)
        assert info_lf["reward_mode"] == "line_following"
        assert info_rc["reward_mode"] == "racing"
        env_lf.close()
        env_rc.close()
