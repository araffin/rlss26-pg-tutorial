"""Tests for the LineFollower-v0, LineFollowerDrift-v0, and LineFollowerRacing-v0 environments."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

# Ensure all environments are registered
import pg_tutorial  # noqa: F401
from pg_tutorial.envs.drift import LineFollowerDriftEnv
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
# Base environment (LineFollower-v0)
# ---------------------------------------------------------------------------


class TestGymEnvChecker:
    """Use Gymnasium's built-in ``check_env`` to validate the base environment."""

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
        check_env(env.unwrapped, skip_render_check=True)
        env.close()

    def test_base_info_has_no_drift_keys(self) -> None:
        """The base env should NOT expose drift-specific info keys."""
        env = LineFollowerEnv(action_noise_std=0.0)
        _obs, info = env.reset(seed=0)
        assert "drift" not in info
        assert "slip_angle" not in info
        assert "reward_mode" not in info
        env.close()


# ---------------------------------------------------------------------------
# Drift environment (LineFollowerDrift-v0)
# ---------------------------------------------------------------------------


class TestDriftDynamics:
    """Tests for the drift / tyre-slip mode."""

    def test_check_env_drift(self) -> None:
        """Gymnasium check_env passes with the drift env."""
        env = LineFollowerDriftEnv(action_noise_std=0.0)
        check_env(env, skip_render_check=True)
        env.close()

    def test_check_env_drift_via_gym_make(self) -> None:
        env = gym.make("LineFollowerDrift-v0", action_noise_std=0.0)
        check_env(env.unwrapped, skip_render_check=True)
        env.close()

    def test_check_env_drift_rgb_render(self) -> None:
        env = gym.make(
            "LineFollowerDrift-v0",
            action_noise_std=0.0,
            render_mode="rgb_array",
        )
        check_env(env.unwrapped, skip_render_check=False)
        env.close()

    def test_drift_info_keys_present(self) -> None:
        """Info dict contains drift-related keys."""
        env = LineFollowerDriftEnv(action_noise_std=0.0)
        _obs, info = env.reset(seed=0)
        assert info["drift"] is True
        assert "slip_angle" in info
        assert "lateral_velocity" in info
        assert "total_progress" in info
        assert info["reward_mode"] == "line_following"
        env.close()

    def test_drift_nonzero_slip_while_turning(self) -> None:
        """With low grip, aggressive turning should produce non-zero slip."""
        env = LineFollowerDriftEnv(
            lateral_grip=0.4,
            yaw_damping=0.9,
            action_noise_std=0.0,
            inertia=0.5,
            friction=0.02,
            off_track_threshold=500.0,
        )
        env.reset(seed=0)
        # Build up speed
        straight = np.array([0.4, 0.4], dtype=np.float32)
        for _ in range(40):
            _, _, terminated, truncated, _ = env.step(straight)
            if terminated or truncated:
                break
        # Steer hard
        turn = np.array([0.6, -0.3], dtype=np.float32)
        any_slip = False
        for _ in range(40):
            _, _, terminated, truncated, info = env.step(turn)
            if abs(info["slip_angle"]) > 0.01:
                any_slip = True
                break
            if terminated or truncated:
                break
        assert any_slip, "Expected non-zero slip angle during aggressive turning with low grip"
        env.close()

    def test_high_grip_low_damping_approximates_no_slip(self) -> None:
        """With grip=1.0 and yaw_damping=0.0 the drift env should closely
        approximate the base no-slip kinematics."""
        env_std = LineFollowerEnv(action_noise_std=0.0, inertia=0.0, friction=0.0)
        env_drft = LineFollowerDriftEnv(
            lateral_grip=1.0,
            yaw_damping=0.0,
            action_noise_std=0.0,
            inertia=0.0,
            friction=0.0,
        )
        env_std.reset(seed=42)
        env_drft.reset(seed=42)

        action = np.array([0.6, 0.4], dtype=np.float32)
        for _ in range(50):
            env_std.step(action)
            env_drft.step(action)

        dx = abs(env_std.robot_x - env_drft.robot_x)
        dy = abs(env_std.robot_y - env_drft.robot_y)
        assert dx < 5.0, f"X positions diverged: {dx:.2f}"
        assert dy < 5.0, f"Y positions diverged: {dy:.2f}"
        env_std.close()
        env_drft.close()

    def test_default_drift_mild_enough_for_pd(self) -> None:
        """With default mild drift params a simple PD controller should
        survive at least 500 steps without going off-track."""
        env = LineFollowerDriftEnv(action_noise_std=0.0)
        obs, _ = env.reset(seed=0)

        # PD gains matching examples/pd_controller.py
        kp_lat, kd_lat = 0.02, 0.005
        kp_head, kd_head = 0.8, 0.1
        base_speed = 0.4
        dt = env.dt

        prev_lat = float(obs[0])
        prev_head = float(obs[1])

        alive_steps = 0
        for _ in range(500):
            lat = float(obs[0])
            head = float(obs[1])
            steer = kp_lat * lat + kd_lat * (lat - prev_lat) / dt + kp_head * head + kd_head * (head - prev_head) / dt
            action = np.clip(
                np.array([base_speed + steer, base_speed - steer], dtype=np.float32),
                -1.0,
                1.0,
            )
            prev_lat, prev_head = lat, head
            obs, _, terminated, truncated, _ = env.step(action)
            alive_steps += 1
            if terminated or truncated:
                break
        assert alive_steps >= 500, f"PD controller went off-track after only {alive_steps} steps with default drift params"
        env.close()


# ---------------------------------------------------------------------------
# Racing reward variant (LineFollowerRacing-v0)
# ---------------------------------------------------------------------------


class TestRacingEnv:
    """Tests for the LineFollowerRacing-v0 variant (drift + racing reward)."""

    def test_check_env_racing(self) -> None:
        env = LineFollowerDriftEnv(reward_mode="racing", action_noise_std=0.0)
        check_env(env, skip_render_check=True)
        env.close()

    def test_check_env_racing_via_gym_make(self) -> None:
        env = gym.make("LineFollowerRacing-v0", action_noise_std=0.0)
        check_env(env.unwrapped, skip_render_check=True)
        env.close()

    def test_racing_info_reports_mode(self) -> None:
        env = gym.make("LineFollowerRacing-v0", action_noise_std=0.0)
        _obs, info = env.reset(seed=0)
        assert info["reward_mode"] == "racing"
        assert info["drift"] is True
        env.close()

    def test_racing_reward_differs_from_line_following(self) -> None:
        """Sanity check: racing and line-following rewards differ."""
        env_lf = LineFollowerDriftEnv(reward_mode="line_following", action_noise_std=0.0)
        env_rc = LineFollowerDriftEnv(reward_mode="racing", action_noise_std=0.0)
        _obs_lf, _ = env_lf.reset(seed=42)
        _obs_rc, _ = env_rc.reset(seed=42)
        action = np.array([0.5, 0.5], dtype=np.float32)
        _, _reward_lf, _, _, info_lf = env_lf.step(action)
        _, _reward_rc, _, _, info_rc = env_rc.step(action)
        assert info_lf["reward_mode"] == "line_following"
        assert info_rc["reward_mode"] == "racing"
        env_lf.close()
        env_rc.close()

    def test_invalid_reward_mode_raises(self) -> None:
        """Passing a bogus reward_mode should raise ValueError."""
        try:
            LineFollowerDriftEnv(reward_mode="banana", action_noise_std=0.0)
            raise AssertionError("Expected ValueError")
        except ValueError:
            pass
