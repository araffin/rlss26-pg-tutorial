"""Tests for the LineFollower-v0 Gymnasium environment."""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
import pytest

# Ensure the environment is registered
import pg_tutorial  # noqa: F401
from pg_tutorial.envs.line_follower import (
    LineFollowerEnv,
    _closest_point_on_segment,
    _make_figure_eight_track,
)

# ---------------------------------------------------------------------------
# Track helper tests
# ---------------------------------------------------------------------------


class TestMakeFigureEightTrack:
    """Tests for the default track generator."""

    def test_shape(self) -> None:
        track = _make_figure_eight_track(num_points=100)
        assert track.shape == (100, 2)

    def test_dtype(self) -> None:
        track = _make_figure_eight_track()
        assert track.dtype == np.float64

    def test_center(self) -> None:
        center_x, center_y = 200.0, 150.0
        track = _make_figure_eight_track(center_x=center_x, center_y=center_y, num_points=500)
        mean_x = float(np.mean(track[:, 0]))
        mean_y = float(np.mean(track[:, 1]))
        assert abs(mean_x - center_x) < 5.0
        assert abs(mean_y - center_y) < 5.0


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
# Environment construction / space tests
# ---------------------------------------------------------------------------


class TestEnvironmentConstruction:
    """Tests for creating the environment with various parameters."""

    def test_make_via_gym(self) -> None:
        env = gym.make("LineFollower-v0")
        assert isinstance(env.unwrapped, LineFollowerEnv)
        env.close()

    def test_direct_instantiation(self) -> None:
        env = LineFollowerEnv()
        assert env.action_space.shape == (2,)
        assert env.observation_space.shape == (6,)
        env.close()

    def test_custom_parameters(self) -> None:
        env = LineFollowerEnv(
            friction=0.2,
            action_noise_std=0.5,
            wheel_base=30.0,
            wheel_radius=8.0,
            max_wheel_speed=15.0,
        )
        assert env.friction == pytest.approx(0.2)
        assert env.action_noise_std == pytest.approx(0.5)
        assert env.wheel_base == pytest.approx(30.0)
        assert env.wheel_radius == pytest.approx(8.0)
        assert env.max_wheel_speed == pytest.approx(15.0)
        env.close()

    def test_friction_is_clipped(self) -> None:
        env = LineFollowerEnv(friction=5.0)
        assert env.friction <= 1.0
        env.close()

        env = LineFollowerEnv(friction=-1.0)
        assert env.friction >= 0.0
        env.close()

    def test_negative_noise_std_clipped_to_zero(self) -> None:
        env = LineFollowerEnv(action_noise_std=-0.5)
        assert env.action_noise_std >= 0.0
        env.close()

    def test_custom_track(self) -> None:
        square_track = np.array(
            [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]],
            dtype=np.float64,
        )
        env = LineFollowerEnv(track_waypoints=square_track)
        assert env.num_track_segments == 4
        env.close()


# ---------------------------------------------------------------------------
# Reset tests
# ---------------------------------------------------------------------------


class TestReset:
    """Tests for environment reset."""

    def test_reset_returns_observation_and_info(self) -> None:
        env = LineFollowerEnv()
        observation, info = env.reset(seed=0)
        assert observation.shape == (6,)
        assert observation.dtype == np.float32
        assert isinstance(info, dict)
        env.close()

    def test_observation_in_bounds(self) -> None:
        env = LineFollowerEnv()
        observation, _info = env.reset(seed=42)
        assert env.observation_space.contains(observation), f"Observation {observation} not in observation space"
        env.close()

    def test_reset_places_robot_at_start(self) -> None:
        env = LineFollowerEnv()
        _observation, info = env.reset(seed=7)
        start_wp = env.track_waypoints[0]
        assert abs(info["robot_x"] - start_wp[0]) < 1e-6
        assert abs(info["robot_y"] - start_wp[1]) < 1e-6
        env.close()

    def test_reset_is_deterministic_with_seed(self) -> None:
        env = LineFollowerEnv()
        obs_a, _ = env.reset(seed=123)
        obs_b, _ = env.reset(seed=123)
        np.testing.assert_array_equal(obs_a, obs_b)
        env.close()

    def test_step_count_resets_to_zero(self) -> None:
        env = LineFollowerEnv()
        env.reset(seed=0)
        env.step(np.array([0.5, 0.5], dtype=np.float32))
        env.step(np.array([0.5, 0.5], dtype=np.float32))
        assert env.step_count == 2
        env.reset(seed=0)
        assert env.step_count == 0
        env.close()


# ---------------------------------------------------------------------------
# Step tests
# ---------------------------------------------------------------------------


class TestStep:
    """Tests for the step method."""

    def test_step_returns_correct_types(self) -> None:
        env = LineFollowerEnv()
        env.reset(seed=0)
        action = np.array([0.5, 0.5], dtype=np.float32)
        observation, reward, terminated, truncated, info = env.step(action)
        assert observation.shape == (6,)
        assert observation.dtype == np.float32
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        env.close()

    def test_observation_in_bounds_after_step(self) -> None:
        env = LineFollowerEnv(action_noise_std=0.0)
        env.reset(seed=0)
        for _ in range(50):
            action = env.action_space.sample()
            observation, _reward, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                break
            assert env.observation_space.contains(observation), f"Observation {observation} out of bounds"
        env.close()

    def test_forward_motion_straight_line(self) -> None:
        """Equal wheel speeds should drive the robot forward in a straight line."""
        env = LineFollowerEnv(friction=0.0, action_noise_std=0.0)
        env.reset(seed=0)
        initial_x = env.robot_x
        initial_y = env.robot_y
        initial_theta = env.robot_theta

        # Drive forward for several steps
        action = np.array([0.5, 0.5], dtype=np.float32)
        for _ in range(20):
            env.step(action)

        # Robot should have moved forward along initial heading
        displacement_x = env.robot_x - initial_x
        displacement_y = env.robot_y - initial_y
        distance = math.hypot(displacement_x, displacement_y)
        assert distance > 0.0, "Robot should have moved forward"

        # Heading should be unchanged for equal wheel speeds
        heading_change = abs(env.robot_theta - initial_theta)
        heading_change = min(heading_change, 2.0 * math.pi - heading_change)
        assert heading_change < 1e-6, "Heading should not change with equal wheel speeds"
        env.close()

    def test_turning_with_differential_speeds(self) -> None:
        """Unequal wheel speeds should cause the robot to turn."""
        env = LineFollowerEnv(friction=0.0, action_noise_std=0.0)
        env.reset(seed=0)
        initial_theta = env.robot_theta

        # Right wheel faster → should turn left (decrease theta in standard frame)
        action = np.array([0.2, 0.8], dtype=np.float32)
        for _ in range(10):
            env.step(action)

        heading_change = env.robot_theta - initial_theta
        # The robot should have rotated
        assert abs(heading_change) > 0.01, "Robot should have turned"
        env.close()

    def test_no_noise_deterministic(self) -> None:
        """With zero noise the environment should be deterministic."""
        env = LineFollowerEnv(friction=0.0, action_noise_std=0.0)

        env.reset(seed=99)
        action = np.array([0.3, 0.7], dtype=np.float32)
        obs_a, reward_a, _, _, _ = env.step(action)

        env.reset(seed=99)
        obs_b, reward_b, _, _, _ = env.step(action)

        np.testing.assert_array_equal(obs_a, obs_b)
        assert reward_a == reward_b
        env.close()

    def test_friction_reduces_speed(self) -> None:
        """Higher friction should lead to lower effective wheel speed."""
        env_low_friction = LineFollowerEnv(friction=0.0, action_noise_std=0.0)
        env_low_friction.reset(seed=0)
        action = np.array([1.0, 1.0], dtype=np.float32)
        env_low_friction.step(action)
        speed_low = abs(env_low_friction.left_wheel_speed)

        env_high_friction = LineFollowerEnv(friction=0.5, action_noise_std=0.0)
        env_high_friction.reset(seed=0)
        env_high_friction.step(action)
        speed_high = abs(env_high_friction.left_wheel_speed)

        assert speed_low > speed_high, "Higher friction should reduce wheel speed"
        env_low_friction.close()
        env_high_friction.close()

    def test_action_clipping(self) -> None:
        """Actions outside [-1, 1] should be clipped without error."""
        env = LineFollowerEnv(action_noise_std=0.0)
        env.reset(seed=0)
        large_action = np.array([5.0, -5.0], dtype=np.float32)
        observation, _reward, _terminated, _truncated, _info = env.step(large_action)
        assert observation.shape == (6,)
        env.close()

    def test_truncation_at_max_steps(self) -> None:
        """Episode should truncate at max_episode_steps."""
        max_steps = 10
        env = LineFollowerEnv(
            max_episode_steps=max_steps,
            action_noise_std=0.0,
            off_track_threshold=1e6,  # prevent early termination
        )
        env.reset(seed=0)

        truncated = False
        terminated = False
        for _step_idx in range(max_steps + 5):
            action = np.array([0.3, 0.3], dtype=np.float32)
            _observation, _reward, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                break

        assert truncated, "Episode should have been truncated"
        assert not terminated, "Episode should not have terminated (off-track)"
        assert env.step_count == max_steps
        env.close()

    def test_termination_when_off_track(self) -> None:
        """Going far from the track should terminate the episode."""
        env = LineFollowerEnv(
            off_track_threshold=5.0,
            friction=0.0,
            action_noise_std=0.0,
        )
        env.reset(seed=0)

        # First spin in place to turn away from the track direction
        spin_action = np.array([1.0, -1.0], dtype=np.float32)
        for _ in range(20):
            _observation, _reward, terminated, truncated, _info = env.step(spin_action)
            if terminated or truncated:
                break

        # Then drive straight ahead (now perpendicular to the track) to leave it
        forward_action = np.array([1.0, 1.0], dtype=np.float32)
        terminated = False
        for _ in range(500):
            _observation, _reward, terminated, truncated, _info = env.step(forward_action)
            if terminated or truncated:
                break

        assert terminated, "Robot should go off-track and trigger termination"
        env.close()


# ---------------------------------------------------------------------------
# Track error computation tests
# ---------------------------------------------------------------------------


class TestTrackErrors:
    """Tests for _compute_track_errors."""

    def test_zero_error_at_start(self) -> None:
        """After reset the robot sits on the track, so lateral error ≈ 0."""
        env = LineFollowerEnv(action_noise_std=0.0)
        env.reset(seed=0)
        lateral_error, heading_error, _closest_point = env._compute_track_errors()
        assert abs(lateral_error) < 1e-3
        assert abs(heading_error) < 1e-3
        env.close()


# ---------------------------------------------------------------------------
# Rendering tests
# ---------------------------------------------------------------------------


class TestRendering:
    """Tests for the rendering pipeline."""

    def test_rgb_array_mode(self) -> None:
        env = LineFollowerEnv(
            render_mode="rgb_array",
            screen_width=200,
            screen_height=150,
        )
        env.reset(seed=0)
        frame = env.render()
        assert frame is not None
        assert frame.shape == (150, 200, 3)
        assert frame.dtype == np.uint8
        env.close()

    def test_render_without_mode_returns_none(self) -> None:
        env = LineFollowerEnv(render_mode=None)
        env.reset(seed=0)
        result = env.render()
        assert result is None
        env.close()

    def test_rgb_array_after_steps(self) -> None:
        env = LineFollowerEnv(
            render_mode="rgb_array",
            screen_width=200,
            screen_height=150,
        )
        env.reset(seed=0)
        for _ in range(5):
            action = env.action_space.sample()
            env.step(action)
        frame = env.render()
        assert frame is not None
        assert frame.shape == (150, 200, 3)
        env.close()


# ---------------------------------------------------------------------------
# Integration test: gym.make round-trip
# ---------------------------------------------------------------------------


class TestGymIntegration:
    """End-to-end test through the gym.make interface."""

    def test_gym_make_reset_step_close(self) -> None:
        env = gym.make("LineFollower-v0", friction=0.1, action_noise_std=0.0)
        observation, _ = env.reset(seed=0)
        assert observation.shape == (6,)
        for _ in range(10):
            action = env.action_space.sample()
            observation, _reward, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                _observation, _info = env.reset()
        env.close()

    def test_gym_make_with_rgb_render(self):
        env = gym.make(
            "LineFollower-v0",
            render_mode="rgb_array",
            screen_width=160,
            screen_height=120,
        )
        _observation, _info = env.reset(seed=0)
        frame = env.render()
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (120, 160, 3)
        env.close()
