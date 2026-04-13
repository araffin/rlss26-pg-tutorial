"""pg_tutorial - policy-gradient tutorial helpers & environments."""

from gymnasium.envs.registration import register

from pg_tutorial.envs.constant_speed import LineFollowerConstantSpeedEnv
from pg_tutorial.envs.drift import LineFollowerDriftEnv
from pg_tutorial.envs.line_follower import LineFollowerEnv

__all__ = ["LineFollowerConstantSpeedEnv", "LineFollowerDriftEnv", "LineFollowerEnv"]

register(
    id="LineFollower-v0",
    entry_point="pg_tutorial.envs.line_follower:LineFollowerEnv",
    max_episode_steps=1000,
)

register(
    id="LineFollowerDrift-v0",
    entry_point="pg_tutorial.envs.drift:LineFollowerDriftEnv",
    max_episode_steps=1000,
)

register(
    id="LineFollowerRacing-v0",
    entry_point="pg_tutorial.envs.drift:LineFollowerDriftEnv",
    max_episode_steps=1000,
    kwargs={"reward_mode": "racing"},
)

register(
    id="LineFollowerRacingCustom-v0",
    entry_point="pg_tutorial.envs.drift:LineFollowerDriftEnv",
    max_episode_steps=1000,
    kwargs={"reward_mode": "racingv2"},
)

register(
    id="LineFollowerConstantSpeed-v0",
    entry_point="pg_tutorial.envs.constant_speed:LineFollowerConstantSpeedEnv",
    max_episode_steps=1000,
)
