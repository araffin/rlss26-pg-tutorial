"""pg_tutorial - policy-gradient tutorial helpers & environments."""

from gymnasium.envs.registration import register

from pg_tutorial.envs.line_follower import LineFollowerEnv

__all__ = ["LineFollowerEnv"]

register(
    id="LineFollower-v0",
    entry_point="pg_tutorial.envs.line_follower:LineFollowerEnv",
    max_episode_steps=2000,
)

register(
    id="LineFollowerRacing-v0",
    entry_point="pg_tutorial.envs.line_follower:LineFollowerEnv",
    max_episode_steps=2000,
    kwargs={"reward_mode": "racing"},
)
