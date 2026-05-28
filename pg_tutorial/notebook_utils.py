from torch.backends.cudnn import deterministic
import base64
import os
import warnings
from collections.abc import Callable
from pathlib import Path

import torch as th

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
from IPython import display as ipythondisplay

from pg_tutorial.envs.constant_speed import LineFollowerConstantSpeedEnv

# Filter out the RecordVideo warning about overwriting videos
warnings.filterwarnings("ignore", message=".*Overwriting existing videos.*")


def show_videos(video_path: str = "", prefix: str = "") -> None:
    """
    Taken from https://github.com/eleurent/highway-env

    :param video_path: Path to the folder containing videos
    :param prefix: Filter the video, showing only the only starting with this prefix
    """
    html = []
    for mp4 in Path(video_path).glob(f"{prefix}*.mp4"):
        video_b64 = base64.b64encode(mp4.read_bytes())
        html.append("""<video alt="{}" autoplay
                    loop controls style="height: 400px;">
                    <source src="data:video/mp4;base64,{}" type="video/mp4" />
                </video>""".format(mp4, video_b64.decode("ascii")))
    ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))


IDX_LATERAL_ERROR: int = 0


def evaluate(
    controller: Callable,
    env: gym.Env,
    n_eval_episodes: int = 5,
    video_name: str | None = None,
    video_save_path: str | None = None,
    verbose: bool = True,
) -> tuple[float, float]:
    """
    Evaluate a controller on an environment over multiple episodes.

    :param controller: A callable that takes lateral error and previous lateral error as inputs
        and returns an action.
    :param env: The Gymnasium environment to evaluate on.
    :param n_eval_episodes: Number of episodes to evaluate.
    :param video_name: Optional name for recording a video of the evaluation.
    :param video_save_path: Optional directory path to save videos. If None, defaults to the parent
        directory of this file's folder.
    :param verbose: Whether to print evaluation results.
    :return: A tuple containing (best_lap_time, mean_episode_return).
    """
    episode_returns, episode_reward = [], 0.0
    total_episodes = 0
    done = False

    # Setup video recorder
    if video_name is not None and env.render_mode == "rgb_array":
        # Determine video save path
        if video_save_path is None:
            video_save_path = str(Path(__file__).parent.parent / "logs" / "videos")
        else:
            video_save_path = str(Path(video_save_path))

        os.makedirs(video_save_path, exist_ok=True)

        # New gym recorder always wants to cut video into episodes,
        # set video length big enough but not to inf (will cut into episodes)
        env = RecordVideo(
            env,
            video_folder=video_save_path,
            step_trigger=lambda _: False,
            video_length=100_000,
        )
        env.start_recording(video_name)

    # some gym magic to retrieve the original env
    line_follower_env = env.unwrapped
    assert isinstance(line_follower_env, LineFollowerConstantSpeedEnv)

    obs, _ = env.reset()
    lateral_error = prev_lateral_error = float(obs[IDX_LATERAL_ERROR])
    best_lap_time = float("inf")
    lap_count = 0

    while total_episodes < n_eval_episodes:
        # retrieve prev and current lateral error
        lateral_error = obs[IDX_LATERAL_ERROR]
        action = controller(lateral_error, prev_lateral_error)

        obs, reward, terminated, truncated, info = env.step(action)
        prev_lateral_error = lateral_error

        episode_reward += float(reward)

        if info["lap_count"] > lap_count:
            lap_count = info["lap_count"]
            # last_lap_time = info["last_lap_time"]
            best_lap_time = info["best_lap_time"]

        done = terminated or truncated
        if done:
            episode_returns.append(episode_reward)
            episode_reward = 0.0
            total_episodes += 1
            env.reset()

    if isinstance(env, RecordVideo):
        print(f"Saving video to {video_save_path}/{video_name}.mp4")
        env.close()
    if verbose:
        print(f"{best_lap_time=:.2f}s | Total reward = {np.mean(episode_returns):.2f} +/- {np.std(episode_returns):.2f}")
    return best_lap_time, np.mean(episode_returns).item()


def evaluate_policy(
    policy,
    env: gym.Env,
    n_eval_episodes: int = 5,
    video_name: str | None = None,
    video_save_path: str | None = None,
    verbose: bool = True,
    deterministic: bool = True,
) -> tuple[float, float]:
    """
    Evaluate a policy on an environment over multiple episodes.

    :param policy: A policy that takes observation as input and returns an action.
    :param env: The Gymnasium environment to evaluate on.
    :param n_eval_episodes: Number of episodes to evaluate.
    :param video_name: Optional name for recording a video of the evaluation.
    :param video_save_path: Optional directory path to save videos. If None, defaults to the parent
        directory of this file's folder.
    :param verbose: Whether to print evaluation results.
    :param deterministic: Wether to sample or take the most likely action
    :return: A tuple containing (best_lap_time, mean_episode_return).
    """
    episode_returns, episode_reward = [], 0.0
    total_episodes = 0
    done = False

    # Setup video recorder
    if video_name is not None and env.render_mode == "rgb_array":
        # Determine video save path
        if video_save_path is None:
            video_save_path = str(Path(__file__).parent.parent / "logs" / "videos")
        else:
            video_save_path = str(Path(video_save_path))

        os.makedirs(video_save_path, exist_ok=True)

        # New gym recorder always wants to cut video into episodes,
        # set video length big enough but not to inf (will cut into episodes)
        env = RecordVideo(
            env,
            video_folder=video_save_path,
            step_trigger=lambda _: False,
            video_length=100_000,
        )
        env.start_recording(video_name)

    obs, _ = env.reset()
    lateral_error = prev_lateral_error = float(obs[IDX_LATERAL_ERROR])
    best_lap_time = float("inf")
    lap_count = 0

    while total_episodes < n_eval_episodes:
        with th.no_grad():
            action = policy.get_action(th.as_tensor(obs), deterministic=deterministic)

        # Convert to NumPy and clip if necessary
        action_np = action.numpy()
        if isinstance(env.action_space, gym.spaces.Box):
            action_np = np.clip(action_np, env.action_space.low, env.action_space.high)
        elif isinstance(env.action_space, gym.spaces.Discrete):
            action_np = action_np.item()

        obs, reward, terminated, truncated, info = env.step(action_np)

        episode_reward += float(reward)

        if info["lap_count"] > lap_count:
            lap_count = info["lap_count"]
            # last_lap_time = info["last_lap_time"]
            best_lap_time = info["best_lap_time"]

        done = terminated or truncated
        if done:
            episode_returns.append(episode_reward)
            episode_reward = 0.0
            total_episodes += 1
            env.reset()

    if isinstance(env, RecordVideo):
        print(f"Saving video to {video_save_path}/{video_name}.mp4")
        env.close()
    if verbose:
        print(f"{best_lap_time=:.2f}s | Total reward = {np.mean(episode_returns):.2f} +/- {np.std(episode_returns):.2f}")
    return best_lap_time, np.mean(episode_returns).item()
