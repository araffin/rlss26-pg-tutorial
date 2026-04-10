"""
A simple implementation of Policy Gradient with a linear policy for discrete action spaces.

Usage:
    python -m pg.linear_policy_discrete [options]

Options:
    --env-id STR          Environment ID (default: CartPole-v1)
    --seed INT            Random seed (default: 0)
    --n-iterations INT    Number of training iterations (default: 1000)
    --n-steps INT         Number of steps per iteration (default: 500)
    --gamma FLOAT         Discount factor (default: 1.0)
    --learning-rate FLOAT Learning rate (default: 0.01)
    --smoothing-window INT Smoothing window for statistics (default: 50)
    --log-freq INT        Logging frequency in iterations (default: 5)

Example:
    python -m pg.linear_policy_discrete --env-id CartPole-v1 --seed 42 --gamma 0.99
"""

import argparse
import time
import warnings
from collections import deque

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
from torch.distributions import Categorical
from tqdm.rich import TqdmExperimentalWarning, tqdm


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Policy Gradient with Linear Policy")

    # Environment arguments
    parser.add_argument("--env-id", type=str, default="CartPole-v1", help="Environment ID")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")

    # Hyperparameters
    parser.add_argument("--n-iterations", type=int, default=1000, help="Number of training iterations")
    parser.add_argument("--n-steps", type=int, default=500, help="Number of steps per iteration")
    parser.add_argument("--gamma", type=float, default=1.0, help="Discount factor")
    parser.add_argument("--learning-rate", type=float, default=1e-2, help="Learning rate for optimizer")
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=50,
        help="Smoothing window for episode statistics",
    )
    parser.add_argument(
        "--log-freq",
        type=int,
        default=5,
        help="Frequency of logging (in iterations)",
    )

    return parser.parse_args()


warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)


class LinearPolicy(nn.Module):
    """
    A simple PyTorch model to represent a Linear policy
    in the discrete action setting.

    :param obs_dim: Dimension of the observation space (we assume that the observation is a 1D vector)
    :param action_dim: Dimension of the action space (here the number of discrete actions)
    """

    def __init__(self, obs_dim: int = 2, action_dim: int = 2) -> None:
        super().__init__()
        self.net = nn.Linear(obs_dim, action_dim, bias=False)

    def get_action(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # logits are un-normalized probabilities of taking each action
        logits = self.net(observation)
        # A convenience class to sample, compute probabilties and find the argmax
        action_dist = Categorical(logits=logits)
        if deterministic:
            # Same as th.argmax(action_dist, dim=-1), get the most likely action
            return action_dist.mode
        return action_dist.sample()

    def forward(self, observation: th.Tensor) -> th.Tensor:
        return self.get_action(observation)

    def get_log_prob(self, observations: th.Tensor, actions: th.Tensor) -> th.Tensor:
        logits = self.net(observations)
        action_dist = Categorical(logits=logits)
        return action_dist.log_prob(actions)


if __name__ == "__main__":
    args = parse_args()

    env = gym.make(args.env_id)
    # eval_env = gym.make(env_id)

    # Print config
    print(f"{args.seed=}")
    print(f"{args.env_id=}")
    print(f"{args.gamma=}")
    print(f"{args.learning_rate=}")
    print(f"{args.n_iterations=}")
    print(f"{args.n_steps=}")
    print(f"{args.smoothing_window=}")
    print(f"{args.log_freq=}")

    assert isinstance(env.observation_space, gym.spaces.Box)
    # Discrete actions
    assert isinstance(env.action_space, gym.spaces.Discrete)

    # Env info
    obs_shape = env.observation_space.shape
    obs_dim = int(np.prod(obs_shape))
    n_actions = int(env.action_space.n)
    # action_dim = 1  # discrete actions, integer actions
    total_timesteps = 0

    # Pseudo-random generator seeding for reproducible results
    np.random.seed(args.seed)
    th.manual_seed(args.seed)

    # Instantiate the policy
    policy = LinearPolicy(obs_dim, n_actions)

    # Create the optimizer
    optimizer = th.optim.Adam(policy.parameters(), lr=args.learning_rate)

    # Storage for data collection
    observations = th.zeros(args.n_steps, *obs_shape)
    # Discrete actions: th.long is for integers
    actions = th.zeros(args.n_steps, dtype=th.long)
    rewards = th.zeros(args.n_steps)
    terminations = th.zeros(args.n_steps)
    truncations = th.zeros(args.n_steps)
    episode_starts = th.zeros(args.n_steps, dtype=th.bool)

    # Report some statistics, mean over last episodes
    current_episode_reward = 0.0
    current_episode_length = 0
    episode_returns: deque[float] = deque(maxlen=args.smoothing_window)
    episode_lengths: deque[int] = deque(maxlen=args.smoothing_window)
    n_episodes = 0
    start_time = time.monotonic()

    current_obs, _ = env.reset(seed=args.seed)
    last_episode_starts = True

    for iteration in tqdm(range(1, args.n_iterations + 1)):
        # for iteration in tqdm(range(1, args.n_iterations + 1)):
        # TODO: make it episodic? -> collect n episodes
        for step in range(args.n_steps):
            # Sample action with current policy and store
            # the log prob of taking the action for later
            action = policy.get_action(th.as_tensor(current_obs))

            # Convert from th.Tensor to integer
            env_action = int(action)

            # Step in the env
            next_obs, reward, terminated, truncated, _ = env.step(env_action)
            # Store the transition
            observations[step] = th.as_tensor(current_obs)
            actions[step] = action
            rewards[step] = float(reward)
            terminations[step] = terminated
            truncations[step] = truncated
            episode_starts[step] = last_episode_starts
            current_episode_length += 1
            total_timesteps += 1

            # Logging
            current_episode_reward += float(reward)

            last_episode_starts = terminated or truncated
            # New episode
            if last_episode_starts:
                episode_returns.append(current_episode_reward)
                episode_lengths.append(current_episode_length)
                current_episode_reward = 0.0
                current_episode_length = 0
                n_episodes += 1
                next_obs, _ = env.reset()

            # Update current obs
            current_obs = next_obs

        # Note(antonin): the current code doesn't handle truncation properly
        # See https://github.com/DLR-RM/stable-baselines3/issues/633
        # One way is to augment the reward when the episode is truncated:
        # terminal_reward = terminal_reward + gamma * value_fn(terminal_state)
        dones = th.logical_or(terminations, truncations)

        # Compute discounted reward to go
        discounted_returns = th.zeros(args.n_steps)
        # Intialize with terminal reward
        # TODO: bootstrap with value when truncating the episode (truncated | terminated = False)
        current_return = rewards[-1]
        discounted_returns[-1] = rewards[-1]
        for step in reversed(range(args.n_steps - 1)):
            next_step_terminal = not episode_starts[step + 1]
            current_return = rewards[step] + next_step_terminal * args.gamma * current_return
            discounted_returns[step] = current_return

        # Advantage computation
        # baseline = rewards.mean()
        baseline = 0
        advantages = discounted_returns - baseline

        # Normalize advantages (optional, commented out in original)
        # advantages = (advantages - advantages.mean()) / advantages.std()
        # Update the policy with policy gradient loss
        log_probs = policy.get_log_prob(observations, actions)
        pg_loss = -(advantages * log_probs).mean()

        # backpropagate and do the gradient update
        optimizer.zero_grad()
        pg_loss.backward()
        # For later, for stability:
        # nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        optimizer.step()

        # Logging
        if (iteration % args.log_freq) == 0:
            print(f" {iteration=}/{args.n_iterations} ".center(30, "="))
            time_elapsed = time.monotonic() - start_time
            fps = total_timesteps / time_elapsed
            print(f"rollout/{n_episodes=}")
            print(f"rollout/{np.mean(episode_returns)=:.2f} +/- {np.std(episode_returns):.2f}")
            print(f"rollout/{np.mean(episode_lengths)=:.2f} +/- {np.std(episode_lengths):.2f}")
            print(f"time/{total_timesteps=}")
            print(f"time/{time_elapsed=:.0f}")
            print(f"time/{fps=:.2f}")
            print(f"train/{pg_loss=:.4f}")
            print("=" * 30)
