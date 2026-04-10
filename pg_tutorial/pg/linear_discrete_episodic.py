"""
Linear Policy Gradient with Episodic Data Collection for Discrete Action Spaces.

A simple implementation of Policy Gradient that collects one complete episode per iteration
before updating the policy. This is the episodic version of the algorithm.

Usage:
    python pg_tutorial/pg/linear_discrete_episodic.py [options]

Options:
    --env-id STR          Environment ID (default: CartPole-v1)
    --seed INT            Random seed (default: 0)
    --n-iterations INT    Number of training iterations/episodes (default: 1000)
    --gamma FLOAT         Discount factor (default: 1.0)
    --learning-rate FLOAT Learning rate (default: 0.01)
    --smoothing-window INT Smoothing window for statistics (default: 50)
    --log-freq INT        Logging frequency in iterations (default: 5)

Example:
    python pg_tutorial/pg/linear_discrete_episodic.py --env-id CartPole-v1 --seed 42 --gamma 0.99
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
    parser = argparse.ArgumentParser(description="Policy Gradient with Linear Policy (Episodic)")

    # Environment arguments
    parser.add_argument("--env-id", type=str, default="CartPole-v1", help="Environment ID")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")

    # Hyperparameters
    parser.add_argument("--n-iterations", type=int, default=1000, help="Number of training iterations/episodes")
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
        # Here, this is the same as logits = weights @ observations (matrix multiplication)
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

    # Print config
    print(f"{args.seed=}")
    print(f"{args.env_id=}")
    print(f"{args.gamma=}")
    print(f"{args.learning_rate=}")
    print(f"{args.n_iterations=}")
    print(f"{args.smoothing_window=}")
    print(f"{args.log_freq=}")

    assert isinstance(env.observation_space, gym.spaces.Box)
    # Discrete actions
    assert isinstance(env.action_space, gym.spaces.Discrete)

    # Env info
    obs_shape = env.observation_space.shape
    obs_dim = int(np.prod(obs_shape))
    n_actions = int(env.action_space.n)
    total_timesteps = 0

    # Pseudo-random generator seeding for reproducible results
    np.random.seed(args.seed)
    th.manual_seed(args.seed)

    # Instantiate the policy
    policy = LinearPolicy(obs_dim, n_actions)

    # Create the optimizer
    optimizer = th.optim.Adam(policy.parameters(), lr=args.learning_rate)

    # Report some statistics, mean over last episodes
    episode_returns: deque[float] = deque(maxlen=args.smoothing_window)
    episode_lengths: deque[int] = deque(maxlen=args.smoothing_window)
    n_episodes = 0
    start_time = time.monotonic()

    for iteration in tqdm(range(1, args.n_iterations + 1)):
        # for iteration in range(1, args.n_iterations + 1):
        # Collect one episode
        observations: list[th.Tensor] = []
        actions: list[th.Tensor] = []
        rewards: list[float] = []

        # Only seed for the very first episode
        current_obs, _ = env.reset(seed=args.seed if iteration == 0 else None)
        done = False

        while not done:
            # Sample action with current policy
            obs_tensor = th.as_tensor(current_obs)
            action = policy.get_action(obs_tensor)

            # Store transitions
            observations.append(obs_tensor)
            actions.append(action)

            # Step in the env
            next_obs, reward, terminated, truncated, _ = env.step(int(action))
            # Check if the episode is over
            done = terminated or truncated

            # Store the reward
            rewards.append(float(reward))
            total_timesteps += 1

            # Update current obs
            current_obs = next_obs

        # Convert lists to tensors
        obs_tensor = th.stack(observations)
        actions_tensor = th.stack(actions)
        rewards_tensor = th.tensor(rewards)

        # Compute discounted returns
        discounted_returns = th.zeros(len(rewards))
        current_return = 0.0
        for step in reversed(range(len(rewards))):
            current_return = rewards[step] + args.gamma * current_return
            discounted_returns[step] = current_return

        # Compute advantages (with baseline of 0)
        advantages = discounted_returns

        # Update the policy with policy gradient loss
        log_probs = policy.get_log_prob(obs_tensor, actions_tensor)
        pg_loss = -(advantages * log_probs).mean()

        # Backpropagate and update
        optimizer.zero_grad()
        pg_loss.backward()
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

        n_episodes += 1
        episode_lengths.append(len(rewards))
        episode_returns.append(sum(rewards))

    env.close()
