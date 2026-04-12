"""
Policy Gradient with Episodic Data Collection for Continuous Action Spaces.

A simple implementation of Policy Gradient that collects one complete episode per iteration
before updating the policy.

Usage:
    python pg_tutorial/pg/continuous_actions_episodic.py [options]

Options:
    --env-id STR          Environment ID (default: LunarLanderContinuous-v3)
    --seed INT            Random seed (default: 0)
    --n-iterations INT    Number of training iterations/episodes (default: 1000)
    --gamma FLOAT         Discount factor (default: 1.0)
    --learning-rate FLOAT Learning rate (default: 0.01)
    --smoothing-window INT Smoothing window for statistics (default: 50)
    --log-freq INT        Logging frequency in iterations (default: 5)
    --save-freq INT       Save checkpoint every n iterations (default: 0, disabled)
    --save-dir STR        Directory to save models (default: logs/pg-episodic)

Example:
    python pg_tutorial/pg/continuous_actions_episodic.py --env-id LunarLanderContinuous-v3 --seed 42 --gamma 0.99
"""

import argparse
import json
import time
import warnings
from collections import deque
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
from torch.distributions import Normal
from tqdm.rich import TqdmExperimentalWarning, tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Policy Gradient with Linear Policy (Episodic)")

    # Environment arguments
    parser.add_argument("--env-id", type=str, default="LunarLanderContinuous-v3", help="Environment ID")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")

    # Hyperparameters
    parser.add_argument("--n-iterations", type=int, default=1000, help="Number of training iterations/episodes")
    parser.add_argument("--gamma", type=float, default=0.98, help="Discount factor")
    parser.add_argument("--learning-rate", type=float, default=1e-2, help="Learning rate for optimizer")

    # Logging
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

    # Saving
    parser.add_argument(
        "--save-freq",
        type=int,
        default=0,
        help="Save checkpoint every n iterations (default: 0, disabled)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="logs/pg-episodic",
        help="Directory to save models",
    )

    return parser.parse_args()


warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)


class LinearPolicy(nn.Module):
    """
    A simple PyTorch model to represent a Linear policy
    in the discrete action setting.

    :param obs_dim: Dimension of the observation space (we assume that the observation is a 1D vector)
    :param action_dim: Dimension of the action space
    :param std_init: Initial standard deviation
    """

    def __init__(self, obs_dim: int = 2, action_dim: int = 2, std_init: float = 1.0) -> None:
        super().__init__()
        self.net = nn.Linear(obs_dim, action_dim, bias=True)
        # State-Independent log standard deviation
        # We use the log to make sure std > 0
        # Note: we could make it state-dependent by having another network
        # that outputs the std (self.net = Linear(obs_dim, action_dim * 2))
        self.log_std = nn.Parameter(th.ones(action_dim) * np.log(std_init))

    def get_action(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Here, this is the same as action_mean = weights @ observations + bias (matrix multiplication)
        action_mean = self.net(observation)
        action_std = th.ones_like(action_mean) * self.log_std.exp()
        # A convenience class to sample, compute probabilties and find the best action
        action_dist = Normal(action_mean, action_std)
        if deterministic:
            # Get the most likely action (here same as action_mean)
            return action_dist.mode
        return action_dist.sample()

    def forward(self, observation: th.Tensor) -> th.Tensor:
        return self.get_action(observation)

    def get_log_prob(self, observations: th.Tensor, actions: th.Tensor) -> th.Tensor:
        action_mean = self.net(observations)
        action_std = th.ones_like(action_mean) * self.log_std.exp()
        action_dist = Normal(action_mean, action_std)
        # Sum all action dimensions (treating them as independent)
        return action_dist.log_prob(actions).sum(dim=1)


def get_normalizer_state(env: gym.Env) -> dict[str, Any] | None:
    """Extract observation normalizer state from a NormalizeObservation wrapper."""
    # Check if the environment has the normalize observation wrapper
    if isinstance(env, gym.wrappers.NormalizeObservation):
        return {
            "obs_mean": env.obs_rms.mean,
            "obs_var": env.obs_rms.var,
            "epsilon": env.epsilon,
        }
    return None


def save_policy(
    policy: LinearPolicy,
    save_dir: Path,
    iteration: int,
    env_id: str,
    normalizer_state: dict[str, Any] | None = None,
    gamma: float | None = None,
    learning_rate: float | None = None,
    seed: int | None = None,
) -> None:
    """Save the policy, hyperparameters and optionally the observation normalizer statistics."""
    # Create directory if it doesn't exist
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save policy state
    policy_path = save_dir / "policy.pt"
    th.save(policy.state_dict(), policy_path)

    # Save observation normalizer statistics if present
    if normalizer_state is not None:
        normalizer_path = save_dir / "obs_normalizer.npz"
        np.savez(
            normalizer_path,
            obs_mean=normalizer_state["obs_mean"],
            obs_var=normalizer_state["obs_var"],
            epsilon=normalizer_state["epsilon"],
        )

    # Save hyperparameters
    hyperparams = {
        "env_id": env_id,
        "iteration": iteration,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "seed": seed,
    }
    hyperparams_path = save_dir / "hyperparameters.json"
    with open(hyperparams_path, "w") as f:
        json.dump(hyperparams, f, indent=2)

    print(f"Saved policy to {save_dir}")


if __name__ == "__main__":
    args = parse_args()

    env = gym.make(args.env_id)
    # Normalize obs to have mean ~ 0.0, std ~ 1.0
    env = gym.wrappers.NormalizeObservation(env)

    # Create save directory based on environment ID
    save_dir = Path(args.save_dir) / args.env_id

    # Print config
    print(f"{args.seed=}")
    print(f"{args.env_id=}")
    print(f"{args.gamma=}")
    print(f"{args.learning_rate=}")
    print(f"{args.n_iterations=}")
    print(f"{args.smoothing_window=}")
    print(f"{args.log_freq=}")
    print(f"{args.save_freq=}")
    print(f"{args.save_dir=}")
    print(f"{save_dir=}")

    assert isinstance(env.observation_space, gym.spaces.Box)
    # Continuous actions
    assert isinstance(env.action_space, gym.spaces.Box)

    # Env info
    obs_shape = env.observation_space.shape
    obs_dim = int(np.prod(obs_shape))
    action_dim = int(np.prod(env.action_space.shape))
    total_timesteps = 0

    # Pseudo-random generator seeding for reproducible results
    np.random.seed(args.seed)
    th.manual_seed(args.seed)

    # Instantiate the policy
    policy = LinearPolicy(obs_dim, action_dim)

    # Create the optimizer
    optimizer = th.optim.Adam(policy.parameters(), lr=args.learning_rate)

    # Report some statistics, mean over last episodes
    episode_returns: deque[float] = deque(maxlen=args.smoothing_window)
    episode_lengths: deque[int] = deque(maxlen=args.smoothing_window)
    n_episodes = 0
    start_time = time.monotonic()

    for iteration in tqdm(range(1, args.n_iterations + 1)):
        # Collect one episode
        observations: list[th.Tensor] = []
        actions: list[th.Tensor] = []
        rewards: list[float] = []

        # Only seed for the very first episode
        current_obs, _ = env.reset(seed=args.seed if iteration == 1 else None)
        done = False

        while not done:
            # Sample action with current policy
            obs_tensor = th.as_tensor(current_obs)
            action = policy.get_action(obs_tensor)

            # Store transitions
            observations.append(obs_tensor)
            actions.append(action)

            # Convert from Torch Tensor to NumPy array
            action_np = action.numpy()
            # Clip infinite support Gaussian to correct bounds
            action_np = np.clip(action_np, env.action_space.low, env.action_space.high)

            # Step in the env
            next_obs, reward, terminated, truncated, _ = env.step(action_np)
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
            std = policy.log_std.exp().mean().item()
            print(f"rollout/{n_episodes=}")
            print(f"rollout/{np.mean(episode_returns)=:.2f} +/- {np.std(episode_returns):.2f}")
            print(f"rollout/{np.mean(episode_lengths)=:.2f} +/- {np.std(episode_lengths):.2f}")
            print(f"time/{total_timesteps=}")
            print(f"time/{time_elapsed=:.0f}")
            print(f"time/{fps=:.2f}")
            print(f"train/{pg_loss=:.4f}")
            print(f"train/{std=:.2f}")
            print("=" * 30)

        n_episodes += 1
        episode_lengths.append(len(rewards))
        episode_returns.append(sum(rewards))

        # Save checkpoint every n iterations if enabled
        if args.save_freq > 0 and iteration % args.save_freq == 0:
            checkpoint_dir = save_dir / f"checkpoint_{iteration}"
            normalizer_state = get_normalizer_state(env)
            save_policy(
                policy,
                checkpoint_dir,
                iteration,
                args.env_id,
                normalizer_state,
                args.gamma,
                args.learning_rate,
                args.seed,
            )

    # Save final policy
    final_dir = save_dir / "final"
    normalizer_state = get_normalizer_state(env)
    save_policy(
        policy,
        final_dir,
        args.n_iterations,
        args.env_id,
        normalizer_state,
        args.gamma,
        args.learning_rate,
        args.seed,
    )

    env.close()
