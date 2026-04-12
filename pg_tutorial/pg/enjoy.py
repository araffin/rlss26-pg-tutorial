"""
Load and evaluate a saved Policy Gradient policy.

Usage:
    python pg_tutorial/pg/enjoy.py --model-path <path_to_saved_model> [options]

Options:
    --model-path STR      Path to the saved model directory (required)
    --n-episodes INT      Number of episodes to run (default: 10)
    --deterministic      Use deterministic action (default: False)
    --render             Render the environment (default: False)

Example:
    python pg_tutorial/pg/enjoy.py --model-path logs/pg-episodic/LunarLanderContinuous-v3/final --n-episodes 5
"""

import argparse
import json
from pathlib import Path
from pprint import pprint

import gymnasium as gym
import numpy as np
import torch as th

import pg_tutorial.pg.continuous_actions_episodic as continuous_pg
import pg_tutorial.pg.discrete_actions_episodic as discrete_pg

Policy = discrete_pg.LinearPolicy | continuous_pg.LinearPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load and evaluate a saved Policy Gradient policy")

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the saved model directory (e.g., logs/pg-episodic/LunarLanderContinuous-v3/final)",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=10,
        help="Number of episodes to run for evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for evaluation (default: None, no seeding)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=False,
        help="Use deterministic action instead of sampling",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Render the environment",
    )

    return parser.parse_args()


def load_policy(policy: Policy, model_dir: Path) -> Policy:
    """Load the policy state dict from the saved file."""
    policy_path = model_dir / "policy.pt"
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy file not found: {policy_path}")

    state_dict = th.load(policy_path, weights_only=True)
    policy.load_state_dict(state_dict)
    policy.eval()
    print(f"Loaded policy from {policy_path}")
    return policy


def load_hyperparameters(model_dir: Path) -> dict:
    """Load hyperparameters from the saved JSON file."""
    hyperparams_path = model_dir / "hyperparameters.json"
    if not hyperparams_path.exists():
        print(f"Warning: Hyperparameters file not found: {hyperparams_path}")
        return {}

    hyperparams = json.loads(hyperparams_path.read_text())
    print("Loaded hyperparameters:")
    pprint(hyperparams)
    return hyperparams


class NormalizeObservationWrapper(gym.ObservationWrapper):
    """Custom wrapper to apply loaded normalization statistics."""

    def __init__(self, env: gym.Env, obs_mean: np.ndarray, obs_var: np.ndarray, epsilon: float) -> None:
        super().__init__(env)
        self.obs_mean = obs_mean
        self.var = obs_var
        self.epsilon = epsilon

    def observation(self, obs: np.ndarray) -> np.ndarray:  # type: ignore[override]
        return (obs - self.obs_mean) / np.sqrt(self.var + self.epsilon)


def load_normalizer(env: gym.Env, model_dir: Path) -> gym.Env:
    """Load observation normalizer statistics if available."""
    normalizer_path = model_dir / "obs_normalizer.npz"
    if not normalizer_path.exists():
        print("No observation normalizer found, using raw observations")
        return env

    # Load normalizer state using numpy
    normalizer_data = np.load(normalizer_path)
    normalizer_state = {
        "obs_mean": normalizer_data["obs_mean"],
        "obs_var": normalizer_data["obs_var"],
        "epsilon": float(normalizer_data["epsilon"]),
    }

    print("Loaded obs normalizer")
    return NormalizeObservationWrapper(env, **normalizer_state)


def evaluate_policy(
    policy: Policy,
    env: gym.Env,
    n_episodes: int,
    deterministic: bool = False,
    seed: int | None = None,
) -> tuple[list[float], list[int]]:
    """Evaluate the policy and return episode returns and lengths."""
    episode_returns: list[float] = []
    episode_lengths: list[int] = []

    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed if episode == 0 else None)
        done = False
        episode_return = 0.0
        episode_length = 0

        while not done:
            obs_tensor = th.as_tensor(obs)
            with th.no_grad():
                action = policy.get_action(obs_tensor, deterministic=deterministic)
            # Convert to NumPy and clip if necessary
            action_np = action.numpy()
            if isinstance(env.action_space, gym.spaces.Box):
                action_np = np.clip(action_np, env.action_space.low, env.action_space.high)
            elif isinstance(env.action_space, gym.spaces.Discrete):
                action_np = action_np.item()

            obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            episode_return += float(reward)
            episode_length += 1

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        print(f"Episode {episode + 1}/{n_episodes}: return={episode_return:.2f}, length={episode_length}")

    return episode_returns, episode_lengths


if __name__ == "__main__":
    args = parse_args()

    model_dir = Path(args.model_path)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        th.manual_seed(args.seed)
        print(f"Set seed to {args.seed}")

    # Load hyperparameters to get the environment ID
    hyperparams = load_hyperparameters(model_dir)
    env_id = hyperparams.get("env_id", "LunarLanderContinuous-v3")

    # Create the environment
    env = gym.make(env_id, render_mode="human" if args.render else None)

    # Get obs and action dimensions
    assert isinstance(env.observation_space, gym.spaces.Box)
    obs_dim = int(np.prod(env.observation_space.shape))

    if isinstance(env.action_space, gym.spaces.Box):
        action_dim = int(np.prod(env.action_space.shape))
        policy_class: type[Policy] = continuous_pg.LinearPolicy
    elif isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = int(env.action_space.n)
        policy_class = discrete_pg.LinearPolicy

    # Create and load the policy
    policy = policy_class(obs_dim, action_dim)
    policy = load_policy(policy, model_dir)

    # Load and apply observation normalizer
    env = load_normalizer(env, model_dir)

    # Evaluate the policy
    print(f"\nEvaluating policy for {args.n_episodes} episodes...")
    returns, lengths = evaluate_policy(policy, env, args.n_episodes, args.deterministic)

    print("\n" + "=" * 40)
    print("Evaluation Results:")
    print(f"  Mean return: {np.mean(returns):.2f} +/- {np.std(returns):.2f}")
    print(f"  Mean length: {np.mean(lengths):.2f} +/- {np.std(lengths):.2f}")
    print("=" * 40)

    env.close()
