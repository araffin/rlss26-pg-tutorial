import itertools

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DiscreteActionWrapper(gym.ActionWrapper):
    """Wraps a continuous Box action space as a Discrete one.

    Discretizes each action dimension into `n_bins` evenly-spaced values,
    then flattens all combinations into a single Discrete space via
    Cartesian product. Total actions = n_bins ** action_dim.

    Args:
        env: Environment with a Box action space.
        n_bins: Number of bins per action dimension. Defaults to 5.

    Attributes:
        action_map: Array of shape (n_actions, action_dim) mapping each
            discrete index to its continuous action vector.
    """

    def __init__(self, env: gym.Env, n_bins: int = 5) -> None:
        super().__init__(env)
        assert isinstance(env.action_space, spaces.Box)

        if n_bins <= 0:
            raise ValueError(f"{n_bins=} must be a positive integer.")

        low, high = env.action_space.low.flat, env.action_space.high.flat
        self.action_map: np.ndarray = np.array(
            list(itertools.product(*[np.linspace(lo, hi, n_bins) for lo, hi in zip(low, high, strict=True)]))
        )
        self.action_space = spaces.Discrete(len(self.action_map))

    def action(self, action: int) -> np.ndarray:
        """Convert a discrete action index to a continuous action vector."""
        return self.action_map[action].reshape(self.env.action_space.shape)
