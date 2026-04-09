from dataclasses import dataclass

import torch as th
import torch.nn as nn


class LinearPolicy(nn.Module):
    def __init__(self, obs_dim: int = 2, action_dim: int = 2):
        self.net = nn.Linear(obs_dim, action_dim)

    def forward(self, observation: th.Tensor) -> th.Tensor:
        return self.net(observation)


@dataclass
class Buffer:
    obs_dim: int
    action_dim: int
    n_steps: int

    def __post_init__(self):
        self.observations = th.zeros(self.n_steps, self.obs_dim)
        self.actions = th.zeros(self.n_steps, self.action_dim)
        self.rewards = th.zeros(self.n_steps, 1)
        self.rewards = th.zeros(self.n_steps, 1)
