"""Train an RL agent using RL Zoo 3.

This example demonstrates how to train an agent using the RL Zoo 3 framework
with custom environments from pg_tutorial.

Requirements
------------
Install the ``rl`` optional dependency group::

    pip install -e ".[rl]"
    # or with uv:
    uv pip install -e ".[rl]"

Usage
-----
Train with SAC on the LineFollowerDrift environment::

    python examples/train_rl_zoo.py --algo sac --env LineFollowerDrift-v0 -c examples/hyperparams/sac.py

Train with PPO::

    python examples/train_rl_zoo.py --algo ppo --env LineFollower-v0 -c examples/hyperparams/ppo.py

Evaluate a trained agent::

    python examples/enjoy_rl_zoo.py --algo sac --env LineFollowerDrift-v0 -c examples/hyperparams/sac.py

Hyperparameters can also be passed directly via command line::

    python examples/train_rl_zoo.py --algo sac --env LineFollowerDrift-v0 -params learning_rate:1e-3 buffer_size:100000
"""

from __future__ import annotations

# Register custom environments
import pg_tutorial  # noqa: F401


def main() -> None:
    """Entry point for RL Zoo training."""
    from rl_zoo3.train import train

    train()


if __name__ == "__main__":
    main()
