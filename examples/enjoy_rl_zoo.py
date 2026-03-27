"""Evaluate/replay RL agents trained with RL Zoo 3.

This example demonstrates how to evaluate agents trained using the RL Zoo 3 framework
with custom environments from pg_tutorial.

Requirements
------------
Install the ``rl`` optional dependency group::

    pip install -e ".[rl]"
    # or with uv:
    uv pip install -e ".[rl]"

Usage
-----
Evaluate a trained SAC agent::

    python examples/enjoy_rl_zoo.py --algo sac --env LineFollowerDrift-v0 -f logs/rl_zoo_test/sac

Evaluate with deterministic actions (default)::

    python examples/enjoy_rl_zoo.py --algo sac --env LineFollowerDrift-v0 -f logs/rl_zoo_test/sac

Evaluate with stochastic actions::

    python examples/enjoy_rl_zoo.py --algo sac --env LineFollowerDrift-v0 -f logs/rl_zoo_test/sac --stochastic

Load the best model instead of the last one::

    python examples/enjoy_rl_zoo.py --algo sac --env LineFollowerDrift-v0 -f logs/rl_zoo_test/sac --load-best

Evaluate without rendering (useful for headless testing)::

    python examples/enjoy_rl_zoo.py --algo sac --env LineFollowerDrift-v0 -f logs/rl_zoo_test/sac --no-render

Run for more timesteps::

    python examples/enjoy_rl_zoo.py --algo sac --env LineFollowerDrift-v0 -f logs/rl_zoo_test/sac -n 5000

Evaluate multiple environments in parallel::

    python examples/enjoy_rl_zoo.py --algo sac --env LineFollowerDrift-v0 -f logs/rl_zoo_test/sac --n-envs 4
"""

from __future__ import annotations

# Register custom environments
import pg_tutorial  # noqa: F401


def main() -> None:
    """Entry point for RL Zoo evaluation."""
    from rl_zoo3.enjoy import enjoy

    enjoy()


if __name__ == "__main__":
    main()
