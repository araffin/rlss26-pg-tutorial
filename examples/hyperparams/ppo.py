"""PPO hyperparameters for custom environments."""

# Hyperparameters for LineFollower and LineFollowerDrift environments
# PPO works well with these default settings
hyperparams = {
    "LineFollower-v0": {
        # Training timesteps
        "n_timesteps": 100_000,
        # Policy architecture
        "policy": "MlpPolicy",
        # Learning rate
        "learning_rate": 3e-4,
        # Number of steps to run in each environment per update
        "n_steps": 2048,
        # Mini-batch size (should be <= n_steps)
        "batch_size": 64,
        # Number of optimization epochs
        "n_epochs": 10,
        # Entropy bonus coefficient (for exploration)
        "ent_coef": 0.0,
        # Value function loss coefficient
        "vf_coef": 0.5,
        # Maximum gradient norm
        "max_grad_norm": 0.5,
        # Discount factor
        "gamma": 0.99,
        # Generalized advantage estimation lambda
        "gae_lambda": 0.95,
        # Clip range for PPO updates
        "clip_range": 0.2,
        # Normalize advantage
        "normalize_advantage": True,
        # Policy network architecture
        "policy_kwargs": "dict(net_arch=[256, 256])",
    },
    "LineFollowerDrift-v0": {
        # Training timesteps (more for drift environment)
        "n_timesteps": 500_000,
        # Policy architecture
        "policy": "MlpPolicy",
        # Learning rate
        "learning_rate": 3e-4,
        # Number of steps to run in each environment per update
        "n_steps": 1024,
        # Mini-batch size
        "batch_size": 64,
        # Number of optimization epochs
        "n_epochs": 10,
        # Entropy bonus coefficient (encourage exploration for drift)
        "ent_coef": 0.01,
        # Value function loss coefficient
        "vf_coef": 0.5,
        # Maximum gradient norm
        "max_grad_norm": 0.5,
        # Discount factor
        "gamma": 0.99,
        # Generalized advantage estimation lambda
        "gae_lambda": 0.95,
        # Clip range for PPO updates
        "clip_range": 0.2,
        # Normalize advantage
        "normalize_advantage": True,
        # Policy network architecture
        "policy_kwargs": "dict(net_arch=[256, 256])",
    },
    # Default fallback for any other environment
    "default": {
        "n_timesteps": 100_000,
        "policy": "MlpPolicy",
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "normalize_advantage": True,
    },
}
