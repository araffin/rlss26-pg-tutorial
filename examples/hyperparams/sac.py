"""SAC hyperparameters for custom environments."""

# Hyperparameters for LineFollowerDrift-v0
# These are good default values to start with
hyperparams = {
    "LineFollowerDrift-v0": {
        # Norm observation
        "normalize": {"norm_obs": True, "norm_reward": False},
        # Training timesteps
        "n_timesteps": 1_000_000,
        # Policy architecture
        "policy": "MlpPolicy",
        # Learning rate
        "learning_rate": 7e-4,
        # Replay buffer size
        "buffer_size": 1_000_000,
        # Batch size for sampling from replay buffer
        "batch_size": 256,
        # Entropy regularization coefficient
        "ent_coef": "auto_0.1",
        # Number of steps to run before training
        "learning_starts": 1_000,
        # Discount factor
        "gamma": 0.99,
        # Soft target update coefficient
        "tau": 0.005,
        # Number of environment steps before taking gradient step
        "train_freq": 1,
        # Number of gradient steps per train_freq
        "gradient_steps": 1,
        # Policy network architecture
        "policy_kwargs": "dict(net_arch=[128, 128])",
        # Custom callback for logging racing metrics to TensorBoard
        "callback": "pg_tutorial.sb3_callbacks.RacingInfoCallback",
    },
    # Default fallback for any other environment
    "default": {
        "n_timesteps": 100_000,
        "policy": "MlpPolicy",
        "learning_rate": 3e-4,
        "buffer_size": 100_000,
        "batch_size": 256,
        "ent_coef": "auto",
        "learning_starts": 100,
        "gamma": 0.99,
        "tau": 0.005,
        "train_freq": 1,
        "gradient_steps": 1,
        # Custom callback for logging racing metrics to TensorBoard
        "callback": "pg_tutorial.sb3_callbacks.RacingInfoCallback",
    },
}
