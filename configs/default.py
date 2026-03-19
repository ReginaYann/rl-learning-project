"""默认训练配置"""

# Q-Learning (FrozenLake)
QL_CONFIG = dict(
    n_episodes=5000,
    max_steps=100,
    lr=0.1,
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.01,
)

# DQN (CartPole)
DQN_CONFIG = dict(
    n_episodes=500,
    max_steps=500,
    batch_size=64,
    buffer_size=10000,
    lr=1e-3,
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.01,
    target_update=100,
)

# REINFORCE (CartPole)
REINFORCE_CONFIG = dict(
    n_episodes=500,
    max_steps=500,
    lr=1e-3,
    gamma=0.99,
)

# PPO (CartPole)
PPO_CONFIG = dict(
    n_episodes=500,
    max_steps=500,
    n_steps=2048,
    batch_size=64,
    lr=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_eps=0.2,
    n_epochs=4,
)
