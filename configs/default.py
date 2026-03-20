"""默认训练配置

各 Agent/Trainer 接受的参数名（run 脚本需过滤，仅传以下 key）:
"""
# 用于 run 脚本过滤，确保不传错参数（与各 Agent __init__ 一致）
QL_AGENT_KEYS = ("lr", "gamma", "epsilon", "epsilon_decay", "epsilon_min")
DQN_AGENT_KEYS = ("lr", "gamma", "epsilon", "epsilon_decay", "epsilon_min", "target_update_freq")
REINFORCE_AGENT_KEYS = ("lr", "gamma")
PPO_AGENT_KEYS = ("lr", "gamma", "gae_lambda", "clip_eps", "n_epochs", "batch_size")

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
    target_update_freq=100,
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

# DPO (LLM)
DPO_CONFIG = dict(
    n_epochs=3,
    model_name="distilgpt2",
    beta=0.1,
    lr=5e-5,
    device="cpu",
)

# PPO-LLM
PPO_LLM_CONFIG = dict(
    n_epochs=3,
    model_name="distilgpt2",
    lr=1e-5,
    clip_eps=0.2,
    gamma=0.99,
    device="cpu",
)
