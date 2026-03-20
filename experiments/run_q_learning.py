#!/usr/bin/env python3
"""
Q-Learning 训练与验证
环境: FrozenLake (4x4 网格，离散状态动作)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
from classic_rl.q_learning import QLearningAgent
from configs.default import QL_AGENT_KEYS
from utils.logger import Logger


def train(config: dict):
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        **{k: v for k, v in config.items() if k in QL_AGENT_KEYS},
    )
    logger = Logger("logs/q_learning")

    for ep in range(config["n_episodes"]):
        state, _ = env.reset()
        total_reward = 0
        for _ in range(config["max_steps"]):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.update(state, action, reward, next_state, terminated or truncated)
            total_reward += reward
            state = next_state
            if terminated or truncated:
                break
        agent.decay_epsilon()
        logger.log(ep, reward=total_reward, epsilon=agent.epsilon)
        if (ep + 1) % 500 == 0:
            avg = sum(h["reward"] for h in logger.history[-100:]) / 100
            print(f"Episode {ep+1}, Avg Reward (last 100): {avg:.2f}")

    logger.save("q_learning.json")
    return agent


def evaluate(agent: QLearningAgent, n_episodes: int = 100):
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    wins = 0
    for _ in range(n_episodes):
        state, _ = env.reset()
        for _ in range(100):
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                wins += int(reward > 0) if terminated else 0
                break
    print(f"验证: {n_episodes} 局中成功 {wins} 局 ({100*wins/n_episodes:.1f}%)")


if __name__ == "__main__":
    from configs.default import QL_CONFIG

    print("=== Q-Learning 训练 ===")
    agent = train(QL_CONFIG)
    print("\n=== 验证 ===")
    evaluate(agent)
