#!/usr/bin/env python3
"""
REINFORCE 训练与验证
环境: CartPole
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
import numpy as np
from classic_rl.reinforce import REINFORCEAgent
from configs.default import REINFORCE_AGENT_KEYS
from utils.logger import Logger


def train(config: dict):
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = REINFORCEAgent(state_dim, n_actions, **{
        k: v for k, v in config.items() if k in REINFORCE_AGENT_KEYS
    })
    logger = Logger("logs/reinforce")

    for ep in range(config["n_episodes"]):
        state, _ = env.reset()
        for _ in range(config["max_steps"]):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.store_reward(reward)
            state = next_state
            if terminated or truncated:
                break
        total_reward = sum(agent.rewards)
        loss = agent.update()
        logger.log(ep, reward=total_reward, loss=loss)
        if (ep + 1) % 50 == 0:
            recent = [h["reward"] for h in logger.history[-50:] if "reward" in h]
            avg = sum(recent) / len(recent) if recent else 0
            print(f"Episode {ep+1}, Avg Reward (last 50): {avg:.1f}")

    logger.save("reinforce.json")
    return agent


def evaluate(agent: REINFORCEAgent, n_episodes: int = 100):
    env = gym.make("CartPole-v1")
    rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        total = 0
        for _ in range(500):
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break
        rewards.append(total)
    print(f"验证: 平均回报 {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")


if __name__ == "__main__":
    from configs.default import REINFORCE_CONFIG

    print("=== REINFORCE 训练 ===")
    agent = train(REINFORCE_CONFIG)
    print("\n=== 验证 ===")
    evaluate(agent)
