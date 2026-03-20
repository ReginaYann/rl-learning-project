#!/usr/bin/env python3
"""
DQN 训练与验证
环境: CartPole (连续状态 4 维，离散动作 2)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
import numpy as np
import torch
from classic_rl.dqn import DQNAgent
from configs.default import DQN_AGENT_KEYS
from utils.logger import Logger
from utils.replay_buffer import ReplayBuffer


def train(config: dict):
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = DQNAgent(state_dim, n_actions, **{
        k: v for k, v in config.items() if k in DQN_AGENT_KEYS
    })
    buffer = ReplayBuffer(config["buffer_size"])
    logger = Logger("logs/dqn")

    for ep in range(config["n_episodes"]):
        state, _ = env.reset()
        total_reward = 0
        ep_loss = None
        for step in range(config["max_steps"]):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            buffer.push(state, action, reward, next_state, terminated or truncated)
            total_reward += reward

            if len(buffer) >= config["batch_size"]:
                batch = buffer.sample(config["batch_size"])
                states = torch.FloatTensor(np.array([b[0] for b in batch]))
                actions = torch.LongTensor([b[1] for b in batch])
                rewards = torch.FloatTensor([b[2] for b in batch])
                next_states = torch.FloatTensor(np.array([b[3] for b in batch]))
                dones = torch.FloatTensor([b[4] for b in batch])
                loss = agent.update(states, actions, rewards, next_states, dones)
                ep_loss = loss
                agent.decay_epsilon()

            state = next_state
            if terminated or truncated:
                break

        log_data = {"reward": total_reward, "epsilon": agent.epsilon}
        if ep_loss is not None:
            log_data["loss"] = ep_loss
        logger.log(ep, **log_data)
        if (ep + 1) % 50 == 0:
            recent = logger.history[-50:]
            avg_reward = sum(h["reward"] for h in recent) / 50
            losses = [h["loss"] for h in recent if "loss" in h]
            msg = f"Episode {ep+1}, Avg Reward (last 50): {avg_reward:.1f}"
            if losses:
                msg += f", Avg Loss: {sum(losses)/len(losses):.4f}"
            print(msg)

    logger.save("dqn.json")
    return agent


def evaluate(agent: DQNAgent, n_episodes: int = 100):
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
    from configs.default import DQN_CONFIG

    print("=== DQN 训练 ===")
    agent = train(DQN_CONFIG)
    print("\n=== 验证 ===")
    evaluate(agent)
