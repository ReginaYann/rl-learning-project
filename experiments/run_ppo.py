#!/usr/bin/env python3
"""
PPO 训练与验证
环境: CartPole
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
import numpy as np
import torch
from classic_rl.ppo import PPOAgent
from utils.logger import Logger


def train(config: dict):
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = PPOAgent(state_dim, n_actions, **{
        k: v for k, v in config.items()
        if k in ["lr", "gamma", "gae_lambda", "clip_eps", "n_epochs", "batch_size"]
    })
    logger = Logger("logs/ppo")

    n_steps = config.get("n_steps", 2048)
    ep_count = 0

    while ep_count < config["n_episodes"]:
        states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []
        state, _ = env.reset()
        ep_rewards = []

        for _ in range(n_steps):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(float(done))
            values.append(value)
            ep_rewards.append(reward)

            state = next_state
            if done:
                state, _ = env.reset()
                ep_count += 1
                if ep_count >= config["n_episodes"]:
                    break

        # 最后一步的 value 用于 GAE
        if dones and dones[-1]:
            next_value, next_done = 0.0, 1.0
        else:
            with torch.no_grad():
                x = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                _, _, _, next_value = agent.model.get_action_and_value(x)
            next_value = next_value.item()
            next_done = 0.0

        advantages, returns = agent.compute_gae(
            rewards, values, dones, next_value, next_done
        )
        stats = agent.update(
            np.array(states),
            np.array(actions),
            np.array(log_probs),
            np.array(advantages),
            np.array(returns),
        )
        total_reward = sum(ep_rewards)
        logger.log(ep_count, reward=total_reward, **stats)
        if ep_count % 50 == 0 and ep_count > 0:
            recent = [h["reward"] for h in logger.history[-50:] if "reward" in h]
            avg = sum(recent) / len(recent) if recent else 0
            print(f"Episode {ep_count}, Avg Reward (last 50): {avg:.1f}")

    logger.save("ppo.json")
    return agent


def evaluate(agent: PPOAgent, n_episodes: int = 100):
    env = gym.make("CartPole-v1")
    rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        total = 0
        for _ in range(500):
            action, _, _ = agent.select_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break
        rewards.append(total)
    print(f"验证: 平均回报 {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")


if __name__ == "__main__":
    from configs.default import PPO_CONFIG

    print("=== PPO 训练 ===")
    agent = train(PPO_CONFIG)
    print("\n=== 验证 ===")
    evaluate(agent)
