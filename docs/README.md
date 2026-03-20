# 算法文档索引

本目录包含各算法的说明文档，包括公式、适用场景和典型应用。

> **公式显示**：文档使用 `$...$`（行内）和 `$$...$$`（块级）书写数学公式。若预览中公式不显示，可安装支持 Math 的 Markdown 扩展（如 VS Code 的 Markdown Preview Enhanced、Markdown+Math 等）。

## 经典 RL 算法

| 算法 | 文档 | 代码 | 环境 |
|------|------|------|------|
| [Q-Learning](Q-Learning.md) | 表格型、off-policy | `classic_rl/q_learning.py` | FrozenLake |
| [DQN](DQN.md) | 深度 Q 网络、经验回放 | `classic_rl/dqn.py` | CartPole |
| [REINFORCE](REINFORCE.md) | 策略梯度、Monte Carlo | `classic_rl/reinforce.py` | CartPole |
| [PPO](PPO.md) | 近端策略优化、clip | `classic_rl/ppo.py` | CartPole |

## LLM 微调算法

| 算法 | 文档 | 代码 | 数据 |
|------|------|------|------|
| [DPO](DPO.md) | 直接偏好优化 | `llm_rl/dpo.py` | preference_data.jsonl |
| [PPO-LLM](PPO-LLM.md) | PPO 用于语言模型 | `llm_rl/ppo_llm.py` | preference_data.jsonl |

## 面试复习

| 文档 | 说明 |
|------|------|
| [面试考点-RL与大模型对齐](面试考点-RL与大模型对齐.md) | 大模型算法岗常见 RL / RLHF / DPO 问答 |
