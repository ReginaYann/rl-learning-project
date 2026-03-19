# PPO (Proximal Policy Optimization)

## 概述

PPO 由 OpenAI 在 2017 年提出，是一种**策略梯度**算法，通过 **clip 机制**限制策略更新幅度，使训练更稳定。广泛应用于机器人、游戏、大模型微调等领域。

## 核心思想

- **Actor-Critic**：同时学习策略（Actor）和价值函数（Critic）
- **Clip 目标**：限制新旧策略比率 \(r_t = \pi(a|s) / \pi_{old}(a|s)\) 在 \([1-\epsilon, 1+\epsilon]\)，避免更新过大
- **GAE**：用广义优势估计降低优势估计的方差

## 公式

### 1. 策略比率

$$
r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)}
$$

### 2. PPO-Clip 目标

$$
L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

- \(\hat{A}_t\)：优势估计（如 GAE）
- \(\epsilon\)：clip 范围，常用 0.2

直观理解：当 \(\hat{A}_t > 0\) 时，限制 \(r_t\) 不要过大；当 \(\hat{A}_t < 0\) 时，限制 \(r_t\) 不要过小。

### 3. 总损失

$$
L = L^{CLIP} - c_1 L^{VF} + c_2 H[\pi]
$$

- \(L^{VF}\)：价值函数 MSE 损失
- \(H[\pi]\)：策略熵，鼓励探索

### 4. 广义优势估计 (GAE)

$$
\hat{A}_t^{GAE} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

其中 TD 误差：
$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

\(\lambda \in [0,1]\) 控制偏差-方差权衡，常用 0.95。

## 算法流程

1. 用当前策略收集一批轨迹，记录 \(s, a, r, \log \pi(a|s), V(s)\)
2. 用 GAE 计算优势 \(\hat{A}_t\) 和回报目标
3. 对同一批数据做多轮（如 4 轮）mini-batch 更新
4. 每轮随机打乱，按 mini-batch 计算 clip 损失 + 价值损失 + 熵项，更新参数

## 适用场景

| 场景 | 说明 |
|------|------|
| 连续/离散动作 | 策略网络输出分布 |
| 复杂环境 | 稳定性好，适合高维控制 |
| 样本复用 | 可对同一批数据多轮更新 |

## 典型应用

- **机器人控制**：OpenAI 用于机械臂、行走
- **游戏 AI**：Dota 2、星际争霸
- **大模型微调**：ChatGPT 的 RLHF 阶段（PPO-LLM）

## 本仓库实现

- 文件：`classic_rl/ppo.py`
- 环境：CartPole-v1
- 运行：`python experiments/run_ppo.py`

## 优缺点

| 优点 | 缺点 |
|------|------|
| 训练稳定 | 超参数较多 |
| 实现相对简单 | 需调 clip 范围、GAE λ 等 |
| 样本可复用 | 比 off-policy 算法样本效率略低 |
