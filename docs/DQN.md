# DQN (Deep Q-Network)

## 概述

DQN 由 DeepMind 在 2015 年提出，用**神经网络**近似 Q 函数，解决了 Q-Learning 无法处理**高维状态**（如图像）的问题。通过**经验回放**和**目标网络**显著提升了训练稳定性。

## 核心思想

- **函数近似**：用神经网络 \(Q(s, a; \theta)\) 替代 Q 表
- **经验回放**：将转移 \((s, a, r, s')\) 存入缓冲区，随机采样训练，打破数据相关性
- **目标网络**：用独立的 \(Q(s', a'; \theta^-)\) 计算目标，减少目标值剧烈变化

## 公式

### 1. 损失函数（均方误差）

$$
L(\theta) = \mathbb{E}_{(s,a,r,s') \sim D} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]
$$

其中：
- \(\theta\)：当前网络参数
- \(\theta^-\)：目标网络参数（定期从 \(\theta\) 复制）
- \(D\)：经验回放缓冲区

### 2. 目标值

$$
y = r + \gamma \max_{a'} Q(s', a'; \theta^-) \cdot (1 - \text{done})
$$

若回合结束（done=1），则 \(y = r\)。

### 3. 动作选择（ε-greedy）

与 Q-Learning 相同，基于 \(Q(s, \cdot; \theta)\) 做 ε-greedy。

## 算法流程

1. 初始化策略网络 \(Q_\theta\) 和目标网络 \(Q_{\theta^-}\)，\(\theta^- \leftarrow \theta\)
2. 每步：用 ε-greedy 选动作，执行，将 \((s, a, r, s')\) 存入回放缓冲区
3. 从缓冲区采样 mini-batch，计算 TD 目标，最小化 MSE 更新 \(\theta\)
4. 每 \(C\) 步：\(\theta^- \leftarrow \theta\)
5. 衰减 \(\epsilon\)

## 适用场景

| 场景 | 说明 |
|------|------|
| 高维状态 | 图像、连续观测（如 CartPole 的 4 维状态） |
| 离散动作 | 动作空间有限 |
| 无模型 | 不需要环境模型 |

## 典型应用

- **Atari 游戏**：原始论文用 84×84 图像作为输入
- **CartPole**：经典控制任务
- **推荐系统**：状态为用户特征，动作为推荐项
- **机器人控制**：离散动作的简单控制

## 本仓库实现

- 文件：`classic_rl/dqn.py`
- 环境：CartPole-v1
- 运行：`python experiments/run_dqn.py`

## 改进变体（扩展阅读）

- **Double DQN**：用当前网络选动作，目标网络评估，减轻过估计
- **Dueling DQN**：分解 \(Q = V + A\)，学习更稳定
- **Prioritized Replay**：按 TD 误差优先采样重要经验

## 优缺点

| 优点 | 缺点 |
|------|------|
| 可处理高维状态 | 仅支持离散动作 |
| 经验回放提高样本效率 | 超参数敏感（学习率、目标更新频率等） |
| 目标网络提升稳定性 | 对连续动作需 DDPG、SAC 等 |
