# REINFORCE

## 概述

REINFORCE 是一种**策略梯度**算法，由 Williams 于 1992 年提出。它直接优化策略 $\pi_\theta(a|s)$，属于 **on-policy**、**Monte Carlo** 方法：每回合结束后用整条轨迹的回报来更新策略。

## 核心思想

- **策略梯度**：对策略参数 $\theta$ 做梯度上升，最大化期望回报
- **Monte Carlo**：用完整轨迹的回报 $G_t$ 作为无偏估计，无需价值函数
- **On-policy**：只用当前策略采样的数据更新

## 公式

### 1. 策略梯度定理

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot G_t \right]
$$

其中 $G_t$ 为从时刻 $t$ 开始的回报：

$$
G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k
$$

### 2. 实际更新（单条轨迹）

对一条轨迹，损失函数为：

$$
L = -\sum_t \log \pi_\theta(a_t | s_t) \cdot G_t
$$

最小化 $L$ 等价于最大化策略梯度目标。

### 3. 回报标准化（减小方差）

实践中常对 $G_t$ 做标准化：

$$
\hat{G}_t = \frac{G_t - \mu}{\sigma + \epsilon}
$$

其中 $\mu, \sigma$ 为该轨迹内 $G_t$ 的均值和标准差。

## 算法流程

1. 用当前策略 $\pi_\theta$ 采样完整轨迹
2. 计算每个时刻的 $G_t$
3. （可选）对 $G_t$ 标准化
4. 计算 $L = -\sum_t \log \pi(a_t|s_t) \cdot G_t$
5. 反向传播，更新 $\theta$

## 适用场景

| 场景 | 说明 |
|------|------|
| 离散/连续动作 | 策略网络可输出任意分布 |
| 回合制任务 | 每局有明确结束 |
| 无需价值函数 | 实现简单 |

## 典型应用

- **CartPole**：经典控制
- **简单游戏**：回合制决策
- **序列生成**：可作为早期策略梯度方法的代表

## 本仓库实现

- 文件：`classic_rl/reinforce.py`
- 环境：CartPole-v1
- 运行：`python experiments/run_reinforce.py`

## 优缺点

| 优点 | 缺点 |
|------|------|
| 实现简单 | 高方差，收敛慢 |
| 支持连续动作 | 仅 on-policy，样本效率低 |
| 无需价值函数 | 需完整轨迹才能更新 |

## 与 Actor-Critic 的关系

REINFORCE 用 $G_t$ 作为梯度权重；Actor-Critic 用**价值函数**或 **优势函数** $A(s,a)$ 替代 $G_t$，可降低方差、支持单步更新。
