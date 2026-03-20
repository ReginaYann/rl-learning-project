# Q-Learning

## 概述

Q-Learning 是一种**无模型**、**off-policy** 的强化学习算法，由 Watkins 于 1989 年提出。它通过维护一张 Q 表来学习最优动作价值函数，适用于**离散状态、离散动作**的环境。

## 核心思想

- **Q 函数** $Q(s, a)$：在状态 $s$ 下执行动作 $a$ 的期望累积回报
- **表格型**：用二维数组存储每个 (状态, 动作) 的 Q 值，状态/动作数量需有限
- **Off-policy**：更新时使用 $\max_{a'} Q(s', a')$，即贪心目标，与当前探索策略无关

## 公式

### 1. Q 值更新（TD 更新）

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
$$

其中：
- $\alpha$：学习率
- $\gamma$：折扣因子
- $r$：即时奖励
- $s'$：下一状态
- $\max_{a'} Q(s', a')$：下一状态的最优 Q 值（off-policy 体现）

### 2. TD 误差

$$
\delta = r + \gamma \max_{a'} Q(s', a') - Q(s, a)
$$

### 3. 动作选择（ε-greedy）

$$
a = \begin{cases}
\text{random action} & \text{with probability } \epsilon \\
\arg\max_a Q(s, a) & \text{with probability } 1 - \epsilon
\end{cases}
$$

## 算法流程

1. 初始化 Q 表为 0
2. 每步：用 ε-greedy 选动作 $a$
3. 执行 $a$，得到 $r, s'$
4. 用上述公式更新 $Q(s, a)$
5. 衰减 $\epsilon$ 以逐渐减少探索

## 适用场景

| 场景 | 说明 |
|------|------|
| 离散状态 | 状态可枚举（如格子世界） |
| 离散动作 | 动作数量有限 |
| 中小规模 | 状态/动作组合不宜过大，否则 Q 表过大 |
| 无模型 | 不需要环境转移概率 $P(s' \mid s, a)$ |

## 典型应用

- **FrozenLake**：网格寻路
- **简单游戏**：如井字棋、迷宫
- **资源调度**：离散决策问题
- **教学**：理解 RL 的入门算法

## 本仓库实现

- 文件：`classic_rl/q_learning.py`
- 环境：FrozenLake-v1 (4×4)
- 运行：`python experiments/run_q_learning.py`

## 优缺点

| 优点 | 缺点 |
|------|------|
| 实现简单 | 仅适用于离散、小规模问题 |
| 无需环境模型 | 高维/连续状态无法用表格表示 |
| 收敛性有理论保证 | 探索策略需精心设计 |
