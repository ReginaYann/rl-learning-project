# PPO for LLM (PPO-LLM)

## 概述

PPO-LLM 是将 **PPO** 应用于**语言模型**的强化学习方法，常用于 RLHF（基于人类反馈的强化学习）阶段。模型根据 prompt 生成文本，由奖励模型（或人类）打分，再用 PPO 优化策略使生成内容获得更高奖励。

## 核心思想

- **策略**：语言模型 $\pi_\theta$，输入 prompt $x$，输出 response $y$ 的分布
- **奖励**：由奖励模型 $r(x, y)$ 或人类反馈给出
- **参考策略**：通常为 SFT 模型 $\pi_{ref}$，用于 KL 约束，防止偏离过远
- **PPO**：用 clip 目标优化策略，使 $r$ 提高且不过度偏离 $\pi_{ref}$

## 公式

### 1. 策略与动作

- 状态 $s$：prompt + 已生成的部分 token
- 动作 $a$：下一个 token
- 策略 $\pi(a|s)$：语言模型的 next-token 概率

### 2. 奖励设计

$$
R(x, y) = r_{RM}(x, y) - \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}
$$

- $r_{RM}$：奖励模型得分
- 第二项：KL 惩罚，限制与参考策略的偏离

### 3. PPO 目标（与经典 PPO 相同）

$$
L^{CLIP} = \mathbb{E} \left[ \min \left( r_t \hat{A}_t,\; \text{clip}(r_t, 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

其中 $r_t = \pi_\theta(a|s) / \pi_{ref}(a|s)$。

### 4. 优势估计

在 LLM 场景中，通常对整条 response 计算一个奖励，优势可简化为：

$$
\hat{A} = R - \bar{R}
$$

即当前 response 的奖励减去 batch 内平均奖励。

## 算法流程

1. 用当前策略对 prompt 采样生成 response
2. 用奖励模型计算 $r(x, y)$
3. 计算策略与参考策略的 log prob，得到 ratio
4. 用 PPO clip 损失更新策略

## 适用场景

| 场景 | 说明 |
|------|------|
| RLHF | 有奖励模型或人类反馈 |
| 复杂奖励 | 多目标、多维度奖励 |
| 需要在线生成 | 可边生成边优化 |

## 典型应用

- **ChatGPT**：RLHF 阶段使用 PPO
- **Claude、Gemini**：类似 RLHF 流程
- **定制化助手**：根据业务奖励微调

## 本仓库实现

- 文件：`llm_rl/ppo_llm.py`
- 奖励模型：简化的规则模型（可替换为真实 RM）
- 运行：`python experiments/run_ppo_llm.py`

## 与 DPO 的对比

| 方面 | PPO-LLM | DPO |
|------|---------|-----|
| 奖励 | 需要奖励模型 | 不需要，用偏好数据 |
| 数据 | 在线生成 + RM 打分 | 离线偏好三元组 |
| 优化 | 策略梯度 + clip | 监督式分类损失 |
| 复杂度 | 高 | 低 |
| 适用 | 复杂、多目标奖励 | 简单偏好、快速迭代 |

## 优缺点

| 优点 | 缺点 |
|------|------|
| 可表达复杂奖励 | 需训练奖励模型 |
| 支持在线学习 | 实现和调参复杂 |
| 与 RLHF 流程一致 | 训练不稳定风险较高 |
