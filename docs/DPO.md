# DPO (Direct Preference Optimization)

## 概述

DPO 由 Rafailov 等人在 2023 年提出，是一种**直接从偏好数据优化语言模型**的方法。它**无需训练奖励模型**，将 RLHF 的优化目标转化为一个简单的分类损失，训练更稳定、实现更简单。

## 核心思想

- **偏好数据**：每条数据为 $(x, y_w, y_l)$，其中 $y_w$ 优于 $y_l$
- **隐式奖励**：通过 Bradley-Terry 模型，偏好可表示为奖励差
- **无奖励模型**：用策略与参考策略的 log ratio 替代显式奖励，推导出闭式损失

## 公式

### 1. Bradley-Terry 偏好模型

人类偏好可用奖励差建模：

$$
P(y_w \succ y_l | x) = \sigma \left( r(x, y_w) - r(x, y_l) \right)
$$

其中 $\sigma$ 为 sigmoid 函数。

### 2. 最优策略与奖励的关系

在 RLHF 中，最优策略可写为：

$$
\pi^*(y|x) \propto \pi_{ref}(y|x) \exp\left( \frac{1}{\beta} r(x,y) \right)
$$

可推出：
$$
r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)
$$

### 3. DPO 损失（核心）

将上述代入 Bradley-Terry，消去 $r$ 和 $Z(x)$，得到：

$$
\mathcal{L}_{DPO} = -\mathbb{E}_{(x,y_w,y_l)} \left[ \log \sigma \left( \beta \left( \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right) \right]
$$

简记 log ratio：
$$
\log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)} = \log \pi_\theta(y|x) - \log \pi_{ref}(y|x)
$$

对于因果 LM，$\log \pi(y|x) = \sum_t \log P(y_t | x, y_{<t})$。

### 4. 直观理解

- 希望 $\pi_\theta(y_w|x)$ 相对 $\pi_{ref}$ 提高
- 希望 $\pi_\theta(y_l|x)$ 相对 $\pi_{ref}$ 降低
- $\beta$ 控制偏离参考策略的程度，越大越保守

## 算法流程

1. 准备偏好数据：$(prompt, chosen, rejected)$
2. 加载策略模型 $\pi_\theta$ 和参考模型 $\pi_{ref}$（通常为 SFT 模型）
3. 对每个 batch：计算 chosen 和 rejected 的 log prob（策略与参考）
4. 计算 log ratio，代入 DPO 损失
5. 反向传播更新 $\pi_\theta$

## 适用场景

| 场景 | 说明 |
|------|------|
| 对齐大模型 | 让模型更符合人类偏好 |
| 有偏好数据 | 需 (prompt, 好回答, 差回答) 三元组 |
| 无需奖励模型 | 省去 RM 训练和 PPO 的复杂流程 |

## 典型应用

- **Chat 模型对齐**：提升有用性、无害性
- **代码模型**：偏好更优解
- **翻译/摘要**：偏好更流畅、准确的输出

## 本仓库实现

- 文件：`llm_rl/dpo.py`
- 数据：`data/preference_data.jsonl`
- 运行：`python experiments/run_dpo.py`

## 与 RLHF / PPO 的关系

| 方法 | 奖励模型 | 优化方式 |
|------|----------|----------|
| RLHF (PPO) | 需要 | 用 RM 做奖励，PPO 优化策略 |
| DPO | 不需要 | 直接从偏好推导损失，监督学习式优化 |

## 优缺点

| 优点 | 缺点 |
|------|------|
| 无需奖励模型 | 依赖高质量偏好数据 |
| 训练稳定 | 难以融入复杂奖励（如多目标） |
| 实现简单 | 对超参数 $\beta$ 敏感 |
