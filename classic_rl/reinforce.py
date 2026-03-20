"""
REINFORCE: 策略梯度算法 (Monte Carlo)
on-policy，每回合结束后用整条轨迹的回报更新策略
核心: 梯度上升，log_prob * G_t，其中 G_t 为从 t 开始的回报
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class PolicyNet(nn.Module):
    """策略网络: 输入状态，输出动作概率分布"""

    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return F.softmax(logits, dim=-1)


class REINFORCEAgent:
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.device = torch.device(device)
        self.policy = PolicyNet(state_dim, n_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        # 存整条轨迹的 log_prob 和 reward，回合结束后一次性更新（Monte Carlo）
        self.saved_log_probs = []
        self.rewards = []

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        # 离散动作上用 Categorical 表示策略分布 π(a|s)，目的：
        # 1) probs：网络 softmax 输出，各动作概率之和为 1
        # 2) Categorical(probs)：PyTorch 中的离散分布，可 sample、可算 log_prob
        # 3) sample()：按 π 随机选动作（探索来自策略随机性，非 ε-greedy）
        # 4) log_prob(action)：得到 log π(a|s)，REINFORCE 需 -log π * G_t，且对参数可导
        # 若用 argmax 则策略确定、无有效 log_prob，策略梯度无法更新
        x = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.policy(x)
        dist = Categorical(probs)
        action = dist.sample()
        if training:
            self.saved_log_probs.append(dist.log_prob(action))
        return action.item()

    def store_reward(self, reward: float):
        self.rewards.append(reward)

    def update(self) -> float:
        # 计算回报 G_t = sum_{k>=t} gamma^{k-t} r_k（从 t 到回合结束的折扣回报）
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        # 转成 tensor 放到 GPU，与 log_prob 同设备才能相乘、反传
        returns = torch.FloatTensor(returns).to(self.device)
        # 标准化：(G_t - 均值) / 标准差。不改变“好轨迹权重大”的相对关系，但缩小数值尺度，
        # 减轻策略梯度的方差（类似 baseline，严格说会略改无偏性，实践中很常用）
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # REINFORCE 要最大化 E[ sum_t log π(a_t|s_t) * G_t ]（G_t 为从 t 起的回报）。
        # 优化器做梯度下降，故损失 = - sum_t log_prob * R（这里 R 已是标准化后的 G_t）。
        # G_t>0（相对本回合平均更好）时，增大该步 log π → 更常选这类动作；G_t<0 则相反。
        policy_loss = []
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        loss = torch.stack(policy_loss).sum()  # 整条轨迹一步反传

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.saved_log_probs = []
        self.rewards = []
        return loss.item()
