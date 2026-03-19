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
        self.saved_log_probs = []
        self.rewards = []

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
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
        # 计算回报 G_t = sum_{k>=t} gamma^{k-t} r_k
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # 标准化

        # 策略梯度: -log_prob * G_t (最小化负的 = 最大化)
        policy_loss = []
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        loss = torch.stack(policy_loss).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.saved_log_probs = []
        self.rewards = []
        return loss.item()
