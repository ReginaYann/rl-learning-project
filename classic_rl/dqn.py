"""
DQN: Deep Q-Network
用神经网络近似 Q(s,a)，支持高维连续状态 (如 CartPole 的 4 维状态)
核心: 经验回放 + 目标网络
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DQN(nn.Module):
    """Q 网络: 输入状态，输出每个动作的 Q 值"""

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
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        target_update_freq: int = 100,
        device: str = "cpu",
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)
        self.update_count = 0

        self.policy_net = DQN(state_dim, n_actions).to(self.device)
        self.target_net = DQN(state_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        with torch.no_grad():
            x = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q = self.policy_net(x)
            return int(q.argmax(dim=1).item())

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> float:
        # 当前 Q 值
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        # 目标 Q 值: r + gamma * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q * (1 - dones)
        loss = F.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
