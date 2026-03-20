"""
PPO: Proximal Policy Optimization
带 clip 的策略梯度，训练更稳定
核心: L = min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)，其中 r_t = pi(a|s)/pi_old(a|s)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """Actor: 策略网络 π(a|s); Critic: 价值网络 V(s)。共享特征提取"""

    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, n_actions)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        features = self.shared(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value

    def get_action_and_value(
        self, x: torch.Tensor, action=None
    ) -> tuple:
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value.squeeze(-1)


class PPOAgent:
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,   # GAE 的 λ，权衡偏差与方差
        clip_eps: float = 0.2,      # PPO clip 范围，限制策略更新幅度
        n_epochs: int = 4,          # 每批数据重复训练轮数
        batch_size: int = 64,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.model = ActorCritic(state_dim, n_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def select_action(self, state: np.ndarray, training: bool = True) -> tuple:
        with torch.no_grad():
            x = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, _, value = self.model.get_action_and_value(x)
        return action.item(), log_prob.item(), value.item()

    def compute_gae(
        self,
        rewards: list,
        values: list,
        dones: list,
        next_value: float,
        next_done: bool,
    ) -> tuple:
        """计算广义优势估计 GAE: A_t = δ_t + (γλ)δ_{t+1} + ...，其中 δ_t = r_t + γV(s_{t+1}) - V(s_t)"""
        advantages = []
        lastgaelam = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
            lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            advantages.insert(0, lastgaelam)
        returns = [a + v for a, v in zip(advantages, values)]  # return = advantage + value
        return advantages, returns

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        log_probs_old: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
    ) -> dict:
        indices = np.arange(len(states))
        total_loss = 0
        for _ in range(self.n_epochs):  # 每批数据重复训练，提高样本利用率
            np.random.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                end = start + self.batch_size
                mb_indices = indices[start:end]

                mb_states = torch.FloatTensor(states[mb_indices]).to(self.device)
                mb_actions = torch.LongTensor(actions[mb_indices]).to(self.device)
                mb_log_probs_old = torch.FloatTensor(log_probs_old[mb_indices]).to(self.device)
                mb_advantages = torch.FloatTensor(advantages[mb_indices]).to(self.device)
                mb_returns = torch.FloatTensor(returns[mb_indices]).to(self.device)

                _, log_prob, entropy, value = self.model.get_action_and_value(mb_states, mb_actions)
                ratio = torch.exp(log_prob - mb_log_probs_old)  # π(a|s) / π_old(a|s)
                pg_loss1 = ratio * mb_advantages
                pg_loss2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_advantages
                pg_loss = -torch.min(pg_loss1, pg_loss2).mean()  # PPO clip
                vf_loss = F.mse_loss(value, mb_returns)
                loss = pg_loss + 0.5 * vf_loss - 0.01 * entropy.mean()  # 熵项鼓励探索

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
        return {"loss": total_loss / (self.n_epochs * (len(indices) // self.batch_size + 1))}
