"""
Q-Learning: 表格型、无模型、off-policy 算法
适用于: 离散状态、离散动作 (如 FrozenLake)
核心: Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
"""
import numpy as np


class QLearningAgent:
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.1,
        lr: float = None,
        gamma: float = 0.99,
        epsilon: float = 1.0,      # 探索率: 1=完全随机探索，0=纯贪心。初期 Q 表为空，从 1 开始合理
        epsilon_decay: float = 0.995,  # 每 episode 后 epsilon *= decay，逐渐减少探索
        epsilon_min: float = 0.01,  # 探索率下限，保留少量随机避免过早收敛到次优
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr if lr is not None else learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        # Q表: Q[state, action]
        self.Q = np.zeros((n_states, n_actions))

    def select_action(self, state: int, training: bool = True) -> int:
        # ε-greedy: 以 epsilon 概率随机探索，否则选 Q 值最大的动作（利用）
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        # Q-Learning 更新: 使用 max_a' Q(s',a') 作为目标 (off-policy)
        if done:
            target = reward  # 回合结束，无后续回报
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])
        td_error = target - self.Q[state, action]
        self.Q[state, action] += self.lr * td_error

    def decay_epsilon(self):
        # 每 episode 调用一次，使 epsilon 逐渐从 1 降到 epsilon_min，实现「先探索后利用」
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
