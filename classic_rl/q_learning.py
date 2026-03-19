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
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        # Q表: Q[state, action]
        self.Q = np.zeros((n_states, n_actions))

    def select_action(self, state: int, training: bool = True) -> int:
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        # Q-Learning 更新: 使用 max_a' Q(s',a') 作为目标 (off-policy)
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])
        td_error = target - self.Q[state, action]
        self.Q[state, action] += self.lr * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
