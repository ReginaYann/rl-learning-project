"""经验回放缓冲区，用于 DQN 等 off-policy 算法"""
import random
from collections import deque
from typing import Any, List, Tuple


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: Any, action: int, reward: float, next_state: Any, done: bool):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Tuple]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)
