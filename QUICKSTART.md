# Quickstart 快速上手

## 1. 环境安装

**推荐 Python：3.10 或 3.11**

```bash
cd rl-learning-project

# 推荐：使用安装脚本（自动安装 CUDA 版 PyTorch）
bash install.sh

# 或手动安装
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

> 环境为 CUDA 12.2 时，使用 `cu121` 构建的 PyTorch 即可。

## 2. 命令行运行（完整训练 + 验证）

### 经典 RL 算法

```bash
# Q-Learning (FrozenLake, 约 2–5 分钟)
python experiments/run_q_learning.py

# DQN (CartPole, 约 5–10 分钟)
python experiments/run_dqn.py

# REINFORCE (CartPole, 约 5–10 分钟)
python experiments/run_reinforce.py

# PPO (CartPole, 约 5–15 分钟)
python experiments/run_ppo.py
```

### LLM 微调算法

```bash
# DPO (需下载 distilgpt2，首次较慢)
python experiments/run_dpo.py

# PPO-LLM
python experiments/run_ppo_llm.py
```

## 3. 快速代码示例（不跑完整训练）

在项目根目录下执行：

```python
# === Q-Learning 快速验证 ===
import sys
sys.path.insert(0, ".")
import gymnasium as gym
from classic_rl.q_learning import QLearningAgent

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
agent = QLearningAgent(env.observation_space.n, env.action_space.n)
state, _ = env.reset()
for _ in range(10):
    action = agent.select_action(state)
    next_state, reward, term, trunc, _ = env.step(action)
    agent.update(state, action, reward, next_state, term or trunc)
    state = next_state
    if term or trunc:
        break
print("Q-Learning OK")
```

```python
# === DQN 快速验证 ===
import sys
sys.path.insert(0, ".")
import gymnasium as gym
import numpy as np
from classic_rl.dqn import DQNAgent
from utils.replay_buffer import ReplayBuffer

env = gym.make("CartPole-v1")
agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
buffer = ReplayBuffer(1000)
state, _ = env.reset()
for _ in range(100):
    action = agent.select_action(state)
    next_state, reward, term, trunc, _ = env.step(action)
    buffer.push(state, action, reward, next_state, term or trunc)
    state = next_state
    if term or trunc:
        state, _ = env.reset()
    if len(buffer) >= 32:
        batch = buffer.sample(32)
        import torch
        states = torch.FloatTensor(np.array([b[0] for b in batch]))
        actions = torch.LongTensor([b[1] for b in batch])
        rewards = torch.FloatTensor([b[2] for b in batch])
        next_states = torch.FloatTensor(np.array([b[3] for b in batch]))
        dones = torch.FloatTensor([b[4] for b in batch])
        agent.update(states, actions, rewards, next_states, dones)
print("DQN OK")
```

```python
# === REINFORCE 快速验证 ===
import sys
sys.path.insert(0, ".")
import gymnasium as gym
from classic_rl.reinforce import REINFORCEAgent

env = gym.make("CartPole-v1")
agent = REINFORCEAgent(env.observation_space.shape[0], env.action_space.n)
state, _ = env.reset()
for _ in range(20):
    action = agent.select_action(state)
    next_state, reward, term, trunc, _ = env.step(action)
    agent.store_reward(reward)
    state = next_state
    if term or trunc:
        break
agent.update()
print("REINFORCE OK")
```

```python
# === PPO 快速验证 ===
import sys
sys.path.insert(0, ".")
import gymnasium as gym
import numpy as np
from classic_rl.ppo import PPOAgent

env = gym.make("CartPole-v1")
agent = PPOAgent(env.observation_space.shape[0], env.action_space.n)
states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []
state, _ = env.reset()
for _ in range(64):
    action, log_prob, value = agent.select_action(state)
    next_state, reward, term, trunc, _ = env.step(action)
    states.append(state)
    actions.append(action)
    log_probs.append(log_prob)
    rewards.append(reward)
    dones.append(float(term or trunc))
    values.append(value)
    state = next_state
    if term or trunc:
        state, _ = env.reset()
adv, ret = agent.compute_gae(rewards, values, dones, 0.0, 1.0)
agent.update(np.array(states), np.array(actions), np.array(log_probs), np.array(adv), np.array(ret))
print("PPO OK")
```

```python
# === DPO 快速验证（需联网下载模型）===
import sys
sys.path.insert(0, ".")
from llm_rl.dpo import DPOTrainer

trainer = DPOTrainer(model_name="distilgpt2", beta=0.1, lr=5e-5)
batch = {"prompt": ["你好"], "chosen": ["你好！有什么可以帮你的？"], "rejected": ["嗯"]}
loss = trainer.train_step(batch)
print(f"DPO OK, loss={loss:.4f}")
```

## 4. 缩短训练时间（调试用）

在 `configs/default.py` 中可临时修改：

```python
# 例如将 QL_CONFIG 的 n_episodes 改为 500
QL_CONFIG = dict(
    n_episodes=500,  # 原 5000
    ...
)
```

或在运行脚本中直接覆盖：

```python
# experiments/run_q_learning.py 末尾
if __name__ == "__main__":
    from configs.default import QL_CONFIG
    QL_CONFIG["n_episodes"] = 500  # 快速测试
    agent = train(QL_CONFIG)
    evaluate(agent)
```

## 5. 输出与日志

- 训练过程会打印到终端
- 日志保存在 `logs/` 目录，如 `logs/q_learning/q_learning.json`
