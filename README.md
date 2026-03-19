# RL 学习代码仓库

面向初学者的强化学习代码库，包含**经典RL算法**和**大模型微调算法**（DPO、PPO等），结构清晰，便于阅读与实操。

## 目录结构

```
rl-learning-project/
├── classic_rl/          # 经典强化学习算法
│   ├── q_learning.py     # Q-Learning (表格型)
│   ├── dqn.py            # Deep Q-Network
│   ├── reinforce.py      # REINFORCE (策略梯度)
│   └── ppo.py            # PPO (近端策略优化)
├── llm_rl/               # 大模型微调算法
│   ├── dpo.py            # Direct Preference Optimization
│   └── ppo_llm.py        # PPO for LLM
├── envs/                 # 环境封装
├── data/                 # 小规模数据集
├── experiments/          # 训练与验证脚本
├── configs/              # 配置文件
└── utils/                # 工具函数
```

## 快速开始

### 1. 安装依赖

**推荐 Python 版本：3.10 或 3.11**

```bash
# 方式一：使用安装脚本（推荐，自动安装 CUDA 版 PyTorch）
bash install.sh

# 方式二：手动安装
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

> 你的环境为 CUDA 12.2 / RTX 4090，使用 PyTorch cu121 构建即可兼容。

### 2. 命令行运行（完整训练 + 验证）

**经典 RL：**

```bash
python experiments/run_q_learning.py   # Q-Learning (FrozenLake)
python experiments/run_dqn.py         # DQN (CartPole)
python experiments/run_reinforce.py   # REINFORCE (CartPole)
python experiments/run_ppo.py         # PPO (CartPole)
```

**LLM 微调：**

```bash
python experiments/run_dpo.py         # DPO
python experiments/run_ppo_llm.py      # PPO-LLM
```

### 3. 详细说明

- 完整命令列表：见 [QUICKSTART.md](QUICKSTART.md)
- 快速代码示例（不跑完整训练）：见 QUICKSTART.md 第 3 节

## 算法说明

详细文档见 [docs/](docs/) 目录：

### 经典RL
- [Q-Learning](docs/Q-Learning.md)：表格型、off-policy，适合离散动作空间
- [DQN](docs/DQN.md)：用神经网络近似Q函数，支持高维状态
- [REINFORCE](docs/REINFORCE.md)：策略梯度，on-policy
- [PPO](docs/PPO.md)：稳定训练的策略梯度，带clip机制

### LLM微调
- [DPO](docs/DPO.md)：直接从偏好数据优化策略，无需奖励模型
- [PPO-LLM](docs/PPO-LLM.md)：将PPO应用于语言模型，结合奖励模型

## 数据集

- 经典RL: 使用 Gymnasium 内置环境 (CartPole, FrozenLake)
- DPO/PPO-LLM: `data/preference_data.jsonl` 小规模偏好数据
