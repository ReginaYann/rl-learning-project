#!/usr/bin/env python3
"""
PPO-LLM 训练与验证
使用小模型 distilgpt2，简化奖励模型（规则）演示
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_rl.ppo_llm import PPOTrainerLLM
from utils.logger import Logger


def train(config: dict):
    data_path = Path(__file__).parent.parent / "data" / "preference_data.jsonl"
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            import json
            data.append(json.loads(line.strip()))

    prompts = [d["prompt"] for d in data]
    trainer = PPOTrainerLLM(
        model_name=config.get("model_name", "distilgpt2"),
        lr=config.get("lr", 1e-5),
        device=config.get("device", "cpu"),
    )
    logger = Logger("logs/ppo_llm")
    n_epochs = config.get("n_epochs", 3)

    for epoch in range(n_epochs):
        total_loss = 0
        total_reward = 0
        for i, prompt in enumerate(prompts):
            stats = trainer.train_step([prompt])
            total_loss += stats["loss"]
            total_reward += stats["mean_reward"]
        avg_loss = total_loss / len(prompts)
        avg_reward = total_reward / len(prompts)
        logger.log(epoch, loss=avg_loss, reward=avg_reward)
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}, Reward: {avg_reward:.4f}")

    logger.save("ppo_llm.json")
    return trainer


def evaluate(trainer: PPOTrainerLLM, n_samples: int = 3):
    """简单验证: 生成示例"""
    import json
    data_path = Path(__file__).parent.parent / "data" / "preference_data.jsonl"
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    prompts = [d["prompt"] for d in data[:n_samples]]

    trainer.model.eval()
    print("\n=== 验证: 生成示例 ===")
    for prompt in prompts:
        responses, _, _ = trainer.generate([prompt])
        reward = trainer.reward_model([prompt], responses)[0].item()
        print(f"Prompt: {prompt}")
        print(f"Generated: {responses[0][:80]}...")
        print(f"Reward: {reward:.4f}")
        print()


if __name__ == "__main__":
    config = dict(n_epochs=3, model_name="distilgpt2", lr=1e-5)
    print("=== PPO-LLM 训练 ===")
    trainer = train(config)
    evaluate(trainer)
