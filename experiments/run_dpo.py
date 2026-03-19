#!/usr/bin/env python3
"""
DPO 训练与验证
使用小规模偏好数据，小模型 distilgpt2 演示
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_rl.dpo import DPOTrainer
from utils.logger import Logger


def load_preference_data(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def train(config: dict):
    data_path = Path(__file__).parent.parent / "data" / "preference_data.jsonl"
    data = load_preference_data(data_path)
    trainer = DPOTrainer(
        model_name=config.get("model_name", "distilgpt2"),
        beta=config.get("beta", 0.1),
        lr=config.get("lr", 5e-5),
        device=config.get("device", "cpu"),
    )
    logger = Logger("logs/dpo")
    n_epochs = config.get("n_epochs", 3)

    for epoch in range(n_epochs):
        total_loss = 0
        for i, item in enumerate(data):
            batch = {
                "prompt": [item["prompt"]],
                "chosen": [item["chosen"]],
                "rejected": [item["rejected"]],
            }
            loss = trainer.train_step(batch)
            total_loss += loss
        avg_loss = total_loss / len(data)
        logger.log(epoch, loss=avg_loss)
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")

    logger.save("dpo.json")
    return trainer


def evaluate(trainer: DPOTrainer, data_path: str, n_samples: int = 3):
    """简单验证: 对几个 prompt 生成，观察输出质量"""
    data = load_preference_data(data_path)
    trainer.model.eval()
    print("\n=== 验证: 生成示例 ===")
    for i, item in enumerate(data[:n_samples]):
        prompt = item["prompt"]
        import torch
        inputs = trainer.tokenizer(prompt, return_tensors="pt").to(trainer.device)
        with torch.no_grad():
            out = trainer.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=trainer.tokenizer.eos_token_id,
            )
        gen = trainer.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"Prompt: {prompt}")
        print(f"Chosen: {item['chosen'][:50]}...")
        print(f"Generated: {gen[:50]}...")
        print()


if __name__ == "__main__":
    config = dict(n_epochs=3, model_name="distilgpt2", beta=0.1, lr=5e-5)
    print("=== DPO 训练 ===")
    trainer = train(config)
    data_path = Path(__file__).parent.parent / "data" / "preference_data.jsonl"
    evaluate(trainer, str(data_path))
