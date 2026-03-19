"""
PPO for LLM: 将 PPO 应用于语言模型
需要: 策略模型、参考模型、奖励模型(或人工反馈)
核心: 用 PPO 的 clip 目标优化策略，使生成内容获得更高奖励
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional, Tuple


class RewardModel:
    """
    简化的奖励模型: 基于规则或小模型的打分
    实际应用中可替换为训练好的 RM 或人类反馈
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """返回每个 (prompt, response) 的奖励，范围 [0,1]"""
        scores = []
        for prompt, response in zip(prompts, responses):
            # 简单规则: 长度适中、包含完整句子给高分
            length = len(response)
            has_period = "." in response or "。" in response
            score = 0.3 * min(length / 50, 1.0) + 0.7 * float(has_period)
            scores.append(score)
        return torch.tensor(scores, dtype=torch.float32)


class PPOTrainerLLM:
    """
    简化的 PPO-LLM 训练器
    演示如何将 PPO 应用于语言模型生成
    """

    def __init__(
        self,
        model_name: str = "distilgpt2",
        ref_model_name: Optional[str] = None,
        lr: float = 1e-5,
        clip_eps: float = 0.2,
        gamma: float = 0.99,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)

        self.ref_model = AutoModelForCausalLM.from_pretrained(ref_model_name or model_name)
        self.ref_model.to(self.device)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        self.reward_model = RewardModel(self.tokenizer)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

    def get_log_probs(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        """计算 response 部分的 log pi(a|s)"""
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1]
        labels = input_ids[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs, 2, labels.unsqueeze(-1)
        ).squeeze(-1)
        # mask 掉 prompt 部分
        response_token_log_probs = token_log_probs * response_mask[:, 1:]
        return (response_token_log_probs.sum(-1) / (response_mask[:, 1:].sum(-1) + 1e-8))

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 32,
        do_sample: bool = True,
        temperature: float = 0.7,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        """生成 response，返回文本、log_probs、ref_log_probs"""
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(self.device)
        prompt_lengths = inputs["attention_mask"].sum(dim=1)

        # 策略模型生成
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        responses = []
        for i, out in enumerate(outputs):
            response_ids = out[prompt_lengths[i] :]
            responses.append(self.tokenizer.decode(response_ids, skip_special_tokens=True))

        # 计算 log probs (简化: 用 teacher forcing)
        full_ids = outputs
        full_attention = (full_ids != self.tokenizer.pad_token_id).long()
        if full_attention.sum() == 0:
            full_attention = torch.ones_like(full_ids, dtype=torch.long)
        response_mask = torch.zeros_like(full_ids, dtype=torch.float)
        for i in range(len(prompts)):
            pl = prompt_lengths[i].item()
            response_mask[i, pl:] = 1.0

        policy_log_probs = self.get_log_probs(
            self.model, full_ids, full_attention, response_mask
        )
        with torch.no_grad():
            ref_log_probs = self.get_log_probs(
                self.ref_model, full_ids, full_attention, response_mask
            )
        return responses, policy_log_probs, ref_log_probs

    def compute_ppo_loss(
        self,
        policy_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """PPO clip 损失"""
        ratio = torch.exp(policy_log_probs - ref_log_probs)
        advantages = rewards - rewards.mean()
        pg_loss1 = ratio * advantages
        pg_loss2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        loss = -torch.min(pg_loss1, pg_loss2).mean()
        return loss

    def train_step(self, prompts: List[str]) -> Dict[str, float]:
        self.model.train()
        responses, policy_log_probs, ref_log_probs = self.generate(prompts)
        rewards = self.reward_model(prompts, responses).to(self.device)
        loss = self.compute_ppo_loss(policy_log_probs, ref_log_probs, rewards)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item(), "mean_reward": rewards.mean().item()}
