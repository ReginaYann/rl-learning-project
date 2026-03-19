"""
DPO: Direct Preference Optimization
直接从偏好数据优化策略，无需训练奖励模型
核心损失: L = -log sigma(beta * (log pi(y_w|x)/pi_ref(y_w|x) - log pi(y_l|x)/pi_ref(y_l|x)))
其中 y_w=chosen, y_l=rejected, sigma=sigmoid
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional


def get_batch_log_probs(
    model: nn.Module,
    tokenizer,
    prompts: List[str],
    responses: List[str],
    device: torch.device,
) -> torch.Tensor:
    """
    计算 log pi(y|x)，即给定 prompt x 生成 response y 的对数概率
    对于因果 LM: log pi(y|x) = sum_t log P(y_t | x, y_{<t})
    """
    batch_log_probs = []
    for prompt, response in zip(prompts, responses):
        full_text = prompt + response
        inputs = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=True))
        need_grad = model.training
        with torch.no_grad() if not need_grad else torch.enable_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = inputs["input_ids"][..., 1:].contiguous().view(-1)
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            token_log_probs = -loss_fct(shift_logits, shift_labels)
            # 只取 response 部分 (从 prompt_len 开始)
            response_token_log_probs = token_log_probs.view(1, -1)[0, prompt_len - 1 : -1]
            if response_token_log_probs.numel() > 0:
                log_prob = response_token_log_probs.mean()
            else:
                log_prob = torch.tensor(0.0, device=device)
        batch_log_probs.append(log_prob)
    return torch.stack(batch_log_probs)


class DPOTrainer:
    """
    简化的 DPO 训练器，便于理解算法
    使用小模型 (如 distilgpt2) 进行演示
    """

    def __init__(
        self,
        model_name: str = "distilgpt2",
        ref_model_name: Optional[str] = None,
        beta: float = 0.1,
        lr: float = 5e-5,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.beta = beta
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)

        self.ref_model = AutoModelForCausalLM.from_pretrained(ref_model_name or model_name)
        self.ref_model.to(self.device)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

    def compute_loss(
        self,
        prompts: List[str],
        chosen: List[str],
        rejected: List[str],
    ) -> torch.Tensor:
        """DPO 损失: -log sigma(beta * (log_ratio_chosen - log_ratio_rejected))"""
        # log pi(y|x) - log pi_ref(y|x) 即策略与参考策略的 log ratio
        policy_chosen_logps = get_batch_log_probs(
            self.model, self.tokenizer, prompts, chosen, self.device
        )
        policy_rejected_logps = get_batch_log_probs(
            self.model, self.tokenizer, prompts, rejected, self.device
        )
        ref_chosen_logps = get_batch_log_probs(
            self.ref_model, self.tokenizer, prompts, chosen, self.device
        )
        ref_rejected_logps = get_batch_log_probs(
            self.ref_model, self.tokenizer, prompts, rejected, self.device
        )

        # log_ratio = log pi/pi_ref
        chosen_log_ratios = policy_chosen_logps - ref_chosen_logps
        rejected_log_ratios = policy_rejected_logps - ref_rejected_logps
        # DPO 目标: chosen 的 ratio 应大于 rejected
        logits = self.beta * (chosen_log_ratios - rejected_log_ratios)
        loss = -F.logsigmoid(logits).mean()
        return loss

    def train_step(self, batch: Dict[str, List[str]]) -> float:
        self.model.train()
        loss = self.compute_loss(
            batch["prompt"],
            batch["chosen"],
            batch["rejected"],
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
