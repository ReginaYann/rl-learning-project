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
        need_grad = model.training  # 策略模型需梯度，参考模型不需
        with torch.no_grad() if not need_grad else torch.enable_grad():
            # 对 prompt+response 整条序列前向，得到每个位置「预测下一 token」的 logits
            outputs = model(**inputs)
            logits = outputs.logits
            # 因果 LM：位置 t 的 logits 预测的是 token t+1，故 logits[:-1] 与 input_ids[1:] 对齐
            shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = inputs["input_ids"][..., 1:].contiguous().view(-1)
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            # CE = -log P(真实下一 token | 上文)，取负得到每个位置上的 log P
            token_log_probs = -loss_fct(shift_logits, shift_labels)
            # DPO 只要「在 x 条件下生成 y」的 log π：去掉 prompt 段，只保留 response 对应位置
            # （切片边界依赖 prompt_len 与拼接 token 化一致，可能有轻微近似）
            response_token_log_probs = token_log_probs.view(1, -1)[0, prompt_len - 1 : -1]
            if response_token_log_probs.numel() > 0:
                # 对 response 各 token 的 log prob 取平均，作为 log π(y|x) 的标量估计（也可用 sum）
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
        beta: float = 0.1,  # 控制偏离参考策略的程度，越大越保守
        lr: float = 5e-5,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.beta = beta
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)

        # 参考模型（通常为 SFT 模型）冻结，用于计算 log ratio
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

        # log_ratio = log π/π_ref。希望 chosen 的 ratio 高、rejected 的 ratio 低
        chosen_log_ratios = policy_chosen_logps - ref_chosen_logps
        rejected_log_ratios = policy_rejected_logps - ref_rejected_logps
        # DPO 损失: -log σ(β * (chosen_ratio - rejected_ratio))，等价于偏好分类
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
