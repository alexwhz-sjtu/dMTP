"""
References:

Simple and Effective Masked Diffusion Language Models:
https://arxiv.org/abs/2406.07524

Large Language Diffusion Models:
https://arxiv.org/abs/2502.09992
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from typing import Any

from dllm.core.schedulers import BaseAlphaScheduler, LinearAlphaScheduler
from dllm.utils.data import prepend_bos


class MDLMTrainer(transformers.Trainer):
    """
    Masked Diffusion Language Model Trainer.
    """

    def __init__(
        self,
        scheduler: BaseAlphaScheduler | None = None,
        time_epsilon: float = 1e-3,
        loss_weight_type: str = "scheduler",  # "ones"
        right_shift_logits: bool = False,
        *args,
        train_dataloader: object | None = None,
        eval_dataloader: object | None = None,
        time_start: float = 0.45,
        time_end: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if not (0.0 < time_epsilon < 1.0):
            raise ValueError("time_epsilon must be in (0, 1)")

        self.scheduler = scheduler or LinearAlphaScheduler()
        self.time_epsilon = time_epsilon
        self.loss_weight_type = loss_weight_type
        self.right_shift_logits = right_shift_logits
        # Optional overrides for dataloaders (so Trainer uses our prebuilt DataLoaders)
        self._override_train_dataloader = train_dataloader
        self._override_eval_dataloader = eval_dataloader
        self.time_start = time_start
        self.time_end = time_end

    def get_train_dataloader(self):
        if getattr(self, "_override_train_dataloader", None) is not None:
            return self._override_train_dataloader
        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset=None):
        if getattr(self, "_override_eval_dataloader", None) is not None:
            return self._override_eval_dataloader
        return super().get_eval_dataloader(eval_dataset)

    def _preprocess_inputs(self, inputs):
        if self.right_shift_logits:
            labels = inputs.get("labels", None)

            # If labels exist and EVERY sequence already starts with -100,
            # we treat them as is and skip prepending BOS.
            if labels is not None:
                # shape: [bsz, seq_len]
                if torch.all(labels[:, 0] == -100):
                    return inputs

            # # Otherwise, prepend BOS (and corresponding labels / attention_mask).
            # inputs = prepend_bos(
            #     inputs,
            #     bos_token_id=self.processing_class.bos_token_id,
            #     label_pad_token_id=-100,
            # )
        return inputs

    def _postprocess_outputs(self, outputs):
        if self.right_shift_logits:
            logits = outputs.logits
            outputs.logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        return outputs

    def _compute_loss_weights(
        self,
        t: torch.Tensor,
        inputs: dict[str, Any],
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Compute loss weights given timestep t and other arguments."""
        b, l = inputs["input_ids"].shape
        if self.loss_weight_type == "scheduler":
            loss_weights = self.scheduler.weight(t).unsqueeze(1).repeat(1, l)  # b, 1
        elif self.loss_weight_type == "ones":
            loss_weights = torch.ones_like(inputs["input_ids"])
        else:
            raise NotImplementedError
        return loss_weights

    @torch.no_grad()
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        if prediction_loss_only:
            return (loss.detach(), None, None)

        logits = getattr(outputs, "logits", outputs)
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().contiguous()

        labels = inputs.get("labels")
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().contiguous()

        return (loss.detach(), logits, labels)

    def compute_loss(
        self,
        model: transformers.PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        **kwargs,
    ):
        # assert self.processing_class.padding_side == "right"
        inputs = self._preprocess_inputs(inputs)
        input_ids, in_hidden_states, labels, attention_mask = (
            inputs["input_ids"], # noise(masked) input
            inputs["in_hidden_states"], # add conditional hidden states
            inputs["labels"], # ground-truth label
            inputs.get("attention_mask", None),
        )
        b, l = labels.shape  # Use labels shape instead of input_ids

        # === 1. Sample diffusion timesteps ===
        # Each example draws a random timestep t ∈ [ε, 1), where ε avoids degenerate values near 0.
        # The scheduler defines the masking rate α(t); we convert it to a masking probability p_mask = 1 - α(t).

        clip = False
        if clip:
            # noise clip
            t = self.time_start +  (self.time_end - self.time_start) * torch.rand(b, device=labels.device)
        else:
            #no clip
            t = self.time_epsilon +  (1 - self.time_epsilon) * torch.rand(b, device=labels.device)

        p_mask = 1 - self.scheduler(t).unsqueeze(1).expand(b, l)

        # === 2. Apply stochastic masking ===
        # Tokens are masked independently according to p_mask(t).
        # Positions with label = -100 are excluded (ignored in loss).
        masked_indices = (torch.rand((b, l), device=labels.device) < p_mask) & (
            labels != -100
        )
        # Replace masked tokens with the special [MASK] token.
        # Use labels (target tokens) as the "clean" reference, not input_ids
        noised_input_ids = torch.where(
            masked_indices, 151666, labels
        )

        # === 3. Forward pass through the model ===
        # The model predicts clean tokens given noised inputs and conditional hs.
        # Expand in_hidden_states: [batch, hidden_size] -> [batch, 1(seq_len), hidden_size]
        if in_hidden_states.dim() == 2:
            in_hidden_states = in_hidden_states.unsqueeze(1)
        outputs = model(
            input_ids=noised_input_ids,
            inputs_embeds=in_hidden_states, 
            attention_mask=attention_mask
            )
        outputs = self._postprocess_outputs(outputs)
        logits = outputs.logits

        # === 4. Handle degenerate cases (no tokens masked) ===
        # If no positions were masked, return a zero loss to keep gradients valid.
        # This step is necessary for Deepspeed Zero-{2,3}
        if not masked_indices.any():
            return (
                (logits.sum() * 0.0, outputs) if return_outputs else logits.sum() * 0.0
            )

        # === 5. Compute per-token loss weights ===
        # Depending on the configuration, weights may depend on timestep t
        # (e.g., scheduler-based) or be uniform (ones).
        loss_weights = self._compute_loss_weights(
            t=t, inputs=inputs, masked_indices=masked_indices
        )

        position_loss = False  # 1. 原始：无位置权重
        
         # 2. 新增：位置权重（越靠前权重越高）
        if position_loss:
            # 方式一：线性衰减（可调）
            pos_weights = torch.linspace(1.0, 0.5, l, device=loss_weights.device)  # 从 1.0 到 0.1

            # 方式二：指数衰减（更平滑）
            # decay_factor = 0.99  # 越接近 1，衰减越慢；越小，前面权重远大于后面
            # pos_weights = torch.pow(decay_factor, torch.arange(l, device=loss_weights.device))  # [l]

            # 方式三：倒数衰减（强调开头极少数 token）
            # epsilon = 1e-6
            # pos_weights = 1.0 / (torch.arange(l, device=loss_weights.device) + 1 + epsilon)

            # 扩展到 batch 维度
            pos_weights = pos_weights.unsqueeze(0).expand(b, -1)  # [b, l]
        else:
            pos_weights = torch.ones_like(loss_weights) # [b, l]

        

        # === 6. Compute weighted cross-entropy ===
        # Only masked tokens contribute to the loss.
        assert (labels[masked_indices] == labels[masked_indices]).all()  # labels should match labels
        # 
        token_loss = F.cross_entropy(
            logits[:,-l:][masked_indices], labels[masked_indices], reduction="none"
        )
        token_loss = token_loss * (loss_weights * pos_weights)[masked_indices]

        # === 7. Normalize loss per effective token length ===
        # Normalize each sequence’s contribution by its number of valid tokens,
        # then average over the batch for stability across variable-length inputs.
        effective_lengths = torch.sum(labels != -100, dim=1, keepdim=True).expand(b, l)
        loss = torch.sum(token_loss / effective_lengths[masked_indices]) / b

        # === 8. Return final loss (and optionally model outputs) ===
        return (loss, outputs) if return_outputs else loss
