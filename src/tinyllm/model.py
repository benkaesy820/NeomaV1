from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TinyConfig:
    vocab_size: int = 8000
    seq_len: int = 256
    n_layers: int = 6
    d_model: int = 256
    n_heads: int = 4
    n_kv_heads: int = 2
    d_ff: int = 768
    dropout: float = 0.0
    rope_base: float = 10000.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TinyConfig":
        fields = cls.__dataclass_fields__
        return cls(**{key: data[key] for key in fields if key in data})

    def validate(self) -> None:
        integer_fields = {
            "vocab_size": self.vocab_size,
            "seq_len": self.seq_len,
            "n_layers": self.n_layers,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "d_ff": self.d_ff,
        }
        for name, value in integer_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")
        if self.rope_base <= 0:
            raise ValueError("rope_base must be greater than zero")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        head_dim = self.d_model // self.n_heads
        if head_dim % 2 != 0:
            raise ValueError("attention head dimension must be even for RoPE")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute the variance in float32 for numerical stability, then cast back.
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(variance + self.eps).to(dtype=x.dtype)
        return self.weight * normalized


def _build_rope_cache(
    seq_len: int,
    head_dim: int,
    base: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(seq_len, dtype=torch.float32)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos(), freqs.sin()


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    seq_len = x.size(-2)
    cos = cos[:seq_len][None, None, :, :].to(dtype=x.dtype)
    sin = sin[:seq_len][None, None, :, :].to(dtype=x.dtype)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack(
        (x_even * cos - x_odd * sin, x_even * sin + x_odd * cos),
        dim=-1,
    )
    return rotated.flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TinyConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.dropout = config.dropout
        self._native_gqa_supported: bool | None = None

        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        dropout_p = self.dropout if self.training else 0.0
        if self.n_heads == self.n_kv_heads:
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=True,
            )

        # Modern PyTorch can execute grouped-query attention directly. Older
        # versions fall back to explicit KV expansion so the project remains
        # portable across CPU-only installations.
        if self._native_gqa_supported is not False:
            try:
                output = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=dropout_p,
                    is_causal=True,
                    enable_gqa=True,
                )
                self._native_gqa_supported = True
                return output
            except (TypeError, RuntimeError, NotImplementedError):
                self._native_gqa_supported = False

        repeats = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = _apply_rope(q, rope_cos, rope_sin)
        k = _apply_rope(k, rope_cos, rope_sin)
        y = self._attention(q, k, v)
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, self.n_heads * self.head_dim)
        return self.out_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, config: TinyConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: TinyConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.ffn_norm = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ffn = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), rope_cos, rope_sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TinyLanguageModel(nn.Module):
    def __init__(self, config: TinyConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        head_dim = config.d_model // config.n_heads
        rope_cos, rope_sin = _build_rope_cache(config.seq_len, head_dim, config.rope_base)
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

        self.apply(self._init_weights)
        self._scale_residual_projections()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residual_projections(self) -> None:
        # GPT-style residual scaling keeps activation variance controlled as
        # layers are stacked while retaining simple random initialization.
        scale = (2.0 * self.config.n_layers) ** -0.5
        with torch.no_grad():
            for block in self.blocks:
                block.attn.out_proj.weight.mul_(scale)
                block.ffn.down_proj.weight.mul_(scale)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.size(1) > self.config.seq_len:
            raise ValueError(f"sequence length exceeds configured limit of {self.config.seq_len}")
        if targets is not None and targets.shape != input_ids.shape:
            raise ValueError("targets must have the same shape as input_ids")
        if loss_mask is not None:
            if targets is None:
                raise ValueError("loss_mask requires targets")
            if loss_mask.shape != targets.shape:
                raise ValueError("loss_mask must have the same shape as targets")

        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        logits = self.lm_head(self.norm(x))

        loss = None
        if targets is not None:
            flat_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="none",
            )
            if loss_mask is None:
                loss = flat_loss.mean()
            else:
                flat_mask = loss_mask.reshape(-1).to(dtype=flat_loss.dtype)
                denominator = flat_mask.sum()
                if denominator.item() == 0:
                    loss = flat_loss.sum() * 0.0
                else:
                    loss = (flat_loss * flat_mask).sum() / denominator
        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 50,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if top_k < 0:
            raise ValueError("top_k must be non-negative")

        was_training = self.training
        self.eval()
        finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)
        try:
            for _ in range(max_new_tokens):
                idx = input_ids[:, -self.config.seq_len :]
                logits, _ = self(idx)
                logits = logits[:, -1, :]

                if temperature == 0:
                    next_id = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    logits = logits / temperature
                    if top_k > 0:
                        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                        logits = logits.masked_fill(logits < values[:, [-1]], -float("inf"))
                    probs = F.softmax(logits, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1)

                if eos_id is not None:
                    next_id = torch.where(
                        finished[:, None],
                        torch.full_like(next_id, eos_id),
                        next_id,
                    )
                    finished |= next_id.squeeze(1).eq(eos_id)

                input_ids = torch.cat((input_ids, next_id), dim=1)
                if eos_id is not None and bool(finished.all()):
                    break
        finally:
            self.train(was_training)
        return input_ids


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
