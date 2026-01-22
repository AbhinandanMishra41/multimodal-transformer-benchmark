"""Llama-style decoder: RMSNorm, RoPE, grouped-query attention, SwiGLU."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TextConfig
from .kv_cache import KVCache, PagedBatch
from .rope import RotaryEmbedding

KVPair = tuple[torch.Tensor, torch.Tensor]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # do the reduction in fp32, otherwise fp16 overflows on large activations
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class SwiGLUMLP(nn.Module):
    def __init__(self, config: TextConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.up_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.down_proj = nn.Linear(config.intermediate_dim, config.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[B, Hkv, T, D] -> [B, Hkv * n_rep, T, D] so each query head has a matching kv head."""
    if n_rep == 1:
        return x
    b, h, t, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_q_heads = config.num_q_heads
        self.num_kv_heads = config.num_kv_heads
        self.num_kv_groups = config.num_kv_groups
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(config.hidden_dim, self.num_q_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_q_heads * self.head_dim, config.hidden_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding,
        start_pos: int | torch.Tensor = 0,
        kv_cache: KVCache | None = None,
        past_kv: KVPair | None = None,
        paged: PagedBatch | None = None,
    ) -> tuple[torch.Tensor, KVPair]:
        """Three ways to get the history:
        - kv_cache: preallocated KVCache (eager inference)
        - past_kv:  explicit tensors, concatenated (this is what the ONNX graph uses)
        - paged:    PagedKVCache via a PagedBatch
        """
        bsz, q_len, _ = x.shape
        q = self.q_proj(x).view(bsz, q_len, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        pad_mask = None
        if paged is not None:
            q, k = rope(q, k, position_ids=paged.position_ids(x.device))
            k, v, pad_mask = paged.update(self.layer_idx, k, v)
        else:
            q, k = rope(q, k, start_pos=start_pos)
            if kv_cache is not None:
                if not isinstance(start_pos, int):
                    raise TypeError("KVCache needs an int start_pos")
                k, v = kv_cache.update(self.layer_idx, k, v, start_pos=start_pos)
            elif past_kv is not None:
                k = torch.cat([past_kv[0], k], dim=2)
                v = torch.cat([past_kv[1], v], dim=2)
        present = (k, v)

        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        # Written out by hand instead of F.scaled_dot_product_attention so the
        # exported ONNX graph is plain MatMul/Softmax that every runtime handles.
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        kv_len = k.shape[2]
        # queries are the last q_len positions, which works for prefill and decode alike
        q_pos = torch.arange(kv_len - q_len, kv_len, device=x.device)[:, None]
        k_pos = torch.arange(kv_len, device=x.device)[None, :]
        mask = (k_pos > q_pos)[None, None]
        if pad_mask is not None:
            mask = mask | pad_mask[:, None, None, :]
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(x.dtype)
        out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.o_proj(out), present


class DecoderBlock(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.attn = GroupedQueryAttention(config, layer_idx=layer_idx)
        self.ffn_norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.mlp = SwiGLUMLP(config)

    def forward(self, x, rope, start_pos=0, kv_cache=None, past_kv=None, paged=None):
        h, present = self.attn(
            self.attn_norm(x), rope=rope, start_pos=start_pos,
            kv_cache=kv_cache, past_kv=past_kv, paged=paged,
        )
        x = x + h
        x = x + self.mlp(self.ffn_norm(x))
        return x, present


class LanguageDecoder(nn.Module):
    def __init__(self, config: TextConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.rope = RotaryEmbedding(config.head_dim, config.max_seq_len, config.rope_theta)
        self.layers = nn.ModuleList([DecoderBlock(config, i) for i in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward_embeddings(
        self,
        inputs_embeds: torch.Tensor,
        start_pos: int | torch.Tensor = 0,
        kv_cache: KVCache | None = None,
        past_key_values: list[KVPair] | None = None,
        paged: PagedBatch | None = None,
    ) -> tuple[torch.Tensor, list[KVPair]]:
        h = inputs_embeds
        presents = []
        for i, layer in enumerate(self.layers):
            past = past_key_values[i] if past_key_values is not None else None
            h, present = layer(h, self.rope, start_pos=start_pos, kv_cache=kv_cache, past_kv=past, paged=paged)
            presents.append(present)
        return self.lm_head(self.norm(h)), presents

    def forward(self, input_ids, start_pos=0, kv_cache=None, past_key_values=None, paged=None):
        return self.forward_embeddings(
            self.embed_tokens(input_ids), start_pos=start_pos,
            kv_cache=kv_cache, past_key_values=past_key_values, paged=paged,
        )
