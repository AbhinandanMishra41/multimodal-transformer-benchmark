"""Rotary position embeddings.

Uses real cos/sin tables and rotate_half instead of torch.polar / complex64.
The complex version is shorter but the ONNX exporter and TensorRT parser
don't support complex tensors, so this keeps one code path for everything.
"""

import torch
import torch.nn as nn


def precompute_rope_cos_sin(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (cos, sin), each [max_seq_len, head_dim]."""
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")

    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(half, device=device, dtype=torch.float32) / half))
    pos = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(pos, inv_freq)  # [T, half]

    # same angle for both halves, matching the rotate_half layout (HF llama does the same)
    emb = torch.cat([angles, angles], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    start_pos: int | torch.Tensor = 0,
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """q: [B, Hq, L, D], k: [B, Hkv, L, D].

    Positions are either start_pos .. start_pos+L-1 for the whole batch, or an
    explicit [B, L] `position_ids` (needed by the paged cache, where every
    sequence in the batch sits at a different position).
    """
    seq_len = q.shape[2]
    if position_ids is not None:
        cos_s = cos[position_ids].unsqueeze(1)  # [B, 1, L, D]
        sin_s = sin[position_ids].unsqueeze(1)
    elif isinstance(start_pos, torch.Tensor):
        # tensor start_pos only shows up during ONNX tracing; index_select keeps it dynamic
        pos = torch.arange(seq_len, device=q.device, dtype=torch.long) + start_pos.to(torch.long)
        cos_s = cos.index_select(0, pos)[None, None]
        sin_s = sin.index_select(0, pos)[None, None]
    else:
        cos_s = cos[start_pos : start_pos + seq_len][None, None]
        sin_s = sin[start_pos : start_pos + seq_len][None, None]

    cos_s = cos_s.to(q.dtype)
    sin_s = sin_s.to(q.dtype)
    q_rot = q * cos_s + rotate_half(q) * sin_s
    k_rot = k * cos_s + rotate_half(k) * sin_s
    return q_rot, k_rot


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 2048, theta: float = 10000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        cos, sin = precompute_rope_cos_sin(head_dim, max_seq_len, theta)
        # non-persistent: cheap to rebuild, no point storing them in checkpoints
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(self, q, k, start_pos=0, position_ids=None):
        return apply_rotary_pos_emb(
            q, k, self.cos_cached, self.sin_cached, start_pos=start_pos, position_ids=position_ids
        )
