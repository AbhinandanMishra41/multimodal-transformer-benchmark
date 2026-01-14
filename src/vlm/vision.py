"""Plain ViT encoder (pre-norm, CLS token, learned position embeddings)."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import VisionConfig


class PatchEmbedding(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.config = config
        # a conv with kernel = stride = patch_size is the same as "cut into patches + linear"
        self.proj = nn.Conv2d(config.in_channels, config.hidden_dim, config.patch_size, stride=config.patch_size)
        if config.include_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.register_parameter("cls_token", None)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.num_visual_tokens, config.hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.proj(images).flatten(2).transpose(1, 2)  # [B, N_patches, D]
        if self.cls_token is not None:
            x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        return x + self.pos_embed.to(x.dtype)


class VisionAttention(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qkv = nn.Linear(config.hidden_dim, 3 * config.hidden_dim)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # unbind, not tuple-unpacking, so the ONNX tracer doesn't warn
        # no mask: image patches attend to each other in both directions
        attn = F.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1, dtype=torch.float32).to(x.dtype)
        out = (attn @ v).transpose(1, 2).reshape(b, n, d)
        return self.out_proj(out)


class VisionMLP(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        hidden = int(config.hidden_dim * config.mlp_ratio)
        self.fc1 = nn.Linear(config.hidden_dim, hidden)
        self.fc2 = nn.Linear(hidden, config.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class VisionBlock(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        self.attn = VisionAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        self.mlp = VisionMLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class VisionEncoder(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbedding(config)
        self.blocks = nn.ModuleList([VisionBlock(config) for _ in range(config.num_layers)])
        self.post_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """[B, C, H, W] -> [B, N_vis, D_vis]"""
        x = self.patch_embed(images)
        for block in self.blocks:
            x = block(x)
        return self.post_norm(x)
