import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import VLMConfig


class MultimodalProjector(nn.Module):
    """LayerNorm + 2-layer GELU MLP from vision width to LM width (LLaVA-1.5 style).

    token_pooling_factor > 1 is a crude token reduction: every p consecutive tokens
    (raster order, CLS included) are concatenated along the channel dim. Unlike
    SmolVLM's pixel shuffle it doesn't merge 2D neighbourhoods, and leftover tokens
    at the end are dropped. All the benchmarks use p = 1.
    """

    def __init__(self, config: VLMConfig) -> None:
        super().__init__()
        self.pooling_factor = config.token_pooling_factor
        in_dim = config.vision.hidden_dim * self.pooling_factor
        self.norm = nn.LayerNorm(in_dim, eps=config.vision.layer_norm_eps)
        self.linear_1 = nn.Linear(in_dim, config.projector_hidden_dim)
        self.linear_2 = nn.Linear(config.projector_hidden_dim, config.text.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        p = self.pooling_factor
        if p > 1:
            n_keep = (n // p) * p
            x = x[:, :n_keep].reshape(b, n_keep // p, d * p)
        return self.linear_2(F.gelu(self.linear_1(self.norm(x))))
