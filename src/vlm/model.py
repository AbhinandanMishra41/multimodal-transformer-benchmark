"""ViT + projector + decoder glued together.

The decoder input is [image tokens | text tokens]. There is no <image>
placeholder token; the image always comes first, which is enough for the
single-image setup used here.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import VLMConfig
from .kv_cache import KVCache, PagedBatch, PagedKVCache
from .language import LanguageDecoder
from .projector import MultimodalProjector
from .vision import VisionEncoder


class MultimodalTransformer(nn.Module):
    def __init__(self, config: VLMConfig) -> None:
        super().__init__()
        self.config = config
        self.vision_encoder = VisionEncoder(config.vision)
        self.projector = MultimodalProjector(config)
        self.decoder = LanguageDecoder(config.text)

    def parameter_breakdown(self) -> dict[str, int]:
        count = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
        parts = {
            "vision_encoder": count(self.vision_encoder),
            "multimodal_projector": count(self.projector),
            "language_decoder": count(self.decoder),
        }
        parts["total"] = sum(parts.values())
        return parts

    def create_kv_cache(self, batch_size, max_seq_len=None, device=None, dtype=None) -> KVCache:
        p = next(self.parameters())
        return KVCache(
            batch_size=batch_size,
            num_layers=self.config.text.num_layers,
            num_kv_heads=self.config.text.num_kv_heads,
            max_seq_len=max_seq_len or self.config.text.max_seq_len,
            head_dim=self.config.text.head_dim,
            device=device or p.device,
            dtype=dtype or p.dtype,
        )

    def create_paged_cache(self, num_blocks: int, block_size: int = 16) -> PagedKVCache:
        p = next(self.parameters())
        return PagedKVCache(
            num_layers=self.config.text.num_layers,
            num_kv_heads=self.config.text.num_kv_heads,
            head_dim=self.config.text.head_dim,
            num_blocks=num_blocks,
            block_size=block_size,
            device=p.device,
            dtype=p.dtype,
        )

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """[B, C, H, W] -> [B, N_vis, D_llm]"""
        return self.projector(self.vision_encoder(images))

    def build_multimodal_embeddings(self, images: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.encode_image(images), self.decoder.embed_tokens(input_ids)], dim=1)

    def forward(self, images, input_ids, labels=None):
        """Returns (logits [B, N_vis + N_txt, V], loss or None).

        `labels` are the already-shifted targets: labels[:, j] is the token that
        should follow input_ids[:, j]. Positions equal to pad_token_id are ignored.
        """
        logits, _ = self.decoder.forward_embeddings(self.build_multimodal_embeddings(images, input_ids))
        loss = None
        if labels is not None:
            text_logits = logits[:, self.config.num_projected_visual_tokens :]
            loss = F.cross_entropy(
                text_logits.reshape(-1, text_logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=self.config.pad_token_id,
            )
        return logits, loss

    # -- inference with a contiguous KVCache ------------------------------------

    @torch.inference_mode()
    def prefill(self, images, prompt_ids, kv_cache: KVCache | None = None):
        logits, present = self.decoder.forward_embeddings(
            self.build_multimodal_embeddings(images, prompt_ids), start_pos=0, kv_cache=kv_cache
        )
        return logits[:, -1], present

    @torch.inference_mode()
    def decode_one_token(self, token_ids, start_pos: int, kv_cache=None, past_key_values=None):
        logits, present = self.decoder(
            token_ids, start_pos=start_pos, kv_cache=kv_cache, past_key_values=past_key_values
        )
        return logits[:, -1], present

    @torch.inference_mode()
    def generate(
        self,
        images: torch.Tensor,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 32,
        use_kv_cache: bool = True,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        """Returns prompt + max_new_tokens generated ids.

        use_kv_cache=False recomputes the whole sequence every step. It's only
        here as a baseline for the benchmark.
        """
        if max_new_tokens <= 0:
            return prompt_ids
        n_vis = self.config.num_projected_visual_tokens
        total = n_vis + prompt_ids.shape[1] + max_new_tokens
        if total > self.config.text.max_seq_len:
            raise ValueError(
                f"{n_vis} image + {prompt_ids.shape[1]} prompt + {max_new_tokens} new tokens = {total}, "
                f"which is more than max_seq_len ({self.config.text.max_seq_len})"
            )

        if not use_kv_cache:
            visual = self.encode_image(images)
            out = prompt_ids
            for _ in range(max_new_tokens):
                x = torch.cat([visual, self.decoder.embed_tokens(out)], dim=1)
                logits, _ = self.decoder.forward_embeddings(x)
                out = torch.cat([out, self._sample(logits[:, -1], temperature)], dim=1)
            return out

        cache = self.create_kv_cache(images.shape[0], max_seq_len=total, device=images.device, dtype=images.dtype)
        logits, _ = self.prefill(images, prompt_ids, kv_cache=cache)
        tok = self._sample(logits, temperature)
        out = [prompt_ids, tok]
        pos = n_vis + prompt_ids.shape[1]
        for _ in range(max_new_tokens - 1):
            logits, _ = self.decode_one_token(tok, start_pos=pos, kv_cache=cache)
            tok = self._sample(logits, temperature)
            out.append(tok)
            pos += 1
        return torch.cat(out, dim=1)

    # -- inference with a PagedKVCache ---------------------------------------------

    @torch.inference_mode()
    def prefill_paged(self, image, prompt_ids, cache: PagedKVCache, seq_id: int) -> torch.Tensor:
        """Prefill a single request (image [1, C, H, W], prompt [1, L]) into the paged cache."""
        x = self.build_multimodal_embeddings(image, prompt_ids)
        cache.allocate_sequence(seq_id)
        batch = PagedBatch(cache, [seq_id], num_new=x.shape[1])
        logits, _ = self.decoder.forward_embeddings(x, paged=batch)
        batch.finish()
        return logits[:, -1]

    @torch.inference_mode()
    def decode_paged(self, token_ids, cache: PagedKVCache, seq_ids: list[int]) -> torch.Tensor:
        """One decode step for a batch of sequences that may all be at different lengths."""
        batch = PagedBatch(cache, seq_ids, num_new=1)
        logits, _ = self.decoder(token_ids, paged=batch)
        batch.finish()
        return logits[:, -1]

    @staticmethod
    def _sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
        if temperature <= 0:
            return logits.argmax(dim=-1, keepdim=True)
        return torch.multinomial(F.softmax(logits / temperature, dim=-1), num_samples=1)
