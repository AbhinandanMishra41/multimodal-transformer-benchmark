"""KV caches.

KVCache        - one preallocated [layers, B, Hkv, T_max, D] slab. Simple and fast.
PagedKVCache   - vLLM-style block pool. Memory is handed out in fixed-size blocks
                 as sequences grow, so short sequences don't hold on to T_max slots.
PagedBatch     - the sequences that take part in one forward pass over a PagedKVCache.
                 Attention layers call .update() on it the same way they'd call
                 KVCache.update().
"""

from dataclasses import dataclass
import math

import torch


@dataclass
class KVCacheStats:
    allocated_bytes: int
    active_bytes: int
    utilization_ratio: float
    seq_len: int
    max_seq_len: int

    @property
    def allocated_mb(self) -> float:
        return self.allocated_bytes / 2**20

    @property
    def active_mb(self) -> float:
        return self.active_bytes / 2**20


class KVCache:
    def __init__(
        self,
        batch_size: int,
        num_layers: int,
        num_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.device = torch.device(device)
        self.dtype = dtype
        self.seq_len = 0

        shape = (num_layers, batch_size, num_kv_heads, max_seq_len, head_dim)
        self.k_cache = torch.zeros(shape, device=self.device, dtype=dtype)
        self.v_cache = torch.zeros(shape, device=self.device, dtype=dtype)

    def update(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor, start_pos: int):
        """Write k/v [B, Hkv, n, D] at start_pos, return everything up to start_pos + n."""
        end = start_pos + k_new.shape[2]
        if end > self.max_seq_len:
            raise ValueError(f"KV cache overflow: need {end} slots, have {self.max_seq_len}")

        self.k_cache[layer_idx, :, :, start_pos:end] = k_new
        self.v_cache[layer_idx, :, :, start_pos:end] = v_new
        if layer_idx == self.num_layers - 1:
            self.seq_len = end
        return self.k_cache[layer_idx, :, :, :end], self.v_cache[layer_idx, :, :, :end]

    def stats(self) -> KVCacheStats:
        elem = self.k_cache.element_size()
        allocated = (self.k_cache.numel() + self.v_cache.numel()) * elem
        active = 2 * self.num_layers * self.batch_size * self.num_kv_heads * self.seq_len * self.head_dim * elem
        return KVCacheStats(
            allocated_bytes=allocated,
            active_bytes=active,
            utilization_ratio=active / allocated if allocated else 0.0,
            seq_len=self.seq_len,
            max_seq_len=self.max_seq_len,
        )


class PagedKVCache:
    """Block-pool KV cache.

    k_pool / v_pool: [num_layers, num_blocks, Hkv, block_size, D]
    block_tables[seq_id] maps logical block i -> physical block index.

    A sequence only owns ceil(len / block_size) blocks, so the waste per
    sequence is < block_size tokens instead of (max_seq_len - len).
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_blocks: int = 256,
        block_size: int = 16,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.device = torch.device(device)
        self.dtype = dtype

        shape = (num_layers, num_blocks, num_kv_heads, block_size, head_dim)
        self.k_pool = torch.zeros(shape, device=self.device, dtype=dtype)
        self.v_pool = torch.zeros(shape, device=self.device, dtype=dtype)

        # used as a stack: pop() from the end, so block 0 is handed out first
        self.free_blocks: list[int] = list(range(num_blocks - 1, -1, -1))
        self.block_tables: dict[int, list[int]] = {}
        self.seq_lengths: dict[int, int] = {}
        self.peak_blocks_in_use = 0

    # -- bookkeeping --------------------------------------------------------

    @property
    def blocks_in_use(self) -> int:
        return self.num_blocks - len(self.free_blocks)

    @property
    def bytes_per_block(self) -> int:
        return 2 * self.num_layers * self.num_kv_heads * self.block_size * self.head_dim * self.k_pool.element_size()

    def allocate_sequence(self, seq_id: int) -> None:
        if seq_id in self.block_tables:
            raise ValueError(f"sequence {seq_id} already allocated")
        self.block_tables[seq_id] = []
        self.seq_lengths[seq_id] = 0

    def free_sequence(self, seq_id: int) -> None:
        blocks = self.block_tables.pop(seq_id, [])
        self.seq_lengths.pop(seq_id, None)
        self.free_blocks.extend(reversed(blocks))

    def reserve(self, seq_id: int, num_new_tokens: int) -> None:
        """Make sure seq_id has room for num_new_tokens more tokens."""
        blocks = self.block_tables[seq_id]
        needed = math.ceil((self.seq_lengths[seq_id] + num_new_tokens) / self.block_size)
        while len(blocks) < needed:
            if not self.free_blocks:
                raise RuntimeError("PagedKVCache is out of blocks")
            blocks.append(self.free_blocks.pop())
        self.peak_blocks_in_use = max(self.peak_blocks_in_use, self.blocks_in_use)

    def advance(self, seq_id: int, num_tokens: int) -> None:
        """Mark num_tokens as written. Call once after all layers have written them."""
        self.seq_lengths[seq_id] += num_tokens

    # -- data ---------------------------------------------------------------

    def _slots(self, seq_id: int, start: int, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        pos = torch.arange(start, start + n)
        table = torch.tensor(self.block_tables[seq_id], dtype=torch.long)
        blocks = table[pos // self.block_size].to(self.device)
        offsets = (pos % self.block_size).to(self.device)
        return blocks, offsets

    def write(self, seq_id: int, layer_idx: int, k: torch.Tensor, v: torch.Tensor, start: int) -> None:
        """k, v: [Hkv, n, D], written at token positions start .. start+n-1."""
        n = k.shape[1]
        blocks, offsets = self._slots(seq_id, start, n)
        # advanced indexing on dims 1 and 3 puts the n dim first -> [n, Hkv, D]
        self.k_pool[layer_idx, blocks, :, offsets] = k.transpose(0, 1).to(self.dtype)
        self.v_pool[layer_idx, blocks, :, offsets] = v.transpose(0, 1).to(self.dtype)

    def gather(self, seq_id: int, layer_idx: int, length: int | None = None):
        """Rebuild the contiguous [Hkv, length, D] view of one sequence."""
        length = self.seq_lengths[seq_id] if length is None else length
        if length == 0:
            empty = torch.empty((self.num_kv_heads, 0, self.head_dim), device=self.device, dtype=self.dtype)
            return empty, empty
        n_blocks = math.ceil(length / self.block_size)
        idx = torch.tensor(self.block_tables[seq_id][:n_blocks], device=self.device, dtype=torch.long)
        k = self.k_pool[layer_idx].index_select(0, idx)  # [nb, Hkv, bs, D]
        v = self.v_pool[layer_idx].index_select(0, idx)
        k = k.permute(1, 0, 2, 3).reshape(self.num_kv_heads, -1, self.head_dim)[:, :length]
        v = v.permute(1, 0, 2, 3).reshape(self.num_kv_heads, -1, self.head_dim)[:, :length]
        return k, v

    def append_tokens(self, seq_id: int, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Convenience for filling a cache by hand: write all layers, then advance on the last."""
        if layer_idx == 0:
            self.reserve(seq_id, k.shape[1])
        self.write(seq_id, layer_idx, k, v, start=self.seq_lengths[seq_id])
        if layer_idx == self.num_layers - 1:
            self.advance(seq_id, k.shape[1])

    def memory_comparison(self, max_seq_len: int) -> dict[str, float]:
        """Blocks held right now vs. what a contiguous cache of max_seq_len per sequence would hold."""
        per_token = self.bytes_per_block // self.block_size
        n_seqs = len(self.block_tables)
        paged = sum(len(b) for b in self.block_tables.values()) * self.bytes_per_block
        contiguous = n_seqs * max_seq_len * per_token
        active = sum(self.seq_lengths.values()) * per_token
        return {
            "num_sequences": float(n_seqs),
            "paged_allocated_mb": paged / 2**20,
            "contiguous_allocated_mb": contiguous / 2**20,
            "active_token_mb": active / 2**20,
            "vram_savings_pct": (1 - paged / contiguous) * 100 if contiguous else 0.0,
            "paged_internal_fragmentation_pct": (1 - active / paged) * 100 if paged else 0.0,
        }


class PagedBatch:
    """One forward pass over a set of sequences stored in a PagedKVCache.

    Every sequence gets `num_new` new tokens (the prompt length for prefill,
    1 for decode). Sequences can have different lengths, so update() returns
    right-padded K/V plus a padding mask.

    Gathering per sequence in Python is slow compared to vLLM's fused paged
    attention kernel, but the memory layout and bookkeeping are the same.
    """

    def __init__(self, cache: PagedKVCache, seq_ids: list[int], num_new: int) -> None:
        # The causal mask in attention assumes right-aligned queries, which only holds
        # for single-token decode or a batch of one. That's all the scheduler needs.
        if num_new > 1 and len(seq_ids) > 1:
            raise ValueError("multi-token steps over a PagedKVCache are only supported for one sequence")
        self.cache = cache
        self.seq_ids = list(seq_ids)
        self.num_new = num_new
        self.starts = [cache.seq_lengths[s] for s in self.seq_ids]
        for s in self.seq_ids:
            cache.reserve(s, num_new)

    def position_ids(self, device: torch.device) -> torch.Tensor:
        starts = torch.tensor(self.starts, device=device, dtype=torch.long)
        return starts[:, None] + torch.arange(self.num_new, device=device)[None, :]

    def update(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor):
        """k_new, v_new: [B, Hkv, num_new, D]. Returns (k, v, pad_mask) with pad_mask [B, T] True = padding."""
        ks, vs, lengths = [], [], []
        for i, seq_id in enumerate(self.seq_ids):
            self.cache.write(seq_id, layer_idx, k_new[i], v_new[i], start=self.starts[i])
            length = self.starts[i] + self.num_new
            k, v = self.cache.gather(seq_id, layer_idx, length)
            ks.append(k)
            vs.append(v)
            lengths.append(length)

        t_max = max(lengths)
        k_out = k_new.new_zeros((len(ks), self.cache.num_kv_heads, t_max, self.cache.head_dim))
        v_out = torch.zeros_like(k_out)
        for i, (k, v) in enumerate(zip(ks, vs)):
            k_out[i, :, : k.shape[1]] = k
            v_out[i, :, : v.shape[1]] = v

        lengths_t = torch.tensor(lengths, device=k_new.device)
        pad_mask = torch.arange(t_max, device=k_new.device)[None, :] >= lengths_t[:, None]
        return k_out, v_out, pad_mask

    def finish(self) -> None:
        for s in self.seq_ids:
            self.cache.advance(s, self.num_new)
