"""Continuous batching over the paged KV cache, at a few batch sizes.

The workload is a fixed list of requests with different prompt lengths and
different output lengths, so sequences finish at different times and new ones
get swapped in. Memory numbers compare the peak number of KV blocks actually
held against what a contiguous cache would need for the same number of
concurrent slots:
  - sized for the longest request in this workload (the fair comparison)
  - sized for max_seq_len (what you'd do if you don't know request lengths up front)
"""

from dataclasses import dataclass
import copy

import torch

from vlm.model import MultimodalTransformer
from vlm.scheduler import Request, run_continuous_batching


@dataclass
class BatchingResult:
    max_batch_size: int
    num_requests: int
    generated_tokens: int
    wall_time_s: float
    throughput_tokens_per_sec: float
    mean_decode_step_ms: float
    paged_peak_kv_mb: float
    contiguous_kv_mb_workload_max: float
    contiguous_kv_mb_max_seq_len: float

    @property
    def savings_vs_workload_max_pct(self) -> float:
        return 100 * (1 - self.paged_peak_kv_mb / self.contiguous_kv_mb_workload_max)

    @property
    def savings_vs_max_seq_len_pct(self) -> float:
        return 100 * (1 - self.paged_peak_kv_mb / self.contiguous_kv_mb_max_seq_len)


def make_workload(model: MultimodalTransformer, num_requests: int, max_new_tokens: int, seed: int = 0) -> list[Request]:
    cfg = model.config
    g = torch.Generator().manual_seed(seed)
    reqs = []
    for i in range(num_requests):
        prompt_len = int(torch.randint(4, 33, (1,), generator=g))
        n_new = int(torch.randint(max(1, max_new_tokens // 4), max_new_tokens + 1, (1,), generator=g))
        reqs.append(
            Request(
                request_id=i,
                image=torch.randn(3, cfg.vision.image_size, cfg.vision.image_size, generator=g),
                prompt_ids=torch.randint(4, cfg.text.vocab_size, (prompt_len,), generator=g),
                max_new_tokens=n_new,
            )
        )
    return reqs


def benchmark_continuous_batching(
    model: MultimodalTransformer,
    batch_sizes: tuple[int, ...] = (1, 4, 8, 16),
    num_requests: int = 32,
    max_new_tokens: int = 32,
    block_size: int = 16,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> list[BatchingResult]:
    model = copy.deepcopy(model).to(device=torch.device(device), dtype=dtype).eval()
    cfg = model.config
    t = cfg.text
    n_vis = cfg.num_projected_visual_tokens
    bytes_per_token = 2 * t.num_layers * t.num_kv_heads * t.head_dim * torch.tensor([], dtype=dtype).element_size()

    # warm-up so the first batch size doesn't pay for cuda init / allocator growth
    run_continuous_batching(model, make_workload(model, 2, 4, seed=99), max_batch_size=2, num_blocks=64, block_size=block_size)

    results = []
    for bs in batch_sizes:
        reqs = make_workload(model, num_requests, max_new_tokens)
        longest = max(n_vis + len(r.prompt_ids) + r.max_new_tokens for r in reqs)
        blocks_per_seq = -(-longest // block_size)
        stats = run_continuous_batching(
            model, reqs, max_batch_size=bs, num_blocks=bs * blocks_per_seq + 8, block_size=block_size
        )
        results.append(
            BatchingResult(
                max_batch_size=bs,
                num_requests=stats.num_requests,
                generated_tokens=stats.generated_tokens,
                wall_time_s=stats.wall_time_s,
                throughput_tokens_per_sec=stats.tokens_per_sec,
                mean_decode_step_ms=stats.mean_decode_step_ms,
                paged_peak_kv_mb=stats.peak_kv_mb,
                contiguous_kv_mb_workload_max=bs * longest * bytes_per_token / 2**20,
                contiguous_kv_mb_max_seq_len=bs * t.max_seq_len * bytes_per_token / 2**20,
            )
        )
    return results
