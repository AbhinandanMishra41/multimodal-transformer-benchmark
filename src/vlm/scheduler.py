"""A small continuous-batching loop on top of PagedKVCache.

Each step:
  1. admit waiting requests while there's a free batch slot (prefill them one by one)
  2. run one decode step for every running request together
  3. drop requests that hit their token budget and return their blocks to the pool

This is the core idea of vLLM's scheduler without preemption, swapping or
chunked prefill. Greedy decoding only.
"""

from collections import deque
from dataclasses import dataclass, field
import time

import torch

from .model import MultimodalTransformer


@dataclass
class Request:
    request_id: int
    image: torch.Tensor  # [C, H, W]
    prompt_ids: torch.Tensor  # [L]
    max_new_tokens: int
    output: list[int] = field(default_factory=list)

    @property
    def done(self) -> bool:
        return len(self.output) >= self.max_new_tokens


@dataclass
class SchedulerStats:
    num_requests: int
    generated_tokens: int
    wall_time_s: float
    decode_steps: int
    mean_decode_step_ms: float
    peak_blocks_in_use: int
    peak_kv_mb: float

    @property
    def tokens_per_sec(self) -> float:
        return self.generated_tokens / max(self.wall_time_s, 1e-9)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def run_continuous_batching(
    model: MultimodalTransformer,
    requests: list[Request],
    max_batch_size: int,
    num_blocks: int,
    block_size: int = 16,
) -> SchedulerStats:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    cache = model.create_paged_cache(num_blocks=num_blocks, block_size=block_size)
    n_vis = model.config.num_projected_visual_tokens
    for r in requests:
        if n_vis + len(r.prompt_ids) + r.max_new_tokens > model.config.text.max_seq_len:
            raise ValueError(f"request {r.request_id} would exceed max_seq_len ({model.config.text.max_seq_len})")
        if r.max_new_tokens < 1:
            raise ValueError(f"request {r.request_id} has max_new_tokens < 1")

    waiting = deque(requests)
    running: list[Request] = []
    step_times = []
    _sync(device)
    t0 = time.perf_counter()

    while waiting or running:
        while waiting and len(running) < max_batch_size:
            req = waiting.popleft()
            logits = model.prefill_paged(
                req.image[None].to(device=device, dtype=dtype),
                req.prompt_ids[None].to(device),
                cache,
                seq_id=req.request_id,
            )
            req.output.append(int(logits.argmax(-1)))
            if req.done:
                cache.free_sequence(req.request_id)
            else:
                running.append(req)

        if not running:
            continue

        _sync(device)
        ts = time.perf_counter()
        last = torch.tensor([[r.output[-1]] for r in running], device=device)
        logits = model.decode_paged(last, cache, [r.request_id for r in running])
        next_ids = logits.argmax(-1).tolist()
        _sync(device)
        step_times.append(time.perf_counter() - ts)

        still_running = []
        for req, tok in zip(running, next_ids):
            req.output.append(tok)
            if req.done:
                cache.free_sequence(req.request_id)
            else:
                still_running.append(req)
        running = still_running

    _sync(device)
    wall = time.perf_counter() - t0
    return SchedulerStats(
        num_requests=len(requests),
        generated_tokens=sum(len(r.output) for r in requests),
        wall_time_s=wall,
        decode_steps=len(step_times),
        mean_decode_step_ms=1000 * sum(step_times) / max(len(step_times), 1),
        peak_blocks_in_use=cache.peak_blocks_in_use,
        peak_kv_mb=cache.peak_blocks_in_use * cache.bytes_per_block / 2**20,
    )
