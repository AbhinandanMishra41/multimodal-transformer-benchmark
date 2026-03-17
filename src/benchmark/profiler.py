"""Latency / throughput / memory measurements for the different backends.

Peak memory is only reported for CUDA backends (torch.cuda.max_memory_allocated).
For CPU / ONNX Runtime there's no clean per-backend number - process RSS includes
torch, onnxruntime and everything loaded before - so it's left as None.

Every profile function works on a copy of the model: they cast it to different
devices/dtypes, and a round trip through fp16 would quietly change the weights
used by later steps.
"""

from dataclasses import asdict, dataclass
import copy
import gc
import time

import numpy as np
import torch

from quantization.onnx_export import ONNXVLMRunner
from vlm.model import MultimodalTransformer


@dataclass
class BackendBenchmarkMetrics:
    backend_name: str
    device: str
    precision: str
    use_kv_cache: bool
    batch_size: int
    prompt_tokens: int
    visual_tokens: int
    generated_tokens: int
    ttft_ms: float
    itl_p50_ms: float
    itl_p95_ms: float
    total_latency_ms: float
    total_latency_std_ms: float
    timed_runs: int
    throughput_tokens_per_sec: float
    model_weights_mb: float
    kv_cache_allocated_mb: float
    peak_cuda_memory_mb: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _pct(xs: list[float], q: float) -> float:
    return float(np.percentile(xs, q)) if xs else 0.0


@torch.inference_mode()
def profile_pytorch_backend(
    model: MultimodalTransformer,
    backend_name: str,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    use_kv_cache: bool = True,
    batch_size: int = 1,
    prompt_len: int = 16,
    max_new_tokens: int = 32,
    warmup_runs: int = 3,
    timed_runs: int = 10,
) -> BackendBenchmarkMetrics:
    dev = torch.device(device)
    gc.collect()
    model = copy.deepcopy(model).to(device=dev, dtype=dtype).eval()
    cfg = model.config
    n_vis = cfg.num_projected_visual_tokens

    images = torch.randn(batch_size, 3, cfg.vision.image_size, cfg.vision.image_size, device=dev, dtype=dtype)
    prompt = torch.randint(4, cfg.text.vocab_size, (batch_size, prompt_len), device=dev)

    for _ in range(warmup_runs):
        model.generate(images, prompt, max_new_tokens=4, use_kv_cache=use_kv_cache)
    _sync(dev)
    if dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)

    ttft, itl, totals = [], [], []
    kv_mb = 0.0
    for _ in range(timed_runs):
        _sync(dev)
        t0 = time.perf_counter()
        if use_kv_cache:
            # cache allocation is part of TTFT on purpose: a real server pays for it too
            cache = model.create_kv_cache(batch_size, max_seq_len=n_vis + prompt_len + max_new_tokens, device=dev, dtype=dtype)
            kv_mb = cache.stats().allocated_mb
            logits, _ = model.prefill(images, prompt, kv_cache=cache)
            tok = logits.argmax(-1, keepdim=True)
            _sync(dev)
            ttft.append((time.perf_counter() - t0) * 1000)

            pos = n_vis + prompt_len
            for _ in range(max_new_tokens - 1):
                ts = time.perf_counter()
                logits, _ = model.decode_one_token(tok, start_pos=pos, kv_cache=cache)
                tok = logits.argmax(-1, keepdim=True)
                _sync(dev)
                itl.append((time.perf_counter() - ts) * 1000)
                pos += 1
        else:
            # baseline: re-run the decoder over the whole sequence for every new token
            visual = model.encode_image(images)
            ids = prompt
            for step in range(max_new_tokens):
                ts = time.perf_counter()
                x = torch.cat([visual, model.decoder.embed_tokens(ids)], dim=1)
                logits, _ = model.decoder.forward_embeddings(x)
                ids = torch.cat([ids, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
                _sync(dev)
                now = time.perf_counter()
                if step == 0:
                    ttft.append((now - t0) * 1000)
                else:
                    itl.append((now - ts) * 1000)
        totals.append((time.perf_counter() - t0) * 1000)

    peak = torch.cuda.max_memory_allocated(dev) / 2**20 if dev.type == "cuda" else None

    mean_total = float(np.mean(totals))
    return BackendBenchmarkMetrics(
        backend_name=backend_name,
        device=str(dev),
        precision="FP16" if dtype == torch.float16 else "FP32",
        use_kv_cache=use_kv_cache,
        batch_size=batch_size,
        prompt_tokens=prompt_len,
        visual_tokens=n_vis,
        generated_tokens=max_new_tokens,
        ttft_ms=float(np.median(ttft)),
        itl_p50_ms=_pct(itl, 50),
        itl_p95_ms=_pct(itl, 95),
        total_latency_ms=mean_total,
        total_latency_std_ms=float(np.std(totals)),
        timed_runs=timed_runs,
        throughput_tokens_per_sec=batch_size * max_new_tokens / (mean_total / 1000),
        model_weights_mb=sum(p.numel() * p.element_size() for p in model.parameters()) / 2**20,
        kv_cache_allocated_mb=kv_mb,
        peak_cuda_memory_mb=peak,
    )


def profile_onnx_backend(
    runner: ONNXVLMRunner,
    backend_name: str,
    precision: str,
    model_size_mb: float,
    batch_size: int = 1,
    prompt_len: int = 16,
    max_new_tokens: int = 32,
    warmup_runs: int = 3,
    timed_runs: int = 10,
) -> BackendBenchmarkMetrics:
    cfg = runner.config
    n_vis = cfg.num_projected_visual_tokens
    rng = np.random.default_rng(42)
    images = rng.standard_normal((batch_size, 3, cfg.vision.image_size, cfg.vision.image_size), dtype=np.float32)
    prompt = rng.integers(4, cfg.text.vocab_size, size=(batch_size, prompt_len), dtype=np.int64)

    for _ in range(warmup_runs):
        runner.generate(images, prompt, max_new_tokens=4)

    ttft, itl, totals = [], [], []
    for _ in range(timed_runs):
        t0 = time.perf_counter()
        _, first, steps = runner.generate(images, prompt, max_new_tokens=max_new_tokens)
        totals.append((time.perf_counter() - t0) * 1000)
        ttft.append(first)
        itl.extend(steps)

    # ORT grows the cache by concatenation, so this is the size at the end of generation
    t = cfg.text
    kv_mb = 2 * t.num_layers * batch_size * t.num_kv_heads * (n_vis + prompt_len + max_new_tokens) * t.head_dim * 4 / 2**20
    mean_total = float(np.mean(totals))
    return BackendBenchmarkMetrics(
        backend_name=backend_name,
        device=runner.active_provider,
        precision=precision,
        use_kv_cache=True,
        batch_size=batch_size,
        prompt_tokens=prompt_len,
        visual_tokens=n_vis,
        generated_tokens=max_new_tokens,
        ttft_ms=float(np.median(ttft)),
        itl_p50_ms=_pct(itl, 50),
        itl_p95_ms=_pct(itl, 95),
        total_latency_ms=mean_total,
        total_latency_std_ms=float(np.std(totals)),
        timed_runs=timed_runs,
        throughput_tokens_per_sec=batch_size * max_new_tokens / (mean_total / 1000),
        model_weights_mb=model_size_mb,
        kv_cache_allocated_mb=kv_mb,
        peak_cuda_memory_mb=None,
    )


@torch.inference_mode()
def profile_kv_cache_scaling(
    model: MultimodalTransformer,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float16,
    token_lengths: list[int] | None = None,
) -> dict[str, list[float]]:
    """Total generate() time with and without the KV cache for a few output lengths."""
    token_lengths = token_lengths or [8, 16, 32, 64, 96]
    dev = torch.device(device)
    model = copy.deepcopy(model).to(device=dev, dtype=dtype).eval()
    cfg = model.config
    images = torch.randn(1, 3, cfg.vision.image_size, cfg.vision.image_size, device=dev, dtype=dtype)
    prompt = torch.randint(4, cfg.text.vocab_size, (1, 16), device=dev)
    model.generate(images, prompt, max_new_tokens=4)

    def timed(**kw) -> float:
        _sync(dev)
        t = time.perf_counter()
        model.generate(images, prompt, **kw)
        _sync(dev)
        return (time.perf_counter() - t) * 1000

    cached, uncached = [], []
    for n in token_lengths:
        cached.append(timed(max_new_tokens=n, use_kv_cache=True))
        uncached.append(timed(max_new_tokens=n, use_kv_cache=False))
    return {
        "token_lengths": [float(n) for n in token_lengths],
        "kv_cache_latency_ms": cached,
        "no_cache_latency_ms": uncached,
        "speedup_ratio": [u / c for u, c in zip(uncached, cached)],
    }
