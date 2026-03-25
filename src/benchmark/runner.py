"""Run everything and write results/ + assets/.

    python -m benchmark.runner --profile compact --max-new-tokens 32 --train-steps 300

Steps: build model -> short training on the toy dataset -> ONNX fp32/int8 +
HF checkpoint export -> per-backend latency -> int8 fidelity -> KV cache
scaling -> continuous batching -> charts + report.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform

import torch

from quantization.onnx_export import ONNXVLMRunner, evaluate_quantization_quality, export_vlm_to_onnx
from quantization.tensorrt_engine import detect_tensorrt_support, profile_vision_encoder_trt
from quantization.vllm_adapter import export_for_vllm, is_vllm_available
from vlm.config import VLMConfig
from vlm.model import MultimodalTransformer
from vlm.train import caption_accuracy, train_synthetic_vlm

from .batching import benchmark_continuous_batching
from .profiler import profile_kv_cache_scaling, profile_onnx_backend, profile_pytorch_backend
from .visualize import generate_all_charts


def run_full_benchmark(
    profile: str = "compact",
    max_new_tokens: int = 32,
    train_steps: int = 300,
    train_batch_size: int = 8,
    grad_accum_steps: int = 4,
    root_dir: str | Path = ".",
) -> dict:
    root = Path(root_dir).resolve()
    artifacts = root / "artifacts" / profile
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    gpu = torch.cuda.get_device_name(0) if cuda else "none"
    print(f"{platform.system()} {platform.release()} | torch {torch.__version__} | GPU: {gpu}")

    torch.manual_seed(42)
    cfg = VLMConfig.from_name(profile)
    model = MultimodalTransformer(cfg)
    params = model.parameter_breakdown()
    print(f"model '{profile}': {params['total'] / 1e6:.1f}M params "
          f"(vision {params['vision_encoder'] / 1e6:.1f}M, projector {params['multimodal_projector'] / 1e6:.2f}M, "
          f"decoder {params['language_decoder'] / 1e6:.1f}M)")

    print(f"training {train_steps} steps (batch {train_batch_size} x {grad_accum_steps} accumulation) on the toy shapes dataset...")
    train = train_synthetic_vlm(
        model, steps=train_steps, batch_size=train_batch_size, grad_accum_steps=grad_accum_steps,
        lr=1e-3, device=device, use_amp=cuda, log_every=50,
    )
    if cuda:
        torch.cuda.empty_cache()  # give the optimizer state's memory back before profiling
    acc = caption_accuracy(model)
    print(f"  loss {train.initial_loss:.3f} -> {train.final_loss:.3f}, caption exact-match {acc:.0%}")

    print("exporting ONNX (fp32 + int8) and HF checkpoint...")
    onnx_paths = export_vlm_to_onnx(model, artifacts / "onnx")
    export_for_vllm(model, artifacts / "vllm_hf_checkpoint")
    print(f"  onnx fp32 {onnx_paths.total_fp32_mb():.1f} MB, int8 {onnx_paths.total_int8_mb():.1f} MB")

    cpu = ["CPUExecutionProvider"]
    ort_fp32 = ONNXVLMRunner(cfg, onnx_paths.vision_fp32, onnx_paths.decoder_fp32, onnx_paths.token_embed_npy, providers=cpu)
    ort_int8 = ONNXVLMRunner(cfg, onnx_paths.vision_int8, onnx_paths.decoder_int8, onnx_paths.token_embed_int8, providers=cpu)
    # fidelity first, against the exact fp32 weights that were exported
    quant = evaluate_quantization_quality(model, ort_int8, onnx_paths)
    print(f"  int8 vs fp32: top-1 agreement {quant.top1_token_agreement_pct:.1f}%, "
          f"perplexity {quant.pytorch_perplexity:.3f} -> {quant.onnx_perplexity:.3f}")

    print("profiling backends...")
    backends = []
    if cuda:
        for kv in (False, True):
            backends.append(profile_pytorch_backend(
                model, f"PyTorch CUDA fp16 ({'KV cache' if kv else 'no cache'})",
                device="cuda", dtype=torch.float16, use_kv_cache=kv, max_new_tokens=max_new_tokens,
            ))
    backends.append(profile_pytorch_backend(
        model, "PyTorch CPU fp32 (KV cache)", device="cpu", dtype=torch.float32, max_new_tokens=max_new_tokens,
    ))
    # ORT runs on CPU with the same thread count as torch, so these compare with the row above
    backends.append(profile_onnx_backend(ort_fp32, "ONNX Runtime CPU fp32", "FP32", onnx_paths.total_fp32_mb(), max_new_tokens=max_new_tokens))
    backends.append(profile_onnx_backend(ort_int8, "ONNX Runtime CPU int8", "INT8", onnx_paths.total_int8_mb(), max_new_tokens=max_new_tokens))
    for b in backends:
        print(f"  {b.backend_name:<34} {b.throughput_tokens_per_sec:7.1f} tok/s  itl p50 {b.itl_p50_ms:6.2f} ms")

    trt = detect_tensorrt_support()
    trt_vision = None
    if trt.has_ort_trt_ep:
        print("TensorRT EP available, timing the vision encoder...")
        try:
            trt_vision = profile_vision_encoder_trt(onnx_paths.vision_fp32, cfg.vision.image_size, artifacts / "trt_cache")
        except Exception as e:  # usually libnvinfer not found; keep going, but record why
            print(f"  skipped: {e}")
            trt_vision = {"error": str(e)}

    print("KV cache scaling, continuous batching...")
    dtype = torch.float16 if cuda else torch.float32
    kv_scaling = profile_kv_cache_scaling(model, device=device, dtype=dtype)
    batching = benchmark_continuous_batching(model, max_new_tokens=max_new_tokens, device=device, dtype=dtype)

    report = {
        "environment": {
            "os": f"{platform.system()} {platform.release()}",
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "cuda_available": cuda,
            "gpu_name": gpu,
            "tensorrt_status": asdict(trt),
            "vllm_installed": is_vllm_available(),
        },
        "model_profile": profile,
        "parameters": params,
        "training": {**asdict(train), "caption_exact_match": acc},
        "backend_benchmarks": [b.to_dict() for b in backends],
        "tensorrt_vision_encoder": trt_vision,
        "quantization_accuracy": asdict(quant),
        "kv_cache_scaling": kv_scaling,
        "continuous_batching": [
            {**asdict(r), "savings_vs_workload_max_pct": r.savings_vs_workload_max_pct,
             "savings_vs_max_seq_len_pct": r.savings_vs_max_seq_len_pct}
            for r in batching
        ],
    }
    (results_dir / "benchmark_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (results_dir / "BENCHMARK_RESULTS.md").write_text(format_markdown(report), encoding="utf-8")
    generate_all_charts(report, root / "assets")
    print(f"wrote {results_dir / 'benchmark_report.json'}, {results_dir / 'BENCHMARK_RESULTS.md'} and assets/*.png")
    return report


def format_markdown(r: dict) -> str:
    env, q = r["environment"], r["quantization_accuracy"]
    out = [
        "# Benchmark results",
        "",
        f"`{r['model_profile']}` profile, {r['parameters']['total'] / 1e6:.1f}M params. "
        f"GPU: {env['gpu_name']}, {env['os']}, torch {env['pytorch_version']}.",
        f"Toy training: loss {r['training']['initial_loss']:.3f} -> {r['training']['final_loss']:.3f}, "
        f"caption exact-match {r['training']['caption_exact_match']:.0%}.",
        "",
        "## Batch size 1",
        "",
        "| backend | TTFT ms | ITL p50 ms | ITL p95 ms | tok/s | total ms (mean ± std) | weights MB | CUDA peak MB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for b in r["backend_benchmarks"]:
        peak = f"{b['peak_cuda_memory_mb']:.0f}" if b["peak_cuda_memory_mb"] is not None else "-"
        out.append(
            f"| {b['backend_name']} | {b['ttft_ms']:.1f} | {b['itl_p50_ms']:.2f} | {b['itl_p95_ms']:.2f} | "
            f"{b['throughput_tokens_per_sec']:.1f} | {b['total_latency_ms']:.0f} ± {b['total_latency_std_ms']:.0f} | "
            f"{b['model_weights_mb']:.1f} | {peak} |"
        )
    n_runs = r["backend_benchmarks"][0]["timed_runs"]
    out += [
        "",
        f"{n_runs} timed runs per row after warm-up. ONNX Runtime and PyTorch CPU use the same number of threads.",
        "",
        "## INT8 (ONNX Runtime dynamic quantization)",
        "",
        f"- size on disk: {q['fp32_size_mb']:.1f} MB -> {q['int8_size_mb']:.1f} MB ({q['compression_ratio']:.2f}x), "
        "graphs plus the host-side embedding table each variant loads",
        f"- logits vs PyTorch fp32 (text positions): cosine {q['logit_cosine_similarity']:.5f}, "
        f"mean abs err {q['logit_mean_abs_error']:.4f}, top-1 agreement {q['top1_token_agreement_pct']:.1f}%",
        f"- perplexity on fresh toy samples: {q['pytorch_perplexity']:.3f} (fp32) vs {q['onnx_perplexity']:.3f} (int8)",
        "",
        "## Continuous batching (paged KV cache)",
        "",
        "Contiguous columns = KV memory a preallocated cache would hold for the same number of slots, "
        "sized either for the longest request in the workload or for max_seq_len.",
        "",
        "| max batch | tok/s | decode step ms | paged peak KV MB | contiguous, longest req (MB) | contiguous, max_seq_len (MB) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for b in r["continuous_batching"]:
        out.append(
            f"| {b['max_batch_size']} | {b['throughput_tokens_per_sec']:.1f} | {b['mean_decode_step_ms']:.1f} | "
            f"{b['paged_peak_kv_mb']:.2f} | {b['contiguous_kv_mb_workload_max']:.2f} | {b['contiguous_kv_mb_max_seq_len']:.2f} |"
        )
    t = r.get("tensorrt_vision_encoder")
    if t and "error" not in t:
        out += ["", "## TensorRT (vision encoder only, batch 1)", "",
                f"ONNX Runtime CUDA EP fp32: {t['cuda_ep_fp32_ms']:.2f} ms, TensorRT EP fp16: {t['tensorrt_ep_fp16_ms']:.2f} ms"]
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="VLM inference benchmark")
    ap.add_argument("--profile", choices=["tiny", "compact", "base"], default="compact")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--train-steps", type=int, default=300)
    ap.add_argument("--train-batch-size", type=int, default=8, help="micro-batch; 8 fits a 4GB GPU")
    ap.add_argument("--grad-accum", type=int, default=4, help="micro-batches per optimizer step")
    args = ap.parse_args()
    run_full_benchmark(args.profile, args.max_new_tokens, args.train_steps, args.train_batch_size, args.grad_accum)


if __name__ == "__main__":
    main()
