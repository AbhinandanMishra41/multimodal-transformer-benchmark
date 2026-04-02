"""Plots for the README. Works off the JSON report, so you can re-plot without re-running:

    python -m benchmark.visualize results/benchmark_report.json
"""

import json
from pathlib import Path
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})


def _bar_labels(ax, bars, fmt):
    for b in bars:
        ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")


def plot_backends(report: dict, out: Path) -> Path:
    rows = report["backend_benchmarks"]
    names = ["\n".join(textwrap.wrap(r["backend_name"], 14)) for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    _bar_labels(a1, a1.bar(names, [r["throughput_tokens_per_sec"] for r in rows], color="C0"), "{:.0f}")
    a1.set_ylabel("tokens / s")
    a1.set_title("throughput, batch 1")
    _bar_labels(a2, a2.bar(names, [r["itl_p50_ms"] for r in rows], color="C1"), "{:.1f}")
    a2.set_ylabel("ms / token")
    a2.set_title("median inter-token latency")
    for a in (a1, a2):
        a.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    path = out / "throughput_latency_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_kv_scaling(report: dict, out: Path) -> Path:
    s = report["kv_cache_scaling"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(s["token_lengths"], s["no_cache_latency_ms"], "o-", label="no cache (recompute everything)")
    a1.plot(s["token_lengths"], s["kv_cache_latency_ms"], "s-", label="KV cache")
    a1.set_xlabel("generated tokens")
    a1.set_ylabel("total time (ms)")
    a1.legend()
    a2.plot(s["token_lengths"], s["speedup_ratio"], "o-", color="C2")
    for x, y in zip(s["token_lengths"], s["speedup_ratio"]):
        a2.annotate(f"{y:.1f}x", (x, y), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8)
    a2.set_xlabel("generated tokens")
    a2.set_ylabel("speedup")
    a2.set_ylim(0, max(s["speedup_ratio"]) * 1.2)
    fig.tight_layout()
    path = out / "kv_cache_scaling_curve.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_memory(report: dict, out: Path) -> Path:
    q = report["quantization_accuracy"]
    b = report["continuous_batching"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    _bar_labels(a1, a1.bar(["ONNX fp32", "ONNX int8"], [q["fp32_size_mb"], q["int8_size_mb"]], color=["C0", "C2"]), "{:.0f} MB")
    a1.set_ylabel("MB on disk (incl. embedding table)")
    a1.set_title("model size")

    x = range(len(b))
    w = 0.27
    a2.bar([i - w for i in x], [r["contiguous_kv_mb_max_seq_len"] for r in b], w, label="contiguous, max_seq_len slots")
    a2.bar(list(x), [r["contiguous_kv_mb_workload_max"] for r in b], w, label="contiguous, longest request")
    a2.bar([i + w for i in x], [r["paged_peak_kv_mb"] for r in b], w, label="paged (peak blocks held)")
    a2.set_xticks(list(x))
    a2.set_xticklabels([f"batch {r['max_batch_size']}" for r in b])
    a2.set_ylabel("KV cache MB")
    a2.set_title("KV cache memory, same number of slots")
    a2.legend(fontsize=8)
    fig.tight_layout()
    path = out / "memory_vram_profile.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_batching(report: dict, out: Path) -> Path:
    b = report["continuous_batching"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar([str(r["max_batch_size"]) for r in b], [r["throughput_tokens_per_sec"] for r in b], color="C0")
    _bar_labels(ax, bars, "{:.0f}")
    ax.set_xlabel("max concurrent requests")
    ax.set_ylabel("tokens / s")
    ax.set_title(f"continuous batching, {b[0]['num_requests']} requests")
    fig.tight_layout()
    path = out / "batched_throughput_scaling.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_all_charts(report: dict, output_dir: str | Path) -> list[Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return [f(report, out) for f in (plot_backends, plot_kv_scaling, plot_memory, plot_batching)]


if __name__ == "__main__":
    report_path = Path(sys.argv[1] if len(sys.argv) > 1 else "results/benchmark_report.json")
    for p in generate_all_charts(json.loads(report_path.read_text()), sys.argv[2] if len(sys.argv) > 2 else "assets"):
        print(p)
