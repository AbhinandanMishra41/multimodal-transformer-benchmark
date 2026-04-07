# Benchmark results

`compact` profile, 42.0M params. GPU: NVIDIA GeForce GTX 1650, Windows 11, torch 2.6.0+cu124.
Toy training: loss 8.260 -> 0.402, caption exact-match 5%.

## Batch size 1

| backend | TTFT ms | ITL p50 ms | ITL p95 ms | tok/s | total ms (mean ± std) | weights MB | CUDA peak MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| PyTorch CUDA fp16 (no cache) | 80.9 | 59.61 | 62.15 | 16.5 | 1940 ± 10 | 80.2 | 450 |
| PyTorch CUDA fp16 (KV cache) | 81.0 | 16.70 | 25.03 | 50.1 | 639 ± 29 | 80.2 | 439 |
| PyTorch CPU fp32 (KV cache) | 108.8 | 11.98 | 13.06 | 65.6 | 488 ± 17 | 160.3 | - |
| ONNX Runtime CPU fp32 | 96.8 | 5.69 | 6.42 | 115.4 | 277 ± 17 | 193.2 | - |
| ONNX Runtime CPU int8 | 62.6 | 3.41 | 4.33 | 182.4 | 175 ± 14 | 50.4 | - |

10 timed runs per row after warm-up. ONNX Runtime and PyTorch CPU use the same number of threads.

## INT8 (ONNX Runtime dynamic quantization)

- size on disk: 193.2 MB -> 50.4 MB (3.83x), graphs plus the host-side embedding table each variant loads
- logits vs PyTorch fp32 (text positions): cosine 1.00000, mean abs err 0.0302, top-1 agreement 100.0%
- perplexity on fresh toy samples: 1.488 (fp32) vs 1.488 (int8)

## Continuous batching (paged KV cache)

Contiguous columns = KV memory a preallocated cache would hold for the same number of slots, sized either for the longest request in the workload or for max_seq_len.

| max batch | tok/s | decode step ms | paged peak KV MB | contiguous, longest req (MB) | contiguous, max_seq_len (MB) |
|---:|---:|---:|---:|---:|---:|
| 1 | 34.8 | 25.6 | 1.00 | 0.98 | 4.00 |
| 4 | 72.7 | 38.7 | 3.75 | 3.94 | 16.00 |
| 8 | 87.5 | 54.7 | 7.44 | 7.88 | 32.00 |
| 16 | 99.2 | 75.7 | 14.81 | 15.75 | 64.00 |
