"""Run the exported text decoder in vLLM and check it matches our implementation.

    python -m benchmark.vllm_run --checkpoint artifacts/compact/vllm_hf_checkpoint

Linux + NVIDIA GPU only (that's a vLLM requirement). Text-only: see the note
at the top of quantization/vllm_adapter.py.
"""

import argparse
import json
from pathlib import Path
import time

from quantization.vllm_adapter import greedy_text_only, load_decoder_from_hf_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="artifacts/compact/vllm_hf_checkpoint")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--out", default="results/vllm_report.json")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams  # imported here so --help works without vllm

    max_len = json.loads((Path(args.checkpoint) / "config.json").read_text())["max_position_embeddings"]
    # no tokenizer: we feed token ids directly
    llm = LLM(model=args.checkpoint, skip_tokenizer_init=True, dtype="float16", gpu_memory_utilization=0.6, max_model_len=max_len)
    params = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, ignore_eos=True, detokenize=False)

    prompts = [[1] + [4 + (7 * i + 13 * j) % 500 for j in range(8 + i % 8)] for i in range(args.batch_size)]
    llm.generate([{"prompt_token_ids": p} for p in prompts[:2]], params)  # warm-up

    t0 = time.perf_counter()
    outs = llm.generate([{"prompt_token_ids": p} for p in prompts], params)
    elapsed = time.perf_counter() - t0
    n_tokens = sum(len(o.outputs[0].token_ids) for o in outs)

    # Compare the first few prompts against our own (fp32, no cache) greedy decode.
    # fp16 vs fp32 can flip a near-tie argmax, so report agreement rather than assert.
    ref = load_decoder_from_hf_checkpoint(args.checkpoint)
    agree = []
    for p, o in list(zip(prompts, outs))[:4]:
        ours = greedy_text_only(ref, p, args.max_new_tokens)
        theirs = list(o.outputs[0].token_ids)
        agree.append(sum(a == b for a, b in zip(ours, theirs)) / len(ours))

    report = {
        "batch_size": args.batch_size,
        "generated_tokens": n_tokens,
        "elapsed_s": elapsed,
        "throughput_tokens_per_sec": n_tokens / elapsed,
        "token_agreement_with_reference": agree,
    }
    print(json.dumps(report, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
