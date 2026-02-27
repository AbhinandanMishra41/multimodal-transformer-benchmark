"""ONNX export, dynamic INT8 quantization and an ONNX Runtime generation loop.

The model is split into two graphs:
  vision_projector.onnx  images [B, 3, H, W] -> visual_embeds [B, N_vis, D]
  decoder_kv.onnx        inputs_embeds [B, L, D], start_pos, past_key/value_i
                         -> logits, present_key/value_i

L and past_seq_len are dynamic, so the same decoder graph handles prefill
(L = N_vis + prompt, empty past) and decode (L = 1).

Token embedding lookup happens in numpy on the host because the decoder graph
takes embeddings, not ids - the image tokens have to be concatenated in front
of the prompt first. The fp32 runner uses token_embeddings.npy; the int8 runner
uses a per-row int8 copy (token_embeddings_int8.npz). Because embeddings are
tied, the same matrix also lives inside the decoder graph as lm_head, so both
variants carry it twice. Removing that would mean doing the lookup in-graph.
"""

from dataclasses import dataclass
from pathlib import Path
import copy
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
import torch
import torch.nn as nn
import torch.nn.functional as F

from vlm.config import VLMConfig
from vlm.model import MultimodalTransformer
from vlm.train import SyntheticMultimodalDataset


def _mb(path: Path) -> float:
    return path.stat().st_size / 2**20


@dataclass
class ExportedONNXPaths:
    vision_fp32: Path
    decoder_fp32: Path
    vision_int8: Path
    decoder_int8: Path
    token_embed_npy: Path
    token_embed_int8: Path

    # everything each variant needs on disk to run, embedding table included
    def total_fp32_mb(self) -> float:
        return _mb(self.vision_fp32) + _mb(self.decoder_fp32) + _mb(self.token_embed_npy)

    def total_int8_mb(self) -> float:
        return _mb(self.vision_int8) + _mb(self.decoder_int8) + _mb(self.token_embed_int8)


class _VisionProjector(nn.Module):
    def __init__(self, model: MultimodalTransformer) -> None:
        super().__init__()
        self.vision_encoder = model.vision_encoder
        self.projector = model.projector

    def forward(self, images):
        return self.projector(self.vision_encoder(images))


class _DecoderWithPast(nn.Module):
    """Flattens the per-layer (k, v) tuples into positional args so ONNX can name them."""

    def __init__(self, model: MultimodalTransformer) -> None:
        super().__init__()
        self.decoder = model.decoder
        self.num_layers = model.config.text.num_layers

    def forward(self, inputs_embeds, start_pos, *past_flat):
        past = [(past_flat[2 * i], past_flat[2 * i + 1]) for i in range(self.num_layers)]
        logits, present = self.decoder.forward_embeddings(inputs_embeds, start_pos=start_pos, past_key_values=past)
        return (logits, *[t for kv in present for t in kv])


def export_vlm_to_onnx(
    model: MultimodalTransformer,
    output_dir: str | Path,
    opset_version: int = 17,
    quantize_int8: bool = True,
) -> ExportedONNXPaths:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = ExportedONNXPaths(
        vision_fp32=out / "vision_projector_fp32.onnx",
        decoder_fp32=out / "decoder_kv_fp32.onnx",
        vision_int8=out / "vision_projector_int8.onnx",
        decoder_int8=out / "decoder_kv_int8.onnx",
        token_embed_npy=out / "token_embeddings.npy",
        token_embed_int8=out / "token_embeddings_int8.npz",
    )

    # export from an fp32 cpu copy so we don't move the caller's model around
    m = copy.deepcopy(model).to("cpu", torch.float32).eval()
    cfg = m.config
    table = m.decoder.embed_tokens.weight.detach().numpy()
    np.save(paths.token_embed_npy, table)
    if quantize_int8:
        # symmetric per-row int8, dequantized row by row at lookup time
        scale = np.abs(table).max(axis=1, keepdims=True) / 127.0
        scale[scale == 0] = 1.0
        q = np.clip(np.round(table / scale), -127, 127).astype(np.int8)
        np.savez(paths.token_embed_int8, q=q, scale=scale.astype(np.float32))

    images = torch.randn(1, cfg.vision.in_channels, cfg.vision.image_size, cfg.vision.image_size)
    torch.onnx.export(
        _VisionProjector(m).eval(),
        (images,),
        str(paths.vision_fp32),
        opset_version=opset_version,
        input_names=["images"],
        output_names=["visual_embeds"],
        dynamic_axes={"images": {0: "batch"}, "visual_embeds": {0: "batch"}},
        dynamo=False,  # the torch.export path doesn't handle the *past_flat varargs yet
    )

    t = cfg.text
    inputs = ["inputs_embeds", "start_pos"]
    outputs = ["logits"]
    axes = {"inputs_embeds": {0: "batch", 1: "seq"}, "logits": {0: "batch", 1: "seq"}}
    past = []
    for i in range(t.num_layers):
        for kind in ("key", "value"):
            inputs.append(f"past_{kind}_{i}")
            outputs.append(f"present_{kind}_{i}")
            axes[f"past_{kind}_{i}"] = {0: "batch", 2: "past_seq"}
            axes[f"present_{kind}_{i}"] = {0: "batch", 2: "total_seq"}
            past.append(torch.randn(1, t.num_kv_heads, 2, t.head_dim))

    torch.onnx.export(
        _DecoderWithPast(m).eval(),
        (torch.randn(1, 4, t.hidden_dim), torch.tensor(2), *past),
        str(paths.decoder_fp32),
        opset_version=opset_version,
        input_names=inputs,
        output_names=outputs,
        dynamic_axes=axes,
        dynamo=False,
    )

    onnx.checker.check_model(str(paths.vision_fp32))
    onnx.checker.check_model(str(paths.decoder_fp32))

    if quantize_int8:
        # dynamic quantization: int8 weights, activations quantized on the fly per batch
        for src, dst in [(paths.vision_fp32, paths.vision_int8), (paths.decoder_fp32, paths.decoder_int8)]:
            quantize_dynamic(str(src), str(dst), weight_type=QuantType.QUInt8, op_types_to_quantize=["MatMul", "Gemm"])

    return paths


class ONNXVLMRunner:
    def __init__(
        self,
        config: VLMConfig,
        vision_onnx_path: str | Path,
        decoder_onnx_path: str | Path,
        token_embed_npy_path: str | Path,
        providers: list | None = None,
        num_threads: int | None = None,
    ) -> None:
        self.config = config
        if str(token_embed_npy_path).endswith(".npz"):
            data = np.load(token_embed_npy_path)
            self._embed_q, self._embed_scale = data["q"], data["scale"]
            self.embed_table = None
        else:
            self.embed_table = np.load(token_embed_npy_path).astype(np.float32)
        # default to the same thread count torch uses, so CPU comparisons are fair
        num_threads = num_threads or torch.get_num_threads()

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = num_threads
        if providers is None:
            available = ort.get_available_providers()
            providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]

        self.vision_session = ort.InferenceSession(str(vision_onnx_path), opts, providers=providers)
        self.decoder_session = ort.InferenceSession(str(decoder_onnx_path), opts, providers=providers)
        self.active_provider = self.decoder_session.get_providers()[0]

    def embed_tokens(self, ids: np.ndarray) -> np.ndarray:
        if self.embed_table is not None:
            return self.embed_table[ids]
        return self._embed_q[ids].astype(np.float32) * self._embed_scale[ids]

    def empty_past_kv(self, batch_size: int) -> dict[str, np.ndarray]:
        t = self.config.text
        empty = np.zeros((batch_size, t.num_kv_heads, 0, t.head_dim), dtype=np.float32)
        return {f"past_{kind}_{i}": empty for i in range(t.num_layers) for kind in ("key", "value")}

    def _run_decoder(self, embeds: np.ndarray, start_pos: int, past: dict[str, np.ndarray]):
        feeds = dict(past, inputs_embeds=embeds, start_pos=np.array(start_pos, dtype=np.int64))
        logits, *present = self.decoder_session.run(None, feeds)
        n = self.config.text.num_layers
        new_past = {}
        for i in range(n):
            new_past[f"past_key_{i}"] = present[2 * i]
            new_past[f"past_value_{i}"] = present[2 * i + 1]
        return logits, new_past

    def prefill(self, images: np.ndarray, prompt_ids: np.ndarray):
        """Returns (logits for every position, past kv, sequence length so far)."""
        vis = self.vision_session.run(["visual_embeds"], {"images": images})[0]
        x = np.concatenate([vis, self.embed_tokens(prompt_ids)], axis=1)
        logits, past = self._run_decoder(x, 0, self.empty_past_kv(images.shape[0]))
        return logits, past, x.shape[1]

    def decode_one_token(self, token_ids: np.ndarray, start_pos: int, past_kv: dict[str, np.ndarray]):
        logits, past = self._run_decoder(self.embed_tokens(token_ids), start_pos, past_kv)
        return logits[:, -1], past

    def generate(self, images: np.ndarray, prompt_ids: np.ndarray, max_new_tokens: int = 32):
        """Greedy. Returns (ids, ttft_ms, per-step latencies in ms)."""
        if max_new_tokens <= 0:
            return prompt_ids, 0.0, []
        t0 = time.perf_counter()
        logits, past, pos = self.prefill(images, prompt_ids)
        tok = logits[:, -1].argmax(-1)[:, None]
        ttft = (time.perf_counter() - t0) * 1000

        out, itl = [prompt_ids, tok], []
        for _ in range(max_new_tokens - 1):
            ts = time.perf_counter()
            step_logits, past = self.decode_one_token(tok, pos, past)
            tok = step_logits.argmax(-1)[:, None]
            itl.append((time.perf_counter() - ts) * 1000)
            out.append(tok)
            pos += 1
        return np.concatenate(out, axis=1), ttft, itl


@dataclass
class QuantizationAccuracyReport:
    logit_cosine_similarity: float
    logit_max_abs_error: float
    logit_mean_abs_error: float
    top1_token_agreement_pct: float
    pytorch_perplexity: float
    onnx_perplexity: float
    perplexity_delta: float
    fp32_size_mb: float
    int8_size_mb: float
    compression_ratio: float


def evaluate_quantization_quality(
    pytorch_model: MultimodalTransformer,
    onnx_runner: ONNXVLMRunner,
    paths: ExportedONNXPaths,
    num_batches: int = 4,
    batch_size: int = 8,
) -> QuantizationAccuracyReport:
    """Compare PyTorch fp32 against an ONNX runner on held-out synthetic samples.

    Only the text positions are compared - logits at image positions aren't
    trained to predict anything, so agreement there wouldn't mean much.
    """
    m = copy.deepcopy(pytorch_model).to("cpu", torch.float32).eval()
    cfg = m.config
    n_vis = cfg.num_projected_visual_tokens
    ds = SyntheticMultimodalDataset(num_batches * batch_size, image_size=cfg.vision.image_size, seed=777)

    cos, max_err, mean_err, agree, pt_nll, ort_nll = [], [], [], [], [], []
    with torch.inference_mode():
        for b in range(num_batches):
            sl = slice(b * batch_size, (b + 1) * batch_size)
            images, ids, labels = ds.images[sl], ds.input_ids[sl], ds.labels[sl]
            valid = labels != cfg.pad_token_id

            pt = m(images, ids)[0][:, n_vis:][valid]
            ort_logits = onnx_runner.prefill(images.numpy(), ids.numpy())[0]
            ox = torch.from_numpy(ort_logits)[:, n_vis:][valid]
            y = labels[valid]

            cos.append(F.cosine_similarity(pt, ox, dim=-1).mean().item())
            diff = (pt - ox).abs()
            max_err.append(diff.max().item())
            mean_err.append(diff.mean().item())
            agree.append((pt.argmax(-1) == ox.argmax(-1)).float().mean().item() * 100)
            pt_nll.append(F.cross_entropy(pt, y).item())
            ort_nll.append(F.cross_entropy(ox, y).item())

    pt_ppl = float(np.exp(np.mean(pt_nll)))
    ort_ppl = float(np.exp(np.mean(ort_nll)))
    fp32_mb, int8_mb = paths.total_fp32_mb(), paths.total_int8_mb()
    return QuantizationAccuracyReport(
        logit_cosine_similarity=float(np.mean(cos)),
        logit_max_abs_error=float(np.max(max_err)),
        logit_mean_abs_error=float(np.mean(mean_err)),
        top1_token_agreement_pct=float(np.mean(agree)),
        pytorch_perplexity=pt_ppl,
        onnx_perplexity=ort_ppl,
        perplexity_delta=ort_ppl - pt_ppl,
        fp32_size_mb=fp32_mb,
        int8_size_mb=int8_mb,
        compression_ratio=fp32_mb / int8_mb,
    )
