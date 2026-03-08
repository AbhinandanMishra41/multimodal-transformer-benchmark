"""TensorRT for the vision encoder + projector.

Two routes:
- build_vision_engine: TensorRT builder API -> serialized .engine (FP16)
- profile_vision_encoder_trt: ONNX Runtime's TensorrtExecutionProvider vs its
  CUDA provider on the same ONNX file

Only the vision graph goes through TensorRT so far. The KV decoder graph sees a
different past_seq_len on every step and needs proper optimization profiles
before it's worth benchmarking under TRT - that's on the TODO list.

No INT8 here: the dynamically-quantized ONNX files use MatMulInteger, which
TensorRT doesn't parse. TRT INT8 needs a calibrator or a QDQ-quantized model.
"""

from dataclasses import dataclass
from pathlib import Path
import importlib.util
import shutil
import subprocess
import time

import numpy as np
import onnxruntime as ort

from vlm.config import VLMConfig


@dataclass
class TensorRTStatus:
    has_tensorrt_pkg: bool
    has_trtexec_cli: bool
    has_ort_trt_ep: bool  # compiled into this onnxruntime build; doesn't mean the TRT libs load
    tensorrt_version: str | None


def detect_tensorrt_support() -> TensorRTStatus:
    version = None
    has_pkg = importlib.util.find_spec("tensorrt") is not None
    if has_pkg:
        try:
            import tensorrt as trt  # type: ignore

            version = trt.__version__
        except Exception:  # installed but broken (missing libs etc.)
            has_pkg = False
    return TensorRTStatus(
        has_tensorrt_pkg=has_pkg,
        has_trtexec_cli=shutil.which("trtexec") is not None,
        has_ort_trt_ep="TensorrtExecutionProvider" in ort.get_available_providers(),
        tensorrt_version=version,
    )


def build_vision_engine(
    onnx_path: str | Path,
    engine_path: str | Path,
    config: VLMConfig,
    fp16: bool = True,
    max_batch_size: int = 8,
) -> Path:
    """Compile vision_projector_*.onnx into a TensorRT engine with a dynamic batch dim."""
    onnx_path, engine_path = Path(onnx_path), Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    chw = (config.vision.in_channels, config.vision.image_size, config.vision.image_size)
    shapes = {"min": (1, *chw), "opt": (max(1, max_batch_size // 2), *chw), "max": (max_batch_size, *chw)}
    status = detect_tensorrt_support()

    if status.has_tensorrt_pkg:
        import tensorrt as trt  # type: ignore

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(0)  # TRT 10: always explicit batch
        parser = trt.OnnxParser(network, logger)
        if not parser.parse(onnx_path.read_bytes()):
            errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"TensorRT could not parse {onnx_path}:\n{errors}")

        cfg = builder.create_builder_config()
        cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        if fp16:
            cfg.set_flag(trt.BuilderFlag.FP16)
        profile = builder.create_optimization_profile()
        profile.set_shape("images", shapes["min"], shapes["opt"], shapes["max"])
        cfg.add_optimization_profile(profile)

        engine = builder.build_serialized_network(network, cfg)
        if engine is None:
            raise RuntimeError("TensorRT engine build failed (see the log above)")
        engine_path.write_bytes(engine)
        return engine_path

    if status.has_trtexec_cli:
        fmt = lambda s: "images:" + "x".join(map(str, s))  # noqa: E731
        cmd = [
            "trtexec", f"--onnx={onnx_path}", f"--saveEngine={engine_path}",
            f"--minShapes={fmt(shapes['min'])}", f"--optShapes={fmt(shapes['opt'])}", f"--maxShapes={fmt(shapes['max'])}",
        ]
        if fp16:
            cmd.append("--fp16")
        subprocess.run(cmd, check=True)
        return engine_path

    raise RuntimeError("No TensorRT found (neither the `tensorrt` package nor `trtexec`).")


def get_tensorrt_ort_providers(cache_dir: str | Path, fp16: bool = True) -> list:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    trt_opts = {
        "trt_fp16_enable": fp16,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": str(cache_dir),
        "trt_max_workspace_size": 1 << 30,
    }
    return [("TensorrtExecutionProvider", trt_opts), "CUDAExecutionProvider", "CPUExecutionProvider"]


def profile_vision_encoder_trt(
    vision_onnx_path: str | Path,
    image_size: int,
    cache_dir: str | Path,
    batch_size: int = 1,
    warmup: int = 5,
    runs: int = 50,
) -> dict:
    """Median latency of the vision encoder + projector: TRT EP (fp16) vs CUDA EP (fp32).

    Raises if ORT silently falls back from the TensorRT EP (it does that when
    libnvinfer can't be loaded), so a CUDA number never gets labelled as TensorRT.
    """
    x = np.random.default_rng(0).standard_normal((batch_size, 3, image_size, image_size), dtype=np.float32)
    results = {}
    for label, providers, expected in [
        ("cuda_ep_fp32_ms", ["CUDAExecutionProvider", "CPUExecutionProvider"], "CUDAExecutionProvider"),
        ("tensorrt_ep_fp16_ms", get_tensorrt_ort_providers(cache_dir, fp16=True), "TensorrtExecutionProvider"),
    ]:
        sess = ort.InferenceSession(str(vision_onnx_path), providers=providers)
        if sess.get_providers()[0] != expected:
            raise RuntimeError(f"wanted {expected}, but onnxruntime fell back to {sess.get_providers()[0]}")
        for _ in range(warmup):  # the first TRT call builds the engine, which takes a while
            sess.run(None, {"images": x})
        times = []
        for _ in range(runs):
            t = time.perf_counter()
            sess.run(None, {"images": x})
            times.append((time.perf_counter() - t) * 1000)
        results[label] = float(np.median(times))
    return results
