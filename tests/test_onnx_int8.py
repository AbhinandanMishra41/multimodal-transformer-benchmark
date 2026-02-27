import json

import torch

from quantization.onnx_export import ONNXVLMRunner, evaluate_quantization_quality, export_vlm_to_onnx
from quantization.vllm_adapter import export_for_vllm, load_decoder_from_hf_checkpoint
from vlm.config import VLMConfig
from vlm.model import MultimodalTransformer


def test_onnx_export_and_int8(tmp_path):
    torch.manual_seed(7)
    cfg = VLMConfig.tiny()
    model = MultimodalTransformer(cfg).eval()
    paths = export_vlm_to_onnx(model, tmp_path / "onnx")

    for p in (paths.vision_fp32, paths.decoder_fp32, paths.vision_int8, paths.decoder_int8, paths.token_embed_npy, paths.token_embed_int8):
        assert p.exists()
    # weights + embedding table both go to int8; scales and fp32 LayerNorm/bias keep it under 4x
    assert paths.total_int8_mb() < paths.total_fp32_mb() / 2.5

    cpu = ["CPUExecutionProvider"]
    fp32 = ONNXVLMRunner(cfg, paths.vision_fp32, paths.decoder_fp32, paths.token_embed_npy, providers=cpu)
    images = torch.randn(1, 3, cfg.vision.image_size, cfg.vision.image_size)
    prompt = torch.randint(4, cfg.text.vocab_size, (1, 5))

    pt_tokens = model.generate(images, prompt, max_new_tokens=6)
    ort_tokens, ttft, itl = fp32.generate(images.numpy(), prompt.numpy(), max_new_tokens=6)
    assert ttft > 0 and len(itl) == 5
    assert torch.equal(pt_tokens, torch.from_numpy(ort_tokens))

    int8 = ONNXVLMRunner(cfg, paths.vision_int8, paths.decoder_int8, paths.token_embed_int8, providers=cpu)
    report = evaluate_quantization_quality(model, int8, paths, num_batches=2, batch_size=2)
    assert report.logit_cosine_similarity > 0.95
    assert report.top1_token_agreement_pct > 90


def test_hf_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(1)
    cfg = VLMConfig.tiny()
    model = MultimodalTransformer(cfg).eval()
    ckpt = export_for_vllm(model, tmp_path / "hf", dtype=torch.float32)

    hf_cfg = json.loads((ckpt / "config.json").read_text())
    assert hf_cfg["num_key_value_heads"] == cfg.text.num_kv_heads

    reloaded = load_decoder_from_hf_checkpoint(ckpt)
    ids = torch.randint(4, cfg.text.vocab_size, (2, 7))
    with torch.inference_mode():
        torch.testing.assert_close(reloaded(ids)[0], model.decoder(ids)[0])
