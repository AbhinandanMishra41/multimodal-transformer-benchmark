"""Export the text decoder as a HuggingFace LlamaForCausalLM checkpoint for vLLM.

The decoder uses the same building blocks and the same weight names as Llama
(q/k/v/o_proj, gate/up/down_proj, RMSNorm, rotate_half RoPE), so the mapping is
mostly a rename.

Only the *text* decoder is exported. vLLM has no idea about this ViT +
projector, so through vLLM the model is text-only (prompt token ids in,
tokens out). Wiring the vision tower in would mean writing a vLLM model plugin.
"""

import importlib.util
import json
from pathlib import Path

import torch

from vlm.config import TextConfig
from vlm.language import LanguageDecoder
from vlm.model import MultimodalTransformer

# our name -> HF name, per layer
_LAYER_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "attn.q_proj.weight": "self_attn.q_proj.weight",
    "attn.k_proj.weight": "self_attn.k_proj.weight",
    "attn.v_proj.weight": "self_attn.v_proj.weight",
    "attn.o_proj.weight": "self_attn.o_proj.weight",
    "mlp.gate_proj.weight": "mlp.gate_proj.weight",
    "mlp.up_proj.weight": "mlp.up_proj.weight",
    "mlp.down_proj.weight": "mlp.down_proj.weight",
}


def is_vllm_available() -> bool:
    return importlib.util.find_spec("vllm") is not None


def export_for_vllm(model: MultimodalTransformer, output_dir: str | Path, dtype=torch.float16) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    t = model.config.text

    hf_config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": t.vocab_size,
        "hidden_size": t.hidden_dim,
        "intermediate_size": t.intermediate_dim,
        "num_hidden_layers": t.num_layers,
        "num_attention_heads": t.num_q_heads,
        "num_key_value_heads": t.num_kv_heads,
        "max_position_embeddings": t.max_seq_len,
        "rms_norm_eps": t.rms_norm_eps,
        "rope_theta": t.rope_theta,
        "hidden_act": "silu",
        "tie_word_embeddings": t.tie_word_embeddings,
        "bos_token_id": model.config.bos_token_id,
        "eos_token_id": model.config.eos_token_id,
        "pad_token_id": model.config.pad_token_id,
        "torch_dtype": str(dtype).replace("torch.", ""),
    }
    (out / "config.json").write_text(json.dumps(hf_config, indent=2), encoding="utf-8")

    sd = {k: v.detach().cpu().to(dtype) for k, v in model.decoder.state_dict().items()}
    hf = {
        "model.embed_tokens.weight": sd["embed_tokens.weight"],
        "model.norm.weight": sd["norm.weight"],
        "lm_head.weight": sd["lm_head.weight"],
    }
    for i in range(t.num_layers):
        for ours, theirs in _LAYER_MAP.items():
            hf[f"model.layers.{i}.{theirs}"] = sd[f"layers.{i}.{ours}"]
    # tied weights share storage; clone so torch.save writes two plain tensors
    hf = {k: v.clone() for k, v in hf.items()}
    torch.save(hf, out / "pytorch_model.bin")
    return out


def load_decoder_from_hf_checkpoint(ckpt_dir: str | Path) -> LanguageDecoder:
    """Inverse of export_for_vllm. Used to check that vLLM and our decoder agree."""
    ckpt_dir = Path(ckpt_dir)
    c = json.loads((ckpt_dir / "config.json").read_text(encoding="utf-8"))
    cfg = TextConfig(
        vocab_size=c["vocab_size"],
        hidden_dim=c["hidden_size"],
        intermediate_dim=c["intermediate_size"],
        num_layers=c["num_hidden_layers"],
        num_q_heads=c["num_attention_heads"],
        num_kv_heads=c["num_key_value_heads"],
        max_seq_len=c["max_position_embeddings"],
        rope_theta=c["rope_theta"],
        rms_norm_eps=c["rms_norm_eps"],
        tie_word_embeddings=c["tie_word_embeddings"],
    )
    hf = torch.load(ckpt_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)
    sd = {
        "embed_tokens.weight": hf["model.embed_tokens.weight"],
        "norm.weight": hf["model.norm.weight"],
        "lm_head.weight": hf["lm_head.weight"],
    }
    for i in range(cfg.num_layers):
        for ours, theirs in _LAYER_MAP.items():
            sd[f"layers.{i}.{ours}"] = hf[f"model.layers.{i}.{theirs}"]

    dec = LanguageDecoder(cfg)
    dec.load_state_dict({k: v.float() for k, v in sd.items()})
    return dec.eval()


@torch.inference_mode()
def greedy_text_only(decoder: LanguageDecoder, prompt_ids: list[int], max_new_tokens: int) -> list[int]:
    """Reference greedy decode with no KV cache, to compare against vLLM output."""
    ids = torch.tensor([prompt_ids])
    for _ in range(max_new_tokens):
        logits, _ = decoder(ids)
        ids = torch.cat([ids, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
    return ids[0, len(prompt_ids) :].tolist()
