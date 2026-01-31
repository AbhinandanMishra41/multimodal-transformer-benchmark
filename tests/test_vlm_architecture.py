import pytest
import torch
import torch.nn.functional as F

from vlm.config import VLMConfig
from vlm.model import MultimodalTransformer
from vlm.train import SyntheticMultimodalDataset, decode_caption, train_synthetic_vlm


def test_greedy_kv_cache_matches_no_cache():
    torch.manual_seed(99)
    cfg = VLMConfig.tiny()
    model = MultimodalTransformer(cfg).eval()
    images = torch.randn(2, 3, cfg.vision.image_size, cfg.vision.image_size)
    prompt = torch.randint(4, cfg.text.vocab_size, (2, 6))

    cached = model.generate(images, prompt, max_new_tokens=8, use_kv_cache=True)
    uncached = model.generate(images, prompt, max_new_tokens=8, use_kv_cache=False)
    assert cached.shape == (2, 14)
    assert torch.equal(cached, uncached)


def test_loss_scores_each_position_against_the_next_input_token():
    # Regression test for an off-by-one: the logit at the position of input_ids[j]
    # must be scored against input_ids[j + 1]. The expected loss is built from
    # input_ids and explicit positions, not from `labels`, so a wrong slice in
    # forward() (or wrongly shifted labels) changes the result.
    torch.manual_seed(0)
    cfg = VLMConfig.tiny()
    model = MultimodalTransformer(cfg).eval()
    ds = SyntheticMultimodalDataset(4, image_size=cfg.vision.image_size)
    images, ids = ds.images, ds.input_ids

    with torch.no_grad():
        logits, loss = model(images, ids, labels=ds.labels)
    n_vis = cfg.num_projected_visual_tokens
    terms = []
    for b in range(ids.shape[0]):
        n = int((ids[b] != 0).sum())  # bos + caption words, followed by padding
        for j in range(n - 1):
            terms.append(F.cross_entropy(logits[b, n_vis + j], ids[b, j + 1]))
        terms.append(F.cross_entropy(logits[b, n_vis + n - 1], torch.tensor(2)))  # last word -> eos
    torch.testing.assert_close(loss, torch.stack(terms).mean())


def test_dataset_captions_roundtrip():
    ds = SyntheticMultimodalDataset(8, image_size=64)
    for ids, cap in zip(ds.input_ids, ds.captions):
        assert decode_caption(ids) == cap
    # labels are input_ids shifted by one
    assert torch.equal(ds.labels[:, :-1][ds.input_ids[:, 1:] != 0], ds.input_ids[:, 1:][ds.input_ids[:, 1:] != 0])


def test_generate_rejects_too_long_sequences():
    cfg = VLMConfig.tiny()
    model = MultimodalTransformer(cfg).eval()
    images = torch.randn(1, 3, cfg.vision.image_size, cfg.vision.image_size)
    with pytest.raises(ValueError, match="max_seq_len"):
        model.generate(images, torch.ones(1, 4, dtype=torch.long), max_new_tokens=cfg.text.max_seq_len)


def test_training_reduces_loss():
    torch.manual_seed(42)
    model = MultimodalTransformer(VLMConfig.tiny())
    metrics = train_synthetic_vlm(model, steps=12, batch_size=8, lr=2e-3, device="cpu")
    assert metrics.final_loss < metrics.initial_loss


def test_generate_with_zero_new_tokens_returns_prompt():
    cfg = VLMConfig.tiny()
    model = MultimodalTransformer(cfg).eval()
    images = torch.randn(2, 3, cfg.vision.image_size, cfg.vision.image_size)
    prompt = torch.randint(4, cfg.text.vocab_size, (2, 5))
    for use_cache in (True, False):
        assert torch.equal(model.generate(images, prompt, max_new_tokens=0, use_kv_cache=use_cache), prompt)


def test_token_pooling_shapes():
    from dataclasses import replace

    cfg = replace(VLMConfig.tiny(), token_pooling_factor=4)
    model = MultimodalTransformer(cfg).eval()
    images = torch.randn(1, 3, cfg.vision.image_size, cfg.vision.image_size)
    # 17 tokens (16 patches + CLS) -> 4 groups of 4, last token dropped
    assert model.encode_image(images).shape == (1, 4, cfg.text.hidden_dim)
    assert cfg.num_projected_visual_tokens == 4
