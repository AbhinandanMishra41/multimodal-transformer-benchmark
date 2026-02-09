import pytest
import torch

from vlm.config import VLMConfig
from vlm.kv_cache import PagedKVCache
from vlm.model import MultimodalTransformer
from vlm.scheduler import Request, run_continuous_batching


@pytest.fixture
def tiny_model():
    torch.manual_seed(42)
    return MultimodalTransformer(VLMConfig.tiny()).eval()


def test_kv_cache_matches_full_forward(tiny_model):
    cfg = tiny_model.config
    images = torch.randn(2, 3, cfg.vision.image_size, cfg.vision.image_size)
    ids = torch.randint(4, cfg.text.vocab_size, (2, 10))
    n_vis = cfg.num_projected_visual_tokens

    with torch.inference_mode():
        full, _ = tiny_model(images, ids)
        cache = tiny_model.create_kv_cache(batch_size=2, max_seq_len=64, dtype=torch.float32)
        last, _ = tiny_model.prefill(images, ids[:, :6], kv_cache=cache)
        torch.testing.assert_close(last, full[:, n_vis + 5], atol=1e-5, rtol=0)

        pos = n_vis + 6
        for j in range(6, 10):
            step, _ = tiny_model.decode_one_token(ids[:, j : j + 1], start_pos=pos, kv_cache=cache)
            torch.testing.assert_close(step, full[:, pos], atol=1e-5, rtol=0)
            pos += 1


def test_paged_cache_roundtrip_and_free():
    torch.manual_seed(7)
    paged = PagedKVCache(num_layers=2, num_kv_heads=2, head_dim=16, num_blocks=32, block_size=4, dtype=torch.float32)
    paged.allocate_sequence(101)
    k = torch.randn(2, 9, 16)  # 9 tokens -> 3 blocks of 4
    v = torch.randn(2, 9, 16)

    # write in two chunks that don't line up with block boundaries
    for sl in (slice(0, 5), slice(5, 9)):
        for layer in range(2):
            paged.append_tokens(101, layer, k[:, sl], v[:, sl])

    assert len(paged.block_tables[101]) == 3
    assert paged.seq_lengths[101] == 9
    k_out, v_out = paged.gather(101, layer_idx=1)
    torch.testing.assert_close(k_out, k)
    torch.testing.assert_close(v_out, v)

    paged.free_sequence(101)
    assert len(paged.free_blocks) == 32
    # freed blocks get reused
    paged.allocate_sequence(5)
    paged.reserve(5, 4)
    assert paged.block_tables[5] == [0]


def test_paged_cache_runs_out_of_blocks():
    paged = PagedKVCache(num_layers=1, num_kv_heads=1, head_dim=8, num_blocks=2, block_size=4, dtype=torch.float32)
    paged.allocate_sequence(0)
    paged.reserve(0, 8)
    paged.advance(0, 8)
    with pytest.raises(RuntimeError, match="out of blocks"):
        paged.reserve(0, 1)


def test_paged_decode_matches_contiguous(tiny_model):
    """Two sequences of different lengths decoded together through the paged cache
    should give the same logits as decoding each one alone with KVCache."""
    cfg = tiny_model.config
    torch.manual_seed(3)
    imgs = torch.randn(2, 3, cfg.vision.image_size, cfg.vision.image_size)
    prompts = [torch.randint(4, cfg.text.vocab_size, (1, n)) for n in (3, 8)]
    steps = torch.randint(4, cfg.text.vocab_size, (2, 4))
    n_vis = cfg.num_projected_visual_tokens

    expected = []
    for i in range(2):
        cache = tiny_model.create_kv_cache(1, max_seq_len=64, dtype=torch.float32)
        tiny_model.prefill(imgs[i : i + 1], prompts[i], kv_cache=cache)
        pos = n_vis + prompts[i].shape[1]
        out = []
        for j in range(4):
            logits, _ = tiny_model.decode_one_token(steps[i : i + 1, j : j + 1], pos + j, kv_cache=cache)
            out.append(logits)
        expected.append(torch.cat(out))

    paged = tiny_model.create_paged_cache(num_blocks=32, block_size=4)
    for i in range(2):
        tiny_model.prefill_paged(imgs[i : i + 1], prompts[i], paged, seq_id=i)
    for j in range(4):
        logits = tiny_model.decode_paged(steps[:, j : j + 1], paged, seq_ids=[0, 1])
        for i in range(2):
            torch.testing.assert_close(logits[i], expected[i][j], atol=1e-5, rtol=0)


def test_continuous_batching_matches_generate(tiny_model):
    cfg = tiny_model.config
    torch.manual_seed(11)
    reqs, ref = [], []
    for i, (plen, n_new) in enumerate([(3, 5), (7, 2), (1, 6), (5, 4), (2, 1)]):
        img = torch.randn(3, cfg.vision.image_size, cfg.vision.image_size)
        prompt = torch.randint(4, cfg.text.vocab_size, (plen,))
        reqs.append(Request(i, img, prompt, n_new))
        ref.append(tiny_model.generate(img[None], prompt[None], max_new_tokens=n_new)[0, plen:].tolist())

    stats = run_continuous_batching(tiny_model, reqs, max_batch_size=2, num_blocks=64, block_size=4)
    assert [r.output for r in reqs] == ref
    assert stats.generated_tokens == sum(n for _, n in [(3, 5), (7, 2), (1, 6), (5, 4), (2, 1)])
