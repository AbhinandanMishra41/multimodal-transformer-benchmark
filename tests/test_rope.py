import torch

from vlm.rope import RotaryEmbedding, apply_rotary_pos_emb, precompute_rope_cos_sin


def test_rope_preserves_norm():
    # RoPE is a rotation of each (i, i + d/2) pair, so vector lengths can't change
    torch.manual_seed(0)
    q = torch.randn(2, 8, 16, 64)
    k = torch.randn(2, 2, 16, 64)
    cos, sin = precompute_rope_cos_sin(head_dim=64, max_seq_len=64)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin, start_pos=5)
    torch.testing.assert_close(q.norm(dim=-1), q_rot.norm(dim=-1))
    torch.testing.assert_close(k.norm(dim=-1), k_rot.norm(dim=-1))


def test_rope_dot_product_depends_only_on_distance():
    torch.manual_seed(1)
    q = torch.randn(1, 1, 1, 32)
    k = torch.randn(1, 1, 1, 32)
    rope = RotaryEmbedding(head_dim=32, max_seq_len=128)

    def score(m, n):
        q_m, _ = rope(q, k, start_pos=m)
        _, k_n = rope(q, k, start_pos=n)
        return (q_m * k_n).sum()

    torch.testing.assert_close(score(3, 10), score(25, 32), atol=1e-5, rtol=0)


def test_rope_int_tensor_and_position_ids_agree():
    torch.manual_seed(2)
    q = torch.randn(2, 4, 6, 32)
    k = torch.randn(2, 2, 6, 32)
    rope = RotaryEmbedding(head_dim=32, max_seq_len=64)

    q_int, k_int = rope(q, k, start_pos=7)
    q_t, k_t = rope(q, k, start_pos=torch.tensor(7))
    pos = torch.arange(7, 13).expand(2, 6)
    q_p, k_p = rope(q, k, position_ids=pos)

    for a, b in [(q_int, q_t), (k_int, k_t), (q_int, q_p), (k_int, k_p)]:
        torch.testing.assert_close(a, b)
