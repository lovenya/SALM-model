"""Silent-correctness checks for the oracle transcript -> content-code bridge."""

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from salm.a2s.text_content_lm import TextContentLM, roll_text_pair  # noqa: E402


def model() -> TextContentLM:
    torch.manual_seed(0)
    return TextContentLM(
        num_codes=16, text_dim=24, content_frames=12, max_text_tokens=8,
        d_model=32, n_layers=2, n_heads=4, dropout=0.0,
    ).eval()


def batch():
    hidden = torch.randn(3, 8, 24)
    mask = torch.tensor([
        [1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
    ], dtype=torch.bool)
    codes = torch.randint(0, 16, (3, 12))
    inp = torch.cat([torch.full((3, 1), 16), codes[:, :-1]], dim=1)
    return hidden, mask, codes, inp


def test_forward_shape_and_finite():
    m = model()
    h, mask, _, inp = batch()
    out = m(h, mask, inp)
    assert out.shape == (3, 12, 16)
    assert torch.isfinite(out).all()


def test_decoder_is_causal():
    m = model()
    h, mask, _, inp = batch()
    with torch.no_grad():
        a = m(h, mask, inp)
        changed = inp.clone()
        changed[:, -1] = (changed[:, -1] + 1) % 16
        b = m(h, mask, changed)
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-6)


def test_padding_text_tokens_do_not_affect_output():
    m = model()
    h, mask, _, inp = batch()
    changed = h.clone()
    changed[~mask] = torch.randn_like(changed[~mask]) * 1000
    with torch.no_grad():
        a = m(h, mask, inp)
        b = m(changed, mask, inp)
    assert torch.allclose(a, b, atol=1e-5)


def test_roll_text_pair_keeps_memory_and_mask_together():
    memory = torch.arange(3 * 4 * 2).view(3, 4, 2)
    pad = torch.tensor([[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=torch.bool)
    rm, rp = roll_text_pair(memory, pad)
    assert torch.equal(rm[0], memory[-1])
    assert torch.equal(rp[0], pad[-1])
    assert all(not torch.equal(rm[i], memory[i]) for i in range(3))


def test_sample_shape_range_and_seeded_reproducibility():
    m = model()
    h, mask, _, _ = batch()
    a = m.sample(h, mask, top_k=5, generator=torch.Generator().manual_seed(7))
    b = m.sample(h, mask, top_k=5, generator=torch.Generator().manual_seed(7))
    assert a.shape == (3, 12)
    assert int(a.min()) >= 0 and int(a.max()) < 16
    assert torch.equal(a, b)


def test_greedy_sampling_is_deterministic():
    m = model()
    h, mask, _, _ = batch()
    a = m.sample(h, mask, greedy=True)
    b = m.sample(h, mask, greedy=True)
    assert torch.equal(a, b)


def test_rejects_wrong_text_width_and_too_many_tokens():
    m = model()
    try:
        m.encode_text(torch.randn(2, 8, 25))
        raise AssertionError("wrong width should fail")
    except RuntimeError as exc:
        assert "text width" in str(exc)
    try:
        m.encode_text(torch.randn(2, 9, 24))
        raise AssertionError("too many tokens should fail")
    except RuntimeError as exc:
        assert "max_text_tokens" in str(exc)
