"""Correctness checks for ContentLMDedup and its teacher-forcing construction.

Every failure this file guards against is SILENT: a misaligned target, a leaked future token,
or an unmasked pad still produces a decreasing loss curve. The model would simply learn to
predict the wrong thing, and no metric downstream would say so.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from salm.a2s.content_dedup import dedup_batch, expand_sequence  # noqa: E402
from salm.a2s.content_lm_dedup import (  # noqa: E402
    ContentLMDedup,
    build_teacher_forcing,
)

K, R = 16, 8


def _model(**kw):
    torch.manual_seed(0)
    return ContentLMDedup(
        num_codes=K, max_run=R, ambient_dim=32, max_tokens=12,
        d_model=32, n_layers=2, n_heads=4, **kw,
    ).eval()


# ----------------------------------------------------------------- teacher forcing


def test_teacher_forcing_alignment_hand_computed():
    # row 0: tokens [3, 7] len 2 ; row 1: tokens [5] len 1, padded
    tokens = torch.tensor([[3, 7, -1], [5, -1, -1]])
    durs = torch.tensor([[2, 4, 0], [6, 0, 0]])
    lens = torch.tensor([2, 1])
    tf = build_teacher_forcing(tokens, durs, lens, bos_id=K, eos_id_in_head=K)

    # inputs: BOS then the tokens, shifted right
    assert tf["codes_in"][0].tolist()[:3] == [K, 3, 7]
    assert tf["durs_in"][0].tolist()[:3] == [0, 2, 4]
    assert tf["codes_in"][1].tolist()[:2] == [K, 5]
    assert tf["durs_in"][1].tolist()[:2] == [0, 6]

    # targets are width l+1 = 4 (sized for the LONGEST row, then per-row masked):
    # token t at position t, EOS right after, -100 thereafter
    assert tf["code_tgt"][0].tolist() == [3, 7, K, -100]     # EOS at index len=2
    assert tf["code_tgt"][1].tolist() == [5, K, -100, -100]  # EOS at index len=1
    # durations shifted to 0-based classes, and NO duration target at the EOS step
    assert tf["dur_tgt"][0].tolist() == [1, 3, -100, -100]
    assert tf["dur_tgt"][1].tolist() == [5, -100, -100, -100]


def test_exactly_one_eos_per_row():
    tokens = torch.tensor([[1, 2, 3], [4, -1, -1]])
    durs = torch.tensor([[1, 1, 1], [3, 0, 0]])
    lens = torch.tensor([3, 1])
    tf = build_teacher_forcing(tokens, durs, lens, bos_id=K, eos_id_in_head=K)
    for row in tf["code_tgt"]:
        assert (row == K).sum() == 1


def test_pad_mask_keeps_bos_and_all_real_inputs():
    tokens = torch.tensor([[1, 2, -1, -1]])
    durs = torch.tensor([[1, 1, 0, 0]])
    lens = torch.tensor([2])
    tf = build_teacher_forcing(tokens, durs, lens, bos_id=K, eos_id_in_head=K)
    # width 5: inputs 0(BOS),1,2 real -> mask False; 3,4 padding -> True
    assert tf["pad_mask"][0].tolist() == [False, False, False, True, True]


def test_no_row_is_fully_masked():
    # A fully-masked row would produce NaNs in attention. BOS must always survive.
    tokens = torch.full((3, 4), -1)
    durs = torch.zeros((3, 4), dtype=torch.long)
    lens = torch.tensor([0, 0, 0])
    tf = build_teacher_forcing(tokens, durs, lens, bos_id=K, eos_id_in_head=K)
    assert (~tf["pad_mask"]).any(dim=1).all()


def test_durations_never_map_to_a_negative_class():
    tokens = torch.tensor([[1, -1]])
    durs = torch.tensor([[1, 0]])          # padded duration 0 must not become class -1
    lens = torch.tensor([1])
    tf = build_teacher_forcing(tokens, durs, lens, bos_id=K, eos_id_in_head=K)
    real = tf["dur_tgt"][tf["dur_tgt"] != -100]
    assert (real >= 0).all() and (real < R).all()


def test_teacher_forcing_agrees_with_the_real_codec():
    rng = np.random.default_rng(3)
    codes = rng.integers(0, K, size=(4, 24))
    tok, dur, lens = dedup_batch(codes, max_tokens=24, max_run=R)
    tf = build_teacher_forcing(
        torch.from_numpy(tok), torch.from_numpy(dur), torch.from_numpy(lens),
        bos_id=K, eos_id_in_head=K,
    )
    for i, L in enumerate(lens):
        # the code targets before EOS must be exactly the deduped tokens
        assert tf["code_tgt"][i, :L].tolist() == tok[i, :L].tolist()
        # and re-expanding those with their durations reproduces the original frames
        d = tf["dur_tgt"][i, :L].numpy() + 1
        assert np.array_equal(expand_sequence(tok[i, :L], d), codes[i])


# ----------------------------------------------------------------- model


def test_forward_shapes():
    m = _model()
    amb = torch.randn(3, 5, 32)
    ci = torch.randint(0, K, (3, 7))
    di = torch.randint(0, R + 1, (3, 7))
    cl, dl = m(amb, ci, di)
    assert cl.shape == (3, 7, K + 1)      # +1 for EOS
    assert dl.shape == (3, 7, R)


def test_causality_no_future_leak():
    m = _model()
    amb = torch.randn(2, 5, 32)
    ci = torch.randint(0, K, (2, 6))
    di = torch.randint(1, R + 1, (2, 6))
    with torch.no_grad():
        a_c, a_d = m(amb, ci, di)
        ci2 = ci.clone(); ci2[:, -1] = (ci2[:, -1] + 1) % K
        di2 = di.clone(); di2[:, -1] = (di2[:, -1] % R) + 1
        b_c, b_d = m(amb, ci2, di2)
    # changing the LAST input must not move any earlier position, for EITHER head
    assert torch.allclose(a_c[:, :-1], b_c[:, :-1], atol=1e-6)
    assert torch.allclose(a_d[:, :-1], b_d[:, :-1], atol=1e-6)


def test_duration_input_actually_affects_predictions():
    # If duration embeddings were ignored, the model could not track its frame budget.
    m = _model()
    amb = torch.randn(2, 5, 32)
    ci = torch.randint(0, K, (2, 4))
    with torch.no_grad():
        a, _ = m(amb, ci, torch.ones_like(ci))
        b, _ = m(amb, ci, torch.full_like(ci, R))
    assert not torch.allclose(a, b, atol=1e-5)


def test_sampling_respects_the_frame_budget_and_is_expandable():
    m = _model()
    amb = torch.randn(4, 5, 32)
    g = torch.Generator().manual_seed(1)
    codes, durs = m.sample(amb, total_frames=64, temperature=1.0, generator=g)
    assert len(codes) == 4
    for c, d in zip(codes, durs):
        assert len(c) == len(d) and len(c) >= 1
        assert all(0 <= x < K for x in c), "sampled a non-code (EOS leaked into output)"
        assert all(1 <= x <= R for x in d)
        out = expand_sequence(np.array(c), np.array(d), total=64)
        assert out.size == 64 and all(0 <= v < K for v in out)


def test_sampling_is_stochastic():
    m = _model()
    amb = torch.randn(2, 5, 32)
    a = m.sample(amb, total_frames=64, generator=torch.Generator().manual_seed(1))
    b = m.sample(amb, total_frames=64, generator=torch.Generator().manual_seed(2))
    assert a[0] != b[0]


def test_training_step_runs_and_reaches_both_heads():
    m = _model().train()
    rng = np.random.default_rng(5)
    codes = rng.integers(0, K, size=(3, 20))
    tok, dur, lens = dedup_batch(codes, max_tokens=20, max_run=R)
    m = ContentLMDedup(num_codes=K, max_run=R, ambient_dim=32, max_tokens=20,
                       d_model=32, n_layers=2, n_heads=4).train()
    tf = build_teacher_forcing(
        torch.from_numpy(tok), torch.from_numpy(dur), torch.from_numpy(lens),
        bos_id=m.bos_id, eos_id_in_head=m.num_codes,
    )
    amb = torch.randn(3, 5, 32)
    cl, dl = m(amb, tf["codes_in"], tf["durs_in"], pad_mask=tf["pad_mask"])
    loss = (F.cross_entropy(cl.reshape(-1, K + 1), tf["code_tgt"].reshape(-1), ignore_index=-100)
            + F.cross_entropy(dl.reshape(-1, R), tf["dur_tgt"].reshape(-1), ignore_index=-100))
    loss.backward()
    assert torch.isfinite(loss)
    for name in ("code_head", "dur_head", "ambient_proj", "dur_embed"):
        g = getattr(m, name).weight.grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, name


def test_forward_rejects_sequences_longer_than_max_tokens():
    m = _model()                      # max_tokens=12 -> token_pos width 13
    amb = torch.randn(1, 5, 32)
    ci = torch.randint(0, K, (1, 20))
    di = torch.ones_like(ci)
    try:
        m(amb, ci, di)
        raise AssertionError("should refuse a too-long sequence with a clear message")
    except RuntimeError as e:
        assert "max_tokens" in str(e), str(e)
