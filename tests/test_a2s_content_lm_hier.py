"""Tests for the coarse+SEP+fine teacher-forcing construction. Pure tensor logic, no model."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from salm.a2s.content_lm_hier import build_hier_batch, coarse_of  # noqa: E402

K, BOS, SEP, EOS = 16, 16, 17, 18


def _batch(tokens, durs, lens, stride=3):
    return build_hier_batch(torch.tensor(tokens), torch.tensor(durs), torch.tensor(lens),
                            stride, BOS, SEP, EOS)


def test_coarse_is_a_strided_subsequence_of_real_tokens():
    t = np.arange(10)
    assert coarse_of(t, 10, 3).tolist() == [0, 3, 6, 9]
    assert coarse_of(t, 5, 3).tolist() == [0, 3]          # respects length, ignores padding
    assert coarse_of(t, 0, 3).tolist() == []


def test_sequence_layout_is_coarse_sep_fine_then_eos():
    tok = [[1, 2, 3, 4, 5, 0, 0]]
    dur = [[2, 3, 4, 5, 6, 0, 0]]
    b = _batch(tok, dur, [5], stride=2)
    ci, ct = b["codes_in"][0].tolist(), b["code_tgt"][0].tolist()
    assert ci[0] == BOS
    # coarse = tokens[0:5:2] = [1,3,5]; then SEP; then the full 5
    assert ci[1:4] == [1, 3, 5]
    assert ci[4] == SEP
    assert ci[5:10] == [1, 2, 3, 4, 5]
    assert ct[:3] == [1, 3, 5] and ct[3] == SEP
    assert ct[4:9] == [1, 2, 3, 4, 5]
    assert ct[9] == EOS, "exactly one EOS, immediately after the last fine token"


def test_durations_supervised_only_on_fine_positions():
    """The coarse prefix is a plan. Supervising durations there would train the model to
    render its own sketch, which is not what the sketch is for."""
    b = _batch([[1, 2, 3, 4, 0]], [[2, 3, 4, 5, 0]], [4], stride=2)
    dt = b["dur_tgt"][0].tolist()
    fine = b["is_fine"][0].tolist()
    for i, (d, f) in enumerate(zip(dt, fine)):
        if not f:
            assert d == -100, f"position {i} is coarse/SEP/EOS but has duration target {d}"
    assert [d for d, f in zip(dt, fine) if f] == [1, 2, 3, 4], "1..R must map to 0..R-1"


def test_exactly_one_eos_and_one_sep_per_row():
    b = _batch([[1, 2, 3, 0], [4, 5, 0, 0]], [[1, 2, 3, 0], [4, 5, 0, 0]], [3, 2], stride=2)
    for r in range(2):
        ct = b["code_tgt"][r].tolist()
        assert ct.count(EOS) == 1, "an extra EOS would end generation early"
        assert ct.count(SEP) == 1, "an extra SEP would restart the fine pass mid-utterance"


def test_padding_mask_covers_bos_plus_real_inputs_only():
    b = _batch([[1, 2, 3, 0], [4, 0, 0, 0]], [[1, 2, 3, 0], [4, 0, 0, 0]], [3, 1], stride=2)
    for r, n in ((0, 3), (1, 1)):
        pm = b["pad_mask"][r].tolist()
        width = 1 + (len(coarse_of(np.array([1, 2, 3, 0]), n, 2)) + 1 + n)
        assert not any(pm[:width]), "a real input was masked"
        assert all(pm[width:]), "padding was left unmasked"
        assert not pm[0], "BOS must never be masked -- a fully masked row gives NaN attention"


def test_coarse_shorter_than_fine_by_the_stride():
    rng = np.random.default_rng(0)
    tok = rng.integers(0, K, (4, 60)); dur = rng.integers(1, 8, (4, 60))
    b = _batch(tok.tolist(), dur.tolist(), [60] * 4, stride=3)
    assert 19 <= b["mean_coarse"] <= 21, f"expected ~20 coarse tokens, got {b['mean_coarse']}"


def test_no_target_index_exceeds_the_head():
    """code_head has K+2 outputs; an EOS id of K+2 would index out of bounds at loss time."""
    b = _batch([[1, 2, 3, 0]], [[1, 2, 3, 0]], [3], stride=2)
    real = b["code_tgt"][b["code_tgt"] != -100]
    assert int(real.max()) <= K + 2, f"target {int(real.max())} >= head width {K+3}"


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    bad = 0
    for name, fn in fns:
        try:
            fn(); print(f"  PASS  {name}")
        except AssertionError as e:
            bad += 1; print(f"  FAIL  {name}: {e}")
    print(f"\n{len(fns)-bad}/{len(fns)} passed")
    raise SystemExit(1 if bad else 0)
