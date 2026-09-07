"""Tests for run-length truncation at the speech boundary. Pure numpy, no I/O, no model."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from salm.a2s.speech_extent import truncate_to_speech  # noqa: E402


def test_durations_sum_to_the_budget():
    tok = np.array([[5, 6, 7, 8, 0]])
    dur = np.array([[10, 10, 10, 10, 0]])
    t, d, n = truncate_to_speech(tok, dur, np.array([4]), np.array([25]))
    assert d[0, :n[0]].sum() == 25, f"kept {d[0,:n[0]].sum()} frames, budget was 25"


def test_the_straddling_token_is_shortened_not_dropped():
    """Dropping it would lose the end of the utterance -- the part whose coherence is at issue."""
    tok = np.array([[1, 2, 3]])
    dur = np.array([[10, 10, 10]])
    t, d, n = truncate_to_speech(tok, dur, np.array([3]), np.array([15]))
    assert n[0] == 2, "the token crossing the boundary must be kept"
    assert d[0, 1] == 5, f"it must be shortened to 5, got {d[0,1]}"
    assert t[0, 1] == 2, "and keep its own code"


def test_budget_beyond_the_sequence_is_a_no_op():
    tok = np.array([[4, 5, 6, 0]])
    dur = np.array([[3, 3, 3, 0]])
    t, d, n = truncate_to_speech(tok, dur, np.array([3]), np.array([999]))
    assert n[0] == 3 and d[0, :3].tolist() == [3, 3, 3] and t[0, :3].tolist() == [4, 5, 6]


def test_never_returns_an_empty_row():
    """An empty sequence would expand to nothing and crash the renderer's content channel."""
    t, d, n = truncate_to_speech(np.array([[9, 9]]), np.array([[5, 5]]),
                                 np.array([2]), np.array([0]))
    assert n[0] >= 1 and d[0, 0] >= 1 and t[0, 0] == 9


def test_shape_preserved_and_tail_zeroed():
    rng = np.random.default_rng(0)
    tok = rng.integers(1, 50, (6, 20)); dur = rng.integers(1, 9, (6, 20))
    t, d, n = truncate_to_speech(tok, dur, np.full(6, 20), rng.integers(10, 40, 6))
    assert t.shape == tok.shape and d.shape == dur.shape
    for i in range(6):
        assert t[i, n[i]:].sum() == 0 and d[i, n[i]:].sum() == 0, "tail not zeroed"


def test_truncation_actually_shortens_on_realistic_input():
    """Median speech is 104 of 256 frames, so lengths should drop by roughly 60%."""
    rng = np.random.default_rng(1)
    n_clips, L = 200, 160
    dur = rng.integers(1, 4, (n_clips, L))
    tok = rng.integers(0, 2048, (n_clips, L))
    lens = np.full(n_clips, L)
    _, d2, n2 = truncate_to_speech(tok, dur, lens, np.full(n_clips, 104))
    assert n2.mean() < lens.mean() * 0.75, f"barely shortened: {n2.mean():.1f} vs {L}"
    assert all(d2[i, :n2[i]].sum() == 104 for i in range(n_clips)), "budget not honoured"


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
