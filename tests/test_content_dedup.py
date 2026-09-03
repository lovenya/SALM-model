"""Correctness checks for the run-length content codec.

The round-trip exactness property is the whole basis for claiming this representation change
is free. A silent bug here would corrupt every sequence the LM trains on while every loss
curve still looked plausible, so it is asserted directly rather than assumed.
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from salm.a2s.content_dedup import (  # noqa: E402
    dedup_batch,
    dedup_sequence,
    expand_sequence,
    stats,
)


def test_hand_computed_runs():
    codes = np.array([5, 5, 5, 2, 2, 9])
    tok, dur = dedup_sequence(codes)
    assert tok.tolist() == [5, 2, 9]
    assert dur.tolist() == [3, 2, 1]
    assert dur.sum() == len(codes)


def test_roundtrip_is_exact_on_random_sequences():
    rng = np.random.default_rng(0)
    for _ in range(200):
        # Low cardinality so runs actually occur, which is the interesting case.
        codes = rng.integers(0, 6, size=rng.integers(1, 300))
        tok, dur = dedup_sequence(codes)
        assert np.array_equal(expand_sequence(tok, dur), codes)


def test_roundtrip_exact_at_realistic_cardinality():
    rng = np.random.default_rng(1)
    codes = rng.integers(0, 1024, size=(50, 256))
    for row in codes:
        tok, dur = dedup_sequence(row)
        assert np.array_equal(expand_sequence(tok, dur), row)


def test_no_adjacent_duplicates_survive():
    rng = np.random.default_rng(2)
    codes = rng.integers(0, 4, size=500)
    tok, _ = dedup_sequence(codes)
    assert not (tok[1:] == tok[:-1]).any()


def test_constant_sequence_collapses_to_one_token():
    tok, dur = dedup_sequence(np.full(256, 77))
    assert tok.tolist() == [77] and dur.tolist() == [256]


def test_alternating_sequence_does_not_compress():
    codes = np.array([1, 2] * 64)
    tok, dur = dedup_sequence(codes)
    assert len(tok) == 128 and dur.sum() == 128


def test_expand_pads_by_holding_the_last_code():
    # Padding must stay on-manifold: every frame handed to the renderer has to be a real code.
    out = expand_sequence(np.array([3, 8]), np.array([2, 2]), total=6)
    assert out.tolist() == [3, 3, 8, 8, 8, 8]


def test_expand_truncates_when_durations_overshoot():
    out = expand_sequence(np.array([3, 8]), np.array([200, 200]), total=256)
    assert out.size == 256 and out[0] == 3 and out[-1] == 8


def test_expand_rejects_bad_input():
    for bad in (
        (np.array([1, 2]), np.array([1])),          # length mismatch
        (np.array([1]), np.array([0])),             # zero duration
    ):
        try:
            expand_sequence(*bad)
            raise AssertionError(f"should have rejected {bad}")
        except ValueError:
            pass


def test_merge_cosine_absorbs_near_neighbours_and_needs_a_codebook():
    # codes 0 and 1 nearly parallel; code 2 orthogonal to both
    cb = np.array([[1.0, 0.0], [0.999, 0.0447], [0.0, 1.0]])
    cb = cb / np.linalg.norm(cb, axis=1, keepdims=True)
    codes = np.array([0, 1, 1, 2, 2])
    tok, dur = dedup_sequence(codes, cb, merge_cosine=0.99)
    assert tok.tolist() == [0, 2], tok
    assert dur.tolist() == [3, 2]
    # exact dedup keeps them separate
    tok0, _ = dedup_sequence(codes)
    assert tok0.tolist() == [0, 1, 2]
    # and it refuses without a codebook
    try:
        dedup_sequence(codes, None, merge_cosine=0.9)
        raise AssertionError("should require codebook_unit")
    except ValueError:
        pass


def test_merge_compares_against_run_opener_not_previous_frame():
    # A chain where each step is close to the last but the ends are far apart. Comparing to
    # the run OPENER must stop the drift; comparing to the previous frame would not.
    cb = np.array([[1.0, 0.0], [0.92, 0.39], [0.7, 0.71], [0.0, 1.0]])
    cb = cb / np.linalg.norm(cb, axis=1, keepdims=True)
    tok, _ = dedup_sequence(np.array([0, 1, 2, 3]), cb, merge_cosine=0.9)
    # 1 merges into 0 (cos .92); 2 does NOT (cos to opener 0 is .7)
    assert tok.tolist() == [0, 2, 3], tok


def test_dedup_batch_shapes_lengths_and_padding():
    codes = np.array([[1, 1, 2, 3], [7, 7, 7, 7]])
    tok, dur, lens = dedup_batch(codes, max_tokens=5, pad_token=-1)
    assert tok.shape == (2, 5) and dur.shape == (2, 5)
    assert lens.tolist() == [3, 1]
    assert tok[0, :3].tolist() == [1, 2, 3] and (tok[0, 3:] == -1).all()
    assert dur[0, :3].tolist() == [2, 1, 1] and (dur[0, 3:] == 0).all()
    assert tok[1, 0] == 7 and dur[1, 0] == 4
    # every row's real durations must still sum to the original frame count
    for i, L in enumerate(lens):
        assert dur[i, :L].sum() == codes.shape[1]


def test_dedup_batch_refuses_to_truncate():
    codes = np.array([[1, 2, 3, 4]])
    try:
        dedup_batch(codes, max_tokens=2)
        raise AssertionError("should refuse rather than silently drop the tail")
    except ValueError:
        pass


def test_stats_match_a_hand_case():
    codes = np.array([[1, 1, 2, 2], [3, 4, 5, 6]])
    s = stats(codes)
    assert s["frames"] == 4
    assert s["mean_tokens"] == 3.0          # 2 and 4
    assert s["max_tokens"] == 4 and s["min_tokens"] == 2
    assert abs(s["compression"] - 4 / 3) < 1e-9


def test_max_run_splits_rather_than_clamps_and_stays_exact():
    # A 100-frame run with max_run=32 must become 32+32+32+4 of the SAME code, and expanding
    # must still reproduce the original exactly. Clamping instead would drop 68 frames.
    codes = np.full(100, 42)
    tok, dur = dedup_sequence(codes, max_run=32)
    assert tok.tolist() == [42, 42, 42, 42]
    assert dur.tolist() == [32, 32, 32, 4]
    assert dur.max() <= 32
    assert np.array_equal(expand_sequence(tok, dur), codes)


def test_max_run_bounds_every_duration_on_real_shaped_data():
    rng = np.random.default_rng(7)
    # heavy repetition, like silence stretches
    codes = np.repeat(rng.integers(0, 5, size=20), rng.integers(1, 60, size=20))
    for cap in (4, 16, 32):
        tok, dur = dedup_sequence(codes, max_run=cap)
        assert dur.max() <= cap
        assert np.array_equal(expand_sequence(tok, dur), codes)


def test_max_run_is_a_noop_when_runs_are_short():
    codes = np.array([1, 1, 2, 3, 3])
    a = dedup_sequence(codes)
    b = dedup_sequence(codes, max_run=32)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_max_run_rejects_zero():
    try:
        dedup_sequence(np.array([1, 1]), max_run=0)
        raise AssertionError("should reject max_run < 1")
    except ValueError:
        pass


def test_merged_frames_take_the_run_code_not_their_own():
    # With merging, an absorbed frame must not change the run's code -- otherwise expand()
    # would emit a code the run never represented.
    cb = np.array([[1.0, 0.0], [0.999, 0.0447], [0.0, 1.0]])
    cb = cb / np.linalg.norm(cb, axis=1, keepdims=True)
    tok, dur = dedup_sequence(np.array([0, 1, 1]), cb, merge_cosine=0.99)
    assert tok.tolist() == [0] and dur.tolist() == [3]
    assert expand_sequence(tok, dur).tolist() == [0, 0, 0]


def test_merge_and_max_run_compose():
    cb = np.array([[1.0, 0.0], [0.999, 0.0447]])
    cb = cb / np.linalg.norm(cb, axis=1, keepdims=True)
    codes = np.array([0, 1] * 30)          # all mergeable into one long run
    tok, dur = dedup_sequence(codes, cb, merge_cosine=0.99, max_run=8)
    assert dur.max() <= 8
    assert dur.sum() == len(codes)
    assert set(tok.tolist()) == {0}
