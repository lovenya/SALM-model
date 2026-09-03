from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from salm.a2s.content_bpe import AcousticBPE, fit_acoustic_bpe, length_statistics  # noqa: E402


def _slow_apply(sequence: np.ndarray, merges: tuple[tuple[int, int, int], ...]) -> np.ndarray:
    values = [int(x) for x in sequence]
    for left, right, new_id in merges:
        out: list[int] = []
        i = 0
        while i < len(values):
            if i + 1 < len(values) and values[i] == left and values[i + 1] == right:
                out.append(new_id)
                i += 2
            else:
                out.append(values[i])
                i += 1
        values = out
    return np.asarray(values, dtype=np.int64)


def test_fit_encode_decode_is_exact_and_never_crosses_rows() -> None:
    rows = np.asarray(
        [
            [0, 1, 0, 1, 0, 1],
            [0, 1, 2, 2, 2, 2],
            [3, 0, 1, 3, 0, 1],
        ],
        dtype=np.int16,
    )
    model, checkpoints, history = fit_acoustic_bpe(
        rows, 4, 8, min_frequency=2, checkpoint_vocabs=[5, 8]
    )
    assert model.vocab_size == 8
    assert history[0]["left"] == 0 and history[0]["right"] == 1
    assert set(checkpoints) == {5, 8}
    for row in rows:
        encoded = model.encode(row)
        assert np.array_equal(model.decode(encoded), row)
        assert np.array_equal(encoded, _slow_apply(row, model.merges))

    assert model.merges[0][:2] == (0, 1)

    # Width-one rows have no within-sequence adjacency at all. Flattening them without
    # respecting boundaries would invent two merge candidates.
    boundary_only = np.asarray([[0], [1], [0]], dtype=np.int64)
    no_leak, _, _ = fit_acoustic_bpe(boundary_only, 2, 4, min_frequency=1)
    assert no_leak.merges == ()


def test_overlap_is_left_to_right_and_matches_slow_reference() -> None:
    rows = np.asarray([[1, 1, 1, 1, 1], [1, 1, 1, 2, 2]], dtype=np.int64)
    model, _, _ = fit_acoustic_bpe(rows, 3, 7, min_frequency=1)
    for row in rows:
        fast = model.encode(row)
        slow = _slow_apply(row, model.merges)
        assert np.array_equal(fast, slow)
        assert np.array_equal(model.decode(fast), row)


def test_prefix_and_json_roundtrip_preserve_codec() -> None:
    rows = np.asarray([[0, 1, 0, 1], [0, 1, 0, 2]], dtype=np.int64)
    model, _, _ = fit_acoustic_bpe(rows, 3, 6, min_frequency=1)
    prefix = model.prefix(5)
    restored = AcousticBPE.from_dict(json.loads(json.dumps(prefix.to_dict())))
    assert restored == prefix
    for row in rows:
        assert np.array_equal(restored.decode(restored.encode(row)), row)


def test_training_lengths_match_independent_encoder_at_checkpoints() -> None:
    rows = np.asarray(
        [[0, 1, 0, 1, 2, 2], [0, 1, 3, 3, 2, 2], [0, 1, 0, 1, 2, 2]],
        dtype=np.int64,
    )
    model, checkpoints, _ = fit_acoustic_bpe(
        rows, 4, 9, min_frequency=1, checkpoint_vocabs=[5, 7, 9]
    )
    for vocab, linked_lengths in checkpoints.items():
        prefix = model.prefix(vocab)
        encoded_lengths = np.asarray([len(prefix.encode(row)) for row in rows])
        assert np.array_equal(linked_lengths, encoded_lengths)


def test_early_stop_and_input_validation() -> None:
    rows = np.asarray([[0, 1], [2, 3]], dtype=np.int64)
    model, checkpoints, _ = fit_acoustic_bpe(
        rows, 4, 10, min_frequency=2, checkpoint_vocabs=[4]
    )
    assert model.vocab_size == 4
    assert np.array_equal(checkpoints[4], np.asarray([2, 2]))
    try:
        model.encode([4])
    except ValueError as exc:
        assert "base codes" in str(exc)
    else:
        raise AssertionError("out-of-range base code was accepted")


def test_length_statistics_reports_requested_quantiles() -> None:
    stats = length_statistics([1, 2, 3, 4, 100])
    assert stats == {
        "count": 5,
        "mean": 22.0,
        "p50": 3.0,
        "p95": 80.79999999999998,
        "min": 1,
        "max": 100,
    }
