"""Lossless byte-pair encoding for integer acoustic-content code sequences.

The base symbols are content-codebook indices, not bytes.  Every learned symbol is the
concatenation of two earlier symbols, so decoding is exact and needs neither a neural model
nor a duration predictor.  Sequence boundaries are never eligible for merging.

The trainer uses a linked corpus plus an occurrence index.  Selecting one pair therefore
touches only occurrences of that pair and their immediate neighbours instead of rescanning
all ~4.5 million training frames after every merge.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import heapq
from typing import Callable, Iterable, Sequence

import numpy as np


Merge = tuple[int, int, int]


@dataclass(frozen=True)
class AcousticBPE:
    """A reversible integer BPE model.

    ``merges`` are ``(left_id, right_id, new_id)`` in learning order.  New IDs must be
    consecutive from ``base_vocab_size``; enforcing that invariant makes corrupt or
    mismatched artifacts fail loudly.
    """

    base_vocab_size: int
    merges: tuple[Merge, ...]

    def __post_init__(self) -> None:
        if self.base_vocab_size < 1:
            raise ValueError("base_vocab_size must be positive")
        known = self.base_vocab_size
        for left, right, new_id in self.merges:
            if left < 0 or right < 0 or left >= known or right >= known:
                raise ValueError(
                    f"merge {(left, right, new_id)} references an unknown earlier symbol"
                )
            if new_id != known:
                raise ValueError(
                    f"merge new_id {new_id} is not consecutive; expected {known}"
                )
            known += 1

    @property
    def vocab_size(self) -> int:
        return self.base_vocab_size + len(self.merges)

    def prefix(self, vocab_size: int) -> "AcousticBPE":
        if not self.base_vocab_size <= vocab_size <= self.vocab_size:
            raise ValueError(
                f"vocab_size must be in [{self.base_vocab_size}, {self.vocab_size}]"
            )
        return AcousticBPE(
            self.base_vocab_size,
            self.merges[: vocab_size - self.base_vocab_size],
        )

    def to_dict(self) -> dict:
        return {
            "format": "salm_integer_acoustic_bpe_v1",
            "base_vocab_size": self.base_vocab_size,
            "vocab_size": self.vocab_size,
            "merges": [list(m) for m in self.merges],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "AcousticBPE":
        if payload.get("format") != "salm_integer_acoustic_bpe_v1":
            raise ValueError(f"unsupported BPE format: {payload.get('format')!r}")
        model = cls(
            int(payload["base_vocab_size"]),
            tuple(tuple(int(x) for x in m) for m in payload["merges"]),
        )
        if int(payload.get("vocab_size", model.vocab_size)) != model.vocab_size:
            raise ValueError("declared vocab_size disagrees with merge table")
        return model

    @cached_property
    def _merge_ranks(self) -> dict[tuple[int, int], tuple[int, int]]:
        return {
            (left, right): (rank, new_id)
            for rank, (left, right, new_id) in enumerate(self.merges)
        }

    def encode(self, sequence: Sequence[int] | np.ndarray) -> np.ndarray:
        """Encode one base-code sequence using the learned merge ordering.

        A heap applies the earliest-ranked currently available pair.  This is equivalent to
        replaying every learned merge over the whole sequence, but scales with the sequence
        length rather than ``len(sequence) * number_of_merges``.
        """
        values = np.asarray(sequence, dtype=np.int64).reshape(-1)
        if values.size == 0:
            return values.copy()
        if int(values.min()) < 0 or int(values.max()) >= self.base_vocab_size:
            raise ValueError(
                f"base codes must be in [0, {self.base_vocab_size}); got "
                f"[{int(values.min())}, {int(values.max())}]"
            )
        if not self.merges:
            return values.copy()

        tokens = values.tolist()
        n = len(tokens)
        prev = [i - 1 for i in range(n)]
        nxt = [i + 1 for i in range(n)]
        nxt[-1] = -1
        alive = [True] * n
        ranks = self._merge_ranks
        heap: list[tuple[int, int, int, int, int, int]] = []

        def push(left_pos: int) -> None:
            if left_pos < 0 or not alive[left_pos]:
                return
            right_pos = nxt[left_pos]
            if right_pos < 0 or not alive[right_pos]:
                return
            pair = (tokens[left_pos], tokens[right_pos])
            ranked = ranks.get(pair)
            if ranked is not None:
                rank, new_id = ranked
                heapq.heappush(
                    heap, (rank, left_pos, right_pos, pair[0], pair[1], new_id)
                )

        for i in range(n - 1):
            push(i)

        while heap:
            rank, left_pos, right_pos, left_id, right_id, new_id = heapq.heappop(heap)
            if (
                not alive[left_pos]
                or not alive[right_pos]
                or nxt[left_pos] != right_pos
                or tokens[left_pos] != left_id
                or tokens[right_pos] != right_id
                or ranks.get((left_id, right_id)) != (rank, new_id)
            ):
                continue
            before = prev[left_pos]
            after = nxt[right_pos]
            tokens[left_pos] = new_id
            alive[right_pos] = False
            nxt[left_pos] = after
            if after >= 0:
                prev[after] = left_pos
            push(before)
            push(left_pos)

        out: list[int] = []
        pos = 0
        while pos >= 0:
            if alive[pos]:
                out.append(tokens[pos])
            pos = nxt[pos]
        return np.asarray(out, dtype=np.int64)

    @cached_property
    def _expansions(self) -> tuple[tuple[int, ...], ...]:
        result: list[tuple[int, ...]] = [(i,) for i in range(self.base_vocab_size)]
        for left, right, _ in self.merges:
            result.append(result[left] + result[right])
        return tuple(result)

    def expansions(self) -> tuple[tuple[int, ...], ...]:
        """Return the cached base-code expansion of every vocabulary symbol."""
        return self._expansions

    def decode(self, encoded: Sequence[int] | np.ndarray) -> np.ndarray:
        """Expand macro-tokens to the original frame-code sequence exactly."""
        symbols = np.asarray(encoded, dtype=np.int64).reshape(-1)
        if symbols.size == 0:
            return symbols.copy()
        if int(symbols.min()) < 0 or int(symbols.max()) >= self.vocab_size:
            raise ValueError(
                f"encoded IDs must be in [0, {self.vocab_size}); got "
                f"[{int(symbols.min())}, {int(symbols.max())}]"
            )
        expansion = self._expansions
        return np.fromiter(
            (base for symbol in symbols for base in expansion[int(symbol)]),
            dtype=np.int64,
        )


class _LinkedCorpus:
    """Mutable linked representation and exact adjacent-pair occurrence index."""

    def __init__(self, sequences: np.ndarray, base_vocab_size: int) -> None:
        seq = np.asarray(sequences)
        if seq.ndim != 2 or seq.shape[0] < 1 or seq.shape[1] < 1:
            raise ValueError(f"sequences must be non-empty rank-2, got {seq.shape}")
        if int(seq.min()) < 0 or int(seq.max()) >= base_vocab_size:
            raise ValueError(
                f"codes must be in [0, {base_vocab_size}); got "
                f"[{int(seq.min())}, {int(seq.max())}]"
            )

        n_rows, width = seq.shape
        total = n_rows * width
        self.tokens = np.asarray(seq, dtype=np.int64).reshape(-1).copy()
        self.prev = np.arange(total, dtype=np.int64) - 1
        self.nxt = np.arange(total, dtype=np.int64) + 1
        starts = np.arange(0, total, width, dtype=np.int64)
        ends = starts + width - 1
        self.prev[starts] = -1
        self.nxt[ends] = -1
        self.alive = np.ones(total, dtype=np.bool_)
        self.row_of = np.repeat(np.arange(n_rows, dtype=np.int32), width)
        self.lengths = np.full(n_rows, width, dtype=np.int64)
        self.positions: dict[tuple[int, int], set[int]] = {}
        self.heap: list[tuple[int, int, int]] = []

        for left in range(total):
            right = int(self.nxt[left])
            if right >= 0:
                pair = (int(self.tokens[left]), int(self.tokens[right]))
                self.positions.setdefault(pair, set()).add(left)
        for (left_id, right_id), pos in self.positions.items():
            heapq.heappush(self.heap, (-len(pos), left_id, right_id))

    def _remove_pair_at(self, left: int, touched: set[tuple[int, int]]) -> None:
        if left < 0 or not self.alive[left]:
            return
        right = int(self.nxt[left])
        if right < 0 or not self.alive[right]:
            return
        pair = (int(self.tokens[left]), int(self.tokens[right]))
        positions = self.positions.get(pair)
        if positions is None or left not in positions:
            raise RuntimeError(f"pair index lost live occurrence {pair} at {left}")
        positions.remove(left)
        if not positions:
            del self.positions[pair]
        touched.add(pair)

    def _add_pair_at(self, left: int, touched: set[tuple[int, int]]) -> None:
        if left < 0 or not self.alive[left]:
            return
        right = int(self.nxt[left])
        if right < 0 or not self.alive[right]:
            return
        pair = (int(self.tokens[left]), int(self.tokens[right]))
        self.positions.setdefault(pair, set()).add(left)
        touched.add(pair)

    def best_pair(self) -> tuple[tuple[int, int], int] | None:
        while self.heap:
            neg_count, left_id, right_id = heapq.heappop(self.heap)
            pair = (left_id, right_id)
            actual = len(self.positions.get(pair, ()))
            if actual and actual == -neg_count:
                return pair, actual
        return None

    def merge(self, pair: tuple[int, int], new_id: int) -> int:
        occurrences = self.positions.get(pair)
        if not occurrences:
            return 0
        # Sorting defines left-to-right non-overlapping handling for (x, x) runs.
        candidates = sorted(occurrences)
        touched: set[tuple[int, int]] = set()
        replacements = 0
        for left in candidates:
            if not self.alive[left]:
                continue
            right = int(self.nxt[left])
            if (
                right < 0
                or not self.alive[right]
                or (int(self.tokens[left]), int(self.tokens[right])) != pair
            ):
                continue
            before = int(self.prev[left])
            after = int(self.nxt[right])
            self._remove_pair_at(before, touched)
            self._remove_pair_at(left, touched)
            self._remove_pair_at(right, touched)

            self.tokens[left] = new_id
            self.alive[right] = False
            self.nxt[left] = after
            if after >= 0:
                self.prev[after] = left
            self.lengths[int(self.row_of[left])] -= 1
            replacements += 1

            self._add_pair_at(before, touched)
            self._add_pair_at(left, touched)

        for changed in touched:
            count = len(self.positions.get(changed, ()))
            if count:
                heapq.heappush(self.heap, (-count, changed[0], changed[1]))
        return replacements


def fit_acoustic_bpe(
    sequences: np.ndarray,
    base_vocab_size: int,
    target_vocab_size: int,
    *,
    min_frequency: int = 2,
    checkpoint_vocabs: Iterable[int] = (),
    on_progress: Callable[[dict], None] | None = None,
) -> tuple[AcousticBPE, dict[int, np.ndarray], list[dict]]:
    """Learn lossless macro-tokens and optionally snapshot train lengths at vocab budgets.

    Returns ``(model, checkpoint_lengths, merge_history)``.  Training can stop below the
    requested vocabulary if no adjacent pair occurs ``min_frequency`` times.
    """
    if target_vocab_size < base_vocab_size:
        raise ValueError("target_vocab_size cannot be smaller than base_vocab_size")
    if min_frequency < 1:
        raise ValueError("min_frequency must be >= 1")
    wanted = sorted(set(int(v) for v in checkpoint_vocabs))
    if any(v < base_vocab_size or v > target_vocab_size for v in wanted):
        raise ValueError("checkpoint vocabularies must lie between base and target vocab")

    corpus = _LinkedCorpus(sequences, base_vocab_size)
    merges: list[Merge] = []
    history: list[dict] = []
    checkpoints: dict[int, np.ndarray] = {}
    if base_vocab_size in wanted:
        checkpoints[base_vocab_size] = corpus.lengths.copy()

    while base_vocab_size + len(merges) < target_vocab_size:
        selected = corpus.best_pair()
        if selected is None or selected[1] < min_frequency:
            break
        pair, indexed_frequency = selected
        new_id = base_vocab_size + len(merges)
        replacements = corpus.merge(pair, new_id)
        if replacements < 1:
            raise RuntimeError(f"selected pair {pair} had no valid replacements")
        merges.append((pair[0], pair[1], new_id))
        row = {
            "vocab_size": new_id + 1,
            "left": pair[0],
            "right": pair[1],
            "new_id": new_id,
            "indexed_frequency": indexed_frequency,
            "replacements": replacements,
            "mean_train_tokens": float(corpus.lengths.mean()),
        }
        history.append(row)
        if row["vocab_size"] in wanted:
            checkpoints[row["vocab_size"]] = corpus.lengths.copy()
        if on_progress is not None:
            on_progress(row)

    model = AcousticBPE(base_vocab_size, tuple(merges))
    return model, checkpoints, history


def length_statistics(lengths: Sequence[int] | np.ndarray) -> dict:
    values = np.asarray(lengths, dtype=np.int64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("lengths must be a non-empty vector")
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "min": int(values.min()),
        "max": int(values.max()),
    }
