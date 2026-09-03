"""
Run-length codec for content code sequences: 256 frame-level codes <-> (tokens, durations).

WHY. The ContentLM works -- it generates per-scene, syntactically real speech from ambient
audio alone (jobs 436344/438059). Its remaining failure is sentence-level coherence: short
utterances come out clean, long ones fragment after ~3-5 words. The cause is that it plans
over 256 tokens at 25 fps, one 40 ms acoustic state at a time, and holds a plan for ~30 of
them. Both alternative explanations are eliminated by measurement -- more epochs (438059,
flat epoch ladder) and regularization/capacity (438062, scene_gap invariant at +0.021 across
seven configs).

WHAT THIS BUYS, MEASURED BEFORE BUILDING. Exact run-length encoding of the training split:

    K= 512:  256 -> 128.2 tokens  (2.00x)   mean run 2.00 frames
    K=1024:  256 -> 141.4 tokens  (1.81x)   mean run 1.81 frames
    K=2048:  256 -> 151.2 tokens  (1.69x)   mean run 1.69 frames

That is ~2x, NOT the 3-5x initially assumed -- 45% of consecutive frame pairs are identical
and the differing ones sit at cosine ~0.79, so they are genuinely distinct centroids rather
than jitter. Optional near-neighbour merging reaches ~2.8x and then saturates (cosine
threshold 0.98 -> 0.90 moves 101 -> 91 tokens), so no threshold unlocks a larger collapse.

Stated plainly so nobody over-reads this module: halving the sequence length is a real
improvement to the planning horizon and is not expected to be sufficient on its own. The
deeper issue is that ONE codebook cannot be both acoustically faithful (K=256 already renders
muffled) and linguistically coarse (K=2048 still dedups to 151 tokens). Fixing that properly
needs a two-level semantic/acoustic hierarchy, which is the designated follow-up.

LOSSLESS BY DEFAULT. With `merge_cosine=0.0` this codec is exactly invertible: expanding
(tokens, durations) reproduces the original 256 codes bit-for-bit, so the representation
change cannot itself degrade anything and needs no intelligibility gate. Near-neighbour
merging (`merge_cosine > 0`) IS lossy -- it substitutes a nearby centroid for a frame -- so
it must be gated the same way quantization was, by generating from merged ORACLE content and
scoring the audio, before any predictor is trained on it.
"""

from __future__ import annotations

import numpy as np
import torch


def dedup_sequence(
    codes: np.ndarray,
    codebook_unit: np.ndarray | None = None,
    merge_cosine: float = 0.0,
    max_run: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One utterance: (T,) frame codes -> (tokens, durations), both 1-D, summing to T.

    `merge_cosine` > 0 also absorbs a frame into the current run when its centroid is within
    that cosine of the RUN'S OPENING centroid -- the opener, not the previous frame, so a run
    cannot drift arbitrarily far through a chain of individually-close steps. Requires
    `codebook_unit`, the L2-normalised codebook.

    `max_run` SPLITS a longer run into several tokens of the same code rather than clamping
    its duration. This matters: the duration head needs a bounded vocabulary (measured run
    lengths are median 1 and mean ~1.8 but reach ~160 frames on silence, and a head over 160
    classes would be almost entirely dead), yet clamping a 163-frame run to 32 would silently
    discard 131 frames and break the round-trip property this codec is built on. Splitting
    keeps `expand_sequence` exact while bounding every duration to <= max_run.
    """
    codes = np.asarray(codes).ravel()
    if codes.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    if merge_cosine > 0.0 and codebook_unit is None:
        raise ValueError("merge_cosine > 0 requires codebook_unit (L2-normalised codebook)")

    if max_run is not None and max_run < 1:
        raise ValueError("max_run must be >= 1")

    tokens = [int(codes[0])]
    durations = [1]

    def _extend(code: int) -> None:
        """Grow the current run, opening a new token of the same code at the max_run bound."""
        if max_run is not None and durations[-1] >= max_run:
            tokens.append(code)
            durations.append(1)
        else:
            durations[-1] += 1

    for x in codes[1:]:
        x = int(x)
        cur = tokens[-1]
        if x == cur:
            _extend(cur)
            continue
        if merge_cosine > 0.0:
            if float(codebook_unit[cur] @ codebook_unit[x]) >= merge_cosine:
                # Absorbed into the run, so it keeps the RUN'S code, not its own.
                _extend(cur)
                continue
        tokens.append(x)
        durations.append(1)
    return np.asarray(tokens, dtype=np.int64), np.asarray(durations, dtype=np.int64)


def expand_sequence(
    tokens: np.ndarray, durations: np.ndarray, total: int | None = None
) -> np.ndarray:
    """
    (tokens, durations) -> (sum(durations),) frame codes. The exact inverse of
    `dedup_sequence(..., merge_cosine=0)`.

    `total` pads or truncates to a fixed length, which generation needs because the concat
    channel demands exactly 256 frames and sampled durations will not sum to that on their
    own. Padding repeats the LAST token rather than inserting a reserved value: the frames
    are fed to a renderer trained only on real content, so every frame must remain a real
    code -- and holding the final code is the least surprising way to do that.
    """
    tokens = np.asarray(tokens).ravel()
    durations = np.asarray(durations).ravel()
    if tokens.size != durations.size:
        raise ValueError(f"tokens {tokens.size} and durations {durations.size} disagree")
    if tokens.size == 0:
        if not total:
            return np.zeros(0, dtype=np.int64)
        raise ValueError("cannot expand an empty sequence to a fixed total")
    if (durations < 1).any():
        raise ValueError("durations must all be >= 1")

    out = np.repeat(tokens, durations).astype(np.int64)
    if total is None:
        return out
    if out.size >= total:
        return out[:total]
    return np.concatenate([out, np.full(total - out.size, out[-1], dtype=np.int64)])


def dedup_batch(
    codes: np.ndarray,
    codebook_unit: np.ndarray | None = None,
    merge_cosine: float = 0.0,
    max_tokens: int | None = None,
    pad_token: int = -1,
    max_run: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    (N, T) frame codes -> (tokens, durations, lengths), the first two padded to `max_tokens`.

    Returns `lengths` rather than relying on a sentinel so the training mask is derived from
    an explicit count -- a padded duration of 0 and a real duration of 0 would otherwise be
    indistinguishable, and a silent off-by-one there would corrupt every sequence.
    """
    codes = np.asarray(codes)
    rows = [dedup_sequence(r, codebook_unit, merge_cosine, max_run) for r in codes]
    lengths = np.asarray([len(t) for t, _ in rows], dtype=np.int64)
    cap = int(max_tokens or lengths.max())
    if lengths.max() > cap:
        raise ValueError(
            f"longest deduped sequence is {lengths.max()} tokens but max_tokens={cap}; "
            "raise max_tokens rather than truncating, or utterances lose their tails"
        )
    tok = np.full((len(rows), cap), pad_token, dtype=np.int64)
    dur = np.zeros((len(rows), cap), dtype=np.int64)
    for i, (t, d) in enumerate(rows):
        tok[i, : len(t)] = t
        dur[i, : len(d)] = d
    return tok, dur, lengths


def unit_codebook(codes: torch.Tensor) -> np.ndarray:
    """L2-normalised codebook as numpy, for the merge_cosine path."""
    c = codes.float()
    n = c.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return (c / n).cpu().numpy()


def stats(codes: np.ndarray) -> dict:
    """Compression summary for a (N, T) code array -- what the build script reports."""
    codes = np.asarray(codes)
    changes = (codes[:, 1:] != codes[:, :-1]).sum(axis=1) + 1
    t = int(codes.shape[1])
    return {
        "frames": t,
        "mean_tokens": float(changes.mean()),
        "median_tokens": float(np.median(changes)),
        "max_tokens": int(changes.max()),
        "min_tokens": int(changes.min()),
        "compression": float(t / changes.mean()),
        "mean_run_frames": float(t / changes.mean()),
    }
