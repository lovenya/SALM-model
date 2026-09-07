"""
Where does the speech actually end inside the fixed 10.24 s canvas?

WHY THIS MATTERS. The content LM is trained to fill all 256 frames, but speech occupies a
median of 104 of them (41%). The remaining 57% is not silence -- measured at 25 distinct codes
with a 0.35 frame-to-frame change rate against the speech region's 0.87 -- so the model spends
well over half its 151-token planning budget predicting quiet churn that carries no words.

WHY IT IS WORTH FIXING NOW rather than earlier. Job 444750 (coarse-to-fine) produced the most
intelligible audio this project has made, and the measured reason was not better planning: the
model simply STOPPED EARLIER. 11.1 words against the flat model's 17.6, where the human said
13.3, and only 11% of its utterances ran past 1.5x their reference against the flat model's
42%. Length control bought more audible improvement than any planning intervention. Truncating
the target at the speech end attacks that directly -- and it also cuts the sequence the model
must plan from ~151 tokens to ~62, shrinking the horizon shortfall from 7.6x to about 3.1x.

NO RENDERER CHANGE IS NEEDED. `expand_sequence` already pads by holding the last real code, so
a model that emits EOS at the speech end produces exactly the input the renderer sees today
when a sequence ends early. The change is entirely in what the model is TRAINED to emit.

Speech is left-aligned in the window (verified on real clips: the head carries the codes, the
tail is churn), so a single boundary per clip is sufficient -- no need to locate a span.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def speech_frames(split: str, n_expected: int, fps: int = 25, total_frames: int = 256,
                  cache_root: Path | None = None) -> np.ndarray:
    """
    (N,) frame index at which each clip's speech ends, clipped to [1, total_frames].

    Read from the wav headers via `sf.info` (no audio decoded) and cached, because doing it for
    17,596 clips on every job start is wasted I/O. Row order follows the features index, which
    is the same order `load_split` uses for the code arrays -- if that ever diverges the two
    would be silently misaligned, so the length is asserted against the caller's expectation.
    """
    import soundfile as sf

    from configs import paths as project_paths

    root = Path(cache_root) if cache_root else project_paths.OUTPUTS_ROOT
    cache = root / "content_codes" / f"speech_frames_{split}_{fps}fps.npy"
    if cache.exists():
        out = np.load(cache)
    else:
        idx_path = root / "features" / "a2s" / split / "index.jsonl"
        rows = [json.loads(l) for l in idx_path.read_text().splitlines() if l.strip()]
        secs = np.array([
            (lambda i: i.frames / i.samplerate)(sf.info(r["speech_file"])) for r in rows
        ])
        out = np.clip((secs * fps).astype(np.int64), 1, total_frames)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, out)
    if out.shape[0] != n_expected:
        raise SystemExit(
            f"{split}: speech-extent cache has {out.shape[0]} rows but the code arrays have "
            f"{n_expected}. Row alignment between the features index and the code cache is not "
            f"optional -- delete {cache} and let it rebuild."
        )
    return out


def truncate_to_speech(
    tokens: np.ndarray, durations: np.ndarray, lengths: np.ndarray, end_frame: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cut each run-length sequence at its clip's speech boundary.

    Walks the cumulative duration and keeps tokens up to `end_frame`, shortening the token that
    straddles the boundary rather than dropping it -- dropping would lose real speech from the
    end of the utterance, which is exactly the part whose coherence is in question.

    Returns arrays of the same (N, L) shape with the tail zeroed, plus the new lengths, so every
    downstream consumer (the dedup cache contract, `build_teacher_forcing`, `build_hier_batch`)
    keeps working unchanged.
    """
    n, l = tokens.shape
    tok = np.zeros_like(tokens)
    dur = np.zeros_like(durations)
    new_len = np.zeros(n, dtype=lengths.dtype)
    for i in range(n):
        budget = int(end_frame[i])
        spent = 0
        k = 0
        for j in range(int(lengths[i])):
            d = int(durations[i, j])
            if spent >= budget:
                break
            take = min(d, budget - spent)
            tok[i, k] = tokens[i, j]
            dur[i, k] = take
            spent += take
            k += 1
        if k == 0:                      # a clip shorter than one frame; keep one real token
            tok[i, 0] = tokens[i, 0]
            dur[i, 0] = 1
            k = 1
        new_len[i] = k
    return tok, dur, new_len
