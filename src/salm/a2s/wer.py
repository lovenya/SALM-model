"""
Word-error-rate utilities, shared between the A2S gate (scripts/a2s/check_speech_stack.py)
and the CTC content-probe (scripts/a2s/train_ctc_probe.py, salm/train_ldm.py's O-F term).

Hand-rolled rather than `jiwer`: that package is not installed, compute nodes have no
internet, and installs are the user's to run. Plain Levenshtein over word tokens.
"""

from __future__ import annotations

import re

_PUNCT = re.compile(r"[^\w\s']")
_WS = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    """Lowercase, drop punctuation (keeping apostrophes), collapse whitespace."""
    s = _PUNCT.sub(" ", s.lower())
    return _WS.sub(" ", s).strip()


def word_edit_distance(ref: list[str], hyp: list[str]) -> int:
    if not ref:
        return len(hyp)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def corpus_wer(refs: list[str], hyps: list[str]) -> dict:
    """
    Corpus WER = total edits / total reference words.

    Corpus-level, not the mean of per-utterance WERs: a two-word utterance transcribed
    wrongly would otherwise contribute a WER of 1.0 with the same weight as a twenty-word
    one, which at small N is enough to move the verdict.
    """
    import numpy as np

    edits = words = 0
    per_utt = []
    for r, h in zip(refs, hyps):
        rt, ht = normalize_text(r).split(), normalize_text(h).split()
        d = word_edit_distance(rt, ht)
        edits += d
        words += len(rt)
        per_utt.append(d / max(len(rt), 1))
    return {
        "wer": edits / max(words, 1),
        "edits": edits,
        "ref_words": words,
        "per_utterance_wer_mean": float(np.mean(per_utt)) if per_utt else float("nan"),
    }


# ---------------------------------------------------------------------------------- CTC vocab
# Fixed, hand-picked rather than scanned from the corpus: normalize_text() above already
# strips everything down to this exact character set (lowercase letters, space, apostrophe),
# so scanning would only ever rediscover the same 28 symbols with extra ceremony. Index 0 is
# CTCLoss's blank by construction (torch.nn.CTCLoss defaults to blank=0).
CTC_BLANK = 0
_CTC_CHARS = " '" + "abcdefghijklmnopqrstuvwxyz"
CTC_VOCAB_SIZE = len(_CTC_CHARS) + 1  # +1 for blank
_CHAR_TO_ID = {c: i + 1 for i, c in enumerate(_CTC_CHARS)}
_ID_TO_CHAR = {i + 1: c for i, c in enumerate(_CTC_CHARS)}


def ctc_encode(text: str) -> list[int]:
    """Normalized transcript -> character ids (blank excluded; unknown chars dropped)."""
    return [_CHAR_TO_ID[c] for c in normalize_text(text) if c in _CHAR_TO_ID]


def ctc_greedy_decode(ids: list[int]) -> str:
    """Collapse repeats and drop blanks -- the standard CTC greedy decode, for eval only."""
    out = []
    prev = None
    for i in ids:
        if i != CTC_BLANK and i != prev:
            out.append(_ID_TO_CHAR.get(i, ""))
        prev = i
    return "".join(out)
