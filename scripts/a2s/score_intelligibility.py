"""
Rank 0 of the F1/F2 plan (2026-08-26b): a CONTENT-INDEPENDENT intelligibility metric for
generated A2S audio.

WHY THIS EXISTS. WER-vs-ground-truth conflates two separate failures: F1 (which words SHOULD
be said -- information-theoretic, not fixable by architecture) and F2 (CAN the model render
crisp words at all, given content -- purely architectural). Both "right words, wrong content"
and pure babble score WER >= 1.0 against a fixed reference transcript, so WER cannot tell them
apart. That means O-F's result (a real, measured CTC-loss drop during training) cannot be
safely judged by WER alone -- it might have bought real phonetic content that the wrong-content
reference simply never rewards. This script measures F2 in isolation, without needing to know
what SHOULD have been said.

TWO SIGNALS, both content-independent by construction:

  1. Inter-ASR agreement. Transcribe with TWO architecturally different ASR systems --
     Whisper-large-v3 (attention-based encoder-decoder) and our own CTC probe (small
     conv+GRU+CTC, trained on AudioLDM's own mel frontend). Score the edit distance BETWEEN
     the two hypotheses, not against any reference. Two independent, differently-trained
     models converging on the same string is real evidence that actual phonemes are present
     -- babble makes them diverge, because there is nothing real there for either to agree on.
     Neither model is in any training loop this project runs (the CTC probe trains O-F's
     loss, not itself, and is frozen throughout), so this stays non-circular.

  2. Real-word rate. Fraction of Whisper's output tokens that appear in a reference
     vocabulary built from THIS PROJECT'S OWN train-split transcripts (17,596 real
     utterances) -- no external wordlist or download needed. A generated clip producing
     real, in-domain words (even if not the RIGHT ones) scores high; babble that Whisper
     hallucinates into English-shaped nonsense mostly does not.

Pre-registered predictions (written before any number from this script exists), from the
2026-08-26b research: on REAL target speech, inter-ASR distance ~0.2-0.3, real-word rate >0.9.
On the unconditional floor, ~0.9-1.0 / low. On A0/A1/A2/oracle (all pre-O-F), predicted
indistinguishable from the floor -- if that holds, it turns the night's "sounds like a foreign
language" listening tests into a real number. On A3/O-F: genuinely unknown -- this is the
first instrument capable of detecting whether the CTC-loss drop bought anything.

Usage:
    python scripts/a2s/score_intelligibility.py --dirs <gen_dir1> <gen_dir2> ... \
        --ctc-checkpoint /path/to/ctc_probe/checkpoints/checkpoint_best.pt --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from configs import paths as project_paths  # noqa: E402
from salm.a2s.asr import WhisperASR  # noqa: E402
from salm.a2s.ctc_probe import CTCProbeTranscriber  # noqa: E402
from salm.a2s.wer import normalize_text, word_edit_distance  # noqa: E402
from salm.data.manifest import load_manifest  # noqa: E402


def log(msg: str) -> None:
    print(msg, flush=True)


def build_reference_vocab(direction: str = "a2s", split: str = "train") -> set[str]:
    """
    Real-word reference, built from THIS project's own transcripts -- no download, no
    external wordlist. Deliberately the train split (17,596 rows): large enough for broad
    in-domain coverage, and using train rather than val/test avoids any risk of a generation
    lane's own held-out reference leaking into what counts as "real" for it.
    """
    rows = load_manifest(direction, split)
    vocab: set[str] = set()
    for r in rows:
        vocab.update(normalize_text(r.speech_text).split())
    return vocab


def real_word_rate(text: str, vocab: set[str]) -> float:
    words = normalize_text(text).split()
    if not words:
        return 0.0
    return sum(1 for w in words if w in vocab) / len(words)


def inter_asr_distance(hyp_a: str, hyp_b: str) -> float:
    """Normalized edit distance between two hypotheses -- NOT against any reference."""
    wa, wb = normalize_text(hyp_a).split(), normalize_text(hyp_b).split()
    denom = max(len(wa), len(wb), 1)
    return word_edit_distance(wa, wb) / denom


def score_one_dir(gen_dir: Path, whisper: WhisperASR, ctc: CTCProbeTranscriber,
                  vocab: set[str], batch_size: int) -> dict:
    manifest_path = gen_dir / "manifest.jsonl"
    if not manifest_path.exists():
        return {"error": f"no manifest.jsonl in {gen_dir}"}
    rows = [json.loads(l) for l in manifest_path.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("status") == "success"]
    if not rows:
        return {"error": f"no successful rows in {gen_dir}"}

    wavs, ids = [], []
    for r in rows:
        wav, _sr = sf.read(gen_dir / r["generated_file_rel"], dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wavs.append(wav)
        ids.append(r["sample_id"])

    whisper_hyps = whisper.transcribe(wavs, batch_size=batch_size)
    ctc_hyps = ctc.transcribe(wavs, batch_size=batch_size)

    dists = [inter_asr_distance(w, c) for w, c in zip(whisper_hyps, ctc_hyps)]
    rwrs = [real_word_rate(w, vocab) for w in whisper_hyps]

    return {
        "n_scored": len(rows),
        "inter_asr_distance_mean": float(np.mean(dists)),
        "inter_asr_distance_std": float(np.std(dists)),
        "real_word_rate_mean": float(np.mean(rwrs)),
        "real_word_rate_std": float(np.std(rwrs)),
        "examples": [
            {"sample_id": i, "whisper": w, "ctc_probe": c, "inter_asr_dist": round(d, 3),
             "real_word_rate": round(r, 3)}
            for i, w, c, d, r in zip(ids, whisper_hyps, ctc_hyps, dists, rwrs)
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--ctc-checkpoint", required=True,
                    help="frozen CTC probe checkpoint (checkpoint_best.pt from train_ctc_probe.py)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    log("Building real-word reference vocabulary from a2s/train transcripts (17,596 rows) ...")
    vocab = build_reference_vocab()
    log(f"    {len(vocab)} unique words")

    log("Loading Whisper-large-v3 ...")
    whisper = WhisperASR(project_paths.WHISPER_LARGE_V3_PATH, args.device)
    log(f"Loading CTC probe from {args.ctc_checkpoint} ...")
    ctc = CTCProbeTranscriber(args.ctc_checkpoint, args.device)

    results = {}
    t0 = time.time()
    for d in args.dirs:
        gen_dir = Path(d).resolve()
        try:
            label = str(gen_dir.relative_to(project_paths.OUTPUTS_ROOT))
        except ValueError:
            label = str(gen_dir)
        log(f"\n[{label}] scoring {gen_dir} ...")
        r = score_one_dir(gen_dir, whisper, ctc, vocab, args.batch_size)
        results[label] = r
        if "error" in r:
            log(f"    SKIPPED: {r['error']}")
            continue
        log(f"    inter_asr_dist={r['inter_asr_distance_mean']:.4f}  "
            f"real_word_rate={r['real_word_rate_mean']:.4f}  n={r['n_scored']}")
        for ex in r["examples"][:2]:
            log(f"      WHISPER: {ex['whisper'][:60]}")
            log(f"      CTC    : {ex['ctc_probe'][:60]}")

    log("\n" + "=" * 90)
    log("F2 INTELLIGIBILITY (content-independent -- does NOT require knowing WHAT should be said)")
    log("lower inter_asr_dist = more agreement = more likely real phonemes present")
    log("higher real_word_rate = more in-domain-vocabulary words produced")
    log("=" * 90)
    scored = {k: v for k, v in results.items() if "error" not in v}
    for label, r in sorted(scored.items(), key=lambda kv: kv[1]["inter_asr_distance_mean"]):
        log(f"  {label:55s} dist={r['inter_asr_distance_mean']:.4f}  "
            f"rwr={r['real_word_rate_mean']:.4f}  n={r['n_scored']}")
    log("=" * 90)
    log(f"elapsed {(time.time()-t0)/60:.1f} min")

    out_path = Path(args.out) if args.out else project_paths.OUTPUTS_ROOT / "intelligibility_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    log(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
