"""
Retrieval baseline: no model, no training. A lookup table.

For each val utterance, find the nearest TRAINING utterance in embedding space and return that
training example's paired ambient clip verbatim. This is the baseline a reviewer asks for, and it
tests something our generative model must justify: with only 15,033 pairs and a fairly closed set
of scene types, is a database enough?

Expect it to WIN on FAD, and do not be alarmed. Retrieval emits real recordings, so it sits on the
real data manifold by construction, while FAD/FD only measure distributional distance. It can score
beautifully while being wrong on every individual example. That is why this script also reports
per-example agreement between the retrieved caption and the true one -- retrieval winning FAD while
losing per-example agreement is a clean, reportable result rather than an embarrassment.

Two embedding spaces, because which one you retrieve in is itself a question:
  whisper  mean-pooled Whisper encoder features -- the same representation our conditioner consumes,
           so this is the honest "could a lookup replace the conditioner?" test
  clap     CLAP audio embedding of the speech -- also tells us, for free, whether CLAP separates
           these utterances at all (the rung1 premise we have been asserting without evidence)

Output is generate.py-compatible (audio/ + manifest.jsonl + summary.json), so the packaging,
reporting and eval-export scripts all work on it unchanged.

Usage:
    python scripts/retrieval_baseline.py --space whisper --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from configs import paths as project_paths  # noqa: E402
from salm.data.frontend import AUDIOLDM_SR, load_waveform_mono_16k  # noqa: E402
from salm.data.manifest import load_manifest  # noqa: E402

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "is", "are", "was", "were",
    "with", "by", "for", "as", "it", "its", "this", "that", "then", "followed", "while",
}


def log(msg: str) -> None:
    print(msg, flush=True)


def caption_tokens(text: str) -> set[str]:
    """Content words, for a cheap semantic-agreement proxy that needs no extra model."""
    return {w for w in "".join(c.lower() if c.isalnum() else " " for c in text).split()
            if w and w not in STOPWORDS}


def jaccard(a: str, b: str) -> float:
    ta, tb = caption_tokens(a), caption_tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


@torch.no_grad()
def embed_whisper(paths: list[str], device: str, batch_size: int = 16) -> torch.Tensor:
    """Masked mean-pool of Whisper encoder features -- what our conditioner sees before pooling."""
    from salm.encoders.whisper_encoder import WhisperSpeechEncoder

    enc = WhisperSpeechEncoder(project_paths.WHISPER_LARGE_V3_PATH, dtype=torch.float16,
                               device=device)
    out = []
    for i in range(0, len(paths), batch_size):
        chunk = paths[i : i + batch_size]
        wavs = [load_waveform_mono_16k(p) for p in chunk]
        maxlen = max(w.shape[0] for w in wavs)
        padded = np.zeros((len(wavs), maxlen), dtype=np.float32)
        lengths = []
        for j, w in enumerate(wavs):
            padded[j, : w.shape[0]] = w
            lengths.append(w.shape[0])
        feats, mask = enc.encode_with_mask(
            torch.from_numpy(padded).to(device), torch.tensor(lengths)
        )
        m = mask.unsqueeze(-1).to(feats.dtype)
        pooled = (feats * m).sum(1) / m.sum(1).clamp(min=1.0)
        out.append(F.normalize(pooled.float(), dim=-1).cpu())
        if (i // batch_size) % 20 == 0:
            log(f"    whisper {min(i+batch_size, len(paths))}/{len(paths)}")
    del enc
    torch.cuda.empty_cache()
    return torch.cat(out)


@torch.no_grad()
def embed_clap(paths: list[str], device: str, batch_size: int = 16) -> torch.Tensor:
    """CLAP audio embedding of the SPEECH (48 kHz, as LAION-CLAP expects)."""
    import librosa
    from transformers import ClapModel, ClapProcessor

    proc = ClapProcessor.from_pretrained(project_paths.CLAP_MODEL_ID)
    model = ClapModel.from_pretrained(project_paths.CLAP_MODEL_ID).to(device).eval()
    out = []
    for i in range(0, len(paths), batch_size):
        chunk = paths[i : i + batch_size]
        wavs = [
            librosa.resample(load_waveform_mono_16k(p), orig_sr=AUDIOLDM_SR,
                             target_sr=48000, res_type="soxr_hq")
            for p in chunk
        ]
        inputs = proc(audio=wavs, sampling_rate=48000, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        res = model.get_audio_features(**inputs)
        feats = res if torch.is_tensor(res) else res.pooler_output
        out.append(F.normalize(feats.float(), dim=-1).cpu())
        if (i // batch_size) % 20 == 0:
            log(f"    clap {min(i+batch_size, len(paths))}/{len(paths)}")
    del model
    torch.cuda.empty_cache()
    return torch.cat(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default="whisper", choices=["whisper", "clap"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--direction", default="s2a")
    ap.add_argument("--split", default="val")
    ap.add_argument("--num-samples", type=int, default=16, help="-1 for the whole split")
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--train-limit", type=int, default=0, help="0 = all training rows")
    ap.add_argument("--copy", action="store_true",
                    help="copy retrieved wavs instead of symlinking")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)

    train = load_manifest(args.direction, "train")
    val = load_manifest(args.direction, args.split)[args.start_index :]
    if args.num_samples > 0:
        val = val[: args.num_samples]
    if args.train_limit:
        train = train[: args.train_limit]
    log(f"retrieval in '{args.space}' space: {len(val)} queries against {len(train)} training rows")

    t0 = time.time()
    embed = embed_whisper if args.space == "whisper" else embed_clap
    log("embedding training speech ...")
    train_emb = embed([str(r.speech_path) for r in train], args.device)
    log("embedding query speech ...")
    val_emb = embed([str(r.speech_path) for r in val], args.device)

    # Both sides are L2-normalized, so a matmul gives cosine similarity.
    sims = val_emb @ train_emb.T
    best = sims.argmax(dim=1)
    best_sim = sims.gather(1, best.unsqueeze(1)).squeeze(1)

    rows = []
    jacs = []
    exact = 0
    for i, (q, bi) in enumerate(zip(val, best.tolist())):
        hit = train[bi]
        # The "generation" is the retrieved training example's ambient clip, verbatim.
        dst = out_dir / "audio" / f"{q.sample_id}.wav"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if args.copy:
            import shutil
            shutil.copy2(hit.audio_path, dst)
        else:
            os.symlink(Path(hit.audio_path).resolve(), dst)

        j = jaccard(q.audio_cap, hit.audio_cap)
        jacs.append(j)
        exact += int(q.audio_cap.strip().lower() == hit.audio_cap.strip().lower())

        wav = load_waveform_mono_16k(str(hit.audio_path))
        rows.append({
            "sample_id": q.sample_id,
            "status": "success",
            "mode": f"retrieval_{args.space}",
            "generated_file_rel": f"audio/{q.sample_id}.wav",
            "generated_file": str(dst),
            "reference_audio_file": str(q.audio_path),
            "speech_file": str(q.speech_path),
            "audio_cap": q.audio_cap,
            "speech_text": q.speech_text,
            "generated_rms": float(np.sqrt(np.mean(np.square(wav)))),
            # retrieval-specific provenance
            "retrieved_sample_id": hit.sample_id,
            "retrieved_caption": hit.audio_cap,
            "retrieved_speech_text": hit.speech_text,
            "retrieval_similarity": float(best_sim[i]),
            "caption_jaccard": j,
        })

    with (out_dir / "manifest.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    rms = [r["generated_rms"] for r in rows]
    summary = {
        "mode": f"retrieval_{args.space}",
        "space": args.space,
        "checkpoint": None,
        "checkpoint_epoch": None,
        "unet_weights_loaded": None,
        "direction": args.direction,
        "split": args.split,
        "num_samples": len(rows),
        "train_pool": len(train),
        "rms_mean": float(np.mean(rms)) if rms else 0.0,
        "rms_min": float(np.min(rms)) if rms else 0.0,
        "near_silent_count": int(sum(1 for r in rms if r < 1e-4)),
        # The per-example numbers that matter: FAD will flatter retrieval, these will not.
        "mean_retrieval_similarity": float(best_sim.mean()),
        "mean_caption_jaccard": float(np.mean(jacs)),
        "exact_caption_match": exact,
        "exact_caption_match_frac": exact / max(len(rows), 1),
        "elapsed_seconds": time.time() - t0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    log(f"\nretrieved {len(rows)} clips in {time.time()-t0:.1f}s")
    log(f"mean similarity to nearest training utterance: {summary['mean_retrieval_similarity']:.4f}")
    log(f"mean caption Jaccard (retrieved vs true):      {summary['mean_caption_jaccard']:.4f}")
    log(f"exact caption matches:                          {exact}/{len(rows)}")
    log("\nInterpretation: high similarity with LOW caption agreement means the embedding space")
    log("clusters utterances that do not share a scene -- retrieval looks confident and is wrong,")
    log("which is exactly the case where FAD flatters it and per-example metrics expose it.")
    log(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
