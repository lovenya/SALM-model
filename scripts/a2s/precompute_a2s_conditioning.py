"""
Precompute everything the A2S training loop reads, in one pass over the split.

Three arrays, all row-aligned to `index.jsonl` from precompute_latents.py:

  cond_seq.npy      (N, T, D) fp16   CLAP frame features of the ambient INPUT  -- the condition
  cond_mask.npy     (N, T)    bool   which of those frames are real audio
  align_text.npy    (N, 512)  fp32   CLAP-TEXT of `speech_text`      -- O2 candidate B
  align_speech.npy  (N, 512)  fp32   CLAP-AUDIO of the target speech -- O2 candidate A

WHY PRECOMPUTE THE CONDITION AT ALL, when S2A deliberately did not. S2A kept raw speech in the
loop to leave the encoder and its pooling swappable. That reasoning does not survive contact
with CLAP: its sequence is 32x768 per clip, so a whole split is under a gigabyte, and the
conditioner still owns the pooling -- only the *encoder choice* gets baked in, and re-running
this script is minutes. What it buys is that the A2S training loop contains no frozen forward
pass and no CPU-bound decode, which is what makes a wide A2S sweep affordable. S2A's own
roadmap wants the same thing (OPEN_ITEMS item 7).

WHY TWO ALIGNMENT CANDIDATES, MEASURED RATHER THAN ASSUMED. The O2 auxiliary loss is the single
thing that made S2A work, so A2S needs its own version, and O2 pulls the conditioner output
toward an embedding of the TARGET. For S2A the target was ambience and CLAP-audio was the
obvious embedding of it. For A2S the target is speech, and this repo's standing assumption is
that CLAP-audio on speech encodes only that speech is happening -- near content-free. That
assumption has never actually been measured here, and it decides the design, so measure it:

  align_speech  CLAP-audio of the target speech. The direct S2A mirror. Expected to COLLAPSE --
                if its pairwise cosine spread is high (targets nearly collinear) it carries no
                per-example signal and cannot serve as an alignment target.
  align_text    CLAP-text of the transcript. Expected to WIN, and it also resolves a tension
                flagged in OPEN_ITEMS: it is already 512-d and already in CLAP space, which is
                the space the pretrained U-Net's `class_labels` slot understands, so aligning
                to it does not fight the U-Net the way a Whisper-space target would.
                Caveat to keep honest: CLAP's text tower was trained on audio CAPTIONS ("a dog
                barks"), not on arbitrary spoken sentences, so a transcript is somewhat out of
                its distribution. The spread number below is what tells us whether that
                matters in practice.

The comparison is `pairwise_cosine_mean`, the same statistic precompute_clap_targets.py already
reports (S2A ambient targets: 0.291 -- healthy spread). Near 1.0 means the targets are
collinear and the alignment loss would have nothing to teach.

Usage:
    python scripts/a2s/precompute_a2s_conditioning.py --split train
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from configs import paths as project_paths  # noqa: E402
from salm.data.frontend import AUDIOLDM_CLIP_SECONDS, load_waveform_mono_16k  # noqa: E402
from salm.encoders.clap_encoder import ClapAudioSequenceEncoder  # noqa: E402


def log(msg: str) -> None:
    print(msg, flush=True)


def spread(emb: np.ndarray, cap: int = 512) -> dict:
    """Off-diagonal pairwise cosine over a subsample -- does this space separate examples?"""
    x = emb[:: max(1, len(emb) // cap)][:cap]
    x = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)
    sim = x @ x.T
    off = sim[~np.eye(len(sim), dtype=bool)]
    return {"pairwise_cosine_mean": float(off.mean()),
            "pairwise_cosine_std": float(off.std())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--features-dir", default=None,
                    help="latents dir holding index.jsonl (default: FEATURES_ROOT/a2s/<split>)")
    ap.add_argument("--out-dir", default=None,
                    help="default: OUTPUTS_ROOT/features_a2s_cond/<split>")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip-speech-align", action="store_true",
                    help="skip the CLAP-audio-of-target-speech candidate (halves the forwards)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    features_dir = Path(args.features_dir or project_paths.FEATURES_ROOT / "a2s" / args.split)
    out_dir = Path(
        args.out_dir or project_paths.OUTPUTS_ROOT / "features_a2s_cond" / args.split
    )

    index_path = features_dir / "index.jsonl"
    if not index_path.exists():
        log(f"ERROR: {index_path} not found. Run:")
        log(f"  python scripts/precompute_latents.py --direction a2s --target speech "
            f"--split {args.split}")
        return 1
    if (out_dir / "cond_seq.npy").exists() and not args.overwrite:
        log(f"ERROR: {out_dir} already populated. Pass --overwrite.")
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in index_path.read_text().splitlines() if l.strip()]
    log(f"a2s/{args.split}: {len(rows)} rows from {index_path}")

    # Same assertion precompute_clap_targets.py makes. A silent misalignment here would pair
    # every condition with the wrong target and still train to a plausible-looking loss.
    for i, r in enumerate(rows):
        if r["latent_index"] != i:
            log(f"ERROR: index.jsonl not ordered by latent_index at row {i} "
                f"(got {r['latent_index']}).")
            return 1
    missing = [k for k in ("audio_file", "speech_file", "speech_text") if k not in rows[0]]
    if missing:
        log(f"ERROR: index.jsonl rows lack {missing}; re-run precompute_latents.py.")
        return 1

    log("Loading CLAP (laion/clap-htsat-unfused) ...")
    from transformers import ClapModel, ClapProcessor

    processor = ClapProcessor.from_pretrained(project_paths.CLAP_MODEL_ID)
    clap = ClapModel.from_pretrained(project_paths.CLAP_MODEL_ID).to(args.device).eval()
    for p in clap.parameters():
        p.requires_grad_(False)

    # The sequence encoder wraps the SAME audio tower; sharing the loaded weights keeps one
    # copy in VRAM and guarantees the sequence features and the pooled embeddings come from
    # identical parameters.
    encoder = ClapAudioSequenceEncoder(device=args.device, audio_model=clap.audio_model)

    cond_seq = None
    cond_mask = None
    align_text = np.zeros((len(rows), int(clap.config.projection_dim)), dtype=np.float32)
    align_speech = (
        None if args.skip_speech_align
        else np.zeros((len(rows), int(clap.config.projection_dim)), dtype=np.float32)
    )

    t0 = time.time()
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        n = len(batch)

        # ---- condition: the ambient INPUT clip, on the same 10.24 s canvas as S2A used
        amb, amb_len = [], []
        for r in batch:
            w = load_waveform_mono_16k(r["audio_file"], target_seconds=AUDIOLDM_CLIP_SECONDS)
            amb.append(w)
            # The FULL canvas, not each clip's true duration. The mask must be identical at
            # generation time, where only the padded clip is available, and a silent
            # train/inference mask mismatch is worse than pooling a little padding. See the
            # note in ClapAudioSequenceEncoder.encode_with_mask.
            amb_len.append(AUDIOLDM_CLIP_SECONDS * 16000)
        amb_t = torch.from_numpy(np.stack(amb)).float()
        feats, mask = encoder.encode_with_mask(
            amb_t, torch.tensor(amb_len, dtype=torch.long)
        )

        if cond_seq is None:
            T, D = int(feats.shape[1]), int(feats.shape[2])
            log(f"    CLAP sequence: T={T} frames  D={D}  "
                f"({AUDIOLDM_CLIP_SECONDS/T:.3f}s per frame)")
            cond_seq = np.zeros((len(rows), T, D), dtype=np.float16)
            cond_mask = np.zeros((len(rows), T), dtype=bool)
        cond_seq[start : start + n] = feats.cpu().numpy().astype(np.float16)
        cond_mask[start : start + n] = mask.cpu().numpy()

        # ---- alignment candidate B: CLAP-text of the transcript
        texts = [r.get("speech_text", "") or " " for r in batch]
        ti = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        ti = {k: v.to(args.device) for k, v in ti.items()}
        with torch.no_grad():
            out = clap.get_text_features(**ti)
            tf = out if torch.is_tensor(out) else out.pooler_output
            align_text[start : start + n] = F.normalize(tf, dim=-1).float().cpu().numpy()

        # ---- alignment candidate A: CLAP-audio of the TARGET speech
        if align_speech is not None:
            import librosa

            sp = []
            for r in batch:
                w = load_waveform_mono_16k(
                    r["speech_file"], target_seconds=AUDIOLDM_CLIP_SECONDS, align="left"
                )
                sp.append(librosa.resample(w, orig_sr=16000, target_sr=48000,
                                           res_type="soxr_hq")[:480000])
            si = processor(audio=sp, sampling_rate=48000, return_tensors="pt")
            si = {k: v.to(args.device) for k, v in si.items()}
            with torch.no_grad():
                out = clap.get_audio_features(**si)
                af = out if torch.is_tensor(out) else out.pooler_output
                align_speech[start : start + n] = (
                    F.normalize(af, dim=-1).float().cpu().numpy()
                )

        done = start + n
        if done % (args.batch_size * 20) == 0 or done == len(rows):
            rate = done / max(time.time() - t0, 1e-6)
            log(f"    {done}/{len(rows)}  {rate:.1f} rows/s  "
                f"eta {(len(rows)-done)/max(rate,1e-6)/60:.1f} min")

    np.save(out_dir / "cond_seq.npy", cond_seq)
    np.save(out_dir / "cond_mask.npy", cond_mask)
    np.save(out_dir / "align_text.npy", align_text)
    if align_speech is not None:
        np.save(out_dir / "align_speech.npy", align_speech)

    meta = {
        "direction": "a2s",
        "split": args.split,
        "rows": len(rows),
        "clap_model_id": project_paths.CLAP_MODEL_ID,
        "features_dir": str(features_dir),
        "cond_seq_shape": list(cond_seq.shape),
        "cond_feature_dim": int(cond_seq.shape[2]),
        "cond_frames": int(cond_seq.shape[1]),
        "cond_seconds_per_frame": AUDIOLDM_CLIP_SECONDS / int(cond_seq.shape[1]),
        "cond_mask_all_true_frac": float(cond_mask.all(axis=1).mean()),
        "align_text": spread(align_text),
        "elapsed_seconds": time.time() - t0,
    }
    if align_speech is not None:
        meta["align_speech"] = spread(align_speech)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    log(f"\nwrote {out_dir}")
    log(f"  cond_seq  {cond_seq.shape} fp16  "
        f"({cond_seq.nbytes/1e9:.2f} GB)")
    log("\n--- which alignment target should O2 use? (S2A ambient reference: 0.291) ---")
    log(f"  align_text   (CLAP-text of transcript)   pairwise cos "
        f"{meta['align_text']['pairwise_cosine_mean']:.4f} "
        f"+/- {meta['align_text']['pairwise_cosine_std']:.4f}")
    if align_speech is not None:
        log(f"  align_speech (CLAP-audio of speech)      pairwise cos "
            f"{meta['align_speech']['pairwise_cosine_mean']:.4f} "
            f"+/- {meta['align_speech']['pairwise_cosine_std']:.4f}")
        a, b = (meta["align_text"]["pairwise_cosine_mean"],
                meta["align_speech"]["pairwise_cosine_mean"])
        log("")
        if b > 0.9 and a < 0.9:
            log("  => CONFIRMED, with a number: CLAP-audio on speech is near-collinear and")
            log("     carries no per-example signal. Use align_text. This is the first direct")
            log("     measurement of an assumption the repo has relied on since 2026-08-06.")
        elif b < a:
            log("  => SURPRISE: CLAP-audio of the speech separates BETTER than CLAP-text of")
            log("     the transcript. Do not just take align_text; look at both before choosing.")
        else:
            log("  => align_text separates better, as expected, but not by the wide margin the")
            log("     'CLAP is blind to speech' story predicts. Worth a second look.")
    log(f"\nelapsed {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
