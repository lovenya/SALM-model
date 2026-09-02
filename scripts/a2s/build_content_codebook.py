#!/usr/bin/env python
"""
Fit ContentCodebook(s) over the training split's real CTC-probe content frames.

This is step 1 of the discrete-content plan and it is a GATE, not a deliverable. See
src/salm/a2s/content_codebook.py for why discrete content is the answer to the mean-collapse
that killed regression (2026-08-31) and both end-to-end runs (jobs 434308, 434417).

WHAT IT DOES
  1. Runs the frozen CTC probe over every training speech file -> (256, 512) content frames.
  2. Subsamples frames to a memory-safe pool and fits k-means codebooks at several K in one
     pass -- K is chosen from the measured reconstruction cost, not picked as a round number.
  3. Reports, per K: relative L2 reconstruction error, cosine similarity, dead codes, and
     usage perplexity (the effective number of codes actually used).

WHAT IT DOES NOT DO
  It does not tell you whether quantization preserves INTELLIGIBILITY. Cosine similarity of
  0.95 could still destroy phoneme identity, and this project has been burned repeatedly by a
  metric moving while the audio did not. That question is answered by generating with
  `--content-source oracle_quantized` through the proven content_concat_v4 renderer and
  scoring the audio -- which is what the accompanying sbatch does after this script.

Usage:
  python scripts/a2s/build_content_codebook.py --split train --codes 256 512 1024 2048
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from configs import paths as project_paths                        # noqa: E402
from salm.a2s.content_codebook import ContentCodebook, fit_codebook  # noqa: E402
from salm.a2s.ctc_probe import SimpleCTCASR                       # noqa: E402
from salm.data.frontend import (                                  # noqa: E402
    AUDIOLDM_CLIP_SECONDS,
    MelConfig,
    MelExtractor,
    load_waveform_mono_16k,
)


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--features-dir", default=None)
    ap.add_argument("--out-dir", default=None,
                    help="default: OUTPUTS_ROOT/content_codebook")
    ap.add_argument("--ctc-checkpoint", default=None)
    ap.add_argument("--codes", type=int, nargs="+", default=[256, 512, 1024, 2048],
                    help="fit one codebook per K, all from the same frame pool")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-frames", type=int, default=1_500_000,
                    help="frame pool cap for k-means; 1.5M x 512 fp32 ~= 3GB on GPU")
    ap.add_argument("--eval-frames", type=int, default=200_000,
                    help="held-out frames used only to score reconstruction, never fitted on")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    features_dir = Path(args.features_dir or project_paths.FEATURES_ROOT / "a2s" / args.split)
    out_dir = Path(args.out_dir or project_paths.OUTPUTS_ROOT / "content_codebook")
    ctc_ckpt = Path(
        args.ctc_checkpoint
        or project_paths.OUTPUTS_ROOT / "ctc_probe/ctc_probe_v1/checkpoints/checkpoint_best.pt"
    )

    index_path = features_dir / "index.jsonl"
    if not index_path.exists():
        log(f"ERROR: {index_path} not found.")
        return 1
    if not ctc_ckpt.exists():
        log(f"ERROR: CTC checkpoint not found: {ctc_ckpt}")
        return 1
    if (out_dir / "codebook_manifest.json").exists() and not args.overwrite:
        log(f"ERROR: {out_dir} already populated. Pass --overwrite.")
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in index_path.read_text().splitlines() if l.strip()]
    log(f"a2s/{args.split}: {len(rows)} rows")
    if "speech_file" not in rows[0]:
        log("ERROR: index.jsonl rows lack speech_file.")
        return 1

    # ---------------------------------------------------------------- CTC probe
    log(f"Loading frozen CTC probe from {ctc_ckpt} ...")
    ctc = SimpleCTCASR().to(args.device).eval()
    payload = torch.load(ctc_ckpt, map_location=args.device, weights_only=False)
    ctc.load_state_dict(payload["model"])
    for p in ctc.parameters():
        p.requires_grad_(False)
    # Same cuDNN-backward quirk workaround the training script documents; harmless here
    # (no backward is ever taken) but keeps behaviour identical to how content is produced
    # everywhere else, which is the point -- the codebook must live in the SAME space.
    ctc.rnn.train()
    mel = MelExtractor(MelConfig(norm_mode="none"), device=args.device)
    log(f"    content width = {ctc.encoder_dim}")

    # ---------------------------------------------------------------- collect frames
    # Reservoir-free approach: take a fixed stride of frames from every utterance so the pool
    # is spread across the whole corpus rather than the first N files. Every utterance
    # contributes equally, which matters because speakers and scenes are not shuffled on disk.
    total_frames_wanted = args.max_frames + args.eval_frames
    per_utt = max(1, total_frames_wanted // max(len(rows), 1))
    log(f"collecting ~{per_utt} frames/utterance -> ~{per_utt * len(rows)} total")

    pool: list[np.ndarray] = []
    rng = np.random.default_rng(args.seed)
    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            chunk = rows[start : start + args.batch_size]
            wavs = [
                load_waveform_mono_16k(
                    r["speech_file"], target_seconds=AUDIOLDM_CLIP_SECONDS, align="left"
                )
                for r in chunk
            ]
            wav_t = torch.from_numpy(np.stack(wavs)).float().to(args.device)
            content = ctc.encode(mel.batch(wav_t))          # (B, 256, D)
            b, t, d = content.shape
            take = min(per_utt, t)
            sel = torch.from_numpy(
                np.stack([rng.choice(t, size=take, replace=False) for _ in range(b)])
            ).to(args.device)
            picked = torch.gather(content, 1, sel.unsqueeze(-1).expand(b, take, d))
            pool.append(picked.reshape(-1, d).float().cpu().numpy())

            if (start // args.batch_size) % 40 == 0:
                done = min(start + args.batch_size, len(rows))
                log(f"    {done}/{len(rows)} utterances")

    frames = torch.from_numpy(np.concatenate(pool, axis=0))
    del pool
    log(f"frame pool: {tuple(frames.shape)}")

    # Held-out split BEFORE fitting: reconstruction error measured on frames the codebook was
    # fitted on would flatter every K and make the comparison between them meaningless.
    perm = torch.randperm(frames.shape[0], generator=torch.Generator().manual_seed(args.seed))
    frames = frames[perm]
    n_eval = min(args.eval_frames, frames.shape[0] // 5)
    eval_frames = frames[:n_eval].to(args.device)
    fit_frames = frames[n_eval : n_eval + args.max_frames].to(args.device)
    log(f"fit on {tuple(fit_frames.shape)}, evaluate on held-out {tuple(eval_frames.shape)}")

    # ---------------------------------------------------------------- fit each K
    manifest = {
        "split": args.split,
        "ctc_checkpoint": str(ctc_ckpt),
        "content_dim": int(ctc.encoder_dim),
        "fit_frames": int(fit_frames.shape[0]),
        "eval_frames": int(eval_frames.shape[0]),
        "seed": args.seed,
        "codebooks": {},
    }

    for k in args.codes:
        log(f"\n=== fitting K={k} ===")
        book, stats = fit_codebook(fit_frames, k, iters=args.iters, seed=args.seed, log=log)
        recon = book.reconstruction_stats(eval_frames.unsqueeze(0))
        stats.update(recon)
        path = out_dir / f"codebook_k{k}.pt"
        torch.save(
            {
                "codes": book.codes.cpu(),
                "num_codes": k,
                "content_dim": int(ctc.encoder_dim),
                "ctc_checkpoint": str(ctc_ckpt),
                "stats": stats,
            },
            path,
        )
        manifest["codebooks"][str(k)] = {**stats, "path": str(path)}
        log(f"    held-out rel_l2={recon['rel_l2']:.4f}  cosine={recon['cosine']:.4f}")
        log(f"    dead={stats['dead_codes']}  usage_perplexity={stats['usage_perplexity']:.1f}"
            f"  top1_share={stats['top1_share']:.4f}")

    (out_dir / "codebook_manifest.json").write_text(json.dumps(manifest, indent=2))

    # ---------------------------------------------------------------- summary
    log("\n" + "=" * 74)
    log("CODEBOOK RECONSTRUCTION (held-out frames, never fitted on)")
    log("=" * 74)
    log(f"{'K':>6}  {'rel_l2':>8}  {'cosine':>8}  {'dead':>6}  {'eff.codes':>10}  {'top1':>7}")
    for k in args.codes:
        s = manifest["codebooks"][str(k)]
        log(f"{k:>6}  {s['rel_l2']:>8.4f}  {s['cosine']:>8.4f}  {s['dead_codes']:>6}  "
            f"{s['usage_perplexity']:>10.1f}  {s['top1_share']:>7.4f}")
    log("=" * 74)
    log("These numbers do NOT establish intelligibility. The gate is the generation step:")
    log("quantized ORACLE content through the proven renderer must stay close to 0.389")
    log("(unquantized oracle) against the 0.372 VAE-roundtrip ceiling. If it collapses to")
    log("the ~0.95 noise floor, this representation is too lossy and no predictor can fix it.")
    log(f"\nwrote {out_dir}/codebook_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
