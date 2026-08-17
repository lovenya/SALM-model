"""
Score generated audio against the intended scene, per generation directory.

The guidance sweep produces audio at several scales and by itself proves nothing -- listening to
six directories is not a measurement. This turns them into one number per lane so the sweep has an
answer:

  clap_audio_cos  cosine(CLAP_audio(generated), CLAP_audio(true target clip))
  clap_text_cos   cosine(CLAP_audio(generated), CLAP_text(true caption))
  self_sim        mean pairwise cosine BETWEEN this lane's own generations

`self_sim` is the guard against a tempting misreading. If a lane's outputs are all near-identical
(mode collapse, or the condition being ignored so every sample decodes to the same ambience), its
clap_audio_cos can still look respectable while carrying no per-example information. High
clap_audio_cos WITH high self_sim means "one generic output that is vaguely close to everything",
which is a failure. What we want is high clap_audio_cos with self_sim well below it.

BASELINE, and this matters: 1/sqrt(512) = 0.0442 is the chance cosine for RANDOM UNIT VECTORS, and
it is the wrong reference for this data. Real CLAP target embeddings are strongly clustered -- their
measured mean pairwise cosine is 0.2927 on val and 0.2910 on train (from
features_clap/s2a/*/meta.json, which precompute_clap_targets.py computes for exactly this purpose).
So "predicts a random REAL scene" scores ~0.29, not ~0.04, and that is the honest no-information
line. The script reads it from meta.json and prints both.

Usage:
    python scripts/score_generations_clap.py --dirs <gen_dir> [<gen_dir> ...] \
        [--out <summary.json>]

Each <gen_dir> must be a salm.generate / generate_regress output directory, i.e. contain
manifest.jsonl with `generated_file_rel` and `sample_id`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from configs import paths as project_paths  # noqa: E402

# Analytic chance for random unit vectors. Kept only as a reference point -- it is NOT the right
# baseline for CLAP embeddings, which are strongly clustered. See the module docstring.
CHANCE_RANDOM = 1.0 / math.sqrt(512)
AUDIOLDM_CLIP_SECONDS = 10.24


def empirical_baseline(direction: str, split: str) -> float | None:
    """
    Mean pairwise cosine of the real CLAP targets = what "predicts a random REAL scene" scores.
    Computed at precompute time and stored in meta.json, so this is free.
    """
    p = project_paths.FEATURES_CLAP_ROOT / direction / split / "meta.json"
    if not p.exists():
        return None
    try:
        return float(json.loads(p.read_text())["pairwise_cosine_mean"])
    except Exception:
        return None


def log(m: str) -> None:
    print(m, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--direction", default="s2a")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)

    from salm.data.frontend import load_waveform_mono_16k
    from salm.generate import ClapAudioEncoder, load_pipeline

    # The audio tower for scoring generations, and the text tower (which lives on the pipeline)
    # for the caption side. Both frozen, both the non-fusion checkpoint AudioLDM was trained on.
    clap = ClapAudioEncoder(str(device))
    pipe = load_pipeline(str(device), torch.float32)
    from salm.generate import clap_text_embeddings

    # Ground truth for the split: sample_id -> (target audio path, caption).
    #
    # Prefer the precomputed features index when it exists, but fall back to the raw manifest.
    # Reading only the features index coupled scoring to the TRAINING precompute for no reason:
    # generate.py itself reads load_manifest() directly, so inference on a split works with no
    # precompute at all, and requiring the index here would have forced a pointless ~30 min latent
    # job before the test split could be scored. The manifest carries audio_file and audio_cap,
    # which is everything this needs.
    truth = {}
    idx = project_paths.FEATURES_ROOT / args.direction / args.split / "index.jsonl"
    if idx.exists():
        with idx.open() as f:
            for line in f:
                r = json.loads(line)
                truth[r["sample_id"]] = (r["audio_file"], r.get("audio_cap", ""))
        src = f"features index ({idx})"
    else:
        from salm.data.manifest import load_manifest

        for r in load_manifest(args.direction, args.split):
            truth[r.sample_id] = (str(r.audio_path), getattr(r, "audio_cap", "") or "")
        src = f"manifest (no features index for '{args.split}')"
    log(f"ground truth for {len(truth)} {args.split} rows, from {src}")

    def embed_audio(paths: list[str]) -> torch.Tensor:
        out = []
        for i in range(0, len(paths), args.batch_size):
            wavs = [
                load_waveform_mono_16k(p, target_seconds=AUDIOLDM_CLIP_SECONDS)
                for p in paths[i : i + args.batch_size]
            ]
            out.append(F.normalize(clap(wavs).float().cpu(), dim=-1))
        return torch.cat(out) if out else torch.empty(0, 512)

    results = {}
    for d in args.dirs:
        gd = Path(d)
        mf = gd / "manifest.jsonl"
        if not mf.exists():
            log(f"  SKIP {gd.name}: no manifest.jsonl")
            continue
        rows = [json.loads(l) for l in mf.open()]

        gen_paths, tgt_paths, caps, missing = [], [], [], 0
        for r in rows:
            rel = r.get("generated_file_rel")
            gp = gd / rel if rel else None
            sid = r.get("sample_id")
            if gp is None or not gp.exists() or sid not in truth:
                missing += 1
                continue
            tp, cap = truth[sid]
            gen_paths.append(str(gp))
            tgt_paths.append(tp)
            caps.append(cap)
        if not gen_paths:
            log(f"  SKIP {gd.name}: no usable rows ({missing} missing)")
            continue

        g = embed_audio(gen_paths)
        t = embed_audio(tgt_paths)
        with torch.no_grad():
            txt = clap_text_embeddings(pipe, caps, str(device)).float().cpu()

        a_cos = (g * t).sum(-1)
        t_cos = (g * txt).sum(-1)
        # Mean pairwise cosine among this lane's own generations, excluding the diagonal.
        sim = g @ g.t()
        n = sim.shape[0]
        self_sim = ((sim.sum() - sim.diag().sum()) / max(n * (n - 1), 1)).item() if n > 1 else 0.0

        # ------------------------------------------------------------- THE CONTROL
        # Added after the first sweep, because without it clap_audio_cos is uninterpretable.
        # Ambient clips are broadly similar to each other, so a generic output scores well against
        # ANY target. The question is never "is audio_cos high" but "is it higher for the RIGHT
        # target than for the wrong ones".
        #
        #   cross      = mean cosine of generated_i against target_j, i != j
        #   margin     = matched - mismatched. ~0 means the number carries no per-example
        #                information at all, however impressive it looks on its own.
        #   audio_r1   = fraction where the TRUE target is the nearest of all targets in the lane.
        #                Chance is 1/n. This is the sharpest per-example statement available.
        gt = g @ t.t()                                       # (n, n)
        if n > 1:
            off = (gt.sum() - gt.diag().sum()) / (n * (n - 1))
            cross = off.item()
            # `idx`, NOT `truth`: `truth` is the sample_id -> (audio_file, caption) dict built
            # above and read on every subsequent lane. Binding a LongTensor to that name made
            # lane 2's `sid not in truth` raise
            #   RuntimeError: Tensor.__contains__ only supports Tensor or scalar
            # because sample_id is a string -- so the scorer died after all generation was done.
            idx = torch.arange(n)
            audio_r1 = (gt.argmax(1) == idx).float().mean().item()
        else:
            cross, audio_r1 = float("nan"), float("nan")

        results[gd.name] = {
            "dir": str(gd),
            "n": len(gen_paths),
            "missing": missing,
            "clap_audio_cos": a_cos.mean().item(),
            "clap_audio_cos_std": a_cos.std().item() if n > 1 else 0.0,
            "clap_audio_cos_mismatched": cross,
            "margin": a_cos.mean().item() - cross,
            "audio_r1": audio_r1,
            "audio_r1_chance": 1.0 / n if n else float("nan"),
            "clap_text_cos": t_cos.mean().item(),
            "self_sim": self_sim,
        }
        # guidance_scale, when the lane recorded one, so the sweep can be read as a curve
        sj = gd / "summary.json"
        if sj.exists():
            try:
                s = json.loads(sj.read_text())
                for k in ("guidance_scale", "checkpoint_epoch", "mode", "seconds_per_sample"):
                    if k in s:
                        results[gd.name][k] = s[k]
            except Exception:
                pass
        log(f"  scored {gd.name}: n={len(gen_paths)}")

    base = empirical_baseline(args.direction, args.split)
    log("")
    log(f"baselines: random unit vectors +/-{CHANCE_RANDOM:.4f}  |  "
        f"random REAL target {base:.4f}" if base is not None else
        f"baselines: random unit vectors +/-{CHANCE_RANDOM:.4f}  (no meta.json for the real one)")
    log("  Use the REAL-target line. CLAP embeddings are clustered, so 0.0442 flatters everything.")
    log("")
    hdr = (f"{'lane':30s} {'s':>5} {'n':>4} {'matched':>8} {'mismat':>8} {'MARGIN':>8} "
           f"{'r1':>6} {'chance':>7} {'text':>7} {'self':>7}")
    log(hdr)
    log("-" * len(hdr))
    for name, r in sorted(results.items(), key=lambda kv: kv[1].get("guidance_scale", 0)):
        log(f"{name:30s} {r.get('guidance_scale', float('nan')):>5} {r['n']:>4} "
            f"{r['clap_audio_cos']:>8.4f} {r['clap_audio_cos_mismatched']:>8.4f} "
            f"{r['margin']:>8.4f} {r['audio_r1']:>6.3f} {r['audio_r1_chance']:>7.3f} "
            f"{r['clap_text_cos']:>7.4f} {r['self_sim']:>7.4f}")

    log("")
    log("READING THIS -- read MARGIN and r1 FIRST, not `matched`:")
    log("  `matched` alone is nearly meaningless. Ambient clips resemble each other, so a generic")
    log("  output scores well against ANY target. The question is whether it scores higher for the")
    log("  RIGHT one.")
    log("    MARGIN ~ 0 and r1 ~ chance -> the output carries NO per-example information. The")
    log("       model is emitting generic ambience regardless of what was said.")
    log("    MARGIN > 0 and r1 > chance -> generation genuinely tracks the individual utterance.")
    log("  Then, across a guidance sweep:")
    log("    MARGIN rising with s -> guidance is making the output track the intended scene.")
    log("    MARGIN flat across s -> the U-Net is not responding to the condition, so the")
    log("       conditioning INTERFACE is the bottleneck, not the conditioner.")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        log(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
