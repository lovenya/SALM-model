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

Chance reference for a 512-d unit-vector cosine is +/-1/sqrt(512) = +/-0.0442, and the script
prints it alongside every number.

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

CHANCE = 1.0 / math.sqrt(512)
AUDIOLDM_CLIP_SECONDS = 10.24


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

    # Ground truth for the split: sample_id -> (target audio path, caption)
    feat_dir = project_paths.FEATURES_ROOT / args.direction / args.split
    truth = {}
    with (feat_dir / "index.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            truth[r["sample_id"]] = (r["audio_file"], r.get("audio_cap", ""))
    log(f"ground truth for {len(truth)} {args.split} rows")

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

        results[gd.name] = {
            "dir": str(gd),
            "n": len(gen_paths),
            "missing": missing,
            "clap_audio_cos": a_cos.mean().item(),
            "clap_audio_cos_std": a_cos.std().item() if n > 1 else 0.0,
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

    log(f"\nchance for a 512-d cosine: +/-{CHANCE:.4f}\n")
    hdr = f"{'lane':34s} {'s':>5} {'n':>4} {'audio_cos':>10} {'text_cos':>9} {'self_sim':>9}"
    log(hdr)
    log("-" * len(hdr))
    for name, r in sorted(results.items(), key=lambda kv: kv[1].get("guidance_scale", 0)):
        log(f"{name:34s} {r.get('guidance_scale', float('nan')):>5} {r['n']:>4} "
            f"{r['clap_audio_cos']:>10.4f} {r['clap_text_cos']:>9.4f} {r['self_sim']:>9.4f}")

    log("")
    log("READING THIS:")
    log("  audio_cos rising with s = guidance is making the output track the intended scene.")
    log("  audio_cos flat across s  = the U-Net is not responding to the condition; the")
    log("     conditioning INTERFACE is the bottleneck, not the conditioner.")
    log("  high audio_cos WITH high self_sim = one generic output close to everything, which")
    log("     is a failure however good audio_cos looks. Want self_sim well below audio_cos.")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        log(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
