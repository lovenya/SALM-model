"""
Assemble a listening tree that spans DIFFERENT generation runs, with the measured scores attached.

`package_generation_evolution.py` handles one multi-lane directory. The comparison that matters now
spans several: the guidance sweep, the evolution lanes, the retrieval diagnostics and the T0
regression run all live in different places. This pulls chosen lanes from anywhere into one tree:

    listen/
        INDEX.md                     what each file is, and what it SCORED -- read this first
        <sample_id>/
            00_input_speech.wav      what the model was given
            01_reference_ambient.wav the ground-truth paired scene
            02_..., 03_...           one file per lane, in the order given on the command line
            about.txt                the transcript and the target caption for this sample

Copies by default rather than symlinking: this tree exists to be downloaded, and `scp` does not
follow symlinks. The whole set is a few tens of MB.

Usage:
    python scripts/package_listening_set.py --out <dir> --num 8 \
        --lane "ours_s7.5=/path/to/guidance_sweep_.../s7.5" \
        --lane "rung0_caption=/path/to/evolution_.../rung0_caption_ceiling" \
        [--scores /path/to/final_scores.json ...]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from configs import paths as project_paths  # noqa: E402


def log(m: str) -> None:
    print(m, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", action="append", required=True,
                    help="label=path/to/generation_dir  (repeatable; order sets file numbering)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num", type=int, default=8, help="how many samples to package")
    ap.add_argument("--direction", default="s2a")
    ap.add_argument("--split", default="val")
    ap.add_argument("--scores", action="append", default=[],
                    help="score json(s) from score_generations_clap.py, folded into INDEX.md")
    ap.add_argument("--symlink", action="store_true", help="symlink instead of copy")
    args = ap.parse_args()

    lanes = []
    for spec in args.lane:
        if "=" not in spec:
            raise SystemExit(f"--lane must be label=path, got {spec!r}")
        label, path = spec.split("=", 1)
        p = Path(path)
        if not (p / "manifest.jsonl").exists():
            log(f"  SKIP {label}: no manifest.jsonl at {p}")
            continue
        rows = {json.loads(l)["sample_id"]: json.loads(l) for l in (p / "manifest.jsonl").open()}
        lanes.append((label, p, rows))
    if not lanes:
        raise SystemExit("no usable lanes")

    # Sample selection follows the FIRST lane, which should therefore be the main model. Lanes that
    # were generated at a smaller sample count (older comparison runs) simply contribute where they
    # can, instead of capping the whole set to their length -- requiring every lane to be present
    # limited an earlier build to 8 samples because one lane only had 16 rows.
    #
    # Missing files are recorded explicitly in each about.txt so an absent file is never mistaken
    # for a silent or failed generation, which was the original reason for the strict rule.
    ordered = list(lanes[0][2])[: args.num]
    counts = {label: len(set(ordered) & set(rows)) for label, _, rows in lanes}
    log(f"{len(lanes)} lanes, packaging {len(ordered)} samples from '{lanes[0][0]}'")
    for label, n in counts.items():
        flag = "" if n == len(ordered) else f"   <-- only {n}/{len(ordered)}"
        log(f"    {label:22s} {n}{flag}")

    # Fall back to the manifest when a split has no precomputed features. generate.py reads the
    # manifest directly, so the TEST split can be generated and listened to without ever running a
    # latent-precompute job -- requiring the index here would have blocked that for no reason.
    truth = {}
    idx = project_paths.FEATURES_ROOT / args.direction / args.split / "index.jsonl"
    if idx.exists():
        with idx.open() as f:
            for line in f:
                r = json.loads(line)
                truth[r["sample_id"]] = r
        log(f"ground truth from features index ({args.split})")
    else:
        from salm.data.manifest import load_manifest

        for r in load_manifest(args.direction, args.split):
            truth[r.sample_id] = {
                "speech_file": str(r.speech_path),
                "audio_file": str(r.audio_path),
                "speech_text": getattr(r, "speech_text", "") or "",
                "audio_cap": getattr(r, "audio_cap", "") or "",
                "speaker_emotion": getattr(r, "speaker_emotion", "") or "?",
                "speaker_gender": getattr(r, "speaker_gender", "") or "?",
            }
        log(f"ground truth from manifest (no features index for '{args.split}')")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def place(src: Path, dst: Path) -> bool:
        if not src.exists():
            return False
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if args.symlink:
            dst.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dst)
        return True

    placed = 0
    for sid in ordered:
        d = out / sid
        d.mkdir(parents=True, exist_ok=True)
        t = truth.get(sid, {})

        if t.get("speech_file"):
            placed += place(Path(t["speech_file"]), d / "00_input_speech.wav")
        if t.get("audio_file"):
            placed += place(Path(t["audio_file"]), d / "01_reference_ambient.wav")

        missing = []
        for i, (label, p, rows) in enumerate(lanes, start=2):
            rel = rows.get(sid, {}).get("generated_file_rel")
            if rel and place(p / rel, d / f"{i:02d}_{label}.wav"):
                placed += 1
            else:
                missing.append(label)

        note = ""
        if missing:
            note = ("\nNOT PRESENT for this sample (that run generated fewer rows -- this is NOT a\n"
                    "failed or silent generation): " + ", ".join(missing) + "\n")
        (d / "about.txt").write_text(
            f"sample_id: {sid}\n"
            f"what was said: {t.get('speech_text', '?')}\n"
            f"target scene:  {t.get('audio_cap', '?')}\n"
            f"\nemotion: {t.get('speaker_emotion','?')}   gender: {t.get('speaker_gender','?')}\n"
            + note
        )

    # ---------------------------------------------------------------- INDEX.md
    scores = {}
    for s in args.scores:
        sp = Path(s)
        if sp.exists():
            try:
                scores.update(json.loads(sp.read_text()))
            except Exception:
                log(f"  WARN: could not read {sp}")

    lines = [
        "# Listening set",
        "",
        "Each folder is one sample. Play them in order: `00` is what the model heard, `01` is the",
        "scene it was supposed to produce, and the rest are the variants.",
        "",
        "## What each file is, and what it scored",
        "",
        "MARGIN = how much more the output resembles its OWN target than other targets. ~0 means the",
        "audio carries no per-example information, however good it sounds. r1 = how often the true",
        "target is the nearest of all targets (chance is 1/n).",
        "",
        "| file | what it is | MARGIN | r1 |",
        "|---|---|---|---|",
        "| `00_input_speech` | the speech the model was conditioned on | — | — |",
        "| `01_reference_ambient` | the ground-truth scene | — | — |",
    ]
    blurb = {
        "ours": "our model (O2: alignment loss)",
        "rung0": "conditioned on the GROUND-TRUTH caption — measured to be a floor, not a ceiling",
        "rung1": "CLAP audio embedding of the speech — the null control",
        "b0": "unconditional: the model with zero information",
        "pristine": "our conditioner + a NEVER-finetuned U-Net",
        "retrieval": "nearest training utterance's paired clip, returned verbatim (real audio)",
        "t0": "regression baseline, no diffusion — expect washed-out mush",
    }
    for i, (label, p, _) in enumerate(lanes, start=2):
        why = next((v for k, v in blurb.items() if k in label.lower()), "")
        s = scores.get(p.name, {})
        m = f"{s['margin']:.3f}" if "margin" in s else "—"
        r = f"{s['audio_r1']:.3f}" if "audio_r1" in s else "—"
        lines.append(f"| `{i:02d}_{label}` | {why} | {m} | {r} |")
    lines += [
        "",
        "## What to listen for",
        "",
        "- Does the scene match what was *implied* by the speech, not just sound like plausible ambience?",
        "- `rung0` and `rung1` should sound generic and similar to each other across samples — both",
        "  scored at chance. If they sound as good as ours, trust your ears over the metric.",
        "- `pristine` should sound unrelated to the speech: it scored a margin of ~0.",
        "- `t0` should sound smeared and low-energy — its latents carry less than half the variance",
        "  of real ones.",
    ]
    (out / "INDEX.md").write_text("\n".join(lines) + "\n")

    # ---------------------------------------------------- samples.json
    # One machine-readable file for the whole set, so a supervisor or a reviewer can read the
    # transcript and target caption per sample without opening 50 about.txt files.
    meta = {
        "split": args.split,
        "direction": args.direction,
        "num_samples": len(ordered),
        "lanes": [
            {"file_prefix": f"{i:02d}_{label}", "label": label, "source_dir": str(p),
             **{k: v for k, v in scores.get(p.name, {}).items()
                if k in ("n", "clap_audio_cos", "clap_audio_cos_mismatched", "margin",
                         "audio_r1", "audio_r1_chance", "self_sim", "guidance_scale",
                         "checkpoint_epoch")}}
            for i, (label, p, _) in enumerate(lanes, start=2)
        ],
        "samples": [
            {
                "sample_id": sid,
                "speech_text": truth.get(sid, {}).get("speech_text", ""),
                "reference_caption": truth.get(sid, {}).get("audio_cap", ""),
                "speaker_emotion": truth.get(sid, {}).get("speaker_emotion", ""),
                "speaker_gender": truth.get(sid, {}).get("speaker_gender", ""),
                "files": sorted(f.name for f in (out / sid).glob("*.wav")),
            }
            for sid in ordered
        ],
    }
    (out / "samples.json").write_text(json.dumps(meta, indent=2))
    log(f"wrote {out/'samples.json'}")

    log(f"placed {placed} files into {out}")
    log(f"read {out}/INDEX.md first")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
