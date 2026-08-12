"""
Package a multi-lane generation run into a listening tree.

Turns the per-lane output of `run_tamia_s2a_gen_evolution.sbatch` into one directory per sample,
so you can hear the input, the ground truth, and every model variant side by side:

    listen/<sample_id>/
        00_input_speech.wav          what the model was given
        01_reference_ambient.wav     the ground-truth paired scene
        02_trained_ep019.wav         our model at epoch 19
        03_trained_ep039.wav          ...
        07_trained_last.wav
        08_pristineunet_ep019.wav    same conditioner, PRETRAINED U-Net (the A/B)
        10_rung0_caption_ceiling.wav conditioned on the ground-truth caption (ceiling)
        11_rung1_clap_control.wav    CLAP-of-speech control
        NOTES.txt                    the transcript, the caption, and what to listen for

Symlinks by default because scratch is at ~91% space and 908K/1000K inodes, and the wavs already
exist. Pass --copy for a self-contained tree -- note `scp` does not follow symlinks unless you use
`rsync -avL`, so copy if you plan to scp.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

# Lane ordering for filenames: earliest training first, then the ablations, then the reference
# points. Anything unrecognised sorts last, alphabetically.
def lane_sort_key(name: str) -> tuple:
    if name.startswith("trained_ep"):
        return (0, int(name.removeprefix("trained_ep")))
    if name == "trained_best":
        return (1, 0)
    if name == "trained_last":
        return (1, 1)
    if name.startswith("pristineunet"):
        return (2, name)
    if name.startswith("rung0"):
        return (3, 0)
    if name.startswith("rung1"):
        return (3, 1)
    return (4, name)


def link_or_copy(src: Path, dst: Path, do_copy: bool) -> bool:
    if not src.exists():
        return False
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if do_copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-root", required=True, help="dir containing one subdir per lane")
    ap.add_argument("--copy", action="store_true",
                    help="copy instead of symlink (needed if you plan to scp rather than rsync -L)")
    args = ap.parse_args()

    gen_root = Path(args.gen_root)
    lanes = sorted(
        (d for d in gen_root.iterdir() if d.is_dir() and (d / "manifest.jsonl").exists()),
        key=lambda d: lane_sort_key(d.name),
    )
    if not lanes:
        print(f"ERROR: no lanes with manifest.jsonl under {gen_root}")
        return 1
    print(f"{len(lanes)} lanes: {', '.join(d.name for d in lanes)}")

    listen = gen_root / "listen"
    listen.mkdir(parents=True, exist_ok=True)

    # sample_id -> {lane: row}
    per_sample: dict[str, dict[str, dict]] = {}
    for lane in lanes:
        for line in (lane / "manifest.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("status") != "success":
                continue
            per_sample.setdefault(r["sample_id"], {})[lane.name] = r

    written = 0
    missing = 0
    for sid, by_lane in sorted(per_sample.items()):
        d = listen / sid
        d.mkdir(parents=True, exist_ok=True)
        any_row = next(iter(by_lane.values()))

        # Input speech and reference ambient are identical across lanes (same val rows).
        if not link_or_copy(Path(any_row["speech_file"]), d / "00_input_speech.wav", args.copy):
            print(f"  WARNING missing input speech for {sid}: {any_row['speech_file']}")
            missing += 1

        # Try every candidate in order. `reference_copy` points inside a generation dir and so
        # goes stale if that dir is ever moved (ours was); `reference_audio_file` points at the
        # dataset itself and survives. Falling through instead of trusting the first one is what
        # stops the reference silently vanishing from the listening tree.
        ref_candidates = [
            any_row.get("reference_copy"),
            any_row.get("reference_audio_file"),
        ]
        if not any(
            link_or_copy(Path(c), d / "01_reference_ambient.wav", args.copy)
            for c in ref_candidates if c
        ):
            print(f"  WARNING no reference ambient found for {sid}")
            missing += 1

        for i, lane in enumerate(lanes, start=2):
            r = by_lane.get(lane.name)
            if r is None:
                continue
            src = lane / r["generated_file_rel"] if r.get("generated_file_rel") else Path(r["generated_file"])
            if not src.exists():
                src = lane / "audio" / f"{sid}.wav"
            if not link_or_copy(src, d / f"{i:02d}_{lane.name}.wav", args.copy):
                missing += 1

        # Provenance per lane, so a listening session is never ambiguous about which weights
        # produced which file.
        prov = []
        for lane in lanes:
            s = lane / "summary.json"
            if s.exists():
                sm = json.loads(s.read_text())
                prov.append(
                    f"  {lane.name}: mode={sm.get('mode')} epoch={sm.get('checkpoint_epoch')} "
                    f"trained_unet={sm.get('unet_weights_loaded')}"
                )
        (d / "NOTES.txt").write_text(
            f"sample_id: {sid}\n\n"
            f"TRANSCRIPT (what the model heard):\n  {any_row.get('speech_text','')}\n\n"
            f"GROUND-TRUTH CAPTION (the scene it should imply):\n  {any_row.get('audio_cap','')}\n\n"
            "WHAT TO LISTEN FOR\n"
            "  00 is the input speech; 01 is the real scene that accompanied it.\n"
            "  The trained_* files should move toward 01 as the epoch number rises.\n"
            "  pristineunet_* uses the same conditioner with the PRETRAINED U-Net -- the\n"
            "    difference from its trained_* twin is what U-Net finetuning bought.\n"
            "  rung0 saw the ground-truth caption, so it is the ceiling.\n"
            "  rung1 is the CLAP control and is expected to sound generic.\n\n"
            "LANE PROVENANCE\n" + "\n".join(prov) + "\n"
        )
        written += 1

    index = [f"{len(per_sample)} samples, {len(lanes)} lanes", ""]
    for lane in lanes:
        s = lane / "summary.json"
        sm = json.loads(s.read_text()) if s.exists() else {}
        index.append(f"{lane.name:32s} epoch={sm.get('checkpoint_epoch')} "
                     f"trained_unet={sm.get('unet_weights_loaded')} "
                     f"rms={sm.get('rms_mean')}")
    (listen / "INDEX.txt").write_text("\n".join(index) + "\n")

    print(f"wrote {written} sample dirs to {listen}")
    if missing:
        print(f"WARNING: {missing} files could not be linked (missing sources)")
    print(f"mode: {'copied' if args.copy else 'symlinked'}")
    print(f"\npull down with:  rsync -avL <host>:{listen}/ ./listen/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
