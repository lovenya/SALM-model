#!/usr/bin/env python
"""
Reorganise generation lanes into a BY-CLIP listening bundle.

Generation writes one directory per lane, which is right for scoring and wrong for listening:
comparing four decoders on one scene means opening four folders and matching filenames by eye.
This inverts it -- one folder per clip, containing the scene, the true speech, and every
decoder's attempt, numbered so a plain alphabetical play-all lands in the right order:

    clip_01__A_motorcycle_is_started/
        00_INPUT_ambient_scene.wav      <- what the model heard, and all it heard
        01_REFERENCE_true_speech.wav    <- what a human actually said over that scene
        02_greedy_s11.wav ...           <- the decoders
        README.md                       <- the caption and the reference transcript

Play 00, then 01, then the rest. Without 00 and 01 in the same folder there is nothing to judge
"scene-appropriate" against, which is why earlier bundles were hard to assess.

NOTE ON GREEDY LANES. With top_k=1 the content codes are deterministic, so greedy_s11..s66
contain the SAME WORDS and differ only in diffusion and vocoder noise. They are six renderings
of one utterance, not six utterances -- useful for hearing renderer variability, misleading if
read as decoder diversity.

Usage:
  python scripts/a2s/build_listening_bundle.py --gen-root DIR --out DIR \
      [--lanes greedy w1.0 w1.5] [--max-clips 8]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path


def slug(text: str, n: int = 40) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", (text or "clip")).strip("_")
    return (s[:n] or "clip").rstrip("_")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lanes", nargs="*", default=None,
                    help="substrings selecting which lane dirs to include (default: all)")
    ap.add_argument("--max-clips", type=int, default=0, help="0 = every clip")
    args = ap.parse_args()

    root, out = Path(args.gen_root), Path(args.out)
    lane_dirs = sorted(d for d in root.iterdir()
                       if d.is_dir() and (d / "manifest.jsonl").exists())
    if args.lanes:
        lane_dirs = [d for d in lane_dirs if any(k in d.name for k in args.lanes)]
    if not lane_dirs:
        raise SystemExit(f"no lanes matched under {root}")

    # clip -> {lane_name: row}, plus the first row seen for the shared scene/reference audio
    clips: dict[str, dict] = defaultdict(dict)
    meta: dict[str, dict] = {}
    for d in lane_dirs:
        for line in (d / "manifest.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("status") != "success":
                continue
            clips[r["sample_id"]][d.name] = (d, r)
            meta.setdefault(r["sample_id"], r)

    ids = sorted(clips)
    if args.max_clips:
        ids = ids[: args.max_clips]
    out.mkdir(parents=True, exist_ok=True)

    n_files = 0
    for i, sid in enumerate(ids, 1):
        m = meta[sid]
        cdir = out / f"clip_{i:02d}__{slug(m.get('audio_cap'))}"
        cdir.mkdir(parents=True, exist_ok=True)

        # The scene and the reference are identical across lanes -- copy once, from any lane.
        any_dir, any_row = next(iter(clips[sid].values()))
        for src_key, dest in (("__input_ambient.wav", "00_INPUT_ambient_scene.wav"),
                              ("__reference.wav", "01_REFERENCE_true_speech.wav")):
            src = any_dir / "audio" / f"{sid}{src_key}"
            if src.exists():
                shutil.copy2(src, cdir / dest); n_files += 1

        for j, lane in enumerate(sorted(clips[sid]), start=2):
            d, r = clips[sid][lane]
            src = d / r["generated_file_rel"]
            if src.exists():
                shutil.copy2(src, cdir / f"{j:02d}_{lane}.wav"); n_files += 1

        lanes_listed = "\n".join(f"- `{j:02d}_{lane}.wav`"
                                 for j, lane in enumerate(sorted(clips[sid]), start=2))
        (cdir / "README.md").write_text(
            f"# {m.get('audio_cap', '(no caption)')}\n\n"
            f"**sample id:** `{sid}`\n\n"
            f"## Listen in this order\n\n"
            f"1. `00_INPUT_ambient_scene.wav` — the ambient recording. This is the model's "
            f"ONLY input. No speech, no text, no transcript.\n"
            f"2. `01_REFERENCE_true_speech.wav` — what a human actually said over this scene. "
            f"One valid answer, not the only one.\n"
            f"3. The generated lanes below.\n\n"
            f"**Reference transcript:** \"{m.get('speech_text', '')}\"\n\n"
            f"The model never sees this transcript. Judge the generations on whether they "
            f"suit the scene, not on whether they match these words — the scene-to-utterance "
            f"map is one-to-many.\n\n"
            f"## Lanes\n\n{lanes_listed}\n\n"
            f"`greedy_*` lanes share identical content codes and differ only in diffusion "
            f"noise: they are six renderings of ONE utterance. The `w*` lanes are genuinely "
            f"different utterances per seed.\n"
        )

    (out / "README.md").write_text(
        f"# Listening bundle\n\nFrom `{root.name}`, {len(ids)} clips, "
        f"{len(lane_dirs)} lanes each.\n\n"
        f"One folder per clip. In each: the ambient scene the model heard (`00_`), the true "
        f"human speech (`01_`), then the generations. Play in filename order.\n\n"
        f"`w` is the classifier-free guidance scale; `w1.0` is the unguided model. `greedy` is "
        f"argmax decoding — its six seeds share identical words and differ only in vocoder "
        f"noise.\n"
    )
    print(f"wrote {out}")
    print(f"  {len(ids)} clips, {len(lane_dirs)} lanes, {n_files} wav files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
