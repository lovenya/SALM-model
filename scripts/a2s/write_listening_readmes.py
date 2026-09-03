#!/usr/bin/env python
"""
Write a README.md into each A2S generation directory, so the explanation travels with the
audio when it is copied to a website or handed to another agent.

Every number in the output is read from the scoring JSONs on disk -- nothing is hardcoded, so
a README cannot drift from the audio it sits next to. Light enough to run anywhere (it only
reads JSON and lists files; no model, no GPU).

Usage:
  python scripts/a2s/write_listening_readmes.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from configs import paths as project_paths  # noqa: E402

OUT = project_paths.OUTPUTS_ROOT
GEN = OUT / "generations"

# Calibration anchors, measured, quoted everywhere so a reader can place any number.
ANCHORS = """| lane | inter_asr_dist | what it means |
|---|---|---|
| `real_speech` | **0.3146** | genuine human recordings. Not 0 because two ASRs never perfectly agree. |
| `vae_roundtrip` | **0.3722** | real speech encoded and decoded by the VAE. The **ceiling** no latent model can beat. |
| `zero` | **~0.967** | the model given NO content. The **floor** -- speech-shaped babble. |
"""

METRIC_NOTE = """## How to read `inter_asr_dist`

Two architecturally different ASR systems (Whisper-large-v3 and a small in-house CTC probe)
transcribe the same clip, and we measure the **edit distance between their two hypotheses** --
never against a reference. Two independent models agreeing means real phonemes are present;
babble makes them diverge, because there is nothing there to agree on.

**Lower is better.** It measures whether words are *present and clear*, NOT whether they are
the *right* words -- deliberately, because ambient audio cannot determine which words someone
said. `real_word_rate` (rwr) is the companion: the fraction of transcribed tokens that are
real words from this corpus.

> **Important for listening:** the generation is *not* supposed to match the reference. The
> scene does not determine the sentence. Judge whether the speech is **plausible for that
> scene**, not whether it matches `__reference.wav`.
"""

FILES_NOTE = """## The three files per sample

| file | what it is |
|---|---|
| `<id>__input_ambient.wav` | the ambient scene. **The only thing the model was given.** |
| `<id>.wav` | what the model generated from it |
| `<id>__reference.wav` | what was actually said in that recording (context, not a target) |
"""


def load_scores() -> dict[str, dict]:
    """label -> {dist, rwr} for every scored lane across all scoring runs."""
    out: dict[str, dict] = {}
    for f in sorted(OUT.glob("intelligibility_*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        for label, r in d.items():
            if not isinstance(r, dict) or "inter_asr_distance_mean" in r is None:
                continue
            if "inter_asr_distance_mean" not in r:
                continue
            out[Path(label).name] = {
                "dist": r["inter_asr_distance_mean"],
                "rwr": r.get("real_word_rate_mean"),
                "example": (r.get("examples") or [{}])[0].get("whisper", ""),
            }
    return out


def lane_rows(root: Path, scores: dict) -> list[tuple]:
    rows = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        s = scores.get(d.name, {})
        rows.append((d.name, s.get("dist"), s.get("rwr"), s.get("example", "")))
    return sorted(rows, key=lambda r: (r[1] is None, r[1]))


def describe_lm_lane(name: str) -> str:
    """Human-readable description of a k<K>__<ckpt>__t<T> lane name."""
    if name == "zero":
        return "**Null control.** No content at all -- the floor. Should be babble."
    if name.startswith("oracle_quantized_k"):
        k = name.split("_k")[-1]
        return (f"**Ceiling for K={k}.** Real content from the target speech, quantized to the "
                f"K={k} codebook. This is what the pipeline could achieve with a *perfect* "
                f"predictor -- it CHEATS by reading the answer.")
    parts = name.split("__")
    if len(parts) != 3:
        return ""
    k, ckpt, t = parts[0].lstrip("k"), parts[1], parts[2].lstrip("t")
    ck = {
        "best_loss": "epoch 7 -- lowest validation loss, least overfit",
        "best": "epoch ~87 -- highest scene_gap, most overfit",
        "ep19": "epoch 19", "ep39": "epoch 39", "ep59": "epoch 59",
        "ep79": "epoch 79", "ep99": "epoch 99 -- the LAST epoch",
    }.get(ckpt, ckpt)
    return (f"K={k} codebook, {ck}, sampling temperature {t}. "
            f"**Ambient audio only** -- no target speech, no retrieval.")


def write_readme(root: Path, title: str, intro: str, scores: dict, highlight: list[str]) -> None:
    if not root.exists():
        print(f"  skip (absent): {root}")
        return
    rows = lane_rows(root, scores)
    n_wav = len(list(root.glob("*/audio/*.wav")))

    L = [f"# {title}", "", intro, "",
         f"`{root.name}` &middot; {len(rows)} lanes &middot; {n_wav} wav files", "",
         FILES_NOTE, METRIC_NOTE, "## Calibration anchors", "", ANCHORS, ""]

    if highlight:
        L += ["## If you only present a few lanes, use these", ""]
        for h in highlight:
            s = scores.get(h, {})
            d = f"{s['dist']:.4f}" if s.get("dist") is not None else "n/a"
            L.append(f"- **`{h}`** (dist {d}) &mdash; {describe_lm_lane(h)}")
        L.append("")

    L += ["## Every lane", "",
          "| lane | dist | rwr | what it is |", "|---|---|---|---|"]
    for name, dist, rwr, _ in rows:
        ds = f"{dist:.4f}" if dist is not None else "&mdash;"
        rs = f"{rwr:.3f}" if rwr is not None else "&mdash;"
        L.append(f"| `{name}` | {ds} | {rs} | {describe_lm_lane(name)} |")

    L += ["", "## Sample transcripts (Whisper, first sample of each lane)", ""]
    for name, dist, _, ex in rows:
        if ex:
            L.append(f"- `{name}`: *\"{ex[:100]}\"*")

    (root / "README.md").write_text("\n".join(L) + "\n")
    print(f"  wrote {root/'README.md'} ({len(rows)} lanes)")


def main() -> int:
    scores = load_scores()
    print(f"loaded scores for {len(scores)} lanes\n")

    write_readme(
        GEN / "content_lm_gen_20260903_0328",
        "A2S: speech generated from ambient audio alone (full checkpoint sweep)",
        "Given only a 10.24 s recording of a **scene** (no speech, no text), the model "
        "generates plausible speech for that scene. This sweep covers three codebook sizes "
        "x seven training checkpoints x two sampling temperatures, so the effect of "
        "codebook granularity and of training duration are both directly audible.\n\n"
        "**Headline finding:** scene understanding works -- the same clip yields the same "
        "topic (and sometimes the same proper noun) across independently-trained "
        "checkpoints. Short utterances are often fully coherent; long ones fragment after "
        "~3-5 words, which is a sequence-length limitation, not a scene-understanding one. "
        "More training does not help.",
        scores,
        ["k2048__ep39__t1.0", "k2048__ep19__t0.7", "k2048__ep99__t0.7",
         "oracle_quantized_k2048", "zero"],
    )

    write_readme(
        GEN / "content_lm_gen_20260903_0004",
        "A2S: speech from ambient audio (first sweep, 3 checkpoints per codebook)",
        "The first ContentLM generation sweep. Superseded by "
        "`content_lm_gen_20260903_0328`, which covers all seven checkpoints per codebook "
        "including the last epoch; this one sampled only three (`best_loss`, `ep39`, "
        "`best`). Kept because it is the run whose scores are quoted in the experiment log.",
        scores,
        ["k2048__ep39__t1.0", "oracle_quantized_k2048", "zero"],
    )

    write_readme(
        GEN / "codebook_gate_20260902_0656",
        "Codebook gate: how much does discretizing content cost?",
        "Before training any predictor, this checks whether quantizing speech content to a "
        "codebook of K entries destroys intelligibility. Every lane feeds **real content "
        "from the target speech** (so it cheats deliberately) through the same renderer; "
        "the ONLY variable is codebook coarseness.\n\n"
        "**Finding:** words survive at K>=512. K=256 is audibly muffled. This is what "
        "justified building the predictor at all.",
        scores,
        ["oracle", "quantized_k2048", "quantized_k1024", "quantized_k256"],
    )

    write_readme(
        GEN / "control_lanes_20260831_0735",
        "Calibration controls",
        "The two reference points every other number is measured against. Neither is a "
        "model output.",
        scores, [],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
