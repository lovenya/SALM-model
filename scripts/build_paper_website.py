#!/usr/bin/env python3
"""Build the static SpeechAudioLM paper website from existing, completed artifacts.

This performs no inference and no metric computation. It joins the curated survey
bundle with the final-scientific evaluation summaries/ledgers, copies a selected
number of browser-compatible WAVs, and emits a GitHub Pages-ready directory.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


DEFAULT_SURVEY = Path("/scratch/lovenya/salm_model_outputs/user_survey_bundle_20260917")
DEFAULT_EVALS = Path("/scratch/lovenya/salm_migration_clean/results/evals/stages")
DEFAULT_EVAL_REPO = Path("/scratch/lovenya/SpeechAudioLM")
DEFAULT_RESULTS = Path("/scratch/lovenya/salm_migration_clean/results")
DEFAULT_MISMATCH_REVIEW = Path(
    "/scratch/lovenya/salm_migration_clean/exports/geval_mismatch_review_20260917_2138/review_data.json"
)
TEMPLATE = Path(__file__).resolve().parents[1] / "paper_companion"


METRICS = {
    "s2a": {
        "ours": {
            "root": "s2a/s2a/salm_e2e/ttas/test/s2a_clean_v2_test_ep70_seed1337_steps100",
            "label": "SpeechAudioLM",
            "detail": "direct model",
            "test_count": 1878,
        },
        "cascade": {
            "root": "s2a/t2a/cascaded/ttas/test/20260618_1211",
            "label": "Text cascade",
            "detail": "ASR → LLM → Audio-Omni",
            "test_count": 1878,
        },
    },
    "a2s": {
        "ours": {
            "root": "a2s/a2s/salm_e2e/ttas/test/hier_k2048_s12_ep12_seed1337_steps100",
            "label": "SpeechAudioLM · hierarchical",
            "detail": "hierarchical · K=2048",
            "test_count": 2197,
        },
        "flat": {
            "root": "a2s/a2s/salm_e2e/ttas/test/flat_k2048_ep11_seed1337_steps100",
            "label": "SpeechAudioLM · flat",
            "detail": "flat · K=2048",
            "test_count": 2197,
        },
        "cascade": {
            "root": "a2s/t2s/cascaded/ttas/test/20260618_1211",
            "label": "Text cascade",
            "detail": "captioner → LLM → CosyVoice",
            "test_count": 2197,
        },
    },
}

SAMPLE_METRICS = ("utmos", "dnsmos")
CRITERION_ORDER = {
    "a2s": ("target_modality_success", "perceptual_audio_quality", "scenario_plausibility", "controlled_detail", "expressive_delivery_fit"),
    "s2a": ("target_modality_success", "perceptual_audio_quality", "scenario_plausibility", "controlled_detail", "contextual_acoustic_fit"),
}


def read_jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read_selected_jsonl(path: Path, sample_ids: set[str]):
    """Parse only selected rows from a potentially very large one-line JSONL ledger."""
    with path.open() as handle:
        for line in handle:
            match = re.search(r'"sample_id"\s*:\s*"([^"]+)"', line[:600])
            if match and match.group(1) in sample_ids:
                yield json.loads(line)


def latest_summary(root: Path, metric: str) -> tuple[Path, dict] | None:
    summaries = sorted((root / metric).glob("*/summary.json"))
    valid = []
    for path in summaries:
        payload = json.loads(path.read_text())
        validation = payload.get("errors", {}).get("validation", {})
        if payload.get("status") == "complete" and validation.get("scientifically_valid") is True:
            valid.append((path, payload))
    return valid[-1] if valid else None


def geval_runs(evals: Path, direction: str, system: str) -> list[Path]:
    if direction == "s2a" and system == "ours":
        pattern = "s2a/s2a/salm_e2e/ttas/test/s2a_clean_v2_test_ep70_seed1337_steps100_shard_*/geval_audio_plausibility/*/summary.json"
    elif direction == "a2s" and system in {"ours", "flat"}:
        planner = "hier_k2048_s12_ep12" if system == "ours" else "flat_k2048_ep11"
        pattern = f"a2s/a2s/salm_e2e/ttas/test/salm_a2s_flat11_hier12_geval_v6_20260915_v2_{planner}_shard_*/geval_audio_plausibility/*/summary.json"
    else:
        stage = "t2a" if direction == "s2a" else "t2s"
        pattern = f"{direction}/{stage}/cascaded/ttas/test/20260618_1211/geval_audio_plausibility/cascaded_geval_audio_v6_plausibility_20260916_{direction}_shard_*/summary.json"
    return sorted(evals.glob(pattern))


def compact_judgment(row: dict) -> dict:
    details = row["output"].get("details", {})
    criteria = {}
    for name, value in details.get("criteria", {}).items():
        criteria[name] = {"score": value.get("score"), "feedback": value.get("feedback", "")}
    source = row.get("input", {}).get("source", {})
    generated = row.get("input", {}).get("generated", {})
    return {
        "score": row["output"].get("score"),
        "criteria": criteria,
        "rawContext": "\n".join((
            "Prompt version: geval_audio_plausibility_v6",
            f"Direction: {source.get('direction', '')}",
            f"Input kind: {Path(source.get('source_audio_file', '')).parent.name or 'audio'}",
            f"Generated kind: {generated.get('generated_audio_kind', '')}",
            "Evaluation variant: plausibility (reference-free)",
        )),
    }


def collect_geval(
    evals: Path, direction: str, system: str, test_ids: set[str], prompt_ids: set[str]
) -> tuple[float, int, int, dict[str, dict]]:
    paths = geval_runs(evals, direction, system)
    if not paths:
        raise FileNotFoundError(f"No G-Eval summaries for {direction}/{system}")
    rows: dict[str, dict] = {}
    seen: set[str] = set()
    for summary_path in paths:
        summary = json.loads(summary_path.read_text())
        if summary.get("status") != "complete" or not summary.get("errors", {}).get("validation", {}).get("scientifically_valid"):
            raise ValueError(f"Not final-scientific: {summary_path}")
        for row in read_jsonl(summary_path.parent / "manifest.jsonl"):
            sample_id = row["sample_id"]
            if sample_id not in test_ids:
                continue
            if sample_id in seen:
                raise ValueError(f"Duplicate G-Eval row for {direction}/{system}: {sample_id}")
            seen.add(sample_id)
            if row.get("status") == "success" and isinstance(row.get("output", {}).get("score"), (int, float)):
                rows[row["sample_id"]] = compact_judgment(row)
        prompt_ledger = summary_path.parent / "artifacts/dedicated-script/geval_audio_plausibility/criterion_judgments.jsonl"
        if prompt_ledger.is_file():
            for criterion_row in read_selected_jsonl(prompt_ledger, prompt_ids):
                sample_id = criterion_row["sample_id"]
                if sample_id not in prompt_ids or sample_id not in rows or criterion_row.get("status") != "success":
                    continue
                criterion = criterion_row["criterion_id"]
                if criterion in rows[sample_id]["criteria"]:
                    prompt = criterion_row.get("prompt", {})
                    if prompt.get("prompt_version") != "geval_audio_plausibility_v6":
                        raise ValueError(f"Unexpected prompt version: {prompt_ledger}")
                    rows[sample_id]["criteria"][criterion].update({
                        "prompt": prompt.get("rendered_user_message", ""),
                        "promptVersion": prompt["prompt_version"],
                        "judge": Path(criterion_row.get("judge", {}).get("model", "")).name,
                    })
    missing = test_ids - seen
    allowed_missing = 33 if direction == "a2s" and system == "cascade" else 0
    if len(missing) != allowed_missing:
        raise ValueError(f"G-Eval omitted {len(missing)} clean-test rows for {direction}/{system}")
    scores = [row["score"] for row in rows.values()]
    if not scores:
        raise ValueError(f"No successful G-Eval rows for {direction}/{system}")
    return sum(scores) / len(scores), len(test_ids), len(scores), rows


def clean_test_ids(eval_repo: Path, direction: str) -> set[str]:
    path = eval_repo / f"data/manifests/clean_v2/{direction}/ttas_test.jsonl"
    ids = {Path(row["audio_file"]).stem for row in read_jsonl(path)}
    expected = METRICS[direction]["ours"]["test_count"]
    if len(ids) != expected:
        raise ValueError(f"Unexpected {direction.upper()} clean-test count: {len(ids)} != {expected}")
    return ids


def dataset_reasoning(eval_repo: Path, sample_ids: set[str]) -> dict[str, str]:
    """Read the saved Qwen3 pairing rationale from the exact clean-v2 test rows."""
    reasons = {}
    for direction in ("a2s", "s2a"):
        path = eval_repo / f"data/manifests/clean_v2/{direction}/ttas_test.jsonl"
        found = {}
        for row in read_jsonl(path):
            sample_id = Path(row["audio_file"]).stem
            if sample_id in sample_ids:
                value = row.get("reasoning", "").strip()
                if not value:
                    raise ValueError(f"Missing pairing rationale for {sample_id}: {path}")
                found[sample_id] = value
        if set(found) != sample_ids:
            raise ValueError(f"Missing {direction} test rationales: {sample_ids - set(found)}")
        if reasons and found != reasons:
            raise ValueError("Direction-specific test manifests disagree on pairing rationale")
        reasons = found
    return reasons


def collect_scalar_metric(root: Path, metric: str) -> tuple[float, int, dict[str, float]] | None:
    found = latest_summary(root, metric)
    if not found:
        return None
    summary_path, summary = found
    values = {}
    if summary["metric"]["level"] == "sample":
        for row in read_jsonl(summary_path.parent / "manifest.jsonl"):
            score = row.get("output", {}).get("score")
            if row.get("status") == "success" and isinstance(score, (int, float)):
                values[row["sample_id"]] = score
    return summary["scores"]["primary"]["value"], summary["samples"]["included"], values


def criterion_means(rows: dict[str, dict], direction: str) -> dict[str, dict]:
    means = {}
    for name in CRITERION_ORDER[direction]:
        values = [row["criteria"][name]["score"] for row in rows.values()
                  if isinstance(row.get("criteria", {}).get(name, {}).get("score"), (int, float))]
        if not values:
            raise ValueError(f"No {direction} scores for criterion {name}")
        means[name] = {"mean": sum(values) / len(values)}
    return means


def collect_ground_truth(eval_repo: Path, results: Path, direction: str, test_ids: set[str]) -> dict:
    policy = json.loads((eval_repo / "configs/ttas_geval_audio_v6_clean_v2.json").read_text())
    merged = results / policy["source_result"]["merged_root_relative_to_results"]
    path = merged / f"dataset_test_{direction}/backend_output.jsonl"
    rows = {}
    for row in read_jsonl(path):
        sample_id = row["sample_id"]
        if sample_id not in test_ids:
            continue
        if sample_id in rows:
            raise ValueError(f"Duplicate ground-truth judgment: {direction}/{sample_id}")
        if row.get("split") != "test" or row.get("prompt_version") != "geval_audio_plausibility_v6":
            raise ValueError(f"Wrong ground-truth split or version: {direction}/{sample_id}")
        if row.get("status") == "complete" and isinstance(row.get("final_score"), (int, float)):
            rows[sample_id] = {"score": row["final_score"], "criteria": row["criteria"]}
    if set(rows) != test_ids:
        raise ValueError(f"Incomplete clean-v2 ground-truth v6 judgments: {direction}: {len(rows)} / {len(test_ids)}")
    return {
        "id": "ground_truth", "name": "Ground truth", "detail": "paired test recording",
        "mean": sum(row["score"] for row in rows.values()) / len(rows),
        "criteria": criterion_means(rows, direction),
    }


def collect_mismatch(review_path: Path) -> dict:
    review = json.loads(review_path.read_text())
    if (review.get("prompt_version"), review.get("pair_count"), review.get("case_count")) != (
        "geval_audio_plausibility_v6", 120, 240
    ):
        raise ValueError("Unexpected matched/mismatched diagnostic provenance")
    output = {}
    for direction in ("a2s", "s2a"):
        cases = [case for case in review["cases"] if case["direction"] == direction]
        if len(cases) != 120 or len({case["input"]["sample_id"] for case in cases}) != 120:
            raise ValueError(f"Invalid mismatch pairing count for {direction}")
        scores = {}
        for variant in ("matched", "mismatched"):
            scores[variant] = {}
            for criterion in CRITERION_ORDER[direction]:
                values = [case[variant]["criteria"][criterion]["score"] for case in cases]
                if any(not isinstance(value, (int, float)) or not 1 <= value <= 5 for value in values):
                    raise ValueError(f"Invalid mismatch score: {direction}/{variant}/{criterion}")
                scores[variant][criterion] = sum(values) / len(values)
        output[direction] = {"scores": scores}
    return output


def build(args: argparse.Namespace) -> None:
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"Output already exists: {out}")
    if not (1 <= args.examples <= 100):
        raise ValueError("--examples must be between 1 and 100")
    survey_rows = list(read_jsonl(args.survey / "samples.jsonl"))
    if len(survey_rows) != 100:
        raise ValueError(f"Expected curated 100-pair bundle, found {len(survey_rows)}")
    selected = survey_rows[: args.examples]
    publish_ids = {row["sample_id"] for row in selected}
    if len(publish_ids) != len(selected):
        raise ValueError("Selected sample IDs must be unique")
    reasons = dataset_reasoning(args.eval_repo, publish_ids)
    supplements = {}
    if args.supplements:
        for row in read_jsonl(args.supplements):
            sample_id = row["sample_id"]
            if sample_id not in publish_ids or sample_id in supplements:
                raise ValueError(f"Unknown or duplicate supplement ID: {sample_id}")
            for group, fields in (("cascadeStages", {"a2s": {"perception", "bridge"}, "s2a": {"perception", "bridge"}}), ("outputText", {d: set(s) for d, s in METRICS.items()})):
                content = row.get(group, {})
                if not isinstance(content, dict) or set(content) - set(fields):
                    raise ValueError(f"Invalid {group} directions for {sample_id}")
                for direction, values in content.items():
                    if not isinstance(values, dict) or set(values) - fields[direction]:
                        raise ValueError(f"Invalid {group}.{direction} fields for {sample_id}")
                    if any(v is not None and not isinstance(v, str) for v in values.values()):
                        raise ValueError(f"Supplement text must be a string or null: {sample_id}")
            supplements[sample_id] = row

    data = {
        "benchmarks": {},
        "geval": {
            "promptVersion": "geval_audio_plausibility_v6",
            "criteria": {
                "scenario_plausibility": "Could the input and generated output belong to a believable shared situation? Reasonable indirect connections are allowed.",
                "controlled_detail": "Does the output give useful detail without inventing implausible facts or over-specifying the scene?",
                "perceptual_audio_quality": "Does the generated recording sound natural, coherent, and free of obvious artifacts?",
                "target_modality_success": "Did the system produce usable speech or environmental audio in the requested modality?",
                "expressive_delivery_fit": "For generated speech, do the voice, pace, and emotion suit the input scene?",
                "contextual_acoustic_fit": "For generated environmental audio, do its sounds, timing, and mood fit what the speech implies?",
            },
            "criterionOrder": CRITERION_ORDER,
            "matchedMismatch": collect_mismatch(args.mismatch_review),
            "summary": {},
        },
        "release": True,
        "samples": [],
    }
    judgments: dict[tuple[str, str], dict[str, dict]] = {}
    per_sample: dict[tuple[str, str, str], dict[str, float]] = {}
    for direction, systems in METRICS.items():
        test_ids = clean_test_ids(args.eval_repo, direction)
        if not publish_ids <= test_ids:
            raise ValueError(f"Selected IDs outside {direction} test split")
        metric_ids = ("fad", "kad", "inception_score") if direction == "s2a" else ("utmos", "dnsmos")
        output_systems = []
        geval_systems = [collect_ground_truth(args.eval_repo, args.results, direction, test_ids)]
        for system, spec in systems.items():
            print(f"Joining {direction}/{system} results and saved prompts...", flush=True)
            root = args.evals / spec["root"]
            geval, geval_attempted, geval_count, geval_rows = collect_geval(
                args.evals, direction, system, test_ids, publish_ids
            )
            judgments[(direction, system)] = geval_rows
            geval_systems.append({
                "id": system, "name": spec["label"], "detail": spec["detail"],
                "mean": geval,
                "criteria": criterion_means(geval_rows, direction),
            })
            scores = {"geval_audio_plausibility": geval}
            for metric in metric_ids:
                result = collect_scalar_metric(root, metric)
                if result:
                    value, _, values = result
                    scores[metric] = value
                    per_sample[(direction, system, metric)] = values
            output_systems.append({"id": system, "name": spec["label"], "detail": spec["detail"], "scores": scores})
        metric_meta = {metric: {"scope": "corpus" if metric in {"fad", "kad", "inception_score"} else "mean"} for metric in metric_ids}
        metric_meta["geval_audio_plausibility"] = {"scope": "mean · 1–5"}
        data["benchmarks"][direction] = {
            "directionName": "speech-to-audio" if direction == "s2a" else "audio-to-speech",
            "note": "Corpus metrics have no per-example value." if direction == "s2a" else "Cascade UTMOS and DNSMOS means use the successfully scored outputs.",
            "metrics": metric_meta,
            "systems": output_systems,
        }
        data["geval"]["summary"][direction] = {"systems": geval_systems}

    out.mkdir(parents=True)
    for source in TEMPLATE.iterdir():
        if source.suffix in {".html", ".css", ".js"} and source.name != "site-data.js":
            shutil.copy2(source, out / source.name)
    for row in selected:
        item = {key: row[key] for key in ("sample_index", "sample_id", "speaker_gender", "speaker_emotion", "audio_caption", "speech_text")}
        item["reasoning"] = reasons[row["sample_id"]]
        item["files"], item["judgments"], item["scores"] = {}, {"a2s": {}, "s2a": {}}, {"a2s": {}, "s2a": {}}
        supplement = supplements.get(row["sample_id"], {})
        item["cascadeStages"] = supplement.get("cascadeStages", {})
        item["outputText"] = supplement.get("outputText", {})
        media_dir = out / "media" / row["sample_id"]
        media_dir.mkdir(parents=True)
        for key, relative in row["files"].items():
            source = args.survey / relative
            target = media_dir / source.name
            shutil.copy2(source, target)
            item["files"][key] = target.relative_to(out).as_posix()
        for direction in ("a2s", "s2a"):
            for system in METRICS[direction]:
                judgment = judgments[(direction, system)].get(row["sample_id"])
                if judgment:
                    item["judgments"][direction][system] = judgment
                for metric in SAMPLE_METRICS:
                    value = per_sample.get((direction, system, metric), {}).get(row["sample_id"])
                    if value is not None:
                        item["scores"][direction].setdefault(system, {})[metric] = value
        data["samples"].append(item)

    (out / "site-data.js").write_text("window.PAPER_DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
    (out / ".nojekyll").touch()
    missing = []
    for item in data["samples"]:
        for direction in METRICS:
            for field in ("perception", "bridge"):
                if not item["cascadeStages"].get(direction, {}).get(field):
                    missing.append({"sample_id": item["sample_id"], "direction": direction, "field": f"cascadeStages.{direction}.{field}"})
            for system in METRICS[direction]:
                if not item["outputText"].get(direction, {}).get(system):
                    missing.append({"sample_id": item["sample_id"], "direction": direction, "field": f"outputText.{direction}.{system}"})
                judgment = item["judgments"][direction].get(system, {})
                for criterion in data["geval"]["criterionOrder"][direction]:
                    value = judgment.get("criteria", {}).get(criterion, {})
                    for field in ("score", "feedback", "prompt"):
                        if value.get(field) is None or value.get(field) == "":
                            missing.append({"sample_id": item["sample_id"], "direction": direction, "field": f"judgments.{direction}.{system}.{criterion}.{field}"})
    (out / "missing-data.json").write_text(json.dumps(missing, indent=2) + "\n")
    manifest = {
        "examples": len(selected),
        "audio_files": sum(len(row["files"]) for row in selected),
        "selection": "first N rows of the fixed, curated seed-20260917 100-pair bundle",
        "geval_prompt_version": "geval_audio_plausibility_v6",
        "missing_fields": len(missing),
    }
    (out / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(out), **manifest}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=100)
    parser.add_argument("--supplements", type=Path, help="Optional per-ID cascade stages and generated-output text JSONL")
    parser.add_argument("--survey", type=Path, default=DEFAULT_SURVEY)
    parser.add_argument("--evals", type=Path, default=DEFAULT_EVALS)
    parser.add_argument("--eval-repo", type=Path, default=DEFAULT_EVAL_REPO)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--mismatch-review", type=Path, default=DEFAULT_MISMATCH_REVIEW)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
