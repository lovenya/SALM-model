#!/usr/bin/env python
"""Fit and evaluate a lossless BPE compression gate on cached acoustic content codes.

This is a representation-only gate: it does not train a predictor or render audio.  For
each K, it learns one maximal BPE model on TRAIN, evaluates prefixes at the requested total
vocabulary sizes, and exhaustively verifies encode->decode equality on TRAIN and VAL.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from configs import paths as project_paths  # noqa: E402
from salm.a2s.content_bpe import (  # noqa: E402
    AcousticBPE,
    fit_acoustic_bpe,
    length_statistics,
)


def log(message: str) -> None:
    print(message, flush=True)


def _encode_and_verify(model: AcousticBPE, rows: np.ndarray) -> tuple[np.ndarray, int]:
    lengths = np.empty(rows.shape[0], dtype=np.int64)
    for index, row in enumerate(rows):
        encoded = model.encode(row)
        decoded = model.decode(encoded)
        if not np.array_equal(decoded, np.asarray(row, dtype=np.int64)):
            mismatch = np.flatnonzero(decoded != np.asarray(row, dtype=np.int64))
            first = int(mismatch[0]) if mismatch.size else -1
            raise RuntimeError(
                f"lossless verification failed at row={index}, first_difference={first}, "
                f"decoded_length={len(decoded)}, source_length={len(row)}"
            )
        lengths[index] = len(encoded)
    return lengths, int(rows.shape[0])


def _run_k(
    k: int,
    cache_root: str,
    out_dir: str,
    target_vocabs: tuple[int, ...],
    min_frequency: int,
) -> dict:
    started = time.time()
    cache = Path(cache_root)
    output = Path(out_dir)
    train_path = cache / f"codes_codebook_k{k}_train.npy"
    val_path = cache / f"codes_codebook_k{k}_val.npy"
    for path in (train_path, val_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    train = np.load(train_path, mmap_mode="r")
    val = np.load(val_path, mmap_mode="r")
    if train.ndim != 2 or val.ndim != 2 or train.shape[1] != val.shape[1]:
        raise ValueError(f"K={k}: incompatible cache shapes train={train.shape}, val={val.shape}")
    for split, rows in (("train", train), ("val", val)):
        if int(rows.min()) < 0 or int(rows.max()) >= k:
            raise ValueError(f"K={k} {split}: code range is outside [0, {k})")

    log(f"K={k}: fitting through total vocabulary {max(target_vocabs)}")

    def progress(row: dict) -> None:
        vocab = int(row["vocab_size"])
        if vocab in target_vocabs or vocab % 500 == 0:
            log(
                f"K={k}: vocab={vocab} last_replacements={row['replacements']} "
                f"mean_train_tokens={row['mean_train_tokens']:.2f}"
            )

    maximal, fit_lengths, history = fit_acoustic_bpe(
        train,
        k,
        max(target_vocabs),
        min_frequency=min_frequency,
        checkpoint_vocabs=target_vocabs,
        on_progress=progress,
    )
    if maximal.vocab_size < max(target_vocabs):
        log(
            f"K={k}: stopped early at vocab={maximal.vocab_size}; no pair met "
            f"min_frequency={min_frequency}"
        )

    reports: list[dict] = []
    for requested in target_vocabs:
        achieved = min(requested, maximal.vocab_size)
        model = maximal.prefix(achieved)
        model_path = output / (
            f"acoustic_bpe_k{k}_target{requested}_achieved{achieved}.json"
        )
        model_payload = model.to_dict()
        model_payload.update(
            {
                "requested_vocab_size": requested,
                "source_train_cache": str(train_path),
                "source_val_cache": str(val_path),
                "min_frequency": min_frequency,
            }
        )
        model_path.write_text(json.dumps(model_payload, indent=2) + "\n")

        # Exhaustive round-trip checks.  Train lengths from the linked fitter are also
        # cross-checked against an independent encode pass, catching fitter/encoder drift.
        train_encoded_lengths, train_verified = _encode_and_verify(model, train)
        val_encoded_lengths, val_verified = _encode_and_verify(model, val)
        linked_lengths = fit_lengths.get(achieved)
        if linked_lengths is not None and not np.array_equal(
            linked_lengths, train_encoded_lengths
        ):
            raise RuntimeError(
                f"K={k} vocab={achieved}: linked-fit and independent encoder lengths differ"
            )
        report = {
            "k": k,
            "requested_vocab_size": requested,
            "achieved_vocab_size": achieved,
            "merge_budget": achieved - k,
            "model_path": str(model_path),
            "train": length_statistics(train_encoded_lengths),
            "val": length_statistics(val_encoded_lengths),
            "roundtrip": {
                "train_rows_verified": train_verified,
                "val_rows_verified": val_verified,
                "exact": True,
            },
            "passes_mean_val_under_80": bool(val_encoded_lengths.mean() < 80.0),
        }
        reports.append(report)
        log(
            f"K={k} vocab={achieved}: train mean/p50/p95 "
            f"{report['train']['mean']:.2f}/{report['train']['p50']:.1f}/"
            f"{report['train']['p95']:.1f}; val "
            f"{report['val']['mean']:.2f}/{report['val']['p50']:.1f}/"
            f"{report['val']['p95']:.1f}; exact round-trip PASS"
        )

    return {
        "k": k,
        "train_shape": list(train.shape),
        "val_shape": list(val.shape),
        "base_vocab_size": k,
        "max_achieved_vocab_size": maximal.vocab_size,
        "merges_learned": len(maximal.merges),
        "last_merge": history[-1] if history else None,
        "budgets": reports,
        "elapsed_seconds": time.time() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ks", type=int, nargs="+", default=[512, 1024])
    parser.add_argument(
        "--target-vocabs", type=int, nargs="+", default=[2048, 4096, 8192],
        help="total vocabularies, including the K base acoustic codes",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=project_paths.OUTPUTS_ROOT / "content_codes",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument(
        "--workers", type=int, default=2,
        help="fit independent K values concurrently; memory scales with workers",
    )
    parser.add_argument(
        "--pass-mean-val-tokens", type=float, default=80.0,
        help="pre-registered compression gate; strict less-than comparison",
    )
    args = parser.parse_args()

    ks = tuple(dict.fromkeys(args.ks))
    vocabs = tuple(sorted(set(args.target_vocabs)))
    if any(k < 1 for k in ks) or any(v < 1 for v in vocabs):
        parser.error("K and vocabulary sizes must be positive")
    if any(v < max(ks) for v in vocabs):
        parser.error("every target vocabulary must be at least the largest requested K")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.out_dir.exists():
        parser.error(f"output directory already exists; refusing overwrite: {args.out_dir}")
    args.out_dir.mkdir(parents=True)

    started = time.time()
    results: list[dict] = []
    if args.workers == 1 or len(ks) == 1:
        for k in ks:
            results.append(
                _run_k(k, str(args.cache_root), str(args.out_dir), vocabs, args.min_frequency)
            )
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(ks))) as executor:
            futures = {
                executor.submit(
                    _run_k,
                    k,
                    str(args.cache_root),
                    str(args.out_dir),
                    vocabs,
                    args.min_frequency,
                ): k
                for k in ks
            }
            for future in as_completed(futures):
                results.append(future.result())
    results.sort(key=lambda item: item["k"])

    # The largest budget is the actual decision point.  Smaller budgets show the curve and
    # tell us whether 8k is buying anything, but should not veto a successful maximal model.
    largest = max(vocabs)
    decisions = []
    for result in results:
        row = next(b for b in result["budgets"] if b["requested_vocab_size"] == largest)
        passed = row["val"]["mean"] < args.pass_mean_val_tokens
        decisions.append({"k": result["k"], "val_mean_tokens": row["val"]["mean"], "pass": passed})
    overall = all(item["pass"] for item in decisions)
    summary = {
        "schema_version": 1,
        "experiment": "lossless_acoustic_code_bpe_compression_gate",
        "cache_root": str(args.cache_root),
        "target_vocab_sizes": list(vocabs),
        "min_frequency": args.min_frequency,
        "gate": {
            "metric": "mean validation encoded tokens at largest requested vocabulary",
            "threshold": args.pass_mean_val_tokens,
            "comparison": "strictly_less_than",
            "pass_definition": "both K=512 and K=1024 pass and every round-trip is exact",
            "decisions": decisions,
            "overall_pass": overall,
        },
        "results": results,
        "elapsed_seconds": time.time() - started,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    log("\nBPE COMPRESSION GATE")
    for decision in decisions:
        log(
            f"  K={decision['k']}: mean val tokens={decision['val_mean_tokens']:.2f} "
            f"-> {'PASS' if decision['pass'] else 'FAIL'} (< {args.pass_mean_val_tokens:g})"
        )
    log(f"  overall: {'PASS' if overall else 'FAIL'}")
    log(f"  report: {args.out_dir / 'summary.json'}")
    # A scientifically failed compression hypothesis is still a successfully completed job;
    # only implementation/data failures should make Slurm mark the process failed.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
