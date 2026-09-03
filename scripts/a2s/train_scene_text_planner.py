#!/usr/bin/env python
"""Train and audit a non-retrieval ambient-scene -> utterance text planner.

Four intended lanes are exposed by ``--source``/``--target-mode``:

* caption -> speech: oracle semantic-input control (does Flan-T5 learn this dataset's style?);
* audio -> speech: the direct learned planner;
* audio -> structured: caption + reasoning + speech as one explicit chain;
* audio -> multitask: rotating caption/reasoning/speech supervision over shared scene memory.

The decisive diagnostic is paired NLL with the exact same targets under true versus
row-shuffled scene memory.  Generation additionally compares true/shuffled outputs and
measures train-set copying.  None of these lanes retrieves an utterance.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from configs import paths as project_paths  # noqa: E402
from salm.a2s.scene_text_planner import (  # noqa: E402
    AudioConditionedT5,
    PlannerRow,
    TASK_TO_ID,
    build_targets,
    format_target,
    normalize_text,
    novelty_metrics,
    token_f1,
)


FLAN_T5_LARGE = Path(
    "/scratch/l/lovenya/salm_migration_clean/models/hf_cache/"
    "models--google--flan-t5-large/snapshots/0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a"
)


def log(message: str) -> None:
    print(message, flush=True)


def load_rows(split: str) -> list[PlannerRow]:
    """Restore reasoning from the source manifest and prove row identity/order."""
    feature_index = project_paths.FEATURES_ROOT / "a2s" / split / "index.jsonl"
    source_manifest = project_paths.A2S_MANIFESTS[split]
    index = [json.loads(x) for x in feature_index.read_text().splitlines() if x.strip()]
    source = [json.loads(x) for x in source_manifest.read_text().splitlines() if x.strip()]

    by_id = {}
    for row in source:
        sid = Path(row["audio_file"]).stem
        if sid in by_id:
            raise RuntimeError(f"duplicate sample_id {sid} in {source_manifest}")
        by_id[sid] = row

    out = []
    for cached in index:
        sid = cached["sample_id"]
        if sid not in by_id:
            raise RuntimeError(f"cached sample {sid} absent from {source_manifest}")
        raw = by_id[sid]
        # These two fields exist in both places. Equality proves that a same-ID collision or
        # stale cache did not silently attach somebody else's reasoning to this row.
        for field in ("audio_cap", "speech_text"):
            if cached[field].strip() != raw[field].strip():
                raise RuntimeError(f"{split}/{sid}: {field} differs between cache and manifest")
        out.append(
            PlannerRow(
                sample_id=sid,
                audio_cap=raw["audio_cap"],
                reasoning=raw.get("reasoning", ""),
                speech_text=raw["speech_text"],
            )
        )
    if len(out) != len(source):
        log(f"note: {split} feature cache has {len(out)} rows from {len(source)} manifest rows")
    return out


def tokenize_labels(tokenizer, texts: Sequence[str], max_length: int, device: str):
    batch = tokenizer(
        list(texts), padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )
    labels = batch.input_ids.to(device)
    return labels.masked_fill(labels == tokenizer.pad_token_id, -100)


def caption_inputs(tokenizer, rows: Sequence[PlannerRow], max_length: int, device: str):
    prompts = [f"write a natural scene-grounded spoken response: {r.audio_cap}" for r in rows]
    return tokenizer(
        prompts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    ).to(device)


def autocast_context(device: str):
    if str(device).startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def batch_loss(model, tokenizer, rows, indices, labels, task_ids, source, ambient, device):
    selected = [rows[i] for i in indices]
    with autocast_context(device):
        if source == "audio":
            a = torch.from_numpy(np.asarray(ambient[np.sort(indices)], dtype=np.float32)).to(device)
            # np.sort is only safe because callers pass sorted indices, asserted here rather
            # than silently reordering labels relative to audio.
            if list(indices) != sorted(indices):
                raise RuntimeError("batch indices must be sorted to preserve mmap row alignment")
            return model.forward_audio(a, task_ids.to(device), labels=labels).loss
        enc = caption_inputs(tokenizer, selected, 96, device)
        return model.t5(**enc, labels=labels, use_cache=False, return_dict=True).loss


@torch.no_grad()
def evaluate(model, tokenizer, rows, ambient, source, target_mode, batch_size, max_target, device):
    model.eval()
    total = rolled = speech_total = speech_rolled = 0.0
    seen = speech_seen = 0
    for start in range(0, len(rows), batch_size):
        stop = min(start + batch_size, len(rows))
        if stop - start < 2:
            break
        idx = list(range(start, stop))
        chosen = rows[start:stop]
        texts, task_ids = build_targets(chosen, target_mode, epoch=0, indices=idx)
        labels = tokenize_labels(tokenizer, texts, max_target, device)
        n_tok = int((labels != -100).sum())

        with autocast_context(device):
            if source == "audio":
                a = torch.from_numpy(np.asarray(ambient[start:stop], dtype=np.float32)).to(device)
                task_ids = task_ids.to(device)
                mem = model.encode_audio(a, task_ids)
                out = model.forward_audio(a, task_ids, labels, memory=mem)
                out_roll = model.forward_audio(
                    a, task_ids, labels, memory=torch.roll(mem, shifts=1, dims=0)
                )
            else:
                enc = caption_inputs(tokenizer, chosen, 96, device)
                out = model.t5(**enc, labels=labels, use_cache=False, return_dict=True)
                roll_ids = torch.roll(enc.input_ids, shifts=1, dims=0)
                roll_mask = torch.roll(enc.attention_mask, shifts=1, dims=0)
                out_roll = model.t5(
                    input_ids=roll_ids, attention_mask=roll_mask, labels=labels,
                    use_cache=False, return_dict=True,
                )
        total += float(out.loss) * n_tok
        rolled += float(out_roll.loss) * n_tok
        seen += n_tok

        # A structured-only model has never seen the standalone speech task token. Its
        # honest comparable binding number is the full structured target's gap.
        if target_mode == "structured":
            speech_total += float(out.loss) * n_tok
            speech_rolled += float(out_roll.loss) * n_tok
            speech_seen += n_tok
            continue

        # Multi-task includes standalone speech training, so this remains meaningful.
        speech = [format_target(r, "speech") for r in chosen]
        speech_labels = tokenize_labels(tokenizer, speech, max_target, device)
        ns = int((speech_labels != -100).sum())
        with autocast_context(device):
            if source == "audio":
                a = torch.from_numpy(np.asarray(ambient[start:stop], dtype=np.float32)).to(device)
                tids = torch.full((len(chosen),), TASK_TO_ID["speech"], device=device, dtype=torch.long)
                mem = model.encode_audio(a, tids)
                so = model.forward_audio(a, tids, speech_labels, memory=mem)
                sr = model.forward_audio(a, tids, speech_labels, memory=torch.roll(mem, 1, 0))
            else:
                enc = caption_inputs(tokenizer, chosen, 96, device)
                so = model.t5(**enc, labels=speech_labels, use_cache=False, return_dict=True)
                sr = model.t5(
                    input_ids=torch.roll(enc.input_ids, 1, 0),
                    attention_mask=torch.roll(enc.attention_mask, 1, 0),
                    labels=speech_labels, use_cache=False, return_dict=True,
                )
        speech_total += float(so.loss) * ns
        speech_rolled += float(sr.loss) * ns
        speech_seen += ns

    val = total / max(seen, 1)
    val_roll = rolled / max(seen, 1)
    sval = speech_total / max(speech_seen, 1)
    sroll = speech_rolled / max(speech_seen, 1)
    return {
        "val_loss": val,
        "val_loss_shuffled": val_roll,
        "val_scene_gap": val_roll - val,
        "val_ppl": math.exp(min(val, 20)),
        "val_speech_loss": sval,
        "val_speech_loss_shuffled": sroll,
        "val_speech_scene_gap": sroll - sval,
    }


def generated_speech_view(text: str, target_mode: str) -> str:
    """Extract the utterance from a generated structured chain."""
    if target_mode != "structured":
        return text.strip()
    marker = text.lower().rfind("speech:")
    return text[marker + len("speech:") :].strip() if marker >= 0 else text.strip()


@torch.no_grad()
def generate_report(model, tokenizer, val_rows, train_rows, ambient, source, target_mode, count, max_new, device):
    model.eval()
    rows = val_rows[:count]
    generated, shuffled = [], []
    batch_size = 8
    for start in range(0, len(rows), batch_size):
        chosen = rows[start : start + batch_size]
        if source == "audio":
            a = torch.from_numpy(
                np.asarray(ambient[start : start + len(chosen)], dtype=np.float32)
            ).to(device)
            task = "structured" if target_mode == "structured" else "speech"
            tids = torch.full(
                (len(chosen),), TASK_TO_ID[task], device=device, dtype=torch.long
            )
            with autocast_context(device):
                ids = model.generate_audio(a, tids, max_new_tokens=max_new, num_beams=4)
                ids_s = model.generate_audio(torch.roll(a, 1, 0), tids, max_new_tokens=max_new, num_beams=4)
        else:
            enc = caption_inputs(tokenizer, chosen, 96, device)
            with autocast_context(device):
                ids = model.t5.generate(**enc, max_new_tokens=max_new, num_beams=4)
                ids_s = model.t5.generate(
                    input_ids=torch.roll(enc.input_ids, 1, 0),
                    attention_mask=torch.roll(enc.attention_mask, 1, 0),
                    max_new_tokens=max_new, num_beams=4,
                )
        generated.extend(tokenizer.batch_decode(ids, skip_special_tokens=True))
        shuffled.extend(tokenizer.batch_decode(ids_s, skip_special_tokens=True))

    generated = [generated_speech_view(x, target_mode) for x in generated]
    shuffled = [generated_speech_view(x, target_mode) for x in shuffled]

    records = []
    for row, pred, wrong in zip(rows, generated, shuffled):
        records.append(
            {
                "sample_id": row.sample_id,
                "audio_cap": row.audio_cap,
                "reference_reasoning": row.reasoning,
                "reference_speech": row.speech_text,
                "generated_speech": pred,
                "generated_with_shuffled_scene": wrong,
                "target_token_f1": token_f1(pred, row.speech_text),
                "changed_when_shuffled": normalize_text(pred) != normalize_text(wrong),
            }
        )
    metrics = novelty_metrics(generated, (r.speech_text for r in train_rows))
    metrics.update(
        {
            "mean_target_token_f1": float(np.mean([r["target_token_f1"] for r in records])),
            "shuffled_output_change_rate": float(
                np.mean([r["changed_when_shuffled"] for r in records])
            ),
        }
    )
    return records, metrics


def trainable_state(model):
    names = {name for name, p in model.named_parameters() if p.requires_grad}
    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items() if name in names}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--model-path", default=str(FLAN_T5_LARGE))
    ap.add_argument("--source", choices=("audio", "caption"), default="audio")
    ap.add_argument(
        "--target-mode", choices=("speech", "caption", "reasoning", "structured", "multitask"),
        default="speech",
    )
    ap.add_argument("--train-scope", choices=("adapter", "decoder", "full"), default="decoder")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--adapter-lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--max-target-length", type=int, default=192)
    ap.add_argument("--adapter-layers", type=int, default=2)
    ap.add_argument("--adapter-heads", type=int, default=8)
    ap.add_argument("--eval-batch-size", type=int, default=16)
    ap.add_argument("--generate-count", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--save-every", type=int, default=2)
    ap.add_argument("--max-minutes", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.source == "caption" and args.target_mode != "speech":
        raise SystemExit("caption source is the oracle caption->speech control; target-mode must be speech")
    if not Path(args.model_path).is_dir():
        raise SystemExit(f"missing local Flan-T5 snapshot: {args.model_path}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    from transformers import AutoTokenizer, T5ForConditionalGeneration

    train_rows, val_rows = load_rows("train"), load_rows("val")
    amb = {}
    for split, rows in (("train", train_rows), ("val", val_rows)):
        path = project_paths.OUTPUTS_ROOT / "features_a2s_cond" / split / "cond_seq.npy"
        amb[split] = np.load(path, mmap_mode="r")
        if amb[split].shape[0] != len(rows):
            raise SystemExit(f"{split}: ambient {amb[split].shape[0]} rows != metadata {len(rows)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    t5 = T5ForConditionalGeneration.from_pretrained(args.model_path, local_files_only=True)
    model = AudioConditionedT5(
        t5,
        ambient_dim=int(amb["train"].shape[-1]),
        adapter_layers=args.adapter_layers,
        adapter_heads=args.adapter_heads,
    ).to(args.device)
    model.set_train_scope(args.train_scope, args.source)

    adapter_params, base_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (adapter_params if name.startswith("audio_adapter.") else base_params).append(p)
    groups = []
    if adapter_params:
        groups.append({"params": adapter_params, "lr": args.adapter_lr})
    if base_params:
        groups.append({"params": base_params, "lr": args.lr})
    if not groups:
        raise SystemExit("no trainable parameters")
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)

    n = len(train_rows)
    batches = math.ceil(n / args.batch_size)
    updates_per_epoch = math.ceil(batches / args.grad_accum)
    total_updates = max(1, updates_per_epoch * args.epochs)
    warmup = max(1, int(total_updates * args.warmup_ratio))

    def lr_factor(step: int) -> float:
        if step < warmup:
            return max(step, 1) / warmup
        progress = (step - warmup) / max(total_updates - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(
        f"{args.run_name}: source={args.source} target={args.target_mode} scope={args.train_scope}; "
        f"trainable={trainable/1e6:.1f}M/{total/1e6:.1f}M; "
        f"train={n} val={len(val_rows)} updates={total_updates}"
    )

    run_dir = project_paths.OUTPUTS_ROOT / "scene_text_planner" / args.run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_loss, best_state = float("inf"), None
    rng = np.random.default_rng(args.seed)
    started = time.time()
    update = 0

    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(n)
        running = 0.0
        opt.zero_grad(set_to_none=True)
        for bi in range(batches):
            idx = np.sort(order[bi * args.batch_size : (bi + 1) * args.batch_size]).tolist()
            if not idx:
                continue
            selected = [train_rows[i] for i in idx]
            targets, task_ids = build_targets(selected, args.target_mode, epoch, idx)
            labels = tokenize_labels(tokenizer, targets, args.max_target_length, args.device)
            loss = batch_loss(
                model, tokenizer, train_rows, idx, labels, task_ids,
                args.source, amb["train"], args.device,
            )
            (loss / args.grad_accum).backward()
            running += float(loss)
            if (bi + 1) % args.grad_accum == 0 or bi + 1 == batches:
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()
                update += 1

        metrics = evaluate(
            model, tokenizer, val_rows, amb["val"], args.source, args.target_mode,
            args.eval_batch_size, args.max_target_length, args.device,
        )
        metrics.update(
            {
                "epoch": epoch,
                "train_loss": running / max(batches, 1),
                "minutes": (time.time() - started) / 60,
                "lr": [float(g["lr"]) for g in opt.param_groups],
            }
        )
        history.append(metrics)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))
        payload = {"trainable_state": trainable_state(model), "args": vars(args), "metrics": metrics}
        torch.save(payload, ckpt_dir / "checkpoint_last.pt")
        if metrics["val_loss"] < best_loss:
            best_loss = metrics["val_loss"]
            best_state = payload["trainable_state"]
            torch.save(payload, ckpt_dir / "checkpoint_best_loss.pt")
        if args.save_every and (epoch + 1) % args.save_every == 0:
            torch.save(payload, ckpt_dir / f"checkpoint_ep{epoch}.pt")
        log(
            f"epoch {epoch:02d} train={metrics['train_loss']:.4f} "
            f"val={metrics['val_loss']:.4f} GAP={metrics['val_scene_gap']:+.4f} "
            f"speech={metrics['val_speech_loss']:.4f} "
            f"SPEECH_GAP={metrics['val_speech_scene_gap']:+.4f} "
            f"({metrics['minutes']:.1f} min total)"
        )
        if args.max_minutes and metrics["minutes"] >= args.max_minutes:
            log(f"stopping cleanly at --max-minutes={args.max_minutes:g}")
            break

    if best_state is not None:
        missing, unexpected = model.load_state_dict(best_state, strict=False)
        # Missing frozen keys are intentional; any unexpected key is not.
        if unexpected:
            raise RuntimeError(f"unexpected keys while restoring best checkpoint: {unexpected}")
        log(f"restored best-loss trainable state ({len(missing)} frozen keys omitted)")

    records, gen_metrics = generate_report(
        model, tokenizer, val_rows, train_rows, amb["val"], args.source, args.target_mode,
        args.generate_count, args.max_new_tokens, args.device,
    )
    with (run_dir / "heldout_generations.jsonl").open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "run_name": args.run_name,
        "source": args.source,
        "target_mode": args.target_mode,
        "train_scope": args.train_scope,
        "best_val": min(history, key=lambda x: x["val_loss"]),
        "generation": gen_metrics,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
