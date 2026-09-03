#!/usr/bin/env python
"""Cache frozen Flan-T5 encoder states for the oracle transcript bridge.

Only ``speech_text`` is read.  No ambient path or waveform is opened in this stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from configs import paths as project_paths  # noqa: E402


DEFAULT_TEXT_MODEL = Path(
    "/scratch/l/lovenya/salm_migration_clean/models/hf_cache/"
    "models--google--flan-t5-large/snapshots/0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a"
)


def text_digest(texts: list[str]) -> str:
    h = hashlib.sha256()
    for text in texts:
        h.update(text.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def load_texts(split: str) -> tuple[list[str], list[str]]:
    index = project_paths.FEATURES_ROOT / "a2s" / split / "index.jsonl"
    rows = [json.loads(line) for line in index.read_text().splitlines() if line.strip()]
    return [str(r["speech_text"]) for r in rows], [str(r["sample_id"]) for r in rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val", "test"], required=True)
    ap.add_argument("--model", type=Path, default=DEFAULT_TEXT_MODEL)
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if not args.model.is_dir():
        raise SystemExit(f"missing offline text model: {args.model}")
    out = Path(args.out_root or project_paths.OUTPUTS_ROOT / "text_content_features")
    out.mkdir(parents=True, exist_ok=True)
    hidden_path = out / f"{args.split}_hidden.npy"
    mask_path = out / f"{args.split}_mask.npy"
    meta_path = out / f"{args.split}_meta.json"

    texts, ids = load_texts(args.split)
    wanted = {
        "split": args.split,
        "num_rows": len(texts),
        "text_sha256": text_digest(texts),
        "sample_ids_sha256": text_digest(ids),
        "model": str(args.model.resolve()),
        "max_tokens": args.max_tokens,
    }
    if meta_path.exists() or hidden_path.exists() or mask_path.exists():
        if not (meta_path.exists() and hidden_path.exists() and mask_path.exists()):
            raise SystemExit(f"incomplete cache at {out}; remove that split's three files manually")
        have = json.loads(meta_path.read_text())
        if all(have.get(k) == v for k, v in wanted.items()):
            print(f"reusing verified {args.split} cache: {hidden_path}", flush=True)
            return 0
        raise SystemExit(f"refusing incompatible existing cache: {meta_path}")

    from transformers import AutoTokenizer, T5EncoderModel

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    dtype = torch.float16 if str(args.device).startswith("cuda") else torch.float32
    model = T5EncoderModel.from_pretrained(
        str(args.model), local_files_only=True, torch_dtype=dtype
    ).to(args.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    text_dim = int(model.config.d_model)

    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=args.max_tokens,
        return_attention_mask=True,
    )
    input_ids = np.asarray(tokenized["input_ids"], dtype=np.int32)
    masks = np.asarray(tokenized["attention_mask"], dtype=np.bool_)

    hidden_tmp = out / f".{args.split}_hidden.{os.getpid()}.npy"
    mask_tmp = out / f".{args.split}_mask.{os.getpid()}.npy"
    hmap = np.lib.format.open_memmap(
        hidden_tmp, mode="w+", dtype=np.float16,
        shape=(len(texts), args.max_tokens, text_dim),
    )
    np.save(mask_tmp, masks)
    with torch.inference_mode():
        for start in range(0, len(texts), args.batch_size):
            stop = min(start + args.batch_size, len(texts))
            ids_t = torch.from_numpy(input_ids[start:stop]).long().to(args.device)
            mask_t = torch.from_numpy(masks[start:stop]).long().to(args.device)
            states = model(input_ids=ids_t, attention_mask=mask_t).last_hidden_state
            hmap[start:stop] = states.float().cpu().numpy().astype(np.float16)
            print(f"{args.split}: {stop}/{len(texts)}", flush=True)
    hmap.flush()
    del hmap
    os.replace(hidden_tmp, hidden_path)
    os.replace(mask_tmp, mask_path)
    wanted.update({"text_dim": text_dim, "dtype": "float16"})
    meta_path.write_text(json.dumps(wanted, indent=2) + "\n")
    print(f"wrote {hidden_path} and {mask_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
