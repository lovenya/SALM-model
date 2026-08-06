"""
Lightweight node resource sampler for Slurm jobs.

Writes JSONL samples plus a peak summary so future walltime/CPU/RAM/GPU asks are based
on measurement rather than guesswork. Intentionally cheap: one nvidia-smi call and one
/proc read per interval.

Usage (backgrounded by the sbatch script):
    python scripts/resource_monitor.py --out-dir <logs> --tag <job> --interval 60 &
    MONITOR_PID=$!
    ...workload...
    kill $MONITOR_PID
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import time
from pathlib import Path

_STOP = False


def _handle_stop(signum, frame):  # noqa: ARG001
    global _STOP
    _STOP = True


def sample_gpus() -> list[dict]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=20,
        )
    except Exception:
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        gpus.append(
            {
                "index": int(parts[0]),
                "mem_used_mb": float(parts[1]),
                "mem_total_mb": float(parts[2]),
                "util_pct": float(parts[3]),
            }
        )
    return gpus


def sample_host() -> dict:
    info: dict = {}
    try:
        with open("/proc/meminfo") as f:
            mem = {}
            for line in f:
                key, _, rest = line.partition(":")
                mem[key] = float(rest.strip().split()[0])  # kB
        total = mem.get("MemTotal", 0.0)
        avail = mem.get("MemAvailable", 0.0)
        info["ram_used_gb"] = (total - avail) / 1024 / 1024
        info["ram_total_gb"] = total / 1024 / 1024
    except Exception:
        pass
    try:
        with open("/proc/loadavg") as f:
            info["loadavg_1m"] = float(f.read().split()[0])
    except Exception:
        pass
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--interval", type=float, default=60.0)
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{args.tag}_resources.jsonl"
    summary_path = out_dir / f"{args.tag}_resources_summary.json"

    peaks: dict = {"gpu_mem_used_mb": {}, "gpu_util_pct": {}, "ram_used_gb": 0.0, "samples": 0}
    t0 = time.time()

    with jsonl_path.open("a") as f:
        while not _STOP:
            rec = {
                "t": time.time(),
                "elapsed_s": time.time() - t0,
                "gpus": sample_gpus(),
                "host": sample_host(),
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()

            peaks["samples"] += 1
            for g in rec["gpus"]:
                i = str(g["index"])
                peaks["gpu_mem_used_mb"][i] = max(
                    peaks["gpu_mem_used_mb"].get(i, 0.0), g["mem_used_mb"]
                )
                peaks["gpu_util_pct"][i] = max(peaks["gpu_util_pct"].get(i, 0.0), g["util_pct"])
            peaks["ram_used_gb"] = max(peaks["ram_used_gb"], rec["host"].get("ram_used_gb", 0.0))

            peaks["elapsed_s"] = time.time() - t0
            summary_path.write_text(json.dumps(peaks, indent=2))

            # Sleep in slices so SIGTERM is honoured promptly.
            slept = 0.0
            while slept < args.interval and not _STOP:
                time.sleep(min(1.0, args.interval - slept))
                slept += 1.0

    peaks["elapsed_s"] = time.time() - t0
    summary_path.write_text(json.dumps(peaks, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
