---
name: submit-jobs
description: Plan, submit, monitor, and audit TamIA Slurm jobs for this project. Use when preparing sbatch scripts, choosing accounts/resources, checking cluster constraints, submitting dependency chains, monitoring running jobs, inspecting logs, or updating resource estimates. Invoke as /submit-jobs.
---

# Submit jobs

Ported from the project's codex `submit-jobs` skill and extended with failures
actually hit in this repo. Treat job submission as an experiment decision: inspect
scripts and configs first, state assumptions, and do not submit until resources,
account, environment, and dependencies are clear.

Run IDs use Montreal time: `TZ=America/Toronto date +%Y%m%d_%H%M`.

## Hard rules

- Login nodes are for editing, metadata inspection, and submission only. All GPU
  work, training, heavy preprocessing, and large evaluations go through Slurm.
  **Neural-net forward passes are not "lightweight checks"** -- a HiFi-GAN vocoder
  or diffusion sampler on a login node will hang for many minutes. That mistake
  cost ~8 min in this repo; run those on a GPU allocation.
- Never start installs, source builds, or downloads on compute nodes: they have no
  internet. Hand the user exact commands for a login node instead.
- Do not set `#SBATCH --partition` unless the cluster or script requires it.
- Do not commit generated resource JSON, logs, outputs, or checkpoints.
- Full-scale runs should come from a committed checkout. Smoke and fit jobs may use
  a dirty tree -- validating the code is their purpose.
- Record the commit at job start; that is the authoritative code identity for a run.

## Smoke before scale, always

Every real bug found in this repo so far was caught by a cheap smoke, not by
reading code. A 30-second smoke that exercises every stage boundary is worth far
more than a careful re-read, because the failures are environmental:

- a relative `source "$(dirname "${BASH_SOURCE[0]}")/..."` in an sbatch script --
  Slurm copies the script to `/var/spool/slurmd/job<N>/slurm_script`, so
  `$BASH_SOURCE` is the spool dir, not the repo. **Source shared preambles by
  absolute path.**
- `HF_HUB_CACHE="$HF_HOME"` when the download went to `$HF_HOME/hub`. Fails with a
  misleading "does not appear to have a file named config.json" while the cache is
  complete. `HF_HUB_CACHE` must be `$HF_HOME/hub` for anything fetched by a current
  `hf download`.

Use the smallest sample count that exercises every stage boundary and output path.
Ask for minutes of walltime, not hours. Capture logs and resource data exactly as a
full run would.

## What sbatch snapshots, and what it does not

This matters when fixing bugs in queued jobs:

- **Snapshotted at submit time**: the sbatch script text itself. Editing it does
  *not* affect an already-queued job -- cancel and resubmit.
- **Read at run time**: anything the script sources or invokes -- the shared env
  preamble, Python modules, config YAML. Fixes there *do* reach queued jobs.

## TamIA facts

- h100 node: **48 CPUs / 500000M / 4x H100-80GB NVLink**. The 64-CPU/1TB rows in
  `sinfo` are **h200** nodes (8 GPUs) -- do not confuse them.
- A GPU job reserves the **whole node**. CPU and RAM requests are therefore cgroup
  limits inside an already-reserved node: under-asking only starves your own
  dataloader workers and frees nothing for anyone else. Ask for close to the whole
  node (e.g. 48 CPUs, 480G).
- Compute nodes have no internet.
- CPU nodes (`cpubase_*`, 64 CPUs / 500000M) are frequently idle and start
  immediately, while a whole 4-GPU allocation can wait ~45 min. Moving genuinely
  CPU-suitable work there keeps the GPU off the critical path -- but measure the
  throughput first, since a serial-I/O-bound script may be slower on CPU than the
  GPU queue wait.
- Accounts: `aip-csubakan` and `aip-ravanelm`, standing target 60/40.

## Choose the account by measurement

FairShare between the two accounts can differ by enough to dominate queue time. On
2026-08-06 the same whole-node 1 h job was predicted to start at 08:48 on
`aip-ravanelm` versus 11:27 on `aip-csubakan` -- a 2h40m difference.

Always check both before assuming, and never guess:

```bash
sbatch --test-only --account=aip-ravanelm <script>
sbatch --test-only --account=aip-csubakan <script>
sshare -A <account> -l
```

`--test-only` predicts the start time and node without queueing anything. When an
urgent run deviates from the 60/40 split, record it and rebalance later.

## Walltime and job structure

- Plan from measured usage plus a buffer: 1.3x default, 1.4x for unstable or newly
  integrated code.
- Set an **internal workload deadline below the Slurm limit** so the job can
  validate outputs, checkpoint, and write its report instead of being hard-killed
  mid-write. For 1-3 h jobs, leave 5-10 min.
- Shorter jobs backfill into gaps more easily, but **verify the gain** -- measured
  here it was only ~10 min (08:48 vs 08:58), not the large win assumed. The better
  argument for splitting is usually failure isolation: a gate failure costing a 1 h
  allocation instead of 3 h.
- Chain with `--dependency=afterok:<jobid>` and submit the dependent job
  immediately, so it accrues priority while the first runs.
- Prefer decoupling stages through artifacts on disk (a features directory with a
  `meta.json`) over passing run IDs between jobs. Either stage can then be rerun
  alone.
- Reject an existing run directory **before** model loading, not after.

## Packing and sharding

When a model fits on one GPU, shard data across the 4 GPUs with explicit
`CUDA_VISIBLE_DEVICES` rather than leaving GPUs idle or requesting more nodes. Give
each process an independent output path; never let two append to one manifest.
Leave shards unmerged for post-run audit unless in-job merging is requested.

Before assuming a model needs N GPUs, check measured VRAM -- if each of 4 GPUs used
35 GB of 80 GB, test a 2-GPU run and fit two shards per node.

## Monitoring

Start `scripts/resource_monitor.py` in the job. Use `--interval 30` for short
stages so model-load and peak memory are not missed; 300 s for long runs.

```bash
squeue -j <jobid> -o "%.10i %.3t %.11M %.11l %.20R"
sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed,Timelimit,AllocTRES%50,MaxRSS -P
```

Every job gets a post-run audit before being called done: requested vs actual
walltime, overrequest ratio, peak GPU memory, GPU utilization, MaxRSS, throughput,
success/error counts. If the monitor summary is empty because the job finished near
the first poll, report GPU accounting as **unavailable** -- not as zero usage.

For generated-audio jobs, sanity-check artifacts: manifest row counts, readable WAV
count, duration and sample-rate distribution, RMS distribution, near-silent count,
and a few lowest-RMS paths for manual listening. Near-silent output is the typical
signature of a broken generation path.

Record measurements in `references/resource-record-schema.md` shape, under ignored
paths.

## Failure handling

- Read the Slurm log, per-lane logs, and row-level errors before resubmitting.
- Distinguish process failure from a report-gate failure: a job can produce valid
  manifests and still exit nonzero on validation rules.
- Retry only failed rows or stages; do not overwrite successful outputs unless the
  bug affected everything.
- If an HPC or toolchain fix fails twice, stop and summarize what was tried before
  making more environment changes.
