# Weights & Biases on TamIA — how this repo uses it, and why

Scope: how W&B is wired into SALM-model training on TamIA (Alliance Canada / DRAC), what the
cluster docs actually allow, and the exact commands to get metrics from a compute node into
wandb.ai.

Sources this document is built on:

- [Alliance Canada wiki: Weights & Biases (wandb)](https://docs.alliancecan.ca/wiki/Weights_%26_Biases_(wandb))
  — the authoritative page for our clusters. Note: the wiki sits behind bot protection
  ("Access Denied / BotStopper"), so agents and scripts generally cannot fetch it; open it in
  a browser.
- [Mila's mirror of the same guidance](https://docs.mila.quebec/technical_reference/clusters/drac/)
  and [Mila wandb setup](https://docs.mila.quebec/examples/good_practices/wandb_setup/).
- [`wandb offline` CLI reference](https://docs.wandb.ai/models/ref/cli/wandb-offline) and
  [`wandb sync` CLI reference](https://docs.wandb.ai/models/ref/cli/wandb-sync).
- The installed client, **wandb 0.28.1** in
  `/scratch/l/lovenya/salm_migration_clean/.venvs/salm-model/.venv-salm-train-audioldm-py312`
  (behaviour below was verified against its source, not just the docs).

---

## 1. Why offline mode is mandatory here

TamIA compute nodes have no internet route. The Alliance page states it for exactly our
cluster family:

> The compute nodes for Narval, Rorqual and Tamia don't have access to the internet, but
> there is a special module that can be loaded in order to allow training scripts to access
> some specific servers, which includes the necessary servers for using CometML and Wandb.

That module exists on TamIA — verified read-only on the login node:

```
$ module avail httpproxy
   httpproxy/1.0
```

**We deliberately do not use it.** The Alliance page warns against it specifically for wandb:

> Be careful when using Wandb with httpproxy. It does not support sending artifacts and
> wandb's logger will hang in the background when your training is completed, wasting
> resources until the job times out.

> While it is possible to upload basic metrics to Weights&Biases during a job on Narval,
> Rorqual and TamIA, the wandb package will automatically attempt to upload information about
> your environment to a Google Cloud Storage bucket, which is not allowed on the compute nodes
> of these clusters. This will result in a crash during or at the very end of a training run.
> Your job may also freeze until it reaches its wall time, thereby wasting resources.

For a 4×H100 job that is the worst possible failure mode: the science finishes, then the job
sits idle burning the allocation until walltime. Offline mode avoids the network entirely:

> In offline mode, wandb will write all metrics, logs and artifacts to the local disk and will
> not attempt to sync anything to the Weights & Biases service on the internet.

### The MILA / eligible-groups exception (does it apply to us?)

The section *Users from MILA and other eligible groups* says members of Mila may use online
mode on clusters with internet access **provided they log in with a valid `mila-org` W&B
account**, and that other groups are known to have made arrangements with W&B to bypass the
Google Cloud Storage calls — you are told to ask your PI whether your group has one.

Our entity is `lovenya-org`, not `mila-org`, and we have no evidence of a GCS bypass
arrangement for `aip-ravanelm`. **Until someone confirms such an arrangement with the PI,
offline is the only supported mode for this repo.** If an arrangement does turn out to exist,
the change is still not "flip to online": you would also have to accept the artifact and
hanging-logger caveats above, so offline-plus-sync remains the better default for long runs.

---

## 2. The env vars we set, and why each matters

All of these live in `submission_scripts/_salm_env.sh`, which every sbatch script sources.

| Variable | Value | Why |
| --- | --- | --- |
| `WANDB_MODE` | `offline` | The whole point of section 1. Without it `wandb.init` tries to reach wandb.ai from a compute node and blocks on a network timeout (and later tries GCS). |
| `WANDB_ENTITY` | `lovenya-org` | **The entity trap — see below.** |
| `WANDB_PROJECT` | `salm-s2a` | Keeps every rung of the ladder in one project so runs are comparable in one workspace. |
| `WANDB_CACHE_DIR` | `$OUTPUTS_ROOT/.wandb_cache` | Keeps W&B's cache off the `/home` quota (home is at 18/25 GiB, 207K/250K files). |

`src/salm/train_ldm.py` additionally passes `project`, `entity`, `mode` explicitly to
`wandb.init()`, so the code is correct even if someone runs it without sourcing `_salm_env.sh`.
Belt and braces, on purpose.

### The entity trap

The W&B account's `default_entity` is **`gaspardbtt-`** — another project's team. The
available entities are `lovenya` (personal), `gaspardbtt-`, and `lovenya-org`.

If `entity` is unset or empty, wandb resolves it to the account default, and the run lands in
**someone else's team** with no error and no warning. You find out days later when the
workspace is empty. This is why the entity is hard-coded in three places (`_salm_env.sh`,
`TrainConfig.wandb_entity`, `scripts/wandb_sync.sh`) rather than relying on any default.

The same trap applies at sync time, not just at init time: `wandb sync` takes `--entity` /
`--project` overrides, and if you sync an offline directory without them you can push a run
into the default entity even though it was created with the right one recorded locally. Always
sync via `scripts/wandb_sync.sh`, which passes both.

### Where the offline data lands

`init_wandb()` passes `dir=<run_dir>/wandb`, and wandb appends its own `wandb/` under that, so
the actual layout is:

```
/scratch/l/lovenya/salm_model_outputs/runs/<arch>/<run_name>/
├── wandb_run_id.txt                       # our persisted run id (see §4)
└── wandb/wandb/
    ├── offline-run-20260812_041432-ncoohd92/
    │   ├── run-ncoohd92.wandb             # the transaction log — this is what gets uploaded
    │   ├── files/                         # config.yaml, requirements.txt, console log
    │   ├── logs/
    │   └── tmp/
    ├── latest-run -> offline-run-...      # symlink
    ├── debug.log
    └── debug-internal.log
```

The doubled `wandb/wandb` is cosmetic (a consequence of `dir=`), not a bug. It does mean the
offline dirs are six levels below `salm_model_outputs`, so any `find` you write by hand needs
enough depth; `scripts/wandb_sync.sh` uses an unbounded `find`, which is correct.

---

## 3. The sync workflow

**Compute nodes cannot sync. Login nodes can.** Auth is already in place: `~/.netrc` holds the
W&B credentials from a previous `wandb login` (never paste an API key into a script or a
commit — `wandb login` on a login node is the only way it should get there).

### Normal case: after a job finishes

```bash
# from a login node, in the repo
bash scripts/wandb_sync.sh                                   # every offline run under runs/
bash scripts/wandb_sync.sh --list                            # dry look at what exists
bash scripts/wandb_sync.sh /scratch/l/lovenya/salm_model_outputs/runs/s2a_audioldm_whisper_xformer/long100
```

The last form is what the long-run sbatch prints at the end of its log:

```
W&B: sync with  bash scripts/wandb_sync.sh $RUN_DIR
```

Under the hood each invocation is:

```bash
wandb sync --project salm-s2a --entity lovenya-org <offline-run-dir>
```

### When to run it

- After the job's last fragment exits (see §4 — auto-resume chains produce several fragments).
- **Not while the job is still running.** `wandb sync` on a live run reads an incomplete
  transaction log; it can upload a partial run and — worse — leave a `.synced` marker behind,
  after which later re-syncs skip the run and the remaining metrics never arrive. If you truly
  need in-flight metrics, `wandb beta sync --live` exists for that, but the supported habit
  here is: job ends, then sync.
- Before any scratch cleanup or purge that would touch `runs/` (see §6).

### Is re-syncing safe?

Yes, with one caveat. Verified in the installed 0.28.1 source:

- `wandb sync` in 0.28.x **reroutes to `wandb beta sync`** unless you pass a legacy-only flag
  (`--sync-all`, `--append`, `--clean`, `--include-globs`, `--skip-console`, `--no-mark-synced`,
  `--sync-tensorboard`, `--view`) or `--legacy`. Our flags (`--project`, `--entity`) are
  allowed in the beta path, so `scripts/wandb_sync.sh` runs through beta sync. It prints
  `Using wandb beta sync. Turn this off with --legacy.` — that message is expected, not a
  warning about our usage.
- Beta sync defaults to `skip_synced=True`, and "synced" means **a `<name>.wandb.synced`
  marker file exists next to the `.wandb` file** (`beta_sync._is_synced`). So a second run of
  the script skips what is already uploaded and prints `Skipped N synced run(s)`.
- To deliberately re-upload, pass `--no-skip-synced` / `--include-synced`.
- Caveat: legacy sync only writes the marker if the run actually *finished*
  (`wandb/sync/sync.py`: `if self._mark_synced and not self._view and finished`). A fragment
  killed by walltime may therefore be re-uploaded on the next sweep. That is harmless
  (idempotent replay into the same run id) but explains a re-upload you didn't expect.

### Current state (as of writing)

Nothing has been synced yet — there is not a single `*.wandb.synced` marker anywhere under
`/scratch/l/lovenya/salm_model_outputs`. Unsynced offline runs on disk:

| run | run id | offline dir size |
| --- | --- | --- |
| `long100` | `ncoohd92` | 4.1 MB |
| `long100_slowval_aborted` | `cbyrr429` | 436 KB |
| `smoke_resume_20260812_0211` | `prnq4ivb` (2 fragments) | 460 KB |

**Action: sync these.** They are small, they are the only copy of those curves, and they sit
on a scratch filesystem with a purge policy.

---

## 4. Resume and W&B run ids

`init_wandb()` in `src/salm/train_ldm.py`:

1. looks for `<run_dir>/wandb_run_id.txt`;
2. if present, calls `wandb.init(id=<that id>, resume="allow", ...)`;
3. if absent, lets wandb mint an id and writes it to that file.

This matters because `run_tamia_s2a_long100.sbatch` is an **auto-resume** script: `RUN_NAME` is
stable (not timestamped) and each 4-hour job resumes from the last checkpoint in the same run
dir. A 100-epoch experiment is therefore many Slurm jobs against one run dir.

**Without the persisted id**, each job would mint a fresh run: N disjoint short curves in the
workspace, none of them showing the actual training trajectory, each with its own step-0
origin — precisely the failure the mechanism exists to prevent.

**With it**, every fragment writes into the same logical run. On disk, each *job* still creates
its own `offline-run-<timestamp>-<id>` directory that shares the run id — visible in
`smoke_resume_20260812_0211`, which has two directories both ending in `prnq4ivb`. That is
normal and expected.

Consequences for syncing fragments:

- All fragments of a run must be synced, not just the newest one.
- Order matters. wandb 0.28.1's beta sync handles this itself *when the fragments are passed to
  one invocation*: "When syncing multiple files that are part of the same run, the files are
  synced sequentially in order of start time regardless of this setting. This happens for
  resumed runs or when using the `--id` parameter" (`wandb/cli/beta.py`). Our script currently
  loops one `wandb sync` per directory instead — it happens to sync in chronological order
  because the timestamped directory names sort that way, but see §5 for the better form.
- Step monotonicity is a real trap. We log with an explicit `step=global_step`, and `global_step`
  is restored from the checkpoint. W&B drops history rows whose step is not greater than the
  highest step already recorded for that run. So if a fragment resumes from a checkpoint that is
  *older* than the last logged step — e.g. resuming from `best` instead of `last`, or a crash
  between a log call and the next checkpoint write — the re-run's early metrics are silently
  discarded, and the curve shows a flat gap rather than an error. If curves ever look truncated
  after a resume, this is the first thing to check.

---

## 5. What the docs recommend that we are NOT doing

This is the section worth acting on. Nothing below has been changed in code by this document —
`_salm_env.sh`, `train_ldm.py`, and `wandb_sync.sh` are being edited elsewhere.

### 5.1 Sync is never actually triggered (highest-value gap)

Alliance's workflow is "run offline, then `wandb sync` on the login node." We implemented the
script and then never ran it — three runs, zero markers. Offline mode is only half a pipeline;
the half that produces a dashboard is a manual step nobody is prompted to take.

Concrete options, in order of effort:

- Sync now, and make it a habit after every job's final fragment.
- Have the sbatch epilogue write a one-line "needs sync" breadcrumb (e.g. touch
  `$RUN_DIR/NEEDS_WANDB_SYNC`) that `wandb_sync.sh` clears, so `--list` shows genuine debt
  rather than "every directory that exists."
- Longer term, the community pattern for exactly this cluster shape is
  [`wandb-osh`](https://pypi.org/project/wandb-osh/) /
  [`wandb-offline-sync-hook`](https://github.com/klieret/wandb-offline-sync-hook): the training
  process drops a trigger file, and a small daemon on the login node syncs when it appears —
  near-live dashboards without any compute-node networking. This needs an install, so it is a
  user decision, not an agent one.

### 5.2 Sync fragments in one invocation, not a loop

Alliance's own example is a single globbed invocation:

> `wandb sync ./wandb/offline-run*`

Our loop calls `wandb sync` once per directory. Two downsides: it re-pays wandb's service
startup per directory, and it forfeits beta sync's built-in "order fragments of the same run by
start time" guarantee, relying on `sort` over directory names instead. Proposed change to
`scripts/wandb_sync.sh`: collect the paths and pass them all in one call, e.g.

```bash
wandb sync --yes --project "$WANDB_PROJECT" --entity "$WANDB_ENTITY" "${OFFLINE[@]}"
```

### 5.3 Add `--yes` for non-interactive syncing

Beta sync prompts for confirmation when invoked with no explicit paths; `--yes` is the
beta-mode flag that skips it and is compatible with the beta path (it is *rejected* in legacy
mode). Harmless when paths are explicit, necessary the moment this is ever run from cron, a
wrapper, or a non-tty shell.

### 5.4 Keep the rest of W&B's state off `/home`

We set `WANDB_CACHE_DIR` but not the sibling directories, all of which default under `$HOME`:

| Variable | Default | Should be |
| --- | --- | --- |
| `WANDB_CONFIG_DIR` | `~/.config/wandb` | under `$OUTPUTS_ROOT` |
| `WANDB_DATA_DIR` | `~/.cache/wandb` (artifact staging) | under `$OUTPUTS_ROOT` |
| `WANDB_ARTIFACT_DIR` | `./artifacts` | under `$OUTPUTS_ROOT` |
| `WANDB_DIR` | cwd | already handled via `dir=` in `wandb.init` |

`$HOME/.cache/wandb` already exists (small — 16 KB, 4 files), so this is prevention rather than
a fire. Home is at 207K/250K **files**, and inodes are the tighter constraint on this system.
(Verified names against `wandb/env.py` in 0.28.1.)

### 5.5 Never call `wandb.save()` / artifacts from a compute node

Alliance: "Uploading artifacts to W&B with `wandb.save()` requires access to Google Cloud
Storage and will cause your job to freeze or crash." Good news: this repo currently makes no
`wandb.save`, `wandb.log_artifact`, `wandb.Artifact`, `wandb.Audio`, `wandb.Image`, or
`wandb.Table` call anywhere — metrics only. Keep it that way. If we later want generated audio
in W&B, log it from a **login-node** script after syncing, or accept that offline media inflate
the offline dir and only upload at sync time.

### 5.6 Version consistency between writer and syncer

The offline `.wandb` file is a versioned transaction log. We currently write and sync with the
same venv (wandb 0.28.1), which is the safe configuration. If the venv is ever rebuilt or the
client upgraded mid-project, sync outstanding offline runs *first*, or keep the old venv around
to sync them.

### 5.7 Prefer the env var over `wandb offline`

The `wandb offline` CLI flips a persisted setting in `WANDB_CONFIG_DIR`. For batch jobs that is
fragile — it depends on a home-directory file being present and correct for whichever user/
node runs the job. `WANDB_MODE=offline` in `_salm_env.sh` plus `mode=` on `wandb.init()` is the
right choice and needs no change; noted here so nobody "fixes" it by adding a `wandb offline`
call to an sbatch script.

---

## 6. Gotchas: disk, inodes, and deleting offline dirs

Current filesystem pressure (`diskusage_report`):

```
/scratch (user lovenya)   932GiB/1024GiB   908K/1000K files
/home    (user lovenya)    18GiB/  25GiB   207K/ 250K files
```

Scratch is at ~91% of both space and inodes. What that means for W&B:

- **Offline runs are cheap.** The largest here is 4.1 MB / 15 inodes; the resumed one is 25
  inodes. A metrics-only offline run grows roughly linearly with logged points, not with model
  size. W&B is not what will fill scratch — checkpoints and precomputed latents are. Do not
  compromise logging to save space.
- **`wandb sync` does not shrink anything.** It reads the `.wandb` file and uploads; the only
  thing it adds locally is a zero-byte `<name>.wandb.synced` marker. Nothing is moved or
  deleted.
- **Deleting after sync is safe, with two conditions.** Once `<run-dir>/run-<id>.wandb.synced`
  exists, the offline directory has served its purpose — the data is on wandb.ai. Two rules:
  1. **Never delete `<run_dir>/wandb_run_id.txt`.** It is not part of the offline data; it is
     what lets a future resume rejoin the same run. Deleting it turns the next resume into a
     brand-new disjoint W&B run.
  2. Per repo policy (`CLAUDE.md`), don't delete artifacts from an agent — move them to a
     `legacy/` directory and let the user delete.
- **wandb's own cleanup** is `wandb sync --clean --clean-old-hours 48 --clean-force`, which
  deletes local data for runs already marked synced. Note `--clean` is legacy-only, so it runs
  the old code path; it also operates on a `wandb` directory it discovers rather than our
  nested layout, so verify with `--dry-run`-style inspection before trusting it. We do not use
  it today; the manual "sync, verify marker, then archive" path is safer given our layout.
- **Purge policy.** Scratch is purge-eligible. An unsynced offline run is a single point of
  failure for a curve that cost GPU-hours. Sync is the backup.

---

## 7. Quick reference

```bash
# Sync everything (login node, from the repo)
bash scripts/wandb_sync.sh

# See what offline runs exist
bash scripts/wandb_sync.sh --list

# Sync one run dir
bash scripts/wandb_sync.sh /scratch/l/lovenya/salm_model_outputs/runs/<arch>/<run_name>

# Force re-upload of an already-synced run
wandb sync --no-skip-synced --project salm-s2a --entity lovenya-org <offline-run-dir>

# Confirm a run was synced
find /scratch/l/lovenya/salm_model_outputs/runs -name '*.wandb.synced'

# Which entity/project am I about to push to?
echo "$WANDB_ENTITY / $WANDB_PROJECT"     # must be lovenya-org / salm-s2a
```

Never: `module load httpproxy` for wandb, `WANDB_MODE=online` in an sbatch script,
`wandb.save()` from a compute node, or an unset `WANDB_ENTITY`.
