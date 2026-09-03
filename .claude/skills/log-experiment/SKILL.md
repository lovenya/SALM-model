---
name: log-experiment
description: Record completed experiments in docs/EXPERIMENT_LOG.md, portable lessons in docs/LEARNINGS.md, and live/rejected directions in docs/RESEARCH_DIRECTIONS.md -- what we tried, why, what the numbers were, and what we concluded. Invoke as /log-experiment.
---

# Log an experiment

Write the experimental record: what was tried, **why it was tried**, what the numbers
actually were, and **what conclusion was drawn**. This is distinct from `/update-docs`, which
keeps `CONTEXT.md`/`README.md` describing the project's *current state*. This skill records
its *history and reasoning* — the thing that is impossible to reconstruct later from code.

The test of a good entry: six months from now, could someone read it and know why a decision
was made, without asking anyone?

## Three destinations, different jobs

| doc | holds | lifetime |
|---|---|---|
| `docs/EXPERIMENT_LOG.md` | dated entries per experiment: hypothesis, config, numbers, verdict | append-only history |
| `docs/LEARNINGS.md` | portable lessons that would apply on a different project | grows slowly, high bar |
| `docs/RESEARCH_DIRECTIONS.md` | live directions (D1..Dn), and rejected ones with reasons | edited in place as status changes |

## What every EXPERIMENT_LOG entry must contain

1. **Date and job IDs.** Slurm IDs are the index into logs, checkpoints and generations.
2. **The question.** What was uncertain before this ran. If you cannot state it, the
   experiment was not worth running and the entry should say so.
3. **What was actually configured** — the values that differed from the baseline, not the
   whole config. Include the run names/paths so artifacts are findable.
4. **The numbers, verbatim.** Real measured values with their units and their calibration
   anchors. Never "improved significantly" — write `0.4438 vs 0.9669 (zero control)`.
5. **The verdict against the PRE-REGISTERED criteria.** The repo rule is that every job
   states success/failure/ambiguous before running. Say which one happened, and if the
   result does not fit any of them, say that explicitly — that is a sign the criteria were
   wrong, which is itself worth recording.
6. **What it changes.** The next action this result implies, or the direction it kills.

## Rules that keep the log trustworthy

- **Record negatives with the same care as positives.** Most entries in this project are
  negative results, and they are the reason the positive ones are believable. A failure with
  its mechanism explained is more valuable than a success with none.
- **Separate "the metric moved" from "the thing works".** This project has repeatedly been
  fooled by a metric improving for reasons unrelated to the mechanism. If a number moved but
  the audio did not, write both facts.
- **Quote listening results as evidence.** Human judgements of the audio are data. Attribute
  them ("2026-09-03, user: ...") and record the exact wording — "3-4 words make sense then
  another group starts" localised a problem no metric had.
- **Own the mistakes, in the log.** Bugs that invalidated results, confounded instruments,
  metrics that measured the wrong component: these belong in the record. Several of this
  project's most useful lessons came from its own errors, and a log that hides them teaches
  nothing.
- **Do not restate what code or git history already says.** Link to the file or commit.
  Record the *reasoning*, which is what neither of those preserves.
- **Verify numbers before writing them.** Read `history.json`, the scoring JSONs, or the job
  log. Never write a figure from memory or from an earlier message in the conversation.

## LEARNINGS.md has a higher bar

Only add an entry if it would be true and useful on a **different project**. "Our CLAP
encoder was misconfigured" is a log entry; "`from_pretrained` on a mismatched class silently
returns a randomly-initialised model, so assert a numeric invariant rather than the absence
of an exception" is a learning. Each entry: the failure, the mechanism, and what to do
instead.

## Procedure

1. Identify what finished since the last entry — check `sacct`, the run directories, and the
   scoring JSONs, not just the conversation.
2. Read the real numbers out of the artifacts on disk.
3. Append the `EXPERIMENT_LOG.md` entry.
4. Update `RESEARCH_DIRECTIONS.md` if a direction's status changed (activated, killed,
   partially confirmed) — including moving things into "Considered and rejected" with the
   reason.
5. Add to `LEARNINGS.md` only if the bar above is met.
6. Commit with a message that states the finding, not just "update docs".
