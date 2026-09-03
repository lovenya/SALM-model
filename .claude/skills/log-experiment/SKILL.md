---
name: log-experiment
description: Record the experimental history across all four continuously-maintained docs -- EXPERIMENT_LOG.md (what we tried, why, the real numbers, the verdict), LEARNINGS.md (transferable ML lessons, written to read cold), RESEARCH_DIRECTIONS.md (live and rejected directions), OPEN_ITEMS.md (the live TODO/decision queue). Invoke as /log-experiment.
---

# Log an experiment

Write the experimental record: what was tried, **why it was tried**, what the numbers
actually were, and **what conclusion was drawn**. This is distinct from `/update-docs`, which
keeps `CONTEXT.md`/`README.md` describing the project's *current state*. This skill records
its *history and reasoning* — the thing that is impossible to reconstruct later from code.

The test of a good entry: six months from now, could someone read it and know why a decision
was made, without asking anyone?

## Four destinations, different jobs

Check every one of these on each invocation. The most common failure of this skill is
updating the log and silently leaving the other three stale.

| doc | holds | who it is for | lifetime |
|---|---|---|---|
| `docs/EXPERIMENT_LOG.md` | dated entry per experiment: the question, the config, the real numbers, the verdict against pre-registered criteria, what it changes | us, and the eventual paper | append-only history |
| `docs/LEARNINGS.md` | transferable knowledge — the mechanism behind a failure, stated so it is useful on a *different* project. Lovenya's reference for interviews and for general ML judgement, so write it to be read cold, without this project's context | Lovenya, long after this project | grows slowly, high bar |
| `docs/RESEARCH_DIRECTIONS.md` | the idea space: live directions (D1..Dn) with status, and "considered and rejected" with the reason | us, when choosing what to try next | edited in place as status changes |
| `docs/OPEN_ITEMS.md` | the live TODO and decision queue — what still needs deciding, building, or is blocked | Lovenya + agents between sessions | edited in place; completed items struck through, never deleted |

`OPEN_ITEMS.md` has its own conventions, stated at the top of that file, and they must be
followed: **[DISCUSS]** needs a decision from Lovenya, **[IMPL]** is agreed and just needs
doing, **[BLOCKED]** waits on something named, and finished items are struck through with
their outcome and a pointer into the experiment log rather than removed. Items we decided
*against* stay struck through too — they are the easiest to forget and the most likely to be
re-proposed.

`CONTEXT.md` and `README.md` are NOT this skill's job — they describe the project's current
state and belong to `/update-docs`. If a result changed the project's state as well as its
history, say so and suggest running that skill too.

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

Only add an entry if it would be true and useful on a **different project, in a different
domain**. "Our CLAP encoder was misconfigured" is a log entry; "`from_pretrained` on a
mismatched class silently returns a randomly-initialised model, so assert a numeric invariant
rather than the absence of an exception" is a learning.

Write these to be read **cold**, by someone with no knowledge of this project — including
Lovenya in an interview two years from now. That means: name the general principle first,
then use the concrete incident as the illustration, not the other way round. Each entry gets
the failure, the *mechanism* that caused it, and what to do instead. A learning without a
mechanism is an anecdote.

## Procedure

1. Identify what finished since the last entry — check `sacct`, the run directories, and the
   scoring JSONs, not just the conversation.
2. Read the real numbers out of the artifacts on disk.
3. Append the `EXPERIMENT_LOG.md` entry.
4. Update `RESEARCH_DIRECTIONS.md` if a direction's status changed (activated, killed,
   partially confirmed) — including moving things into "Considered and rejected" with the
   reason.
5. Update `OPEN_ITEMS.md`: strike through what the experiment settled (with its outcome and a
   pointer into the log), and add whatever it newly raised. This is the step most often
   skipped, and skipping it is why decisions get re-litigated across sessions.
6. Add to `LEARNINGS.md` only if the bar above is met.
7. Commit with a message that states the finding, not just "update docs".
