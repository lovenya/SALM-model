# Repo rules

## HARD RULE: never touch Slurm without explicit instruction

**Do NOT run `sbatch`, `scancel`, `scontrol update`, `scontrol requeue`, or anything else
that submits, cancels, holds, requeues, or modifies a job — unless the user has explicitly
told you to, in that message, for that job.**

This is not a default to be weighed against convenience. It applies even when:

- a job is obviously broken, failing, or wasting the allocation;
- a code fix would make the running job faster or better;
- the agent believes restarting is clearly the better trade;
- the user approved a plan that mentions submitting jobs;
- the user approved a *previous* submission or cancellation.

Approval is per-action, not standing. A plan that says "launch the long run" is not
permission to cancel and relaunch it later.

**What to do instead:** state the situation, state the recommendation and its cost, and
give the exact command for the user to run. Then stop and wait.

> Bad:  "Restarting is worth it" → `scancel 406278` → `sbatch ...`
> Good: "406278 is at ~4.5 min/epoch; the fix roughly halves that, so restarting costs the
>        5 epochs done so far and gains ~40. Want me to, or will you? Command:
>        `scancel 406278 && sbatch submission_scripts/...`"

This rule exists because it was broken: on 2026-08-12 the agent cancelled running job
`406278` on its own judgement to apply a performance fix. Losing queued position and
in-progress compute is the user's call, never the agent's — the user owns the allocation and
the deadlines that depend on it.

Read-only Slurm inspection is always fine: `squeue`, `sacct`, `sinfo`, `sshare`,
`scontrol show <job|node|reservation>`, and `sbatch --test-only` (which queues nothing).

## Other standing rules

- **No installs or downloads from the agent.** Compute nodes have no internet, and
  login-node installs are the user's to run. Hand over exact commands instead.
- **No heavy compute on login nodes.** Neural-network forward passes are not "lightweight
  checks". Read library source to answer API questions rather than running a model.
- **Don't delete artifacts.** Move things to a `legacy/` directory and let the user delete.
- **Verify APIs against installed source** before writing code that depends on them. Two
  separate smoke failures in this repo were guessed-at library signatures that a two-minute
  read of `site-packages` would have caught.
- Heavy artifacts live in `/scratch/l/lovenya/salm_model_outputs`, never in git.

## HARD RULE: every job gets an expected-results writeup before/at submission

**Whenever recommending or submitting a job — every job, not just the notable ones — state
up front: what does success look like, what does failure look like, and what does EACH
outcome actually tell us.** Not a vague "we'll see how it goes" — name the specific metric(s)
that will actually move if the hypothesis is right, confirm (by reading the code, not
assuming) that those metrics are actually wired to reflect what's being tested, and say
what each concrete outcome (success / failure / ambiguous) implies for what to do next.

This surfaces real gaps before they're wasted compute: job 430049 (Rank 2 falsification,
2026-08-27) is the example that prompted this rule — the submission script's own
auto-printed summary reports `val_clap_r1`/`train_align`/`val_d2_gap`, but by construction
NONE of the three reflect the new content cross-attention channel (`val_clap_r1`/`train_align`
come from the conditioner's own output, which the new channel never touches; `val_d2_gap`'s
diagnostic pass is a separate `@torch.no_grad()` call documented as blind to it too). The
metric that actually would show it (`val_loss`/`x0_rel_mse`, since the new channel feeds
into the U-Net's own denoising accuracy) isn't even in the auto-print, though it IS recorded
in `history.json` every epoch. This was only caught by reading the metric computation, not
by reading the job's own summary of itself — a template's default output is not a substitute
for checking what a specific run's success condition actually requires.

Applies to every job going forward, and to the plan/config the job comes from, not just the
sbatch script — the goal is that neither the agent nor the user learns "wait, what would even
count as this job working?" only after it finishes.

See `CONTEXT.md` for project state, decisions, and the data/architecture details.
