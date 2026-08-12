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

See `CONTEXT.md` for project state, decisions, and the data/architecture details.
