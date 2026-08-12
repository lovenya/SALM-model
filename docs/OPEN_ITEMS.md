# Open items

Things still to discuss, decide, or implement. Kept separate from `EXPERIMENT_LOG.md` (which records
what already happened) because at this pace things get lost otherwise.

Conventions: **[DISCUSS]** needs a decision from Lovenya; **[IMPL]** is agreed and just needs doing;
**[BLOCKED]** is waiting on something named. Move items to the experiment log once resolved, with
the outcome.

---

## Waiting to run

- **[IMPL] Multi-snapshot generation.** Ready, not submitted.
  `sbatch submission_scripts/run_tamia_s2a_gen_evolution.sbatch` — 10 lanes, predicted start
  19:33 EDT / 04:03 IST on `aip-csubakan`. Produces `listen/<sample_id>/` for A/B listening plus
  per-lane and total timing.
- **[IMPL] D1 condition-shuffle + D2 conditional-vs-null gap.** The definitive test of whether the
  speech is used at all. Evaluation-time only — runs on `406301`'s existing checkpoints, no
  retraining. **Highest information per GPU-minute of anything outstanding.** Also worth logging D1
  per-epoch in future runs so it becomes a curve.
- **[IMPL] Rerun training with the collective-deadline fix.** `406301` reached 91 of 100 epochs and
  died at its deadline; the fix is committed but unexercised. A short 2-rank run with a tiny
  `max_train_minutes` would verify it before another long job.

## Decisions needed

- **[DISCUSS] Register our model in `SpeechAudioLM/configs/model_capabilities.json`.** Blocks the
  whole metric suite (`fad`, `kad`, `kl_passt`, `target_caption_clap_score`, `inception_score`).
  One line — `"salm_audioldm_whisper": {"tasks": ["S2A","T2A"]}` — but it edits the *other* repo,
  against the clean-room principle. Alternatives: borrow an already-registered name, or you make the
  edit.
- **[DISCUSS] The minimal pipeline baseline** — ASR → transcript → off-the-shelf text-to-audio. You
  ruled out evaluating the heavy 4-stage cascade, which I accepted. This is the *narrow* two-stage
  version, and the literature is emphatic that it is the row reviewers demand, because it decides
  whether end-to-end is needed at all. V2A-Mapper's analogue loses badly (FD 56.40 vs 24.17), which
  is the shape of result we'd want. Worth revisiting for this version only.
- **[DISCUSS] Guidance scale.** `s=3.0` is my arbitrary choice (diffusers' AudioLDM default is 2.5,
  AudioGen uses 3.0). Untuned, and it materially affects quality and diversity. Sweep it?
- **[DISCUSS] Which ladder rungs are worth the compute**, and how much breadth across Tiers 2–5 of
  the menu in `~/.claude/plans/polished-crafting-crown.md`.
- **[DISCUSS] Model selection is currently broken.** "Best val loss" selects on noise (expected min
  of 91 draws ≈ 0.1805, observed 0.1833). We need a selection criterion that isn't noise — candidates:
  low-`t` bucket loss, D1 delta, or a generated-audio metric.

## Agreed direction, not yet built

- **[IMPL] Tier 2 — auxiliary alignment loss (top recommendation).** `L = L_DDPM + λ·MSE` to the
  frozen CLAP audio embedding, `w(n) = 10^(−n/200)`, **λ = 2000**. Published recipe (Latent CLAP
  Loss, arXiv 2403.12182) on almost exactly our setup, worth ≈19% relative FAD there. λ is the
  actionable detail — a naive λ≈1 would do nothing.
- **[IMPL] Tier 2 — two-stage / V2A-Mapper style.** Train the conditioner to regress the target's
  CLAP embedding, then feed a **fully frozen** AudioLDM. Minutes of compute, and it makes the CLAP
  cosine a *valid* metric again because the U-Net stops co-adapting.
- **[IMPL] The ladder rungs** L0 `mean_linear` (0.66M) and L1 `attn_mlp` (7.1M) at matched `epochs`
  and `lr_total_steps`. **No longer optional** — V2A-Mapper's own table shows their 2.1M MLP scoring
  the best CLIP-score in its family, so any reviewer will demand our equivalent row.
- **[IMPL] Retrieval baselines** (nearest training speech → its paired ambient clip, verbatim). Free,
  no training. Expect it to **win FAD** (it returns real audio, so it's on the real manifold) and lose
  per-example — which is a good result if reported with a per-example metric and a novelty check.
- **[IMPL] Efficiency numbers as a first-class result**: parameter count, peak VRAM, wall-clock
  latency per sample, number of models/stages. This is the load-bearing claim, so it needs measuring
  rather than asserting.

## Known-imperfect, deliberately deferred

- **Per-channel latent variance is uneven** (ch0 std 1.355 vs 0.86–0.96). A single scalar scaling
  factor cannot express this. Probably minor; no action.
- **`speech_seconds=16.0` padding is largely pointless** — Whisper re-pads to its own 30 s window
  regardless, so our fixed length only serves DataLoader batching.
- **We use only Whisper's final encoder layer.** LTU-AS attends over *multiple* Whisper layers, which
  is a cheap untried improvement.
- **VAE posterior is nearly deterministic** (`E[σ]=0.088`), so posterior sampling adds almost no
  augmentation — an assumption that turned out not to hold.
- **`best` and `last` are both epoch 90** and probably byte-identical.
- **W&B drops non-increasing steps.** We log with an explicit `step=global_step`, and a resumed job
  rejoins the same run id, so resuming from a checkpoint *behind* the furthest-logged step silently
  loses metrics until it catches up. `checkpoint_last` is always safe; `checkpoint_best` may not be.
  Now warned about loudly at resume time. `history.json` is unaffected either way.

## Questions you raised that we deferred

- **[DISCUSS] Loss and validation metrics in detail** — you explicitly parked this. The material is
  ready: why ε-MSE is structurally uninformative here, what the timestep buckets show, and what to
  replace the selection criterion with.
- **[DISCUSS] GPU utilisation and efficiency of the training loop** — agreed to look at this after we
  have the generation timing data.
- **[DISCUSS] A2S direction.** Not started; S2A assumed throughout.

## Housekeeping

- **[IMPL] `git push`** — several commits ahead of `origin/main`.
- **[IMPL] Account rebalancing.** Standing target is 60% `aip-csubakan` / 40% `aip-ravanelm`; recent
  runs have leaned on whichever had better FairShare. Check both with `sbatch --test-only` rather than
  assuming.
- **[IMPL] Delete `salm_model_outputs/legacy/`** (130 MB of superseded smoke artifacts) when you're
  satisfied nothing there is needed. Scratch is at **91% space, 908K/1000K inodes**.
- **[DONE 2026-08-12] `wandb sync`** — ran; 2 of 4 runs synced. `ncoohd92` (long100) and
  `cbyrr429` failed with a truncated transaction log because both processes were killed without
  flushing. Unrecoverable tail; partial data did upload. `history.json` is the authoritative record
  and is complete. Future syncs retry them automatically (no marker was written).
