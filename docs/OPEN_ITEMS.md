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

- **[DONE 2026-08-12] Registered in `SpeechAudioLM/configs/model_capabilities.json`** as
  **`salm_e2e`** with `["S2A", "T2A", "A2S"]`. Verified: `validate_model_task` accepts all three and
  correctly rejects `T2T`.

  *Why this name and not `salm_audioldm_whisper`:* an arch-specific name would need re-registering for
  every ladder rung (L0, L1, L3, a Qwen encoder, AudioLDM 2), i.e. repeated edits to the other repo.
  One stable umbrella name means we touch it once; the architecture lives in the run path instead.

  *What registration is:* a plain whitelist. `evaluate.py` calls `validate_model_task` and hard-fails
  with "Unknown model ... Known models: ..." otherwise, so the metric suite could not run at all.
  Adding the name changes nothing else. Its *purpose* is that our outputs get scored by the **same**
  metrics into the same results tree as the cascade numbers that already exist.

  The edit is **left uncommitted** in `SpeechAudioLM` — that file also carries the user's own unstaged
  `dataset_pair` change, so staging it is theirs to do. (A first attempt reformatted the whole file via
  `json.dumps`, turning a 3-line diff into 53; reverted and re-done as a surgical insert.)

  **Follow-up, agreed with the user:** the existing cascade S2A/T2A evals ran on 2,277 test rows (the
  unfiltered split) while our clean_v2 S2A test has 1,878. The results will be filtered to matching
  rows before any head-to-head comparison is reported.

- **[RESOLVED 2026-08-12] The pipeline baseline needs no work** — it already exists and is scored.
  For S2A the cascade *is* ASR → LLM → T2A, so my "minimal two-stage vs heavy four-stage" distinction
  was meaningless. `fad`/`kad`/`kl_passt`/`inception_score`/`target_caption_clap_score` summaries are
  under `results/evals/stages/s2a/t2a/`. Caveat for the eventual comparison: those ran on 2,277 test
  rows (unfiltered) vs our clean_v2 S2A test of 1,878 — matching rows needed for fairness.
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

## Efficiency: after the current cohort finishes (raised by the user 2026-08-12)

The user proposed raising batch size once the queued runs complete, on the grounds that we are
wasting compute (measured: 11.6 of 79.6 GB VRAM, util 61-74%). The instinct is right; the lever is
wrong.

**MEASURED: 84.6% of every Whisper pass is padding.** TTAS speech averages 4.63 s (median 4.34,
p95 8.45, max 14.11) but Whisper always processes a fixed 30 s window -- 1500 frames of which ~232
are real. The trainer's own comments note the frozen 635 M encoder dominates runtime, so it is
spending ~6.5x more time on zeros than on speech.

Raising batch size does not fix this; it processes more padding in parallel. Utilisation would climb
toward 90% while the useful work per epoch stays identical. Utilisation measures how busy the GPU is,
not how much of that work matters.

### Recommended first: precompute the frozen Whisper features

The encoder is frozen, so its output is constant across epochs -- exactly the argument that already
justifies precomputing latents and CLAP targets.

| storage | frames/sample | size |
|---|---|---|
| with 30 s padding | 1500 | 57.7 GB |
| **valid frames only** | **232** | **8.9 GB** |

92 GB free on /scratch, so the 8.9 GB variant fits. Removes the 635 M forward pass from every step
of every epoch AND the CPU-bound speech decode/resample, i.e. both suspected bottlenecks. Benefits
L0, L0f, O2 and T0 identically, since they share the encoder. Not built yet.

### Then, and only with these three fixed together, raise the batch size

| per-GPU bs | effective | steps/epoch | steps @100 ep | of pinned 23,400 | r1 chance |
|---|---|---|---|---|---|
| 16 (current) | 64 | 234 | 23,400 | 100% | 0.0625 |
| 32 | 128 | 117 | 11,700 | 50% | 0.0312 |
| 64 | 256 | 58 | 5,800 | 25% | 0.0156 |

1. **`lr_total_steps` breaks SILENTLY.** At 64/GPU, 100 epochs is 5,800 steps against a curve pinned
   to 23,400, so the run consumes 25% of the LR schedule and ends barely annealed. No error, just a
   quietly worse and non-comparable model. Must be updated in lockstep with batch_size.
2. **`val_clap_r1`'s chance baseline is `1/batch_size`**, so r1 is NOT comparable across batch
   sizes. Compare each run against its own logged `val_clap_r1_chance`, never across cohorts.
3. **LR needs rescaling** (linear or sqrt rule). Omit it and the new runs are under-tuned; apply it
   and two variables changed at once, so the cohort boundary must be documented.

Consequence either way: a batch-size change creates an old cohort (the completed 91-epoch L2 at
effective 64) and a new cohort that cannot be compared directly. Worth doing deliberately, once,
rather than drifting.
