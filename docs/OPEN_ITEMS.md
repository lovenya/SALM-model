# Open items

## What this document is

**The live TODO and decision queue.** Everything still to discuss, decide, or build — kept separate
from `EXPERIMENT_LOG.md` (which records what already happened and why) because at this pace items
get lost between sessions otherwise.

Completed items are **struck through and kept in place**, not deleted, so the history of what was
proposed and what actually happened stays readable — including things we decided *not* to do, which
are the easiest to forget and the most likely to be re-proposed.

**Conventions.** **[DISCUSS]** needs a decision from Lovenya. **[IMPL]** is agreed, just needs
doing. **[BLOCKED]** waits on something named. ~~**[DONE]**~~ struck through with the outcome and,
where relevant, a pointer into the experiment log. ~~**[DROPPED]**~~ struck through with the reason
we chose against it.

---

## Waiting to run

- **[IMPL] Multi-snapshot generation.** Ready, not submitted.
  `sbatch submission_scripts/run_tamia_s2a_gen_evolution.sbatch` — 10 lanes, predicted start
  19:33 EDT / 04:03 IST on `aip-csubakan`. Produces `listen/<sample_id>/` for A/B listening plus
  per-lane and total timing.
- ~~**[DONE 2026-08-12] D1 + D2 diagnostics**~~ — ran as job 406885, AND are now logged per-epoch.
  **But the first version of D1 was broken and its result must not be cited.** It computed
  `loss(shuffled) − loss(correct)`, a difference of two numbers at ~0.19 with σ 0.006, so its
  ~0.0002 deltas were two orders of magnitude below its own resolution. Proof it was blind rather
  than correct: it reported +0.00017 for O2, whose conditioner reached 9× chance. **Rebuilt** as a
  relative prediction gap `‖ε(c)−ε(c_shuffled)‖/‖ε(c)‖`, plus a zero-vector reference for D2 because
  the learned null embedding can drift toward the mean condition. See EXPERIMENT_LOG.
- ~~**[DONE 2026-08-12] Collective-deadline fix verified in production.**~~ All three long100 runs
  (L0, L0f, O2) completed a full 100 epochs and exited cleanly — no NCCL watchdog timeout of the kind
  that killed 406301 at epoch 91. Verified by outcome rather than by the planned 2-rank smoke test.

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
- **[IMPL — BUILT, NOT RUN] Guidance-scale sweep.** Now the single highest-value cheap experiment,
  and it has a quantitative motivation rather than "it's untuned". CFG computes
  `ε = ε(null) + s·(ε(c) − ε(null))`, and that bracket **is D2**. O2's measured D2 = 0.0234, so at
  s=3.0 conditioning shifts the prediction only ~7% from unconditional — enough to explain
  condition-blind output from a conditioner that provably works. Reaching ~15% needs s ≈ 6.5.
  Built: `submission_scripts/run_tamia_s2a_guidance_sweep.sbatch` (s ∈ 1/3/5/7.5/10/15, identical
  rows and seed, 4 GPUs) plus `scripts/score_generations_clap.py`, because six directories of audio
  is not a measurement. **Decides between the two live hypotheses:** audio_cos rising with s ⇒ we
  were under-guiding; audio_cos flat ⇒ the conditioning INTERFACE is the bottleneck and Tier 3 is
  next, not more training.
- **[DISCUSS] Which ladder rungs are worth the compute**, and how much breadth across Tiers 2–5 of
  the menu in `~/.claude/plans/polished-crafting-crown.md`.
- ~~**[DONE 2026-08-12] Model selection fixed.**~~ `select_metric` knob, set to `val_clap_r1` in
  `configs/base`. Retrospectively vindicated: O2's val_loss went **up** (0.1978 → 0.1985) across
  exactly the 100 epochs in which its conditioner went from chance to 9× chance, so val_loss is not
  merely uninformative here but *anti-correlated* with learning. Selecting on it would have preferred
  the model that learned nothing. Resume also restores the selection metric's best from history,
  without which a resumed run would overwrite `checkpoint_best` on its first validation.

## Agreed direction, not yet built

- ~~**[DONE 2026-08-12] Tier 2 auxiliary alignment loss (O2) — AND IT WORKED.**~~ The single
  biggest result so far: val_clap_r1 0.325 → **0.564** against chance 0.0625 (**9×**), val CLAP
  cosine **+0.613** against a ±0.0442 band (**14×**), reached within 10 epochs then plateaued.
  Meanwhile L0 and L0f sat at chance for 100 epochs. **So the objective was the problem, not
  conditioner capacity** — a better result than the planned "complexity earns its keep" narrative.

  **λ = 2000 was WRONG and was not used.** That paper's term is squared error on *unnormalized*
  embeddings damped by `w(n)=10^(−n/200)`, so its effective magnitude is tiny. Ours is `(1−cos)` on
  L2-normalized vectors, bounded [0,2] and ~1.0 at init against a diffusion loss of 0.198 — λ=2000
  would have made alignment ~10,000× the objective. Used **λ = 1.0**. Copying the published number
  would have been an expensive silent mistake.
- **[IMPL] Tier 2 — two-stage / V2A-Mapper style.** Train the conditioner to regress the target's
  CLAP embedding, then feed a **fully frozen** AudioLDM. Minutes of compute, and it makes the CLAP
  cosine a *valid* metric again because the U-Net stops co-adapting.
- ~~**[DONE 2026-08-12] L0 (0.66 M) and L0f (frozen U-Net), 100 epochs each.**~~ Both at chance
  (r1 0.065 and 0.070 vs chance 0.0625) with σ 0.0012/0.0025 — genuinely flat, not noisy.
- ~~**[DEPRIORITISED] L1 (7.1 M) and L3 (18.6 M).**~~ Struck not because they are done but because
  L0 (0.66 M) and L2 (32.3 M) both sat at chance under the plain diffusion loss: interpolating
  between two identical endpoints measures nothing. ~29 GPU-h saved. Revisit **only** if the
  guidance sweep shows the interface works, at which point capacity might start to matter.
- ~~**[DONE 2026-08-12] Retrieval baseline (B1), both spaces.**~~ And it produced an unexpected
  finding: nearest training neighbours are **0.97 cosine-similar** while sharing only **0.055**
  caption Jaccard and **0/16** exact scenes. Pooling is masked (`retrieval_baseline.py:94-95`), so
  this is not the padding artifact I first suspected — the conditioner's *input* space genuinely does
  not separate scenes, which is a plausible partial explanation for why the diffusion gradient alone
  never found the mapping.
- **[PARTIAL] Efficiency numbers as a first-class result.** Measured so far: **0.69 s per generated
  clip** at 50 DDIM steps (gen-evolution, 10 lanes in 111 s), and 2.16 min/epoch training. Still
  missing: peak VRAM for a single inference, parameter count table, and stage/model count versus the
  cascade. T0 will supply the one-forward-pass latency floor to compare against.

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

- ~~**[DONE 2026-08-12] Loss and validation metrics.**~~ Resolved by building the replacements
  rather than by discussion: `val_clap_r1`/`r5` (free, real chance baseline of 1/batch_size),
  `val_d1_gap`/`val_d2_gap`/`val_d2_gap_zero`, `val_x0_rel_mse`. Deliberately **rejected**
  `target_caption_clap_score` and FAD as in-training metrics — both need full sampling (50 steps +
  VAE + vocoder) which would dwarf a 2.16 min epoch, and under O2 a CLAP-based score partly measures
  our own training objective, so it must never drive selection. Held-out eval only.
- ~~**[ANSWERED 2026-08-12] GPU utilisation.**~~ The waste is real but not where it looked: **84.6%
  of every Whisper pass is padding**. Raising batch size processes more padding in parallel and does
  not help. See the efficiency section at the end of this doc — the fix is precomputing the frozen
  encoder's features, still unbuilt.
- **[DISCUSS] A2S direction.** Not started; S2A assumed throughout.

## Housekeeping

- **[IMPL] `git push`** — several commits ahead of `origin/main`.
- ~~**[RESOLVED 2026-08-12] Account policy: throughput wins over the ratio.**~~ User clarified the
  60/40 csubakan/ravanelm split is *guidance*, not a constraint — if one account starts sooner, use
  it. Pick by `sshare` FairShare and queue depth at submit time; treat 60/40 as a tiebreaker only;
  never resubmit a queued job just to fix the ratio. **And do NOT use `sbatch --test-only` to compare
  starts** — it simulates a job behind everything else and once reported 09:56 against a real 11:36.
  The only reliable read is `squeue -j <id> -o %S` after submitting.
- **[IMPL] Delete `salm_model_outputs/legacy/`** (130 MB of superseded smoke artifacts) when you're
  satisfied nothing there is needed. Scratch is at **91% space, 908K/1000K inodes**.
- **[DONE 2026-08-12] `wandb sync`** — ran; 2 of 4 runs synced. `ncoohd92` (long100) and
  `cbyrr429` failed with a truncated transaction log because both processes were killed without
  flushing. Unrecoverable tail; partial data did upload. `history.json` is the authoritative record
  and is complete. Future syncs retry them automatically (no marker was written).

## Next up (as of 2026-08-12, end of first real cohort)

Ordered by information per GPU-hour, which is how every earlier decision here was made.

1. ~~**[DONE 2026-08-13] Guidance sweep on O2** (job 407327).~~ Result: `clap_audio_cos` 0.491 at
   s=1, 0.553 at s=3, then **flat-to-declining** — s=3→15 is −0.017 across a 5× guidance increase.
   **But the number is not yet interpretable**: it has no mismatched-target control, and a lane
   emitting one identical output for every input still scores 0.20. Job 415856 is re-scoring with
   MARGIN and r1. Do not draw the Tier-3 conclusion until that lands.
2. ~~**[DONE 2026-08-13] Generate from O2** via gen-evolution (job 407330).~~ 11 lanes, 24 rows
   each, no silent files, 0.58 s/clip. Includes the `pristineunet_*` lanes that test the
   conditioning-collapse hypothesis and the rung0 ceiling. **Scoring pending** (job 415856) — RMS
   alone says nothing semantic.
3. ~~**[DONE 2026-08-13] T0 ran, 96 epochs** (job 407331).~~ Confirmed its prediction: std_ratio
   0.315 → 0.477 against real latents at 1.0, and 0.14 s/sample vs diffusion's 0.69 (**4.9× faster**).
   **But see the T0 overfitting entry in EXPERIMENT_LOG** — its `checkpoint_best` is epoch 4 of 96,
   so the generated audio came from an essentially untrained network.
4. **[IMPL] Re-run D1/D2 with the rebuilt metric** on the existing L0/L0f/O2/L2 checkpoints. Cheap
   (~7 min, no training) and necessary: every D1 number currently in the log came from the blind
   version. Also compare `d2_gap` against `d2_gap_zero` to test whether the learned null drifted.
5. **[DISCUSS] λ sweep for O2 — and sweep DOWN, not up.** λ=1.0 worked, but train alignment loss fell
   to 0.014 (train cosine 0.986) against val 0.613, which is real overfitting on the alignment task.
   λ=1.0 makes alignment ~5× the diffusion term. Try **λ ∈ {0.2, 5.0}**, with 0.2 the interesting
   one. This is now evidence-driven rather than guesswork.
6. **[IMPL] Run-dir collision guard.** **Required by the `submit-jobs` skill** ("Reject an existing
   run directory before model loading... never wait until job completion to report the collision"),
   and its absence already cost us: two T0 jobs were submitted with the same default `RUN_NAME` and
   would have interleaved checkpoints, history.json and one W&B run id. Caught by inspection, not by
   the code. Not built.
7. **[IMPL] Precompute frozen Whisper features.** 8.9 GB storing valid frames only; removes the
   635 M forward pass from every step of every epoch *and* the CPU-bound decode/resample. Bigger win
   than any batch-size change. See the efficiency section below.
8. **[BLOCKED on the sweep] Tier 3 — add a sequence/cross-attention conditioning path.** Only
   justified if the sweep comes back flat. Evidence it might be needed: arXiv 2501.14680 shows pooled
   and sequence conditioning are *complementary* (mean-pooled FiLM alone reached FAD 1.89), so the
   real limitation may be that we have **only** the pooled path, not that 512 floats is too few.

## Deliberately chosen against (so we stop re-proposing these)

- **AudioGen / autoregressive.** Not on tooling grounds — the user correctly pushed back that we can
  fork and write our own code, and `WaveformConditioner` is a purpose-built extension point, so I
  withdrew that argument. What stands: our 42 h at 1.1 pairs/scene is the documented failure zone for
  learning a *new* conditioning pathway (audiocraft#447: crackling in >80% of generations,
  `grad_norm=INF`), DCASE 2023 Task 7 at comparable scale went almost entirely diffusion, and the
  benchmark gap is 0.2–1.5 FAD. Revisit after the interface question is settled — if conditioning
  itself is the problem, changing generator family will not fix it. AudioLDM is **no longer a
  requirement** (that was an initial idea, not a supervisor constraint).
- **`target_caption_clap_score` / FAD during training.** Needs full sampling, and contaminated under
  O2. Held-out eval only.
- **Raising the batch size now.** Attacks utilisation, not the 84.6% padding waste, and silently
  breaks `lr_total_steps` and the r1 chance baseline. After the Whisper precompute, deliberately.
- **More training data as the fix.** At 1.1 pairs per unique scene the constraint is supervision
  *density* per example, not example count — which is exactly what O2 added, and O2 worked.
- **`speaker_emotion` experiments.** Dropped at the user's instruction. Also only ~3 of its 47 labels
  are usable (neutral 8082 / happy 3163 / angry 2777, everything else single or double digits).
- **Evaluating the 4-stage cascade as a quality bar.** User decision; the claim is end-to-end plus
  efficiency. The cascade S2A results already exist and are scored, so the pipeline baseline needs
  no new work — only row-filtering to match our 1,878 clean_v2 rows against their 2,277.

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
