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
- ~~**[SUPERSEDED 2026-08-23] A2S direction.**~~ No longer a discussion item — it is the active phase. See *Added 2026-08-23* below for the plan, the encoder decision and the gate that has to pass first.

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
8. ~~**[DROPPED 2026-08-16] Frozen-U-Net runs (O2f, O2f λ=0.2).**~~ The hypothesis that motivated
   them is refuted: the pristine U-Net scores MARGIN **−0.0036** against the trained U-Net's
   **+0.2707**, so U-Net finetuning is essential rather than harmful and freezing is the wrong
   direction. ~7 GPU-h saved. User decision to drop rather than run one as a recorded negative.
9. **[DEFERRED] Tier 3 — sequence/cross-attention conditioning path.** Was blocked on the sweep
   coming back flat. It did not: with the mismatched-target control, MARGIN rises to a peak at
   s=7.5, so the pooled interface is working. Not the bottleneck, so this drops well down the list.

### Added 2026-08-15

- **[IMPL] Score the RETRIEVAL baseline semantically — it has never been scored.** B1 ran (job
  406885) and produced 16 clips in each of the Whisper and CLAP spaces, but only ever got
  *similarity* numbers, never a quality/semantic score. Its manifests are already
  `score_generations_clap.py`-compatible (`generated_file_rel` + `sample_id` both present), so this
  is a few minutes of GPU. **Not included in the queued rescore job 415856**, which only points at
  the O2 sweep and generation dirs.
  Why it matters: retrieval returns REAL audio, so it sits on the real manifold by construction and
  is expected to win FAD while losing per-example. MARGIN and `audio_r1` are exactly the metrics
  that can demonstrate that, and we now have them. Known so far: similarity to nearest training
  utterance 0.972 (whisper) / 0.971 (clap), caption Jaccard 0.055 / 0.027, **0/16 exact scene
  matches** — high similarity with near-zero scene agreement.
  Dirs: `runs/s2a_audioldm_whisper_xformer/long100/diagnostics_20260812_2155/retrieval_{whisper,clap}`

- **[IMPL] Free up scratch space, then try the frozen MLLM encoder (Qwen2.5-Omni).** User's call,
  2026-08-15. Scratch is tight and the MLLM path needs precomputed features. Do the cleanup first,
  then the encoder swap. Prior art already gathered: Audio-Omni takes thinker hidden states at
  **layer −2** (their Table 7: IS 11.26 vs 9.36, FD 2.75 vs 4.21 against the final layer) with a
  **zero-init linear** adapter, and elaborate adapters (MetaQuery, learned queries) scored *worse*
  than raw penultimate hidden states. Weights are already local at `models/qwen2.5-omni-3b`.

- **[IMPL] Remaining defects from the 2026-08-15 pre-flight review** (the three severe ones are
  fixed and committed; these are the rest):
  - `train_align`/`train_diffusion` are last-batch snapshots logged beside an epoch-mean
    `train_loss`, so `train_loss ≠ train_diffusion + λ·train_align` and both O2 curves are far
    noisier than the total. Accumulate means instead.
  - `val_d1_gap` (single timestep t=125, under bf16 autocast) and `diagnostics.json`'s
    `mean_d1_gap` (mean over four buckets, fp32) are different quantities with near-identical
    names — diffing them will show a phantom regression. Compare against `low_t_d1_gap`, or rename.
  - `diagnose_conditioning.py` loads the U-Net with `strict=False` and discards the result, so a
    key-set divergence would silently diagnose the PRISTINE U-Net while reporting
    `unet_weights_loaded: true`. `generate.py` already handles this correctly — mirror it.
  - `stage2_train_generate.sbatch` and `full_one_node.sbatch` pass `--override output_root=...`,
    which is not a `TrainConfig` field and is only remapped for YAML, so the trainer exits
    immediately. Both also look for checkpoints at the wrong path. Stale scripts, currently unused.
  - `train_regress.py` silently drops unknown `--override` keys (train_ldm raises), and has no
    resume path at all despite inheriting `resume_from`.
  - `cfg_dropout_prob: 0` is a DDP landmine: `null_embedding` only gets gradient through
    `apply_cfg_dropout`, which early-returns at 0, making it an unused parameter that DDP errors on.
    Base config is 0.1 so the cohort is safe, but a sweep of that knob would hit it.
  - `train_ldm.py` saves `cuda_rng` but never restores it, so "exact resume" is not bit-exact.

### Added 2026-08-20 (from a walkthrough of the objective and metrics)

- **[IMPL — IMMEDIATE] Make `val_clap_r1` full-set instead of within-batch.** It currently ranks each
  conditioner output against only the **B candidates in its own batch**, so chance is `1/batch_size`
  (0.0625 at B=16). Two consequences: r1 is **not comparable across batch sizes**, which blocks the
  planned batch-size increase outright; and at B=1 the metric is vacuous (chance = 1.0), which is why
  the code has an `n >= 2` guard. Ranking against all 1,887 val targets instead is a **3.9 MB**
  matrix — trivially cheap — batch-independent, and by the design review roughly **5–9× more
  sensitive** in our operating regime (r1@16 = 0.564 maps to full-set R@1 ≈ 0.03–0.06). Within-batch
  was simply the easy thing to write. Report `cond_R@1_full` and `cond_MRR` alongside.

- **[IMPL] Per-group gradient clipping, so λ can actually be swept.** `clip_grad_norm_` is applied to
  the **joint** conditioner + U-Net parameter list at `grad_clip=1.0`. The alignment term produces
  gradient only into the conditioner, but it inflates the *joint* norm, so clipping rescales the
  U-Net's gradient by the same factor. At λ=2000 the U-Net's effective LR would fall to roughly
  `1e-5 / 2000 = 5e-9` — **silently frozen**. And we know a non-finetuned U-Net produces nothing
  (pristine lane MARGIN −0.0036), so a naive high-λ run would fail for a reason unrelated to the
  objective's merit. Clip per group (or scale `lr_unet` in compensation) before sweeping λ above 1.0.
  **Correction to the record:** "λ makes no difference, question closed" is too strong. What is
  established is that λ ∈ [0.2, 1.0] makes no difference. Above 1.0 is untested and currently
  untestable.

- **[IMPL — carry into the A2S / bidirectional phase] Temporal conditioning is likely mandatory for
  speech, not optional.** User's note, 2026-08-20. Everything measured so far is S2A, where the
  target is ~10 s of *ambience* — largely stationary, and where our 2×2 showed the mapping is
  order-invariant on the input side. **Speech generation is the opposite case**: a spoken utterance
  is strongly ordered, phoneme timing matters, and a single global 512-d vector broadcast identically
  across all 256 latent positions cannot express "say this word, then that one". So when A2S starts:
  - assume **sequence conditioning + cross-attention are required**, not experimental — the
    `CrossAttn` blocks are already pretrained and unused;
  - build the **temporal-margin metric first**, because without it a temporal fix cannot be
    evaluated either way;
  - do not carry the S2A order-invariance finding across as if it generalises — it is a property of
    *ambient targets*, and speech should be expected to behave oppositely.

- **[IMPL — HIGH PRIORITY] Try real cross-attention conditioning (Tier 3).** User's call, 2026-08-20.
  Context that makes this well-posed rather than speculative: our U-Net *already has* cross-attention
  blocks (`CrossAttnDownBlock2D` ×3, `CrossAttnUpBlock2D` ×3, `cross_attention_dim`
  [128,256,384,640]), pretrained and currently unused because we pass `encoder_hidden_states=None`.
  **AudioLDM 1's own pipeline also passes None** (`pipeline_audioldm.py:522`), so we are faithful to
  it — this is not a deviation we introduced. Feeding a *sequence* would let each of the 256 latent
  time positions attend to different content instead of receiving one broadcast bias.
  Caveat to keep honest: a direct ablation (arXiv 2501.14680) found mean-pooled FiLM **beat**
  cross-attention alone and the combination (FAD 1.89 / 2.02 / 1.94), so pooled conditioning is not
  inherently deficient. And the 2×2 ablation showed our mapping is largely **order-invariant** on the
  input side, which weakens the a-priori case. So pair this with the temporal-margin metric, or the
  result will be uninterpretable either way.

- **[IMPL] Restate the data-sparsity motivation.** "13,959 unique captions across 15,033 pairs = 1.1
  per scene" is a statement about caption **strings**. Measured in CLAP-audio space, an ε-net at
  similarity > 0.75 collapses 4,000 sampled clips to ~1,061 groups ≈ **3.8 clips per acoustically
  distinguishable scene** — so caption uniqueness overstates the sparsity by 2–4×. Any reviewer can
  compute this in one line of numpy. Restate as "13,959 unique caption strings, though roughly 2–4
  clips per acoustically distinguishable scene". **The O2 result is empirical and does not depend on
  this framing** — only the motivation paragraph needs the fix.

- **[NOTE] λ=2000 is not translatable to our parameterisation, and we should say so.** Verified from
  the paper: `L = L_DDPM + λ·E_n[w(n)·‖E_audio − E_latent,n‖²]`, `w(n) = 10^(−n/200)`, λ = 2000 (or
  1000 with a tuning layer), FAD 7.47 → 6.52. Their term is **unnormalised MSE** damped by a weight
  whose mean over uniform n is ≈ **0.087**, so ~92% of its magnitude is removed before λ applies.
  Ours is `(1 − cos)` on unit vectors, bounded [0,2]. We can establish the number does not transfer;
  we **cannot** compute our equivalent λ without their embedding magnitudes. So λ=1.0 was a
  scale-matched guess, not a derived translation — state it that way if asked.

### Added 2026-08-23 — A2S: the reverse direction, as a separate model first

User's call, 2026-08-23. **Same architecture, fresh weights, separate run.** We do not warm-start
A2S from any S2A checkpoint. Get A2S working standalone, and only then consider (a) merging the two
sets of weights, (b) joint bidirectional training, or (c) a single model with a **direction
conditioner** — a one-hot / learned direction embedding telling the model whether the input is audio
or speech. That direction conditioner is explicitly **later work**, not part of this phase. Doing it
this way means each direction has an independently measured baseline, so a joint model has to earn
its keep against two real numbers rather than against nothing.

Note the A2S manifests are **larger** than S2A (17,596 / 2,206 / 2,197 vs 15,033 / 1,887 / 1,878)
and are a *different row set* — 2,790 train rows are A2S-only. Nothing precomputed for S2A covers
them; every precompute has to be rerun with `--direction a2s`.

- **[DECIDED] Input encoder = CLAP-HTSAT-unfused audio tower.** This is the mirror of the Whisper
  argument, and it inherits its logic exactly. Whisper was chosen for S2A because that direction
  needs *what was said*. A2S needs *what the scene is* — which is precisely what CLAP-audio encodes,
  because it was trained on environmental audio↔caption pairs of the kind this dataset contains. The
  property that made CLAP the wrong encoder for S2A (it reports "a man speaks", not the meaning) is
  the same property that makes it the right one here.
  Three practical reasons on top of the argument:
  1. **Already computed.** `scripts/precompute_clap_targets.py` embeds `audio_file` — for S2A those
     vectors are the *targets*; for A2S the identical vectors are the *inputs*. Only the a2s row set
     and the test split need generating. Minutes of GPU, no new code path.
  2. **Already local.** `models--laion--clap-htsat-unfused` is in the HF cache; compute nodes have no
     internet, so this matters. (BEATs is **not** local and would need a login-node download —
     that's the blocker if we ever want it.)
  3. **Already the designed extension point.** `src/salm/encoders/base.py:6` states the
     Whisper ↔ CLAP-audio ↔ Qwen2.5-Omni swap as the reason `FrozenEncoder` exists.
  Known limitation, accepted for now: CLAP is one global 512-d vector with no temporal structure.
  That is defensible on the *input* side — a 10 s ambient clip is largely stationary, the same
  argument that held for S2A. It does nothing for the *output* side; see the temporal item below.

- **[IMPL] Cheap symmetric control: Whisper encoder on the ambient input.** It is the literal mirror
  of the S2A control (CLAP-audio on speech), it needs no new code at all, and it should *lose* to
  CLAP. Running it completes a clean 2×2 of encoder × direction, which is a stronger claim than
  "we picked the sensible encoder for each direction". Skip only if compute is tight.

- **[BLOCKER — RUN BEFORE ANY A2S TRAINING] Can this stack reconstruct intelligible SPEECH at all?**
  The frontend is 16 kHz, **64 mel bins**, fmax 8000, and the VAE + HiFi-GAN were trained on general
  audio. 64 mels is low for speech (speech vocoders typically use 80), so the round trip
  `speech wav → mel → VAE encode → VAE decode → vocoder` may not survive as words. If it does not,
  A2S is dead before training and no encoder choice matters. This is the exact A2S analogue of the
  rung-0 gate and it is pure inference.
  **It is also objectively scoreable, unlike the S2A gate:** we hold `speech_text` for every row, so
  run Whisper ASR over the round-tripped audio and report **WER against the ground-truth transcript**.
  A correlation number would not have told us whether words survived; WER does. `check_audioldm_stack.py`
  already performs this round trip — point it at `speech_file` and add the WER leg.

- **[NOTE] The O2 alignment target cannot be CLAP for A2S, and O2 is the thing that made S2A work.**
  O2 aligns the conditioner output to an embedding of the *target*. For S2A the target was ambience
  and CLAP-audio was the right embedding of it. For A2S the target is speech, and this repo's own
  documented reasoning says CLAP encodes only that speech is happening. So the mirror is:

  | | S2A (works) | A2S (proposed) |
  |---|---|---|
  | input encoder | Whisper (speech) | CLAP-audio (ambience) |
  | O2 alignment target | CLAP-audio of target ambience | pooled Whisper of target speech, or a text embedding of `speech_text` |

  ~~**Unresolved tension**~~ **RESOLVED 2026-08-23 by picking CLAP-TEXT of the transcript.** The
  tension was that aligning to a pooled-Whisper target would push the conditioner output out of the
  CLAP-512 space the pretrained U-Net's `class_labels` slot understands — in S2A the alignment space
  and the U-Net's space were the *same* space, which is plausibly part of why O2 worked so cleanly.
  CLAP-text is already 512-d and already in that space, so the tension disappears rather than being
  traded off. Caveat kept on the record: CLAP's text tower was trained on audio *captions* ("a dog
  barks"), not arbitrary spoken sentences, so a transcript is somewhat out of its distribution.
  **Both candidates are computed and their pairwise spreads measured**, so the choice is made from a
  number rather than from this argument — see the *Built* block below.

- **[IMPL] Length handling changes and must not be inherited by accident.** S2A targets are ~10 s
  ambience against a 10.24 s canvas, so `_fix_length`'s centre-crop/pad is harmless. A2S targets are
  speech at mean 4.86 s, median 4.45 s, range 1.15–16.11 s — under half the canvas on average.
  Centre-padding would teach the model to emit silence, then speech, then silence, with the onset
  varying per example for no learnable reason. **Left-align and pad right**, and expect the model to
  have to learn "utterance, then silence". The >10.24 s tail gets truncated either way; count how
  many rows that is before assuming it is negligible.

- **[EXPECTED] The pooled interface is likely to be the real ceiling here — as predicted on
  2026-08-20.** A single 512-d vector broadcast identically across all 256 latent time positions
  cannot express "say this word, then that one". S2A's order-invariance finding is a property of
  *ambient targets* and must not be carried across. Running the pooled version first is still right —
  it is the floor every later number is read against — but **build the temporal-margin metric
  alongside it**, or a cross-attention fix afterwards will be uninterpretable in either direction.
  Budget for the pooled baseline producing speech-like output with the wrong words, and treat that as
  the expected result rather than a failure.

### Built 2026-08-23 — the A2S pipeline, queued but not yet run

Code is in and validated offline (synthetic-cache plumbing test, S2A regression test, every
library API read from installed source). Nothing has been measured yet.

**Module layout.** A2S gets its own surface (`src/salm/a2s/`, `scripts/a2s/`) per the user's call,
but `train_ldm.py` was NOT forked — it carries the DDP collective-deadline fix, the resume path and
the selection-metric logic, and two copies would drift. The direction-specific parts plug in
instead: a new encoder, a new dataset, two registry entries, and four small edits to the trainer.

**No encoder in the training loop.** The CLAP condition is precomputed (32×768 per clip, ~0.9 GB for
the train split), so an A2S step is three memmap reads plus the U-Net. S2A spent most of every step
inside a 635M Whisper forward that was 84.6% padding. This is what makes a wide A2S sweep affordable,
and it is the same mechanism item 7 above wants for S2A.

**Two training lanes, re-earning the S2A ablation rather than assuming it transfers:**

| lane | objective | prediction |
|---|---|---|
| A0 | plain diffusion loss | at chance, as L0 *and* L2 were on S2A |
| A1 | + O2 alignment, λ=1.0, CLAP-text target | learns, and fast — S2A's O2 plateaued inside 10 epochs |

40 epochs each on the 100-epoch LR curve (`a2s_probe40.yaml`), because 40 already separates those
two outcomes and a fixed `lr_total_steps` keeps a later 100-epoch run comparable and resumable.

**Predictions, written down before the numbers exist** so a surprise gets questioned rather than
believed:

- **The gate is the real risk.** 64 mel bins (speech vocoders use 80) and a VAE/vocoder trained on
  general audio. Scored as WER against `speech_text`, with the untouched clip as the floor, so the
  number that matters is the *degradation*.
- **`align_speech` should collapse.** CLAP-audio of speech ought to be near-collinear across rows
  (pairwise cosine → 1.0) against `align_text`'s healthy spread. If it does, that is the first
  direct measurement of an assumption this repo has relied on since 2026-08-06. If it does *not*,
  the encoder-choice argument for A2S needs re-examining, not just the alignment target.
- **The pooled interface should cap A1**, per the 2026-08-20 note: one 512-d vector broadcast across
  256 latent positions cannot express word order. Expect speech-shaped output with the wrong words,
  and treat that as the expected result. `b0` (the learned null embedding through the same U-Net) is
  generated alongside every lane precisely because "it sounds like speech" proves nothing on its own.
- **Truncation is a real cost here, unlike S2A.** Speech averages 4.86 s in a 10.24 s canvas, so most
  of the target is silence the model must learn to emit, and the >10.24 s tail is cropped. The gate
  reports both numbers.

**Still missing, and named so it is not mistaken for done:** the temporal-margin metric (without it a
cross-attention fix is uninterpretable either way), A2S diagnostics in `diagnose_conditioning.py`,
and an A2S entry in the metric suite.

### Added 2026-08-24 — A0/A1 landed, O2 works but the target is content-poor; two follow-ups
### built, neither submitted

- ~~**[DONE 2026-08-24] A0 (plain) and A1 (+O2) probes.**~~ A0 flat at chance for all 40 epochs
  (r1=0.061). A1 reaches 0.327 (5.2x chance) but peaks at epoch 6 and drifts down after, while
  train-side cosine keeps climbing — a widening train/val gap S2A's O2 never showed. **First listen
  (user, real audio): "sounds like speech but incomprehensible, like a foreign language."** Matches
  the theory precisely — register/prosody survives (CLAP-text carries that), words do not (CLAP's
  text tower was trained on captions, not on preserving verbatim word-by-word content). Not a decoder
  problem: the 2026-08-23 gate already measured ~0 WER degradation through the real VAE→vocoder path.
  See EXPERIMENT_LOG for the full writeup, the ridge-probe finding, and the metric-space correction
  (val_clap_r1/cos are in CLAP-TEXT space here, not the near-collapsed CLAP-audio space).

- **[RUNNING 2026-08-24, job 426004] InfoNCE O2 loss.** `align_loss: cosine|infonce` in
  `train_ldm.py`, default `cosine` (bit-identical to every existing config/checkpoint). Motivated by
  a real, measured gap: a closed-form ridge regression on the same data beat A1's trained conditioner
  on R@1@16 (0.45-0.46 vs 0.354), despite using less input information. Verified offline before use:
  centroid collapse scores 1.04 under InfoNCE (still near-random) vs 0.74 under cosine (looks like
  progress) — the exact blind spot this closes.
  **Result so far (14+ of 40 epochs): confirmed AND re-scoped in the same breath.** Worst InfoNCE
  epoch (r1=0.424) already beats A1's best (0.354) — the theory holds, cosine really was leaving
  signal on the table. But it shows the SAME early-peak-then-decline shape (peak ~ep2 at 0.513,
  declining since) from a much higher floor — InfoNCE fixed centroid collapse, not whatever else
  drives the decline. **Re-scoped by the oracle result immediately below**: worth finishing for a
  paper-table retrieval number; no longer expected to fix comprehensibility, since even a perfect
  CLAP-text target already fails to.

- ~~**[DONE 2026-08-24, job 426005] Oracle-conditioning ablation — DECISIVE.**~~ `generate.py
  bypasses the conditioner and feeds the row's own true alignment vector straight to `class_labels`,
  looked up by `sample_id` (survives slicing; verified against the real val cache, 0 missing).
  Decomposes "conditioner underfits the target" from "the target itself is a poor generation signal."
  **Result: the pre-registered prediction held.** User, by ear: both `oracle_text` (perfect target,
  conditioner bypassed) and `real_conditioner` sound identically "incomprehensible, foreign-language-
  like." Removing 100% of conditioner error changed nothing audible. **This resolves the fork the
  whole investigation was built to test: the bottleneck is the target/interface (CLAP-text's
  resolution on word content), not the conditioner's ability to reach it.** No further tuning of the
  current recipe — loss, epochs, architecture — can fix comprehensibility, because the perfect version
  of the current target already fails to produce it. Next: O-F (mel-space CTC content loss, decoded
  via the VAE only, no vocoder) or a two-stage cascade — not more conditioner tuning. See
  EXPERIMENT_LOG 2026-08-24b for the full writeup.

- **[RUNNING 2026-08-26, job 428814, account aip-ravanelm] O-F: content loss on the decoded mel.**
  Built: `scripts/a2s/train_ctc_probe.py` (a small from-scratch CTC ASR model trained on
  AudioLDM's own mel frontend — no off-the-shelf model matches its 64-mel/16kHz config), wired into
  `train_ldm.py` as `lambda_ctc`/`ctc_checkpoint`/`ctc_max_t_frac`, additive to O2 (requires
  `lambda_align > 0`, same training branch).
  **CTC probe (job 428711) converged well**: best_wer=0.4145, legible transcripts
  ("let's see if she' starts this time come on oll gir wak up" vs the real
  "Let's see if she starts this time..."). A real, usable content-loss head.
  **One real bug, caught by the actual job, not preventable by a login-node test**: cuDNN's fused
  GRU kernel needs `.train()` mode to allocate the reserve-space its own backward needs — `.eval()`
  skips that regardless of `requires_grad_(False)`, so gradient flowing through a frozen-but-eval
  GRU crashed with `cudnn RNN backward can only be called in training mode`. This is GPU/cuDNN-
  specific with no CPU equivalent, so it could not have been caught by the CPU-only login-node
  validation adopted earlier this session — the real Slurm job was the only venue that could catch
  it. Fixed: `ctc_head.eval()` overall, `ctc_head.rnn.train()` specifically. See EXPERIMENT_LOG
  2026-08-26 for the full account-switch story (aip-csubakan → aip-ravanelm mid-flight) too.

- **[NOTE] `train_loss` and `val_loss` are different formulas whenever `lambda_align > 0`** — user
  caught this by inspection. `val_loss` is always pure diffusion MSE (verified: `diffusion_loss()`
  never adds the align term, on either side); `train_loss` includes `+ lambda*align` when O2 is
  active. Their divergence is partly a formula artifact, not automatically evidence of anything.
  Separately, `train_align`/`train_diffusion` (already a known defect) are last-batch snapshots, not
  epoch means, which adds noise to any train-vs-val cosine comparison until fixed.

- **[SCOPED, NOT DECIDED] Sequence-level O2 against target-speech Whisper frames.** User's idea,
  arrived at from "shouldn't cross-attention capture the sequential nature of the target's
  understanding better?" Precompute Whisper features of the TARGET SPEECH (never done for A2S; only
  the ambient INPUT side is cached today), pool to a fixed T frames (~8, matching CLAP's order of
  magnitude), and add a framewise O2 term (cosine or InfoNCE per position) on an intermediate
  T-frame conditioner output — POOLED down to the single vector that still reaches `class_labels`,
  so the U-Net's conditioning slot is untouched. Real cross-attention (zero-init gated adapter, so
  it starts as a no-op on the pretrained prior) is the escalation only if this looks promising, not
  the first step.
  **Gate resolved 2026-08-24: the oracle test came back incomprehensible too** (see the
  oracle-ablation entry above), which strengthens the case for a content-richer target. Still not
  built — ranked behind O-F below, since O-F sidesteps the embedding-target question entirely rather
  than trying a higher-resolution version of the same kind of target.
  **Honest expectation, stated before any number exists**: this most likely helps time-varying
  register/prosody, not exact words — a Whisper-framewise target is HIGHER resolution than CLAP-text,
  hence a HARDER thing to predict from ambient audio, not an easier one, so if CLAP-text's own
  realizability was already modest (ridge probe), this should not be expected to suddenly solve word
  content. Judge it on richness of the non-lexical signal, not comprehensibility, unless surprised.
  **SUPERSEDED by the 2026-08-26b F1/F2 reframe below** — a Whisper-framewise target is still a
  pooled-*per-frame* embedding, so it inherits the same "no phoneme-rate channel" problem the oracle
  test and AudioLDM 2's own λ ablation both identify. Rank 2 (real cross-attention) is the version of
  this idea worth building; this item is not.

### Added 2026-08-26b — the F1/F2 reframe: two failures were being measured as one, and a ranked plan

Deep research pass (opus/Plan agent, real web research authorized), triggered by the user asking
whether to try AudioLDM 2, Audio-Omni, or BEATs next. Full writeup: `EXPERIMENT_LOG.md` 2026-08-26b.
Headline: **AudioLDM 2's own paper independently confirms the oracle ablation** — they use λ=1 (full
temporal resolution, ~51.2 tok/s) for TTS conditioning specifically, vs λ=8 (~6.4 tok/s) for general
audio, explicitly because pooled features lose the detail speech needs. Our A2S delivers ~0 tok/s
(one pooled vector). Their ablation and ours agree from opposite directions.

**The reframe**: split "which words should be said" (**F1**, content origination, information-
theoretic — no architecture fixes this) from "can the model render words at all, given content"
(**F2**, phonetic realizability, purely architectural — fixable, and what the oracle test isolated).
**Consequence: WER-vs-ground-truth conflates F1 and F2 and cannot tell "right words, wrong content"
from babble — both score ≥1.0.** This means A3/O-F's WER result (still pending as of this entry)
CANNOT be trusted as a verdict on whether O-F worked, even once it lands — its CTC loss genuinely
dropped (0.3965→0.2968), and whether that bought real phonetic content is invisible to the current
metric. Rank 0 below is required before any O-F conclusion is safe.

**Audio-Omni: verdict DO NOT INVEST** (researched via the actual paper, not just the local checkpoint
config) — transcript is a mandatory input for every speech task in it, same shape as AudioLDM2-
speech but bigger, no training code released, CC BY-NC 4.0, a completely different 44.1kHz stereo
pipeline (nothing transfers), zero independent evaluation.

**BEATs: settled** — its own paper's design goal is discarding fine-grained detail for high-level
semantics, the opposite of what content needs. Only legitimate use: an alternative scene encoder,
decided by the existing `scripts/retrieval_baseline.py` ridge probe (~1 hour, zero training). If it
doesn't beat CLAP's 0.45, drop permanently.

**Ranked plan** (costs/falsification tests in EXPERIMENT_LOG 2026-08-26b and the agent's full report):
- **[RUNNING 2026-08-26c, job 429378] Rank 0**: content-independent intelligibility metric
  (`scripts/a2s/score_intelligibility.py` — inter-ASR agreement between Whisper-large-v3 and our
  own frozen CTC probe, real-word rate against a vocabulary built from our own train transcripts).
  Retro-scoring every generation lane on disk (A0-A3).
- **[RUNNING 2026-08-26c, job 429392] Rank 1**: oracle-transcript cascade
  (`scripts/a2s/run_oracle_tts_cascade.py`, Kokoro-82M + `audioldm2_gigaspeech`, both downloaded).
  Verified against the real checkpoint before running: `max_new_tokens: 8` genuinely ships in
  `language_model/config.json` (set to 512 explicitly), `voice` is a required Kokoro argument
  (`af_heart`, matching the source's own error-message example) — both caught by reading installed
  source, not by a failed run.
- **[RUNNING 2026-08-26c, job 429416] Rank 2**: zero-init gated cross-attention content stream,
  built via diffusers' own `IPAdapterAttnProcessor2_0` rather than hand-rolled attention mechanics
  (`src/salm/a2s/content_adapter.py`) — verified end-to-end against installed diffusers source (see
  EXPERIMENT_LOG 2026-08-26c for the full mechanism). 12-epoch falsification test, ORACLE content
  (frozen CTC probe's encoder on the TARGET SPEECH's own mel — training-only, never at real A2S
  inference). This is the LEAST execution-tested code of the night — could not run the forward pass
  on the login node to iteratively debug, so real risk of a bug static review missed remains until
  the job's own result comes back.
- **[IMPL] Rank 3**: cross-attention over the ambient input — bandwidth/register/prosody, explicitly
  NOT expected to fix content. Score on CLAP-audio similarity / FAD / scene-appropriateness, **never
  WER** — scoring a register win on an intelligibility axis is how a real success gets read as a
  failure.
- **[DISCUSS] Rank 4**: reframe F1 (predicting plausible content from the scene) as the actual
  research contribution — searched prior art (UmbraTTS, Audio-Omni, AudioLDM2-TTS, F5-TTS) and
  confirmed nobody in the field infers speech content from a non-speech scene alone. Cheapest Stage-A
  baseline: retrieval against training transcripts using the existing ridge/conditioner apparatus,
  zero training, immediately gives a complete listenable system paired with Rank 1.

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
