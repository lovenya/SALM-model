# Context (for agents and developers)

> **Job control is the user's, not the agent's.** Never `sbatch`/`scancel`/`scontrol` without
> explicit per-action instruction. See `CLAUDE.md` -- this was violated once (job 406278 was
> cancelled by the agent) and the rule is now written down in three places.


This file is the internal working doc for this repo. It is NOT the public
README -- keep cluster paths, in-progress decisions, and open questions here.
Update it via the `/update-docs` skill after every meaningful milestone.

## A2S: the renderer is settled, the bottleneck is the content LM — 2026-09-07

**The measurement that reframes the direction.** `val_scene_gap` accounts for **1.8–2.8%** of
the content LM's validation loss (k512 +0.0414/1.5002, k1024 +0.0370/1.6213, k2048
+0.0322/1.7588). The other 97%+ is language modelling over code sequences learned from 17,596
utterances. Four independent measurements agree without having been designed to:

- exact-word recovery f1 **0.16** even from a ground-truth caption (job 442799)
- the plan head beat a scene-blind marginal by **0.0040 nats** (443751)
- cross-seed agreement collapses after **~4 words** (442814)
- the scene accounts for **<3%** of the loss (above)

The scene constrains the OPENING and the TOPIC, not word 7. Every conditioning intervention
tried was attempting to extract information that is not present, which is why they all landed
where they did.

**Best results and the deliverable shortlist: `docs/BEST_RESULTS.md`** — ranked by ear, with
exact reproduction paths and a PROTECTED artifact list (5.6 GB) that must be checked before any
cleanup. Written after a working result was closed on the wrong metric and its checkpoints
deleted on the agent's recommendation.

**Operating point (use these):**
- renderer: `runs/a2s_audioldm_clapseq_xformer/rladder_20260906_1654/checkpoints/checkpoint_ep119.pt` — FROZEN
- content LM: `content_lm/dedup_k2048/checkpoints/checkpoint_best_loss.pt` (epoch 9)
- decoding: **greedy** (`--content-lm-top-k 1`)
- measured: `inter_asr_dist` 0.3688 over 128 val clips (real speech 0.3146, VAE ceiling 0.3722)

**`checkpoint_best` is misnamed and must not be used blindly.** It is selected on
`val_scene_gap`, and that metric rises monotonically WITH overfitting (dedup_k512: +0.041 at
ep9 → +0.133 at ep99). So `best` is close to the most overfit checkpoint of each run —
val_loss 5.01 against `best_loss`'s 2.37. Under sampling the two scored alike; under greedy
`best_loss` wins by 0.13–0.17 in all three codebooks. **State the tag with every number.**

**Eliminated by matched, gated experiments (do not re-propose):**

| tried | job | result |
|---|---|---|
| more training | 438059 | flat epoch ladder |
| regularization/capacity, 7 configs | 438062 | scene_gap invariant, spread 0.0019 |
| overfitting as the cause | 442814 | train == val, ruled out |
| content CFG | 442847 | w=1.5 buys two words, w≥2.5 destroys the audio |
| temperature ladder | 443379 | flat |
| nucleus sampling | 443737 | clarity and variety are ONE axis, no free point |
| plan vector (bag-of-codes) | 443751 | null — target carried 0.0040 nats over scene-blind |
| latent think tokens | 443989 | null — horizon flat at 1 across n_think 0/8/32 |
| renderer trained on quantized content | 443895 | **worse** (0.5903 vs 0.3223 control) |
| CosyVoice token space | 444076 | ours more structured, but the margin was overstated — see below |
| CTC "peakiness" (D3 pre-net premise) | — | false: every code used, mean run 68–80 ms |
| speaker entanglement of codes | — | gender explains 0.6% of code entropy |

**The renderer is solved and frozen.** It cannot change WHICH words are produced — only how
they sound. With codes fixed it reproduces the same words at 0.96/1.00/0.93 (442847 greedy
control), and Lovenya's A/B on 2026-09-06 confirmed it by ear: two lanes differing by 0.27
`inter_asr_dist` were heard as "90/95% same", because both share the same LM and greedy codes
and therefore say the same sentence. It generates the whole 10.24 s at once via latent
diffusion (not autoregressive, not chunked) — and the oracle lane proves that design renders
coherent speech when the codes are right, so all-at-once is not the bottleneck.

**METRICS HAVE DISAGREED WITH THE EAR TWICE.** `inter_asr_dist` measures ASR *agreement*, i.e.
machine transcribability. When both lanes sit below human intelligibility, large differences in
it are inaudible. Rank renderer variants BY EAR; the numbers place lanes on a scale and no more.

**The plan-horizon arithmetic (corrected 2026-09-07).** The model holds ~20 run-length tokens
of plan. The full sequence it must generate is **151.4 tokens** (measured, k2048 dedup cache),
not the 56 first claimed — 56 was the speech region only (111 of 256 frames), but the model
must also emit the silent tail. **The shortfall is 7.6×, not 2.8×.** Compression cannot close
it (it shrinks both sides equally), which is why run-length's 2.0× bought so little.

**CORRECTION (2026-09-07), from the independent audit.** The CosyVoice comparison was
confounded. Our +35.9% n-gram reduction was measured on FRAME-LEVEL codes with mean run 1.76,
so a third of it was simply "the next frame repeats"; CosyVoice's 25 Hz tokens are already
run-collapsed (mean run 1.08). Recomputed on our run-length codes, the fair comparison is
**20.7% vs −10.7%** — still decisively ours, but a 31-point gap rather than 47. The direction
stays closed; it was closed on an inflated number. Unresolved: CosyVoice's negative value may
itself be a sparsity artifact (vocab 6,561, ~0.004 observations per trigram context). The
bigram row (−0.202 at ~28 observations per context) argues against that, but settling it needs
re-extracting its tokens over more data.

**No unpaired speech exists.** `datasets/` holds only TTAS (22,793 clips, 17,596 already train).
Pretraining the content LM on speech-only data — which needs no ambient — is a download
decision, not an engineering one.

## MiDasheng memorization job prepared — 2026-09-07

- Calibration jobs 444459 and 444434 both completed; distinct output directories, no
  overwrite. Treat 444459 as canonical and 444434 as duplicate compute.
- Corrected utilization: prior 59–88% figures were peaks. Whole-job sampled means
  were 32–35%; batch32 median updates ~0.277s vs batch16 ~0.199s.
- Next: [64-pair memorization plan](docs/MIDASHENG_MEMORIZATION_20260907.md),
  A2S/joint batch32/64, 16 supervised blocks per pair, 2,000 optimizer updates.
  Explicit 100-update warmup then cosine LR, logged per update. Same eligible
  training subset across lanes; fixed-noise source-swap loss diagnostics added.
- Prepared `submission_scripts/run_tamia_midasheng_memorize.sbatch`: one 4-H100 job,
  aip-csubakan, 55-minute cap. Not submitted. Ten lightweight tests, metadata preflight
  and Slurm test-only validation passed. No additional packages or downloads.
- User prioritizes listenable outputs; accepts unmatched batch-size exposure for
  exploration, with controlled reporting runs later if needed.

## MiDasheng mapping pilot prepared — 2026-09-07

- Gate job 444114 completed in 4:14. Reconstruction speech WER to original ASR was
  0.0 on eight pairs; shuffled control about 1.0763. User listened and liked results.
  This verifies a promising codec pairing, not A2S mapping. AudioLDM runs are a
  separate collaborator's work; do not attribute them to this experiment.
- Implemented source projector + Qwen rank-16 Q/V LoRA + stop-head adaptation through
  a frozen differentiable MiDasheng flow head. Source-only A2S and alternating joint
  A2S/S2A modes; no target text inputs. Encoder and waveform codec stay frozen.
- [Pilot instructions and expected outcomes](docs/MIDASHENG_MAPPING_PILOT_20260907.md).
  Four packed H100 lanes, microbatch16/32 for A2S and joint, 32 pairs, 40-step initial
  calibration, 35-minute cap. User-run command: `sbatch submission_scripts/run_tamia_midasheng_mapping.sbatch`.
  GPU training is not yet validated; no job submitted by the agent.
- [Offline interactive architecture page](docs/midasheng_architecture.html) shows
  training, inference, direction switches, causal history and trainable/frozen modules.
- Seven lightweight tests, Python/Bash syntax and file-metadata preflights passed.
  Every saved checkpoint gets generated controls and native retention examples.
  Full-data streaming/resume and optional internal text are future gated work, not implemented.

## Historical preparation: MiDasheng compatibility job — 2026-09-06

- User downloaded generator snapshot `598b91a0abbdf15b5f979a0f8d8b0ef86533dbc1`,
  encoder candidate `3969e451d1c62618089c1135fe4c58ed33af2457`, and Qwen tokenizer/config
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. Qwen special_tokens_map.json is not required.
- [Setup, submission and expected results](docs/MIDASHENG_GATE_20260906.md): user-run
  dedicated environment followed by `run_tamia_midasheng_gate.sbatch` (4 H100 / 25 minutes).
  Four lanes compare FP32/batch4 with BF16/batch4/8/16. Eight fixed validation pairs;
  native speech/sound, candidate round trips, shuffled-latent controls, independent ASR.
- Metadata preflight passed (637 generator / 473 encoder-checkpoint tensors), four contract
  tests passed, Bash syntax checked. No model inference, installs, downloads or submissions
  performed by the agent. Environment imports and GPU execution remain unvalidated.
- This is an oracle component/compatibility diagnostic, not A2S or joint training. Successful
  reconstruction/native generation licenses the tiny jointly trained bidirectional pilot.
  Existing AudioLDM work remains available; previous cascade diagnostics remain parked.

## Independent review — 2026-09-06

- Read the Codex export and research/user excerpts from all four top-level Claude transcripts;
  excluded `pre_midturn_fix/`. Findings and evidence boundaries are in
  [the independent audit](docs/A2S_INDEPENDENT_AUDIT_20260906.md).
- Internal generated text is acceptable; text-free/shared-generator alternatives remain open.
  The user runs all submissions, downloads, installs and long/heavy commands. Delete nothing.
- **Latest user clarification:** audio/speech → internal text → speech/audio is acceptable
  when integrated and jointly trained; an additional text-to-text model is unnecessary.
  Explain gradient paths and total model size explicitly. AudioLDM is not closed, and the
  dataset has not been shown too small for fine-tuning. See the
  [joint-model design](docs/A2S_JOINT_MODEL_DESIGN_20260906.md).
- Two diagnostic jobs are prepared but **parked, not GPU-validated or submitted**: a 96-scene pretrained
  planner comparison (4 H100 / 35 minutes), and supplied-text rendering stress (4 H100 /
  15 minutes). [Commands and expected artifacts](docs/A2S_REVIEW_JOBS_20260906.md).
- Existing sampled planner rendering has 16.32% ASR WER to its supplied text (33/64 exact
  transcripts); whole-utterance rendering is not established as solved. The first checks
  separate planning quality from text-to-code-to-waveform fidelity.
- Shared pretrained speech/sound generation is an independent branch, not yet a migration
  decision. UNISON's released trainer needs source-conditioning adaptation; MiDashengLM-Gen
  is an inference-only candidate with downloaded assets; joint training still needs implementation.
- Six lightweight contract tests and shell/Python syntax checks cover the new harness;
  actual GPU memory/throughput and generation APIs still require the first compute run.

## Current status — 2026-09-02

- **A2S renderer capability is established but conditional:** `content_concat_v3/v4`
  generated intelligible fresh speech when given a dense CTC trajectory extracted from target
  speech. This proves the AudioLDM rendering path can articulate speech; it does not solve
  ambient-scene-to-content planning.
- **Raw-audio scene planning works zero-shot, imperfectly.** Job 436247 completed 8/8 in 3:08.
  Direct Thinker outputs cross-matched all four scenes. The scaffold correctly named baby/applause
  but misheard motorcycle as chainsaw and water as frogs. Audio controls the planned semantic
  output; direct planning was more reliable than forcing a written scene caption first.
- Qwen is a candidate scene encoder/planner here, not the speech renderer or final model. The
  unproven bridge is planned text/phonemes -> renderer-compatible dense content.

- The user has listened to original speech, mel→vocoder reconstruction, and
  VAE→vocoder reconstruction and confirms all are intelligible. The existing AudioLDM
  VAE/vocoder is therefore not the present A2S gibberish bottleneck.
- **Do not interpret the 2026-08-29 Rank-2 retrieval generations.** Precompute loaded
  `ClapModel(...).audio_model`, but live generation loaded the same full checkpoint as a
  standalone `ClapAudioModel`. With the installed transformers version, all pretrained
  `audio_model.*` tensors were unexpected and the standalone tower's `audio_encoder.*`
  tensors were newly initialized. Retrieval keys were pretrained/cache-derived while live
  queries were random.
- Fixed live CLAP loading to use exactly `ClapModel(...).audio_model`, added an injectable
  tower so precompute does not load CLAP twice, and added a compute-side cache-parity gate.
  Generation now supports explicit `oracle|retrieval|shuffled|zero` Rank-2 content sources
  and records their source sample/file/similarity in every manifest row.
- Prepared but **not submitted**:
  `submission_scripts/run_tamia_a2s_rank2_matched_listening.sbatch`. Smoke runs one checkpoint
  × four samples; full runs six unique checkpoints × four content sources × 16 fixed val rows,
  with identical seeds and four packed H100 lanes. It refuses generation unless live/cache
  cosine is at least 0.999 and every live query retrieves its own cached row at rank 1.
- Decision gate: if target-speech **oracle** content produces recognizable English clearly
  above shuffled/zero, keep the renderer and next learn ambient→content. If oracle remains
  gibberish across the checkpoint spread, stop training this AudioLDM1 cross-attention path
  and move to a released unified generator.
- Current migration preference: UNISON first (one source-audio-conditioned flow-MM-DiT and
  shared VAE, released fine-tuning path; research/non-commercial constraints), UniFlow-Audio
  second (MIT code, smaller models, full training stack, shared generator with separate
  time-aligned speech and cross-attention sound interfaces). AudioLDM2-TTS is an oracle/control,
  not a shared released multitask checkpoint. Audio-Omni now has official inference code and a
  released-weight link, but the local clone has no checkpoint and the repo has no optimizer,
  dataset, loss, or training entry point. Its shared generator is adaptable in principle, but
  requires us to implement and validate the rectified-flow fine-tuning recipe ourselves.
- JEPA is scoped as an auxiliary masked cross-modal latent prediction loss on a shared
  temporal scene/planning memory, not the speech renderer and not a literal action-conditioned
  world model. Try it only after a simple paired contrastive/shared-memory baseline.

## Living docs — read these to catch up

- `docs/EXPERIMENT_LOG.md` — chronological record of what we tried, measured, decided and got
  wrong. The place to look for "why is it like this?" and for every number we have.
- `docs/OPEN_ITEMS.md` — what is still to discuss, decide or implement. Kept separate so nothing
  gets lost at the current pace.
- `docs/ARCHITECTURE_MENU.md` — the options for what to try next, tiered so every added component
  is licensed by a measured gap against something simpler. Carries the research evidence.
- `docs/RESEARCH_DIRECTIONS.md` — the IDEA SPACE: alternatives and architectures considered
  (VoiceLDM, semantic-token cascades, content CFG, F1-as-contribution, world models/JEPA),
  with the reasoning for and against each, and rejected ideas kept with their reasons so they
  are not re-proposed. Separate from EXPERIMENT_LOG, which records only what was actually run.
- `docs/wandb_setup.md` — W&B on this cluster (offline mode, sync workflow, the entity trap).
- `docs/LEARNINGS.md` — portable technical lessons (gotchas, mechanisms, fixes) worth
  remembering independent of this project — interview-ready, not project narrative.
- `CLAUDE.md` — hard rules, chiefly: **never touch Slurm jobs without explicit instruction.**

## What this repo is

`SALM-model` is where **model training** for the SpeechAudio LM project
happens. Full project proposal (motivation, contributions, evaluation plan):
`/scratch/l/lovenya/SpeechAudioLM/SpeechAudioLM.md`.

One-line project summary: a generative model that maps bidirectionally
between environmental audio and speech (audio -> plausible speech, and
speech -> plausible audio), trained on the TTAS paired dataset.

## Repo split

- `SpeechAudioLM` (separate repo): dataset cleaning/filtering/splitting, and
  all eval/metrics infra (`evaluate.py`, wired metrics like FAD/KAD/UTMOS/
  DNSMOS/etc.). Mature, in active use for that side of the project.
- `SALM-model` (this repo): model architecture + training only. **Clean-room**
  -- we do not import SpeechAudioLM's code. We do read its manifests and
  the shared TTAS audio/speech files by absolute path (see
  `configs/paths.py`).

Metrics/eval are explicitly deferred for this repo. We are not wiring eval
here yet -- that happens after initial training runs produce something worth
scoring. When needed, reuse `SpeechAudioLM/evaluate.py` and its metric stack
rather than re-implementing metrics here.

## Data

Dataset: TTAS (paired environmental-audio + speech recordings), already
cleaned/split/filtered upstream in `SpeechAudioLM`.

Two direction-specific manifests, because filtering is direction-dependent:

- **A2S** (`data/manifests/clean_v2/a2s/ttas_{train,val,test}.jsonl` in
  SpeechAudioLM): rows that pass the audio-to-speech G-Eval audio v6
  filtering.
- **S2A** (`data/manifests/clean_v2/s2a/ttas_{train,val,test}.jsonl`): rows
  that pass the speech-to-audio filtering.

Row fields: `audio_file`, `speech_file` (paths, relative to
`TTAS_DATASET_ROOT` in `configs/paths.py`), `speech_text`, `audio_cap`,
`reasoning`, `speaker_gender`, `speaker_emotion`.

**Important**: A2S and S2A manifests are NOT the same set of pairs. Each
direction was filtered independently against its own G-Eval score, so a
given audio/speech pair can appear in one direction's manifest and not the
other's (or in both, or in neither -- rows below threshold are dropped
entirely). Don't assume you can zip the two manifests row-for-row.

Row counts:

| split | A2S    | S2A    |
|-------|--------|--------|
| train | 17,596 | 15,033 |
| val   | 2,206  | 1,887  |
| test  | 2,197  | 1,878  |

### Data audit (2026-08-06)

Ran a full existence + overlap check over all six `clean_v2` manifests
against `TTAS_DATASET_ROOT`:

- **File existence**: 0 missing `audio_file` and 0 missing `speech_file`
  across all 40,797 rows (all six train/val/test x a2s/s2a files).
- **Cross-direction pair overlap** (by `(audio_file, speech_file)`):
  - train: 17,596 A2S / 15,033 S2A, 14,806 common, 2,790 A2S-only, 227
    S2A-only.
  - val: 2,206 A2S / 1,887 S2A, 1,864 common, 342 A2S-only, 23 S2A-only.
  - test: 2,197 A2S / 1,878 S2A, 1,855 common, 342 A2S-only, 23 S2A-only.
  - Confirms the "not the same pairs" note above with real numbers.
- **Cross-split leakage**: 0 collisions -- no `audio_file` appears in more
  than one split within a direction. Split integrity from SpeechAudioLM's
  fixed 80/10/10 split is preserved through the clean_v2 filtering.

Conclusion: clean_v2 manifests are safe to build the training pipeline on
as-is; no missing-file or split-leakage blockers.

### Audio format audit (2026-08-06, 400-file random sample per field)

- **Sample rate**: uniformly 24 kHz for both `audio_file` and `speech_file`
  (400/400 each). No resampling needed across the dataset as delivered.
- **Channels**: `speech_file` is always mono (400/400). `audio_file` is
  **mixed mono/stereo** (~70% stereo / ~30% mono in the sample) -- any
  preprocessing/front-end must explicitly downmix audio to mono (or decide
  to keep stereo and handle mono files as a special case).
- **Duration**: `audio_file` is mostly clipped/padded to 10.00s (median
  10.00s, mean 9.64s) but with a long low tail -- saw one sample as short as
  0.01s, worth a real duration-distribution pass before finalizing
  windowing/padding. `speech_file` ranges ~1.15-16.11s, mean 4.86s, median
  4.45s, generally shorter than its paired audio clip.
- Implication for the AudioLDM-derived recipe: front-end (mel-spec hop/win,
  fixed clip length, mono downmix, silence/near-empty-clip filtering) needs
  to be decided against these real distributions, not assumed from
  AudioLDM's original AudioSet-style training data.

## The architectural fact everything rests on

AudioLDM does **not** condition on a text token sequence. It conditions its
U-Net on a **single global 512-d CLAP embedding**, and it is *trained* on CLAP
**audio** embeddings with CLAP **text** embeddings swapped in at inference --
CLAP's shared audio/text space is what makes that substitution valid.

**Verified against installed source**, not assumed (diffusers 0.39,
`diffusers/pipelines/deprecated/audioldm/pipeline_audioldm.py`):

- `:519-525` -- `self.unet(latent_model_input, t, encoder_hidden_states=None,
  class_labels=prompt_embeds)`. One global vector, no cross-attention sequence.
- `:178` -- `prompt_embeds = F.normalize(prompt_embeds, dim=-1)`. The condition
  is L2-normalized, so the mapper must L2-normalize too; an unnormalized vector
  is out of distribution for the U-Net even with the right direction.
- `:115`, `:147-149` -- `prompt_embeds` is an accepted `__call__` argument and
  bypasses the text encoder when given, so mapper output can be fed straight to
  the pipeline instead of reimplementing the sampler.

Two consequences: the trainable part can be small (which is what ~15k pairs can
support), and the conditioning slot can be aimed at from any modality.

Note AudioLDM now lives under `pipelines/deprecated/` in diffusers -- pin the
version rather than tracking latest.

## Decisions log

- **2026-09-07**: A2S renderer **frozen** at `rladder_20260906_1654/checkpoint_ep119.pt`. The
  150-epoch ladder was flat from ~ep39 (`inter_asr_dist` 0.3161–0.3180, a 0.002 spread) and
  Lovenya preferred the 100+ rungs by ear. Further renderer training is not justified: a
  renderer cannot change which words are produced.
- **2026-09-07**: Generation defaults to **greedy** decoding and **`checkpoint_best_loss`**.
  The "never argmax" rule this codebase followed was an argument about continuous means
  imported into discrete token space; see `LEARNINGS.md`. Greedy moved the project metric from
  0.452 to 0.3688 for one flag and no retraining.
- **2026-09-07**: `content_concat_v3` deleted. Reconstructed first: v2/v3/v4 differ ONLY in
  `epochs` and `lr_total_steps` — v3 is a re-run of v2 that reproduced it to four decimals,
  and v4 is the same config trained twice as long. The design change was at v1 (frame-aligned
  concatenation replacing cross-attention). Lineage recorded in `EXPERIMENT_LOG.md`.
- **2026-08-06**: ~~One bidirectional model~~ **SUPERSEDED same day**: scope
  narrowed to **S2A only** (speech -> ambient audio) for the first model. A2S
  comes later. The manifests are already direction-specific, so this costs
  nothing to revisit.
- **2026-08-06**: Clean-room repo -- no SpeechAudioLM code imports, only
  absolute-path reads of its manifests/data.
- **2026-08-06**: Training recipe is **AudioLDM**'s, unchanged: eps-prediction
  MSE in VAE latent space + classifier-free guidance via condition dropout. The
  only substitution is where the conditioning vector comes from.
- **2026-08-06**: Conditioning encoder = **frozen Whisper-large-v3 encoder**
  (`WhisperModel.encoder` only; the decoder is never constructed and no
  transcript is ever produced). Rationale: the task needs *what was said*
  ("you almost hit me" -> traffic), which is what Whisper's encoder represents.
  A CLAP audio encoder on speech is expected to fail here because CLAP was
  trained on captions like "a man speaks" -- it encodes that speech is
  happening, not its meaning. That makes CLAP-audio a **control**, not the
  model. Encoders sit behind one interface (`src/salm/encoders/base.py`) so the
  Audio-Omni pattern (frozen Qwen2.5-Omni thinker layer -2 hidden states +
  zero-init linear; weights already local) drops in as experiment #2.
- **2026-08-06**: Frozen = speech encoder, mel VAE, HiFi-GAN vocoder.
  Trainable = mapper (32.3M params) + U-Net.
- **2026-08-06**: `cvssp/audioldm-s-full-v2` (small, ~180M U-Net) over
  `-l-full`: at 15k pairs and a short wall-clock budget the large variant costs
  more per step with no guaranteed win. One-line swap in `configs/paths.py`.
- **2026-08-06**: CLAP audio tower must be `laion/clap-htsat-unfused`. The
  locally cached `630k-audioset-fusion-best.pt` is the **fusion** variant with a
  different architecture and embedding space; using it would silently break the
  premise of aiming at the space the U-Net already understands.
- **2026-08-06**: Precompute VAE latents rather than encoding in-loop. Mel
  extraction and VAE encoding are the steps most likely to be subtly wrong and
  they are deterministic, so running them once makes mistakes surface in minutes
  and keeps the VAE out of VRAM during training. Speech stays loaded on the fly
  so the encoder and its pooling remain swappable.

## Training ladder

| Rung | Conditioning | Trains | Purpose |
|---|---|---|---|
| 0 | CLAP **text** of `audio_cap` | nothing | stack validation + ceiling |
| 1 | CLAP **audio** of the speech | nothing | control; expect generic output |
| 2 | mapper only, AudioLDM frozen, MSE+cosine to target CLAP audio embedding | mapper | cheap de-risking probe (NOT AudioLDM's loss) |
| 3 | **AudioLDM recipe proper** -- eps-MSE diffusion + CFG dropout | mapper + U-Net | the model to report |

Rung 0 is a hard gate before training: it is pure inference, and skipping it
risks discovering a broken generation path only after hours of training. It also
selects the mel `norm_mode` empirically by scoring both variants.

## Known-uncertain parameter

The mel post-log normalization (`norm_mode` in `src/salm/data/frontend.py`).
AudioLDM's own codebase has several variants across repos/versions, so it is a
config knob, and `scripts/check_audioldm_stack.py` picks the winner by
round-trip correlation. Do not trust any setting without that check -- a wrong
mel scale makes the VAE encode garbage while everything still appears to run.

## Environment

`.venv-salm-train-audioldm-py312` at
`/scratch/l/lovenya/salm_migration_clean/.venvs/salm-model/`. Built by
`scripts/setup_env.sh`; registered in `configs/runtime_environments.json`.
Verified versions: torch 2.13.0+cu130, diffusers 0.39.0, transformers 5.14.1,
librosa 0.11.0, numpy 2.4.6.

## Cluster notes

TamIA h100 node = **48 CPUs / 500000M / 4x H100-80GB NVLink** (the 64-CPU/1TB
rows in `sinfo` are h200 nodes). A GPU allocation reserves the whole node, so
CPU/RAM asks are cgroup limits inside it -- under-asking only starves our own
dataloader workers. Current request: 48 CPUs, 480G, 4 GPUs, 3 h.

Account: standing target is 60% `aip-csubakan` / 40% `aip-ravanelm`.

**The first runs deliberately went to `aip-ravanelm`, and the balance is owed
back to `aip-csubakan`.** Measured with `sbatch --test-only` at 08:04 on
2026-08-06, for the same 1 h whole-node job:

```text
aip-ravanelm -> predicted start 08:48  (44 min wait)
aip-csubakan -> predicted start 11:27  (3h 23m wait)
```

FairShare (0.123 vs 0.037) dominated, so csubakan would have cost ~2h40m and
missed the deadline. **Send the next non-urgent runs to `aip-csubakan`** to
rebalance. Always check with `sbatch --test-only` on both accounts before
assuming either is faster -- the gap moves.

Useful pattern: `sbatch --test-only --account=<acct> <script>` predicts the start
time without queueing anything.

## Job structure

Split into two chained jobs rather than one 3 h job:

```text
stage 1 (1 h): Rung 0 gate + VAE latent precompute
stage 2 (2 h): Rung 3 training + generation   [--dependency=afterok:stage1]
```

The reason is **failure isolation, not queue time** -- measured backfill gain was
only ~10 min (08:48 vs 08:58). What it buys: a gate failure costs a 1 h
allocation instead of 3 h, and the go/no-go signal on the mel/VAE stack arrives
sooner. Submitting stage 2 immediately with `afterok` lets it accrue priority
while stage 1 runs, so finish time is about the same as the monolithic job.

The stages are decoupled through the features directory -- stage 1 records the
selected mel `norm_mode` in `meta.json`, stage 2 reads it -- so there is no
run-ID coordination and stage 2 can be rerun independently.

## Open / TBD

- Whether the mapper can actually learn scene semantics from Whisper features at
  15k pairs. If it plateaus near the rung1 control, the frozen-MLLM encoder
  (Qwen2.5-Omni) is the next lever, not more U-Net capacity.
- Whether AudioLDM's VAE/vocoder, trained on general audio, is adequate here --
  the rung0 gate's round-trip scores are the evidence to check.
- Metrics/eval wiring: intentionally not started. FAD/IS/KAD/CLAP-score assets
  are already cached and reusable through `SpeechAudioLM/evaluate.py --stage
  s2a` once there is generated audio worth scoring.
- A2S content LM: the 97% that is language modelling, over 2048 acoustic codes from 17,596
  utterances. This is the bottleneck. Untried levers: coarse-to-fine hierarchy (a coarse pass
  at stride ~8 gives ~19 tokens, which fits the measured ~20-token horizon), BPE over content
  codes (`src/salm/a2s/content_bpe.py`, written, never run — its PASS bar of "<80 tokens" is
  four times too lenient given the horizon and must be rewritten first), and variable-length
  output (speech fills a median 111 of 256 frames, so 57% of the planning budget goes to a
  tail that carries no words).
- A per-scene-family diversity metric. `cross_clip_diversity` reads 0.909 for the greedy lane
  and hides that four stock openings cover 46% of 128 clips; splitting pairs by caption
  similarity exposes it (0.794 vs 0.909). Until that exists, no diversity claim about a
  deterministic decoder should be believed.
- Whether CosyVoice's `llm.pt` pretrained weights would help even though its token space does
  not. Much bigger commitment; not justified on current evidence.

