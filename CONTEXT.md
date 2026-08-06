# Context (for agents and developers)

This file is the internal working doc for this repo. It is NOT the public
README -- keep cluster paths, in-progress decisions, and open questions here.
Update it via the `/update-docs` skill after every meaningful milestone.

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

Account: standing target is 60% `aip-csubakan` / 40% `aip-ravanelm`. The first
run went to `aip-ravanelm` because FairShare there was ~3x better (0.123 vs
0.037); rebalance on later, less time-critical runs.

## Open / TBD

- Whether the mapper can actually learn scene semantics from Whisper features at
  15k pairs. If it plateaus near the rung1 control, the frozen-MLLM encoder
  (Qwen2.5-Omni) is the next lever, not more U-Net capacity.
- Whether AudioLDM's VAE/vocoder, trained on general audio, is adequate here --
  the rung0 gate's round-trip scores are the evidence to check.
- Metrics/eval wiring: intentionally not started. FAD/IS/KAD/CLAP-score assets
  are already cached and reusable through `SpeechAudioLM/evaluate.py --stage
  s2a` once there is generated audio worth scoring.
- A2S direction: not started.
