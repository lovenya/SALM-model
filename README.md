# SALM-model

Training code for **SpeechAudio LM** — a generative model for bidirectional mapping
between environmental audio and speech.

Given an audio scene (rain, traffic, a crackling fire), generate a plausible spoken
utterance that fits it; and given an utterance, generate the ambient scene it would
plausibly occur in:

| Ambient audio | Plausible speech |
|---|---|
| Loud vehicle passing | *"Hey, you almost hit me! Slow down!"* |
| Crackling fire | *"Let's sit closer — it's getting cold."* |
| City traffic | *"We'll never make it on time at this rate."* |

This is harder than text-to-speech or audio captioning: it requires reasoning jointly
about acoustic context and spoken content, and the mapping is one-to-many and
inherently creative.

## Approach

The current model targets the **speech → audio (S2A)** direction and is built on
[AudioLDM](https://arxiv.org/abs/2301.12503).

The design rests on one property of AudioLDM: it does not condition on a text token
sequence. It conditions its U-Net on a **single global CLAP embedding**, and it is
*trained* on CLAP **audio** embeddings with CLAP **text** embeddings swapped in at
inference — CLAP's shared audio/text space is what makes that substitution work.

Because the conditioning slot is one vector in a shared multimodal space, it can be
aimed at from any input modality. So a small trainable **mapper** learns to predict that
vector from frozen speech-encoder features, and the pretrained AudioLDM turns it into
audio:

```
speech waveform
      │  front-end: mono, 16 kHz, fixed-length
      ▼
[frozen speech encoder]        Whisper-large-v3 encoder
      │                        (encoder only — no decoding, no transcript)
      ▼  frame-level hidden states
[trainable mapper]             attention pooling + MLP → 512-d, L2-normalized
      ▼
[AudioLDM U-Net]  ◄─ noisy latent, timestep
      ▼
[mel VAE decoder] → mel → [HiFi-GAN] → ambient audio
```

Frozen: speech encoder, mel VAE, vocoder. Trainable: mapper + U-Net.

Deliberately, no transcript is produced anywhere in the model. Routing through text is
what a cascaded ASR→LLM→generator baseline does; keeping this path end-to-end is what
makes the comparison meaningful.

**Objective.** AudioLDM's own training recipe, unchanged: ε-prediction MSE in the VAE
latent space, with classifier-free guidance trained by dropping the condition on a
fraction of steps.

## Data

Trained on TTAS, a paired environmental-audio / speech dataset, with direction-specific
train/val/test splits. Because each direction is filtered independently on its own
quality scores, the S2A and A2S manifests are not the same set of pairs.

Dataset preparation and evaluation live in a separate repository; this repo consumes the
finished manifests and contains model training only.

## Layout

```
configs/           path registry, environment map, run configs
src/salm/
  data/            manifest reading, audio front-end, dataset
  encoders/        frozen conditioning encoders (swappable)
  mapper.py        the trainable mapper
  train_ldm.py     training (AudioLDM diffusion objective)
  generate.py      sampling / inference
scripts/
  setup_env.sh              environment creation
  check_audioldm_stack.py   pre-training validation gate
  precompute_latents.py     one-time VAE latent extraction
  resource_monitor.py       job resource sampling
submission_scripts/         Slurm job scripts
```

## Usage

Create the environment (needs internet — run on a login node):

```bash
bash scripts/setup_env.sh
```

Validate the generative stack before training anything. This checks that weights load
offline, that mel extraction round-trips correctly through the vocoder and VAE, and that
sampling produces audible audio — it also selects the mel normalization variant
empirically:

```bash
python scripts/check_audioldm_stack.py --out-dir <dir> --device cuda
```

Precompute VAE latents for the targets (once per split), then train:

```bash
python scripts/precompute_latents.py --direction s2a --split train
torchrun --nproc_per_node=4 -m salm.train_ldm --config configs/s2a_rung3.yaml
```

Generate. `--mode s2a` uses a trained checkpoint; `--mode rung0` conditions on the
reference caption instead, giving an upper bound on what the generator can do with a
perfect description; `--mode rung1` feeds a CLAP audio embedding of the speech straight
into the frozen model, as a control:

```bash
python -m salm.generate --mode s2a --checkpoint <ckpt.pt> \
    --split val --num-samples 16 --out-dir <dir> --copy-reference
```

On a cluster, the whole chain runs as one job:

```bash
sbatch submission_scripts/run_tamia_s2a_rung3_one_node.sbatch
```

## Status

Early. The S2A training path is implemented and the first runs are in progress; A2S is
planned. Internal design notes, data details, decisions, and open questions are in
`CONTEXT.md`.
