# Paper companion source

This directory is the static-site template. The public GitHub Pages repository
is `/scratch/lovenya/speechaudiolm-paper`, which serves
`https://lovenya.github.io/speechaudiolm-paper/` after its `main` branch is
pushed and GitHub Pages finishes deploying. The template's `site-data.js` is
deliberately empty; publish a built release, not this template.

The site has a paper-title homepage, Dataset, G-Eval Audio, Results, Model, and
Examples pages. The Model architecture content is deferred. The Examples page
uses 100 common clean-v2 test IDs, seven recordings per ID, both reference
texts, dataset pairing rationale, optional speaker metadata, and a direction
switch. Direct and cascade systems appear in readable vertical rows. Each
generated output shows all five v6 criterion scores. Opening a criterion shows
judge feedback; the saved exact prompt is a second disclosure. Cascade stage
text and generated-output descriptions remain explicit placeholders.

The G-Eval page shows full-test criterion means, a 120-pair matched/mismatched
diagnostic in each direction, and highest-/lowest-scoring listening examples
for all ten direction-specific criterion entries. It labels the v6 snapshot
because the paper draft and prompt revisions have provisional newer numbers.
The clean-v2 ground-truth score is a filtered reference point, not independent
validation of the judge. Objective metrics are on the Results page.

## Rebuild

The current validated release is
`/scratch/lovenya/salm_model_outputs/paper_companion_100_geval_20260923/`.
It contains 100 shared IDs, 700 source-identical WAVs, 500 generated-output
judgments, 2,500 exact prompts, and `missing-data.json`. Rebuild to a new
directory so earlier releases remain available:

```bash
UV_CACHE_DIR=/tmp/salm-site-uv-cache uv run --no-project --offline \
  --python /scratch/lovenya/salm_migration_clean/.venvs/salm-model/salm-train-audioldm-py312/bin/python \
  scripts/build_paper_website.py --examples 100 \
  --output /scratch/lovenya/salm_model_outputs/paper_companion_next
```

The exporter reads the curated 100-pair bundle, exact clean-v2 manifests,
completed system evaluations, v6 prompt ledgers, and saved mismatch review. It
does not run models or recalculate scores. A fresh score release should update
all G-Eval sources together and their displayed version label; do not mix v6
criterion results with the draft paper's newer provisional numbers.

To fill missing text, supply one JSONL row per selected `sample_id` via
`--supplements`. Use `supplements.example.jsonl` as the schema. The current
receipt has 400 missing cascade-stage text fields and 500 generated-output
transcript/caption fields. No evaluation scores, feedback, prompts, or audio
are missing. The builder rejects unknown IDs, duplicate IDs and invalid
fields. Reference captions/transcripts are never substituted for missing
generation-stage results.

Run the validations in [VALIDATION.md](VALIDATION.md) before copying the release
to the website repository. The website repo, not this training repo, holds the
public WAVs. GitHub Pages serves relative links from the project subpath and
`.nojekyll` preserves the plain static site.
