# Validation — 2026-09-23

Validated checkout: `/scratch/lovenya/speechaudiolm-paper/`.

Run from `/scratch/lovenya/SALM-model`:

```bash
node --check paper_companion/app.js
node scripts/website/verify_companion.mjs \
  /scratch/lovenya/speechaudiolm-paper
UV_CACHE_DIR=/tmp/salm-site-uv-cache uv run --no-project --offline \
  --python /scratch/lovenya/salm_migration_clean/.venvs/salm-model/salm-train-audioldm-py312/bin/python \
  scripts/website/validate_release.py \
  /scratch/lovenya/speechaudiolm-paper
```

Both validators pass. The release has six linked HTML pages, 100 unique common
test IDs, and 700 copied WAVs that hash exactly to the curated source files.
The validator checks RIFF structure and native sample rates (300 at 16 kHz,
300 at 24 kHz, 100 at 44.1 kHz). All selected pairing rationales match the
exact direction-specific clean-v2 test manifests.

Every one of the 500 generated-output judgments has five integer scores,
feedback and an exact v6 prompt: 2,500 prompts total. Per-example means equal
the five scores. Full-test aggregate tables have the expected systems and
criterion means; each matched/mismatched diagnostic contains 120 paired
cases per direction. Per-system coverage counts are not shown in result tables.
The DOM stand-in checks ten criterion demos, 20 example
links, all 40 demo audio paths, the criterion tables, and 400 sample/direction/
view states. It also exercises filters, direction switching, disclosures,
missing-score fallback and HTML escaping. Public data has no private absolute
filesystem paths.

The remaining 400 cascade intermediate fields are listed in
`missing-data.json`; generated-output transcripts/captions are outside the
site's scope. The website checkout passed both validators after the final
light style and content correction on 2026-09-23. Before these changes, the
user confirmed that deployed examples rendered in Chrome and in Edge after a
hard reload. Browser layout, audio decoding and keyboard/touch
behavior need a focused check after the next push.
