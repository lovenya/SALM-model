---
name: update-docs
description: Update CONTEXT.md (internal, agent/dev-facing) and, if relevant, README.md (public-facing) after a milestone or important decision in the SALM-model project. Invoke as /update-docs.
---

# Update docs

Keep `CONTEXT.md` and `README.md` in sync with reality after a meaningful
milestone: a decision made, a component implemented, a data/config change,
a run completed, a blocker found. Do this whenever the user invokes
`/update-docs`, using the current conversation as the source of what
changed.

## Ground rules

- `CONTEXT.md` is for us (agents + developer). It can and should contain
  cluster paths, decisions with rationale, data quirks, open TBDs, dated
  history. Be precise and concrete -- numbers, paths, dates -- not vague
  status language.
- `README.md` is for the public/consumers. Keep it short: what the project
  is, current high-level status, how to use it once there's something to
  use. No cluster-internal paths, no in-progress design debates, no
  rationale for decisions, no dated history.
- Never invent facts. If something changed in this conversation, use it. If
  you're unsure a detail belongs in CONTEXT.md vs README.md, default to
  CONTEXT.md.
- Don't rewrite sections that haven't changed. Edit incrementally.

## Steps

1. Scan the current conversation since the last doc update for: decisions
   made, files/components added, data findings, open questions resolved or
   newly raised, and anything explicitly flagged TBD/pending.
2. Update `CONTEXT.md`:
   - If it fits an existing section (Data, Decisions log, Open/TBD, etc.),
     edit that section in place.
   - Log dated decisions under "Decisions log" as `- **YYYY-MM-DD**: ...`.
   - Move resolved items out of "Open / TBD"; add newly surfaced ones in.
   - If the doc has grown unwieldy, consider whether older dated entries
     should move to a `PROGRESS_LOG.md`-style history file instead of
     bloating the living-state doc -- ask the user before restructuring.
3. Update `README.md` only if something public-relevant changed (project
   status line, a new usable entry point, setup/usage instructions). Skip
   it if the update is purely internal (e.g. a path convention, an
   unresolved design debate).
4. After editing, summarize for the user in the chat what changed in each
   file (a few bullets), not a full diff dump.
