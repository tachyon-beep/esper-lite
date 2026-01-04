# Phase 3: Addressable Subtargets (Grouped Indexing, Not Slot Explosion)

This folder is a working plan surface. The phase details live in:

- `docs/plans/planning/kasmina2/kasmina_tamiyo_submodule_intervention_roadmap.md` (Phase 3 section)

## Objective

Increase controllability **within** a slot without creating thousands of new slots:

- introduce deterministic inventories of internal subtargets (e.g., head groups, channel groups)
- add a grouped addressing mechanism so Tamiyo can select a subtarget without combinatorial blow-up

## Why this is its own phase

Phase 0–2 keep “microstructure” aggregated (few scalars). Phase 3 introduces:

- new policy head(s) or new factoring in the action space
- new causal relevance masks
- stronger requirements on stable subtarget IDs and inventory ordering

## Candidate design alternative (from `docs/plans/planning/kasmina2/chatgpt_feedback.md`)

Before committing to a new `subtarget_idx` head, evaluate a *headless* addressing scheme:

- Add two ops:
  - `FOCUS_NEXT_SUBTARGET`
  - `FOCUS_PREV_SUBTARGET`
- Maintain a per-slot `focus_idx` (internal state), and interpret `GROW_INTERNAL/SHRINK_INTERNAL` as “apply to focused subtarget”.

Why it might be useful:

- Avoids action-space explosion (adds 2 ops, not a whole head).
- Keeps deterministic, bounded behavior (K small, focus cycles through a fixed inventory).

What it costs:

- Requires new sensors (at least focused subtarget identity + focused level) to keep “sensors match capabilities”.
- Adds another internal state variable that must be DDP-symmetric and fail-fast.

## Docs to write before coding Phase 3

- `leyline_contracts.md` (SubtargetKind/SubtargetId; addressing head spec; causal masks)
- `inventory_rules.md` (stable, deterministic ordering; no float ties; no drift)
- `tamiyo_scaling.md` (representation pressure; when to pivot to Phase α)
- `experiments_and_gates.md` (invalid-action rate, entropy collapse, pruning/fossilize stability)
