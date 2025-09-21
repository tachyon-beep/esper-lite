# Speculative Deltas

Purpose: Curate research concepts that could materially improve the Esper‑Lite prototype and present them as design‑style deltas. Each item is explicitly speculative (not committed to the prototype scope) but framed as if it were a planned design addition, with a minimal adoption path.

How to read: Each subfolder contains a short rationale, proposed behaviour, cross‑subsystem impact, a status/matrix, and a low‑risk adoption sketch for the prototype if desired.

Contents
- `scaffolds/` — Temporary, foldable blueprint deployments (policy overlay; no lifecycle change).
- `tamiyo-narset-split/` — Strategic vs tactical controller split with protected zones and inverse throttle.
- `mycosynth-config-fabric/` — Centralised, real‑time configuration fabric distinct from Leyline.
- `future-pruning/` — Hardware‑aware, structured pruning and torch.ao integration.
- `bsds-lite/` — Blueprint Safety Data Sheet (Urabrask) fields adopted in a lightweight form.

Notes
- These are research‑forward; keep prototype risk low by adopting metadata/contracts first, behaviours later.
- Leyline remains canonical; any contract changes are proposed there first.
