# Current State — Esper        Checkpoint: 2026-07-05 ~17:30 (checkpoint #17 — EV Stage-0 reconciliation + methodology correction; PDR-0028, PDR-0029)

## The bet right now
**EV-stabilization** (esper-lite-f25b71c165, Now bet per PDR-0027). Metric =
the **value-free EV Stage-0 gate** `Cov(R_cf, R)/Var(R) > 0.40` on a Stage-2-OFF
control run, + low `r_main_var_share`. **Delivery is on branch
`feat/ev-stab-stage2-hra`** (adopted PDR-0028 — that branch, not this one, holds
the real EV implementation).

## What just happened (this session)
The "implement Stage-0" task was really a **branch-divergence + methodology**
problem (PDR-0028). Stage-0's control-run gate was implemented NOWHERE: the existing
metric only fires on the Stage-2 ON leg and decomposes V_cf-contaminated GAE
λ-returns (leg-b biased toward passing) — invalid as a gate. **Two independent DRL
reviews converged** on this (opus plan-gate + fable specialist). Owner ratified
"do it right on ev-stab". **Landed** (ev-stab commit `f721a5b3`, TDD 4/4 green):
`compute_return_variance_shares` — the value-free, raw-scale, per-return covariance
decomposition (the canonical gate metric, reads identically on both legs).

## In flight
- **Stage-0 remaining plumbing on `feat/ev-stab-stage2-hra`** (esper-lite-3d67b09687,
  in_progress): per-component buffer SoA → both-legs ppo_agent wiring (demote the
  ON-leg λ-return metric to a Stage-2 diagnostic) → the 6-file telemetry contract →
  read the gate on a control run. Nothing running on GPU.
- Review gate esper-lite-cfbdfdf040 CLOSED (PDR-0029): Stage 3 scoped out (needs its
  own authored plan); MAJOR-1 (`ev_sum` hard floor) + MAJOR-3 (provenance) folded into
  the Stage-2 acceptance criteria.

## Facts the next session must not relitigate
- Do NOT read the Stage-0 gate from the ON-leg λ-return metric — it is V_cf-contaminated
  (PDR-0028). The value-free `compute_return_variance_shares` is the gate.
- Shapley A/B evidence ≠ HRA Stage-0 gate evidence (shapley⊕HRA mutually exclusive;
  MAJOR-3). Capture the value-free gate BEFORE any HRA ON-leg run.
- Two diverging copies of `partition.py`/`reward_variance.py` exist (per-step on this
  Shapley branch; per-return on ev-stab). Reconcile ONLY at owner-gated unification.
- Marginal-V (Stage 1) is ALREADY satisfied (directly-learned op-independent V(s)) —
  approved, do not re-open.

## Open questions / blocked-on-owner
- **Branch consolidation DONE (2026-07-05, owner-authorized + pushed):** product
  workspace + Shapley/Phase-0 line + squashed Stage-2 merged onto `origin/main`
  (`f3758236`). `feat/phase-minus1-scale-falsifier` is now fully contained in main
  (redundant — safe to delete; not done unilaterally). Live branches reduced 4 → 2.
- **REMAINING unification (deliberately deferred):** `feat/ev-stab-stage2-hra` (23
  commits, the active EV WIP) reconciles onto main via a per-hunk merge (Shapley-TOPUP
  → main; Stage-0 SoA → ev-stab; Stage-2 → ev-stab's newer post-`fab280c0`; `partition.py`
  = main superset; `reward_variance.py` = UNION per-step + per-return). Gated on: Stage-0
  complete + OFF-leg byte-identity + drl/pytorch review (CLAUDE.md). Do it off-main on a
  scratch branch, verify, then fast-merge — NOT a rushed core-training-path merge.
- Standing placeholders: north-star / rent TARGETs (metrics.md) still owner-set.
- **STANDING RULE (PDR-0030):** every experiment must reach a KNOWN disposition at its end
  — merge-to-main (default) / justified enduring fork / abandon — to avoid a pile-up of
  incomplete-data orphans (NOT to force premature retirement). Reproducibility is a valid
  reason to keep a fork forever, BUT the fork must STAND ALONE (deps/config/data + a
  reproduce recipe = a checkout-and-run, not a 6-week reintegration). Folded into
  CHECKPOINT. ev-stab's disposition = merge-to-main, deferred to Stage-0 completion.

## Last checkpoint did (checkpoint #17)
- PDR-0028 (reconciliation + branch adoption, owner-ratified) + PDR-0029 (review-gate
  disposition). metrics.md: added the EV Stage-0 gate row (reading pending).
- Committed the Stage-0 core unit on ev-stab (`f721a5b3`) + wardline `.gitignore` hygiene.
- Tracker: cfbdfdf040 closed; esper-lite-3d67b09687 annotated (core unit landed);
  Stage-3 (esper-lite-3defe42928) marked needs-authored-plan.
- Branch consolidation: docs/product + feature line merged & PUSHED to `origin/main`
  (`f3758236`); 4 → 2 live branches. Backup tags `backup-preconsolidation/*` retained.

## Next session, start here
**Continue Stage-0 on `feat/ev-stab-stage2-hra`** (checkout it first). TDD backlog:
port `decompose_additends` + `ADDITEND_SIGN_MAP` into ev-stab's `partition.py` → the
per-component buffer SoA (both legs) → both-legs `ppo_agent` wiring → 6-file contract
→ read `Cov(R_cf,R)/Var(R)` on a capable-host control run. Memory pointers:
`ev-stab-stage2-impl-state` (updated this session), `ev-variance-research-verdict`.
