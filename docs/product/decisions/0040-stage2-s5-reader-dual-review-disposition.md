# PDR-0040 — Stage-2 MAJOR-1 acceptance: S5 impure reader landed + dual-review disposition (Findings 1–4)

Date: 2026-07-07   Status: accepted (within grant: build + accept acceptance-harness code for the active bet; publish the checkpoint)
Author: Claude (Opus 4.8)   Owner sign-off: n/a (autonomous within grant; owner this session dispatched "reconcile the drift, fold the S5 — it already has, just publish — then carry on")
Related: PDR-0039 (wrapper architecture + pure layer S1–S4 — this executes its S5 slice and hardens the pure predicates), PDR-0037 (pure scorer), gate doc `docs/analysis/2026-07-06-stage2-hra-major1-acceptance-gate.md`, task esper-lite-2a4b56e719, epic esper-lite-f25b71c165. Commits `49536a08` (S5), `424bc7e2` (Findings 1–3), `e019665c` (Finding 4).

## Context

PDR-0039 landed the torch-free pure layer (S1–S4) and mapped the impure tail S5–S7. Between
checkpoint #24 (commit `970c11bc`, 03:22) and this checkpoint, a prior working session executed
**S5** (the duckdb reader boundary) and then ran **two independent reviews (drl-expert + external
research agent)** of the whole MAJOR-1 wrapper. The reviews surfaced four findings — one a
verified-live soundness defect in a hard-REJECT safety gate. All four were fixed. This work
committed to code and the task tracker but was **never written back to the workspace**, so
`current-state.md` and `metrics.md` fell a session behind git. RESUME/ORIENT this session detected
the drift; this PDR reconciles it and records the review disposition durably.

## Options considered (for the drift)

1. **Silently rewrite `current-state.md`/`metrics.md` to match git.** Rejected — a decision (the
   review disposition, including a soundness fix to a REJECT gate) with no PDR is indistinguishable
   from drift to the next session, and destroys the append-only audit trail.
2. **Publish a checkpoint: PDR + refreshed artifacts + commit.** Chosen — restores the audit trail;
   the S5 + dual-review work becomes durable and attributable, and the fixes are re-openable via a
   real reversal trigger tied to `metrics.md`.

## What landed (durable)

- **S5** (`49536a08`, `stage2_acceptance_io.py`): `read_leg` — `ppo_updates` → chronologically
  ordered typed `UpdateRow`; `row_to_update` is the **fail-loud** dict→typed boundary (missing
  column = schema drift → raises; NULL per-stream `ev_*` = a valid OFF leg). §6 safety readers over
  `episode_outcomes` — `read_run_val_acc` (G1), `read_run_added_params` (G2), `read_run_churn` (G3).
  duckdb isolated from the pure layer. 15 IO tests over in-memory duckdb.
- **Finding 1 — UNSOUND, verified live → FIXED (`424bc7e2`):** G1/G2 read a **single env, not an
  env-average**. `episode_idx` is a dense **global** counter (`episodes_completed + env_idx`), so
  `MAX(episode_idx)` selects one row (highest env of the final wave) and `AVG` collapses over it —
  the noisiest possible per-seed estimate feeding a **hard-REJECT** gate. Fixed: per-env terminal
  row (`ROW_NUMBER PARTITION BY env_id ORDER BY episode_idx DESC`), then mean across envs.
- **Finding 2 — comment error → FIXED:** the G2 "exact fossilized" claim was false;
  `host_params*(param_ratio-1)` is the **total** added-parameter footprint (superset of fossilized),
  owner-ratified as a **conservative proxy**; relabelled "terminal added-parameter footprint".
- **Finding 3 — DIVERGES → FIXED:** §4 LEG-B covariates (`IQR(Var(returns_total))`, raw
  advantage-std IQR) were computed in `leg_series` but dropped at the report boundary. Now threaded
  into `Stage2Report`, and the **§4 confound-downgrade** implemented (LEG-B PASS → **INCONCLUSIVE**
  when `IQR(Var(returns))` drops as much as the gated residual — a return-regime artifact, not
  actor-path shielding).
- **Finding 4 — §8F(i) config fence → FIXED (`e019665c`):** `validate_pair` enforced only
  `seed`+`reward_mode`, but §10 freezes `per_head_advantage_norm=False` on both legs. Added
  `read_run_uses_per_head_norm` (`BOOL_OR` over `ppo_updates` — catches an any-point toggle) +
  `RunMeta.uses_per_head_norm`; `validate_pair` now rejects the pair if either leg used it.
- **Tests:** 56 wrapper + 216 telemetry-suite green, ruff + mypy clean, still torch-free, **no
  training-code touch**.

## Recorded deferrals (carry into S6/S7 — NOT bugs)

- env-count-completeness validity → **S6** (needs `runs.n_envs`).
- lr/entropy schedule-match → **S7** or parked.
- G4 guard-channel reader + `read_run_meta` (`actor_advantage_source`) → **S7** (the emission that
  creates that column).
- G3/G4 "materially elevated" thresholds + G1/G2 definitions → definitional, pending §6/drl
  confirmation.

## The call

Accept both reviews and their fixes: the gate design survives review with a soundness bug removed
and the §4 confound-downgrade added, all confined to the impure/predicate layer. Publish the
checkpoint (this PDR + artifacts). **Continue to S6** (CLI + markdown verdict packet), then S7
(emission → freeze → A/B). The gate remains **NOT-FREEZABLE until S7**; **no A/B before freeze**
(MAJOR-1 pre-registration, unchanged).

## Rationale

The dual review did exactly what pre-freeze review exists for — caught a hard-REJECT gate reading
the noisiest possible per-seed estimate before any verdict depended on it. That the fixes are
confined to the reader/predicate layer (no training-code touch) confirms PDR-0039's pure→impure,
training-touch-last ordering paid off: the soundness surface was reviewable and cheaply reversible
at the moment the bug surfaced.

## Reversal trigger

- If S6/S7 or the mandated S7 specialist review shows the **per-env-terminal G1/G2 aggregation** or
  the **§4 confound-downgrade** is still mis-specified against the real telemetry → revisit before
  freeze (the gate is not frozen; this is the window for it).
- **Advisor check-2 (PDR-0039) still open:** the §9 emission's **OFF-leg byte-identity** must be
  verified by a runnable GPU-free test before S7 lands, else the emission approach is reworked.
- Unchanged: gate **NOT-FREEZABLE until S7**; **no paired A/B before freeze**.
