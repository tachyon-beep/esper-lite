# PDR-0058 — §1 EV-finiteness population corrected to the scored set (Option B, owner-ruled)

Date: 2026-07-11
Status: accepted (owner ruling, explicit)

## Context

The ON rerun completed 5/5 clean (rc=0 all arms, zero dropped events — the
PDR-0056 reversal trigger never fired; co-tenancy attribution CONFIRMED).
First scoring of the frozen packet returned **INVALID** on one §1 breach:
non-finite `explained_variance` on both arms. Diagnosis
(`docs/analysis/2026-07-11-stage2-invalid-adjudication.md`): exactly 2 rows of
2000 — the only two sub-floor-variance updates in all ten arms — carrying the
emitter's DELIBERATE `explained_variance=NaN` low-return-variance marker
(vectorized.py flagged-exclusion; `value_loss`/`value_nrmse` finite on the same
rows, so no numerical event). The §1 checker ran on the pre-§8B population —
broader than the gate doc's own "scored updates" definition. Step-3 calibration
had read the same OFF data clean (the check arrived post-freeze in the
review-hardening batch).

## Options

- **A — uphold INVALID** (strict letter of one §1 bullet): makes any future
  wave invalidatable by a single documented non-measurement marker; structural
  emitter/gate collision recurring at a rate set by return-variance dynamics.
- **B — rule the finiteness population = the scored set** (gate doc's own
  definition; §11.2's "finiteness on scored updates" language): chosen.

## The call (owner, 2026-07-11)

Option B — "not a discretionary relaxation after an inconvenient result; a
correction of the validity population to match the emitter convention, the
pre-registered §8B exclusion, and the gate document's own scored-updates
definition." Implemented as the three-layer envelope (gate doc **§11.3**):
global numerical health on all post-burn-in candidates (`ev_return_variance`
finite everywhere — it is, 2000/2000); the NaN marker permitted ONLY on
§8B-floored updates (±Inf never; `ev_sum`/`ev_main`/`ev_cf` strict — no marker
convention); §11.2-hard diagnostics finite on scored updates. One shared
`decompose_population` ends the validity/statistics split-brain. Ten new
pre-registered contract tests; e2e CLI fixture's pre-existing `cf_value_loss`
signature gap fixed (was failing at HEAD). No threshold or treatment statistic
changed.

Governance held: no-peek (no leg/guard/verdict quantity read before the
ruling; counterfactual verdict not computed), inputs hashed
(`2026-07-11-stage2-rescore-manifest.txt`), correction committed (`d187402a`)
BEFORE the single rescore, first packet preserved as ADJUDICATION-HOLD.

Declined scope (recorded): `value_loss`/`value_nrmse` as new §1 hard conditions
(post-data gate-tightening; proposed for the n=10 pre-registration instead).
TIP registry item: replace the NaN marker with `ev_defined` +
`ev_undefined_reason` (missingness semantics).

## Reversal trigger

Any future non-finite EV NOT exactly explained by the envelope (NaN on an
unfloored update, any ±Inf, or non-finite `ev_sum`/`ev_main`/`ev_cf` anywhere)
is a hard INVALID and re-opens the envelope design itself — treat as a
numerical failure investigation, never widen the envelope to admit it.
