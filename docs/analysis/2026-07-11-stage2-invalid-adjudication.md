# Stage-2 packet INVALID — spec-vs-code adjudication request (owner-gated)

**Date:** 2026-07-11 00:25
**Status:** ESCALATED — owner ruling required before any scorer change or rescore.
**Precedent pattern:** §11.2 / step-3 escalations (implemented faithfully → collided
with real data semantics → owner adjudicates → amendment recorded pre-interpretation).

## What happened

The full 5-seed ON rerun completed clean (5× rc=0, zero dropped-event lines, all
arms 200/200 updates; PDR-0056 reversal trigger never fired). Packet scored at
00:14 under the frozen spec (`docs/analysis/2026-07-11-stage2-score-spec.json`:
δ=0.050311, ε_rel=0.10, τ_acc=0.3, Δparam_max=1527.8, W=17, budget=150,
g3=1.5, g4=2.0/floor 2, host_params=6418 telemetry-cross-checked).

**Verdict: INVALID** (`docs/analysis/2026-07-11-stage2-major1-packet.txt`) —
§1 breach: "non-finite (NaN/Inf) scored metric(s): explained_variance" on BOTH
arms. No other §1 breach fired (budget, signatures, pairing, provenance,
tracebacks, config equivalence all passed).

## Diagnosis (every step verified in data + code)

1. **Scope:** exactly 2 rows of 2000 — `off_s44` update #65, `on_s45` update #57.
   Both have `ev_return_variance` 0.799/0.868 — BELOW the §8B floor (1.0) — and
   they are the ONLY two sub-floor updates in all 10 arms (NaN ⟺ floored, exactly).
2. **Not a critic failure:** `value_loss` (0.018/0.036) and `value_nrmse`
   (0.54/0.52) are finite on the same rows — a NaN residual would have made
   nrmse NaN. No numerical event occurred.
3. **Emitter convention (frozen training path, common-mode both arms):**
   `vectorized.py:406-420` ("EV-telemetry-robustness flagged-exclusion") emits
   the batch-level `explained_variance` as the mean over UNFLAGGED updates only;
   "if none are unflagged, set NaN." With 1 update/batch, a single
   `ev_low_return_variance=true` update emits `explained_variance=NaN` BY
   DESIGN — an "EV undefined here" marker, not a measurement. Raw event
   confirms: `explained_variance: NaN`, `ev_low_return_variance: true`.
4. **Scorer §8B** (`floored_exclusion`) exists precisely to drop these updates
   from every level/vol statistic, keyed on the (finite, present)
   `ev_return_variance ≤ 1.0`.
5. **The §1 finiteness check** (`_validate_leg`,
   `stage2_acceptance_packet.py:664`) runs on post-burn-in but PRE-§8B rows —
   so it trips on the two convention-markers §8B is about to exclude.
6. **The gate doc's own vocabulary:** "scored updates" is DEFINED as post
   burn-in + post §8B exclusion (gate doc thresholds table, budget row:
   "budget counts scored updates POST burn-in + POST §8B exclusion (verified in
   code)"); the §11.2 amendment says "finiteness on scored updates". The two
   NaN rows are NOT scored updates under the doc's own definition.
7. **Calibration consistency:** step-3 calibration read the SAME off_s44 data
   (incl. the NaN row) clean — the OFF-calibration path does not run
   `_validate_leg`; the score-path finiteness check came in with the
   review-hardening batch after freeze. The data did not change; the checker
   population did.

## The ruling requested

- **Option A — uphold INVALID (strict letter of the §1 bullet):** "any
  non-finite in a scored metric" read as "anywhere post-burn-in in a metric
  that gets scored." Consequence: this wave is invalid over 2/2000 deliberate
  non-measurement markers, and ANY future 200-round arm with a single
  sub-floor-variance update anywhere post-burn-in invalidates its wave — a
  structural collision between the emitter convention and the gate that will
  recur at a rate set by return-variance dynamics, not by run health.
- **Option B — rule the §1 finiteness population = the scored set (gate-doc
  reading):** EV-family finiteness is evaluated on post-§8B scored updates;
  `ev_return_variance` itself stays checked on the full post-burn-in set (it
  is the flooring key and must be finite everywhere — and it is, 2000/2000).
  Fix is read-path-only in the scorer + tests; then rescore the same packet.

**Recommendation: B**, on population-correctness grounds. Blind-discipline
note: the counterfactual verdict has NOT been computed — §1 was the only
breach, but no leg/guard statistics have been read, so this ruling cannot be
influenced by what the verdict would become.

## Also recorded (either way)

- The emitter convention (NaN-by-convention at the run-batch aggregate) is
  exactly the class of population/convention seam the entropy-floor audit and
  TIP metric-registry work target: the flag (`ev_low_return_variance`) and the
  floored value travel in different fields than the headline metric, and two
  independently-documented components disagreed about what NaN means. Add the
  §1-population reconciliation to the TIP validity-envelope registry when TIP
  starts.
- Floored fractions are 1/183 vs 0/183 on the affected pairs — far inside
  `floored_asymmetry_max=0.10`; §8B semantics are untouched by this event.
