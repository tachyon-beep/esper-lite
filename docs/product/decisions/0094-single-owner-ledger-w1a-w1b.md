# PDR-0094 — Re-review W1-A/W1-B adopted: the settlement ledger is the SOLE OWNER of the committed seed's entire per-seed reward row (not provisional-only); Δ_designed pinned to the ledger's always-targeted cadence

Date: 2026-07-14   Status: accepted (plan rev 3 §1; pre-reg §2.5.9 amendment updated; narrow re-check requested from the same reviewer per their scoping)
Follows PDR-0091 (W1). Input: drl-expert rev-2 re-review — REQUEST-CHANGES narrowly (W1 scope only; B1–B5/N1–N5 resolutions APPROVED as written).

## What does this buy?  (REQUIRED — PDR-0068)
It closes the two defects my own W1 fix would have shipped: a residual double-pay vector (the WAIT slot-mask is a
separate tensor that still admitted the committed slot, plus the single-active-seed fallback) and a silent drop of
the window PBRS stream that would have broken BOTH residual gates in the real run (~0.078/epoch × d ≈ 0.4–0.8 miss
against the |Δ| ≤ ~0.14 harness scale). The correction generalizes W1 into a principle the implementation can hold:
one owner per reward row, never two.

## Decisions
1. **Single-owner principle (W1-A):** during [R+1, B−1] the settlement ledger owns the committed seed's ENTIRE
   per-seed reward row — provisional (full live law at actual inputs; contribution=None branches apply on
   unmeasured epochs, no carry/imputation — N-r2b), within-HOLDING PBRS climb/drag, warning semantics, synergy —
   reproducing an always-targeted HOLDING seed. The policy path suppresses ALL per-seed components when
   `seed_info.committed` (defense-in-depth; closes the `slot_by_op[WAIT]` and index-0-fallback leaks by
   construction, not by mask hygiene alone).
2. **Cadence reconciliation (W1-B):** Δ_designed(x) for G-PBRS/G-DESIGNED-DELTA is computed under the
   always-targeted cadence the ledger enforces; the B-path replay models the ledger row, never a policy-targeted
   one. Without this the residual gates cannot read ≈0 in the real run.
3. **Transform-law consistency (N-r2a, documented not split):** the WINDOW keeps the full live law INCLUDING
   ratio_penalty (identical to S's law — maximal B−S continuity; a live spike pays for B exactly as for S); the
   ANNUITY excludes ratio_penalty (smoothed, policy-unselectable input — the spot-spike detector has no referent).
   The law change at B tracks the input regime change and is priced in Δ_designed, not hidden.
4. **Estimand/telemetry (N-r2c):** the every-epoch cadence now covers provisional AND pbrs — both labeled in the
   protocol-package estimand and included in the lock-burden/aliasing telemetry.
5. Process note: this is the second consecutive round in which verifying a fix produced the next finding (W1 found
   verifying B1a; W1-A/B found adversarially checking W1). The reviewer's request was scoped — only reworked §1 +
   the cadence note return for the narrow re-check; the rest of rev 2 is approved.

## Reversal trigger
- If the pre-commit code review finds a per-seed component NOT enumerated in the single-owner row (something beyond
  provisional/pbrs/warning/synergy keyed on seed_info) → the row inventory was incomplete; re-enumerate from
  compute_reward before landing.
- If the replay's B-path residual still misses Δ_designed after the cadence pin → the harness convention and the
  ledger disagree somewhere else (e.g., discount indexing); halt and reconcile before any further build.
