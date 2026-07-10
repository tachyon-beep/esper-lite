# PDR-0060 — Pre-DECIDE evidence base banked; Gram-matrix telemetry contract adopted

Date: 2026-07-11
Status: accepted (analysis + spec within grant; the Path A′/B/C call itself is NOT made here)

## Context

After the Stage-2 REJECT (PDR-0059), the owner-relayed external review recommended an
error decomposition and seed-44 contrast before any critic-redesign DECIDE. Run same
session (task esper-lite-c004d67fb9, closed): first pass `0fceab52`, then a second
external review calibrated the claims and we added two findings it missed
(`fa1cbc42`). Full detail: `docs/analysis/2026-07-11-stage2-error-decomposition.md`
(including the post-review claim-status table).

## The call

1. **Bank the exact-input findings** as the pre-DECIDE evidence base:
   - LEG-A deficit is LATE-training and seed-heterogeneous (first half within δ on
     all seeds; s41/s43 diverge to persistent ~−0.10; s44 transiently positive →
     last-50 parity). Warmup-tax reading falsified.
   - Main head learns its own target well (ev_main 0.68–0.77 Q4); cf target is
     partially learnable (ev_cf 0.44–0.54 Q4) but its scale grows 3.7–11.3× within
     runs — the absolute burden outruns relative-error reduction.
2. **Hold as provisional** (estimated-denominator artifacts): error-covariance
   anticorrelation (suggestive only — corr_e beyond ±1); seed-44 as a
   scale-stabilization mechanism (n=1 association; largest Q1 lag ratio);
   shared-trunk interference (unidentified — no main-only arm exists).
3. **Adopt the sufficient-statistic telemetry contract**: per-update means + Gram
   matrix of (V_main, V_cf, G_main, G_cf) on the raw scale (~14 scalars) —
   supersedes the itemized field list; enables exact decomposition, the identity
   assertion, normalizer-lag reads, and the held-out affine calibration-rescue
   A′/B discriminator on all future windows. Rides the post-freeze observability
   landing (PDR-0058 scope note). Not retroactive: no per-sample data persisted.
4. **Record the endogeneity finding**: per-stream GAE bootstraps each stream's own
   head (`returns_cf = A_cf + V_cf`, rollout_buffer.py:678) — part of the cf
   non-stationarity is architecture-induced, so head-size/loss-weight tuning cannot
   rescue objective A; A′/B change the bootstrap topology, which is the point.
5. **Theory correction recorded**: component MSEs align with the total at the
   population optimum; the durable claim is finite-regime ("total fit left
   unprotected"), not an in-principle indictment of decomposition.

## What is explicitly NOT decided

The Path A′ / B / C selection. Current advisory lean: slightly B, A′ credible —
with pre-committed discriminators (held-out calibration rescue; cf residual after
scale stabilization; diag scale-slope/ev_cf recovery). The DECIDE is owner-gated
and sequenced after the schedule-correct 600-round diagnostic.

## Reversal trigger

If the exact (Gram-matrix) decomposition on the next instrumented run contradicts
the approximate reads — e.g. error covariance proves positive, or Var(e_cf) does
NOT dominate — the provisional findings are discarded wholesale and the Path lean
resets to neutral; the banked exact-input findings (Δ_A splits, EV trajectories,
scale slopes) stand regardless, as they used no estimated denominators.
