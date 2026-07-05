# PDR-0003 — Commission the owner-gated causal-contribution run to resolve the (a)/(b) fork

Date: 2026-06-28   Status: accepted   Author: Claude (agent)   Owner sign-off: yes (grant — dispatch delivery + launch GPU runs within the active bet)
Supersedes: —   Related: PDR-0002, PDR-0004, PDR-0005, docs/plans/concepts/2026-06-28-causal-contribution-run-design-v2.md

## Context
With the cheap-rescale escape hatch closed (PDR-0002), the redesign-build is gated on the
**(a)/(b) fork**: the early-converging r0c0 stem cohort (54% of fossils, negative per-seed
LOO) is either **(a) a freeloader** [reward should penalize it] or **(b) an LOO-undervalued
enabling stem** [reward should credit synergy] — opposite fixes. The removal-cost ablation
is structurally blind to (b): stems commit *before* their neighbours co-reside
(developmental/sequential enabling), so leave-one-out on the always-on baseline cannot see
enabling. The fork is therefore **unresolvable on existing observational data** — it needs
an intervention.

## Options considered
1. **Build an estimand-invariant intervention harness and run a paired causal experiment**
   (SUPPRESS-SLOT makes r0c0 un-committable; a three-domain RNG split + compare-point proves
   the suppression does not offset the controller sampling stream, closing the systematic
   Δ_struct bias channel) — pro: actually identifies a causal effect; con: substantial
   engineering + GPU.
2. Collect more observational seeds and re-fit LOO — REJECTED: more data cannot see a
   developmental effect the estimator is blind to by construction.
3. Drop the (a)/(b) distinction and pick a fix by prior — REJECTED: the two fixes are
   opposite; a wrong guess actively harms the redesign.

## The call
Option 1. Designed (v1 → v2 via workflow + adversarial review), built, independently
reviewed (R2, no blockers), hardened, and **validated on GPU** (Stage-1 compare-point proof
+ R1 full-scale offset-free gate). Committed to the branch (68fca06d harness/patch/docs;
e0134230 escrow-config-fix) — **not pushed** (within grant). The estimand *scope* of what
the run may claim is a separate, owner-gated decision (PDR-0004).

## Rationale
A causal question demands a causal instrument; the harness is the cheapest correct way to
make r0c0's contribution identifiable. Building it also produced reusable infrastructure
(RNG-domain split, offset-parity gates, NaN-safe sampling) that strengthens the whole line.

## Reversal trigger
Reopen / abandon the causal-run approach if, **after the entropy-collapse fix (PDR-0005)**,
the harness still cannot produce a trustworthy read — e.g. the estimand stays unidentified
on a healthy policy, or the realized paired-Δ SD is so wide that the (a)/(b) answer needs
more seeds than the morphogenesis floor (n=10) can supply. At that point the (a)/(b) fork
routes to a different method, not more runs of this one.
