# Metrics — Esper             Last read: 2026-07-03

> Research framework, not a commercial product: the "north-star" is the morphogenetic
> objective J (committed per-param counterfactual gain), not an engagement metric.
> BASELINE values below are real session readings; TARGET numbers/dates are placeholders
> for the owner to set (a target with no number+date is not falsifiable — do not accept).

## North-star
| Metric | Target (falsifiable) | Current (BASELINE) | Read on | Trend |
|--------|----------------------|--------------------|---------|-------|
| **Committed J** = Σ_seed (1/params)·Σ_t cf·α over committed (FOSSILIZED) residency | ≥ placebo-noise-floor + TARGET by `<owner-set date>` | **~0 on control (defect under repair); fossilize/ep 0.207±0.006** | 2026-06-25 (n=5 control) | → |

## Input metrics (the levers that move the north-star)
| Metric | Target | Current | Read on |
|--------|--------|---------|---------|
| corr(reward, J) — reward aligned to the J yardstick | ≥ TARGET `<owner-set>` | **0.212 ± 0.046 (weak)** | 2026-06-25 (n=5) |
| **Placebo noise floor tau** = P99 of SIGNED terminal (φ − c_paid) for a null player — the north-star's "placebo-noise-floor" term and the enablement deadband | calibrated when the full-run P99 CI can bound a false-positive budget (CONSERVATIVE-PROVISIONAL until first ON run) | **PRELIMINARY (smoke, N=72): +0.252 pp, block-bootstrap CI [+0.106, +0.380]; NON-degenerate (77.8% nonzero). Full 2-arm read (3 seeds × std {1e-3, 1e-4}, ≈3240 samples/arm) in flight** | 2026-07-03 (smoke) |
| share_attribution — reward variance carried by the cf-attribution term | context only (magnitude-confounded: `ba`≈68% of |reward|) | 0.925 ± 0.004 | 2026-06-25 (n=5) |

## Guardrails (must NOT degrade)
| Metric | Floor / ceiling | Current | Read on |
|--------|-----------------|---------|---------|
| Host-accuracy contribution (all-seeds-on − all-disabled) | ≥ floor `<owner-set>` (must stay positive) | **+7.69pp mean, n=6 (host-alone ~38% → with-structure ~46%; per-run +6.67…+8.27, tight) — UPGRADES the prior n=1 +6.1%; morphogenesis confirmed REAL** | 2026-06-30 (3 control + 3 suppress) |
| **Policy DECISION-STEP (conditional) entropy** | must NOT collapse (≳ 0.1 on decision steps) | **HEALTHY — slot 1.0→0.30–0.70, op ~0.88 (never <0.1). Prior "⚠ BREACHED" was a FALSE ALARM (PDR-0006): the alarm read `head_*_entropy` = learnable_fraction × conditional — a density proxy diluted by ~60% forced steps; the ~411 anomalies were a telemetry artifact. Raw series → relabel `head_*_entropy_density`.** | 2026-07-01 |
| Rent / efficiency — committed param & compute cost per acc-point | ≤ ceiling `<owner-set>` | **n=5 BANKED (5/5 seeds): control 12–20 vs suppress(no r0c0) 7–9 pp/M-param, median Δeff −7.0 (sign-test p≈0.03) ⇒ r0c0 is an efficiency-enabling stem (b), NOT a freeloader (PDR-0009). DIRECTION banked; MAGNITUDE → owner's n=10 call (the "CI" is a sign test, not a CI). [prior n=1: +252% params / 7× compute for +6% acc]** | 2026-07-02 (n=5 J-read) |
| Churn not reward-farmed — germinate/prune per episode vs contribution | not decoupled from contribution | germ 12.4 / prune 11.6 per ep | 2026-06-25 (n=5) |
