# Metrics — Esper             Last read: 2026-06-30

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
| share_attribution — reward variance carried by the cf-attribution term | context only (magnitude-confounded: `ba`≈68% of |reward|) | 0.925 ± 0.004 | 2026-06-25 (n=5) |

## Guardrails (must NOT degrade)
| Metric | Floor / ceiling | Current | Read on |
|--------|-----------------|---------|---------|
| Host-accuracy contribution (all-seeds-on − all-disabled) | ≥ floor `<owner-set>` (must stay positive) | **+7.69pp mean, n=6 (host-alone ~38% → with-structure ~46%; per-run +6.67…+8.27, tight) — UPGRADES the prior n=1 +6.1%; morphogenesis confirmed REAL** | 2026-06-30 (3 control + 3 suppress) |
| **Policy slot-head entropy (controller health)** | must NOT collapse (≳ 0.1) | **⚠ BREACHED — collapses from ~update 3 (entropy <0.1; 594 anomalies/run R1, 411 GATE-1). General + pre-existing. The active Now fix (PDR-0005).** | 2026-06-30 |
| Rent / efficiency — committed param & compute cost per acc-point | ≤ ceiling `<owner-set>` | **+252% params / 7× compute for +6% acc (n=1; inefficient — the redesign target)** | 2026-06-27 |
| Churn not reward-farmed — germinate/prune per episode vs contribution | not decoupled from contribution | germ 12.4 / prune 11.6 per ep | 2026-06-25 (n=5) |
