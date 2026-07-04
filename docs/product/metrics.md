# Metrics — Esper             Last read: 2026-07-05

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
| corr(reward, J) — reward aligned to the J yardstick (EPISODE-level: episode-summed reward vs episode J) | **THE A/B effect-size floor (PDR-0019): paired Δcorr ≥ +0.10 absolute at n=5; LB > 0 at n=10.** Primary gates = design criteria (i)–(v); the Δfossilize_payable_J overlay is superseded (structurally zero median); step-level is NOT the gate (F2 sampling ceiling) | **0.212 ± 0.046 (weak)**; probe recheck 0.179–0.261 per seed; corr(J,Σba) ≡ corr(J,Σreward) — ba is the reward's entire J-relevant content | 2026-07-03 (n=5) |
| **Placebo noise floor tau** = P99 of SIGNED terminal (φ − c_paid) for a null player — the north-star's "placebo-noise-floor" term and the enablement deadband | **OWNER-RATIFIED (PDR-0018): ACCEPT-PROVISIONAL / LOWER-BOUND ONLY** — sufficient to open F2 calibration; NOT the final ON operating threshold; ON-run recalibration MANDATORY before any n=10 magnitude claim | **tau = +0.28 pp (lower bound). Full 2-arm, 3240 samples/arm: 1e-3 arm +0.2401 CI [+0.2398, +0.2721]; 1e-4 arm +0.1201 CI [+0.1000, +0.1401]. Non-degeneracy PASSED. ⚠ plateau trigger FIRED (PDR-0015): floor is magnitude-dependent — hence lower-bound-only status** | 2026-07-03 (ratified) |
| **A/B coverage** — share of positive pre-fossil value mass reachable by the fossilize-only payment gate (J currency, param-normalized) | pre-registered rule (gate comment #99): <50% blocker, 50–75% yellow, >75% licensed | **66.7% → YELLOW** (raw cf-points 19.4%, caveat not gate — PDR-0018 §2). ⚠ SCOPE CORRECTION (PDR-0019): k=1 pays structurally zero — the true PAYMENT floor is k≥2 = **1.1%** of episodes (~12 paid events/run expected), hence the asymmetric-null rule | 2026-07-03 (n=5) |
| share_attribution — reward variance carried by the cf-attribution term | context only (magnitude-confounded: `ba`≈68% of |reward|) | 0.925 ± 0.004 | 2026-06-25 (n=5) |

## Guardrails (must NOT degrade)
| Metric | Floor / ceiling | Current | Read on |
|--------|-----------------|---------|---------|
| Host-accuracy contribution (all-seeds-on − all-disabled) | ≥ floor `<owner-set>` (must stay positive) | **+7.69pp mean, n=6 (host-alone ~38% → with-structure ~46%; per-run +6.67…+8.27, tight) — UPGRADES the prior n=1 +6.1%; morphogenesis confirmed REAL** | 2026-06-30 (3 control + 3 suppress) |
| **Policy DECISION-STEP (conditional) entropy** | must NOT collapse (≳ 0.1 on decision steps) | **HEALTHY — slot 1.0→0.30–0.70, op ~0.88 (never <0.1). Prior "⚠ BREACHED" was a FALSE ALARM (PDR-0006): the alarm read `head_*_entropy` = learnable_fraction × conditional — a density proxy diluted by ~60% forced steps; the ~411 anomalies were a telemetry artifact. Raw series → relabel `head_*_entropy_density`.** | 2026-07-01 |
| Rent / efficiency — committed param & compute cost per acc-point | ≤ ceiling `<owner-set>` | **n=5 BANKED (5/5 seeds): control 12–20 vs suppress(no r0c0) 7–9 pp/M-param, median Δeff −7.0 (sign-test p≈0.03) ⇒ r0c0 is an efficiency-enabling stem (b), NOT a freeloader (PDR-0009). DIRECTION banked; MAGNITUDE → owner's n=10 call (the "CI" is a sign test, not a CI). [prior n=1: +252% params / 7× compute for +6% acc]** | 2026-07-02 (n=5 J-read) |
| Churn not reward-farmed — germinate/prune per episode vs contribution | not decoupled from contribution | germ 12.4 / prune 11.6 per ep | 2026-06-25 (n=5) |
| **ON-wave mechanism validity (IN-FLIGHT, first 20% of runs)** — TOPUP finiteness, G1 trips, anomaly rates vs OFF baseline | zero non-finite; G1 zero trips; anomaly rates ≤ OFF baseline | **HEALTHY: 275 TOPUP all finite, tau=0.28 stamped 275/275, true episode ids confirmed; G1 trips 0/4 runs; reward-hacking + rollback rates OFF-consistent. ⚠ G4-flag (audit at scoring, NOT a trip): k=2 paid fraction 47% vs ~1% null-exceedance prior; 4/9 payments cap-clipped (tracker comment 121)** | 2026-07-05 (4 runs @ ~40 b.e.) |
