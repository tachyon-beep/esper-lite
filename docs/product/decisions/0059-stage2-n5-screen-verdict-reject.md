# PDR-0059 — Stage-2 MAJOR-1 n=5 screen verdict: REJECT (read banked)

Date: 2026-07-11
Status: accepted (pre-registered gate, mechanically evaluated; read delivered to owner)

## Context

Rescore of the identical frozen spec under §11.3 (PDR-0058): **§1 VALID** —
the two NaN-marker rows are the only §8B exclusions (0.5% on each affected
arm). Packet of record: `docs/analysis/2026-07-11-stage2-major1-packet.txt`
(commit `bac4cc7f`); spec + input hashes committed pre-rescore.

## The verdict (screen predicate LEG-A ∧ MECH ∧ G1–G4, §11.1)

**REJECT**, on two independent hard conditions plus a guard:

- **LEG-A FAIL** (MAJOR-1 hard override): per-seed Δ_A = −0.1022, −0.0876,
  −0.0978, **+0.1742**, −0.0928; median ≈ −0.093 < −δ = −0.0503. The
  decomposed critic fits the total return materially WORSE than the single
  head on 4/5 seeds.
- **MECH fail** (structural): ON `ev_main` volatility not below the OFF
  baseline on ≥⌈0.8n⌉ seeds — the predicted stabilization mechanism did not
  fire.
- **G4 fail**: gradient-pathology on every ON seed (4–7/run vs OFF 0–9 with
  several zeros/ones) and s42 rollbacks 34 vs 3. Mixed picture: ON
  value-collapse 0–2 vs OFF 3–11 (ON better there).
- **G1/G2/G3 PASS** — no host-accuracy regression, no param inflation, no
  churn farming. The kill is about the critic mechanism, not behaviour.
- LEG-B (descriptive, §11.1): INCONCLUSIVE — 4/5 seeds reduced adv-residual
  volatility but the pre-registered confound downgrade fired; seed 44 is an
  outlier in BOTH legs (+0.174 Δ_A, +0.555 Δ_B; noted, not interpreted at n=5).

Descriptive context that travels with the verdict: cf target scale ~14–18 vs
main ~4–5 with Q1→Q4 drift ~2–4 → ~24–28 on every seed — the cf-stream
non-stationarity (same root as the LEG-B demotion and §11.2). Mediator caveat
(PDR-0058 ruling wording): this estimates HRA under the current entropy-floor
objective and 200-round schedule, including downstream interactions.

## Pre-committed consequences (§7 — what REJECT means, no more)

NO n=10 wave for HRA objective-A as implemented. The Stage-0 diagnosis (cf
stream dominates return variance, ratio ~1.0, PDR-0032/0033) is untouched —
what failed is this treatment (decomposition without stabilizing the
non-stationary cf target), not the de-shaping rationale. What to do next with
the EV-stabilization epic (TIP first / reward-efficiency / redesigned HRA with
cf-target stabilization / Stage-1 per-head-norm flag) is an OWNER DECIDE that
this PDR deliberately does not make.

## Reversal trigger

None for this implementation — the §7 screen is final for it. A redesigned HRA
(e.g., cf-target normalization/stabilization first, per the §9 scale evidence)
enters through a NEW pre-registration with its own frozen gate, not by
reopening this one. Post-freeze sequence already licensed and unchanged:
penalty-schedule fix (drl-expert review, PDR-0055) → clean 600-round seed-41
OFF diagnostic.
