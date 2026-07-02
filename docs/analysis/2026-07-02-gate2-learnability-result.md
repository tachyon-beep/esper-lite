# GATE 2 — learnability probe: RESULT — PASS (retro-write mandated); terminal-flush FAILED

**Date:** 2026-07-02 · **Scope:** GATE 2 of the reward-credit term (PDR-0010, esper-lite-254175df90).
**Pre-registration:** `docs/analysis/2026-07-02-gate2-learnability-probe-preregistration.md` (thresholds locked
before the campaign ran). **Data:** `telemetry/gate2_probe/tier2_{control,retro}_s{51..55}.jsonl` (10 runs, K=30
updates each, same-seed pairs, δ_raw=2.0, n=5 regime config verbatim). Readout: `scripts/gate2_tier2_analyze.py`.

## Verdict (against the pre-registered rules)

**G2b — retro-write: PASS.** A once-per-episode credit written into the rollout buffer at the FOSSILIZE
timestep (pre-GAE, `divide_by_std`-scaled) propagates through this recurrent PPO's full production pipeline and
moves the policy's fossilize propensity.

| Pre-registered criterion | Bar | Measured | Result |
|---|---|---|---|
| Paired ΔP(FOSSILIZE\|eligible) > 0 at final K, one-sided | p < 0.05 | 5/5 pairs positive (+0.032…+0.147, median **+0.081** on ~0.13 control baseline); exact sign-flip **p = 0.0312** | ✅ |
| Persistence at final K (no rise-then-collapse) | positive at K | ΔP +0.087/+0.085/+0.084 at batches 25/27/29 — stable | ✅ |
| Magnitude ≥ control's own 5-update natural drift | ≥ 1× | median ΔP +0.081 vs drift 0.026 = **3.1×** | ✅ |

**G2a — terminal-flush: FAIL.** Pooled Tier-1 screen over 236 event-batches: realized Δadv(t_f)/σ_A at δ* =
median **0.0397** (q10 0.0035) — below the pre-registered 0.05 FAIL bar, exactly as the analytic decay predicted
(median (γλ)^(150−t_f) = 0.0139 at the regime γλ=0.94525). Retro-write's same screen: median **0.67 σ_A**/event.

**GATE 2 overall: PASS, with retro-write as a MANDATED BUILD REQUIREMENT** (the pre-registered conditional-pass
structure). Per the design consult, this PASS transfers to candidate B as well — retro-write is mechanically
B's delivery point — so the learnability question is settled for both candidates at once. L2 (sparse-event SNR),
L3 (shared-trunk gradient→logit), and L4 (critic catch-up within 30 updates) all cleared empirically.

## Guardrails (all pass)

- `advantage_std_floored`: 0/300 update batches; `update_skipped`: 0; rollback-contaminated envs: 0.
- Events: 37–79 fossilize events per run (retro arms 42–75; ≈329 injected events total).
- **Pairing verified empirically:** ΔP at the earliest common eligible batch ≈ ±0.00007 on seeds 51/52/54/55
  (trajectories near-identical pre-injection; CRN pairing held). Seed 53's arms first share an eligible batch at
  7 (post-divergence), +0.015 — consistent with treatment drift, endpoint mid-pack.
- Corroborating behavior: retro arms fossilized MORE than their paired controls on 4/5 seeds.
- GAE-linearity self-check (validate6): retro Δadv == δ_buf exactly; terminal Δadv == δ_buf·(γλ)^(T−1−t_f).

## Caveats (held honestly)

- **Learnability ≠ desirability.** The probe pays EVERY fossilize uniformly, so "propensity rises" is the
  learnability signal, not the shipped behavior — the real term is Shapley-gated (null-player = 0, deadband,
  cap, G-clamp), so it credits only enabling stems. The probe deliberately over-credits to test the channel.
- Early-training regime (first 30 updates from scratch): the running reward std is small (δ_buf ≈ 2.6–5.6 for
  δ_raw=2.0 vs ≈1.08 in the mature regime), so per-event signal in σ_A units will be smaller at maturity
  (~0.2–0.4σ_A/event by scale transfer, still ≫ the 0.05 floor). Recorded per batch in the JSONL.
- Injection targets all effective-FOSSILIZE steps (incl. any G5-failed attempts), not only committed seeds —
  identical across arms; a purity refinement for the build, not the probe.
- The n=10 magnitude question on the (b) result itself is untouched by GATE 2 (owner's call, PDR-0009).

## What this unlocks (per the issue's gate order)

1. ✅ GATE 2 learnability — PASSED (this document).
2. ✅ Signal verification — `fossilize_contribution_scale` multiplies the per-seed LOO (premise intact).
3. → **reward-function-reviewer review** of the design + this result, then **owner sign-off**. The flag stays
   `shapley_synergy_scale=0.0` (default-OFF) throughout; enablement is never autonomous.
4. Build requirement banked: the top-up must be delivered via **retro-write** (buffer write at t_f pre-GAE,
   `divide_by_std`, no clip, no normalizer-stat update; unit-test against hand-computed GAE per the design
   consult). The `pending_hindsight_credit` rail and terminal-flush are ruled out.

## Bug found en route

`use_telemetry=False` silently disables ALL blending/fossilization (gradient-health collection is
telemetry-gated; G2 hard-fails unmeasured — KTS-001). Filed: **esper-lite-4fe98055f7**.
