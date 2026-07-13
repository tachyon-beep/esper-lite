# Pre-registration — the offline advantage-vs-LOO read (§5 "the floor already ran the RCT") + §3 KL gate

Date: 2026-07-14 · Written BEFORE any number is read (over-read guard). Governs the two decisive zero-GPU reads that
gate the GPU spend. Runs: `telemetry/stage2_on_longdiag/seed{41,42}` longdiag (~1.08M decisions/seed). Companion:
`2026-07-13-round8-findings-evaluation.md` (§5, §3).

## Why this is a natural experiment (the identifying assumption, stated up front)
The realized op mix at HOLDING is **flat across every current-LOO bin** (~55/17/15/13, both seeds) — the anti-WAIT
floor forced FOSSILIZE at ~15-17% *independent of LOO*. That is **marginal ignorability**: assignment to FOSSILIZE is
(marginally) independent of the "dose" (LOO). So the existing telemetry already estimates the counterfactual the policy
gradient would act on. **This is MARGINAL, not conditional** — residual state-confounds (seed age, slot occupancy,
host-accuracy trend, round) must be stratified out before the differential is trusted.

## Read A — advantage-vs-LOO (§5), the reframe's only unproven bridge

**Quantity:** for HOLDING op-decisions, per-decision advantage `A` for the realized op, binned by current LOO
(`seed_contribution`), gated stage-at-decision = HOLDING, fresh non-null, NO carry-forward.

**Reconstruction (two tiers — REPORT WHICH):**
- **VALID tier:** full GAE if per-step `value` rows exist (V(s_t), V(s_{t+1}), reward-after-all-shaping, γ, GAE λ,
  terminal-vs-truncation masks, rollout+batch boundaries) — reconstruction must pass a **PARITY CHECK**: re-derived
  standardized advantages reproduce the training-time mean≈0 / std≈1 (or a logged aggregate). Parity → VALID.
- **PROXY tier:** if per-step value rows are absent (Tamiyo decisions are sparse), fall to 1-step TD
  `δ = r + γV(s′) − V(s)` at decision rows — **sign-only**, and it **CANNOT alone kill the experiment** (gpt-prime).
  Report the tier explicitly next to every conclusion.

**Primary statistic:** per LOO bin, `E[A | FOSSILIZE]` vs `E[A | SET_ALPHA_TARGET]` vs `E[A | PRUNE]`, and the slope of
`A_FOSSILIZE` (and of the differential `A_FOSSILIZE − A_SET_ALPHA`) vs LOO. Stratify by seed_age / occupancy / host-acc
trend / round. Per seed (n=2); do not pool blindly.

**Pre-registered decision rule (fixed before the number):**
| Outcome | Reading | Action |
|---|---|---|
| **GRADED** — `A_FOSS − A_SETALPHA` rises with LOO (positive slope, survives stratification) | the learning signal exists; the dead-zone is why the policy can't act on it | **PDR-0069 confirmed → build the op-isolated ρ-sweep fix** |
| **FLAT** — no slope | the immediate reward gradient is cancelled by baseline/continuation value; the floor fix alone will NOT restore LOO-selectivity | **reframe collapses → reward/critic line reopens (N2 = prime suspect)** |
| **INVERTED** — `A_FOSS` falls with LOO, or `A_FOSS < A_SETALPHA` at high LOO | committing high-LOO seeds is *bad for (critic-estimated) return*; the policy's low fossilization may be CORRECT | **the epic's founding premise ("under-fossilization is a defect") is itself the over-read → escalate a premise re-examination** |

**Validity caveat (banked now):** the outcome variable is the CRITIC's return, not ground truth. This read tells you
**what the fix would TEACH the policy**, not what is true. A mis-fit critic (the EV-stab epic's own concern) propagates
into `A`. So GRADED confirms "the signal the policy would learn from exists"; it does not prove fossilizing is optimal.
A PROXY-tier result is directional only.

## Read B — the KL trust-region gate (§3, P8)

**Quantity:** fraction of PPO updates where KL early-stop fired (`early_stop` / `early_stop_epoch`, `target_kl=0.015`),
and per-head approx-KL (op / slot / blueprint / others), per seed.

**Pre-registered decision rule:**
- **P8 < ~5% (early-stop rarely fires):** the trust region has been effectively vacuous → a **competing,
  reward-independent root cause for the Dec-2025 WAIT-collapse**, and the λ=0 necessity smoke is confounded. **The
  KL-aggregation fix (per-head or gradient-bearing-sample-only KL for early-stop) must land BEFORE any GPU arm** — a
  harness fix, and (core-training-loop change) an OWNER-GATED escalation.
- **P8 materially non-zero (early-stop fires normally):** the §3 bomb is defused; the aggregate KL is diluted but still
  binds. Note the magnitude; proceed.
- **Per-head KL:** expect op/slot/blueprint ≈ 0 (structural), learning heads materially higher — a diluted aggregate is
  confirmed if the learning-head KL is ≫ the aggregate.

## What these reads do NOT decide
Neither read authorizes a GPU run or a primitive change (owner-gated). Read A decides *whether the reframe's premise
holds*; Read B decides *the experiment's sequencing*. Both are inputs to the owner DECIDE, not the decision.
