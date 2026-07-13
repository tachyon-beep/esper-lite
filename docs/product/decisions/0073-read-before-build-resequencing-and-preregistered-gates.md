# PDR-0073 — Read-before-build: two pre-registered zero-GPU reads gate the GPU spend; op-isolated ρ-sweep + KL-fix drafted but owner-gated

Date: 2026-07-14   Status: accepted (resequencing + pre-registration, within grant); the GPU arm, the primitive change, and the KL-aggregation fix are **owner-gated (proposed), flagged**
Supersedes: — (refines PDR-0071's "harness enablers → λ-sweep" ordering). Related: `docs/analysis/2026-07-14-advantage-loo-read-preregistration.md`, PDR-0072, PDR-0071, tracker esper-lite-7fe21bd091 (harness enablers), esper-lite-f25b71c165 (epic).

## Context
The round-8 evaluation (PDR-0072) established that the floor reframe's one unproven bridge is per-decision advantage-vs-LOO, and that two cheap reads sit upstream of every GPU dollar. Both primes and both experts converged (4/4) on **read before build**. Round-7's plan pointed straight at the λ-sweep; this inverts that: the highest-value work is now reading existing telemetry, not building a primitive.

## What does this buy, and how is that measured?  (REQUIRED — PDR-0068)
It avoids buying a GPU experiment on an unproven premise, and it avoids running a necessity smoke that would lie. Read A tests whether the learning signal the reframe assumes actually exists (and pre-registers an INVERTED branch that could collapse the epic premise for zero GPU). Read B tests whether the experiment even needs re-sequencing (a diluted KL trust region would confound the λ=0 arm). Measured: the pre-registered decision tables in the companion doc fire on data, not mood; the GPU go/no-go is made *after* these reads, with a sharper question.

## Options considered
- **Proceed to the λ-sweep now (PDR-0071 ordering).** Rejected: spends GPU before the premise is tested; the λ=0 smoke is confounded by the KL-dilution finding (N2/§3); the sweep is confounded by fixed-λ (N5).
- **Read before build: two pre-registered zero-GPU reads, then decide.** Chosen. The advantage read (§5) is a natural experiment already on disk (marginal ignorability from the flat op-mix); the KL read (§3) is a cheap sequencing gate.
- **Skip straight to instrumenting a fresh run.** Deferred: slower and more expensive than reading the existing longdiag telemetry first.

## The call
- **Read A (§5, the RCT):** reconstruct per-decision advantage from the existing `value_estimate` telemetry (VALID tier only if a parity check reproduces training-time mean≈0/std≈1; else PROXY/sign-only and cannot alone kill the experiment), read `E[A | op]` per LOO bin, classify GRADED / FLAT / INVERTED per the fixed rule. **Launched this session (in flight).**
- **Read B (§3, the KL gate):** measure P8 (early-stop fire rate) + per-head KL; P8 < ~5% → the KL-aggregation fix lands before any GPU arm. **Launched this session (in flight).**
- **Drafted, but OWNER-GATED (not enacted):** the op-isolated **ρ-sweep** primitive (change only the op head; ρ∈{0,0.5,1}; endpoint-matched per-num_valid; fresh hard-floor control arm) — a core RL-primitive change needing drl-expert review; and the **KL-aggregation fix** (per-head or gradient-bearing-sample-only KL for early-stop) — a core-training-loop change. Both are escalations, flagged for owner sign-off; neither is built here.

## Reversal trigger
- **Read A = INVERTED** → the epic's founding premise is the over-read → escalate a strategy re-examination (a vision-level question — owner-gated).
- **Read A = FLAT** → the floor fix alone won't restore LOO-selectivity → deprioritize the ρ-sweep, reopen the reward/critic line (N2 first).
- **Read A = GRADED (VALID)** and **Read B = P8 non-vacuous** → license the ρ-sweep design for owner GPU authorization.
- **Read B P8 < ~5%** → block the λ=0 necessity smoke until the KL-aggregation fix lands (a re-sequencing that itself needs owner sign-off).
- If a reconstruction parity check is unattainable (PROXY tier only), neither read may unilaterally kill the experiment; the first instrumented smoke run must log exact standardized advantage.

## Outcome — Read B (2026-07-14, same session; Read A still in flight)
Read B fired the "P8 < ~5%" branch (P8 = **0.000%**, both seeds), BUT the read corrected the pre-registered *mechanism*:
the trust region is vacuous because these runs are **K=1** (`recurrent_n_epochs=1` → epoch-0 diagnostic KL = KL(θ‖θ) = 0
by construction; ratio ≡ 1.0, so ratio-clip is also inert), NOT because of head-dilution. So the prescribed **KL-aggregation
fix is misdirected at K=1** (per-head KL is also 0), and N2's dilution claim is untestable on this run (real only at K≥2).
Corrected trigger: a vacuous trust region is confirmed as a competing, reward-independent structural fact; the levers are
**K≥2 epochs or post-step KL gating** (both owner-gated core-loop changes), and the λ=0 necessity smoke stays confounded.
Goes to the owner DECIDE alongside Read A.
