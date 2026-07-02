# GATE 2 — learnability probe: design + pre-registration

**Date:** 2026-07-02 · **Scope:** GATE 2 of the reward-credit term (PDR-0010, esper-lite-254175df90).
**Status:** PRE-REGISTERED (design + threshold *forms* locked before any Tier-2 run; σ_A-relative constants lock
after Tier 0b reports σ_A, before Tier 2 launches). Probe: `scripts/gate2_learnability_probe.py`.
**Design consult:** drl-expert (2026-07-02), full document in the session log; this file is the durable record.

## Question under test

Can a once-per-episode sparse credit attributed to a FOSSILIZE decision propagate into this recurrent PPO's
policy-gradient signal strongly enough to move the policy's fossilize propensity? PDR-0008's "optimizer adequate"
was earned in the dense per-step regime and does NOT transfer here. If the credit cannot propagate, candidate A
(Committed-Shapley top-up) is dead regardless of how correctly φ is computed → salvage is candidate B
(temporal/hindsight credit).

## Code facts established 2026-07-02 (all verified in-tree)

1. **The design doc's routing claim is WRONG as written.** The "deferred scaffold-credit rail"
   (`pending_hindsight_credit`, `action_execution.py:1082–1085`) flushes into the NEXT decision step's reward and
   is zeroed on episode reset (`parallel_env_state.py:258`). A terminal-computed credit routed through it would be
   **silently dropped**. Viable routings: **(i) terminal-flush** — add to the last step's reward; **(ii)
   retro-write** — write into `buffer.rewards[env, t_f]` post-episode, pre-GAE (precedented by
   `mark_terminal_with_penalty`, `rollout_buffer.py:876`; GAE runs inside `agent.update()` after episodes complete).
2. **Owed verification CLOSED:** `fossilize_contribution_scale` multiplies the per-seed LOO
   (`seed_contribution = val_acc − baseline_accs[slot]`, `action_execution.py:833`) — the synergy-blind ~0 signal,
   exactly as the design premise assumed. `SEED_FOSSILIZED.counterfactual` (~10 for r0c0) is the *joint*
   `counterfactual_total_improvement`, used only by gates. No bug; premise intact.
3. GAE is linear in rewards: a terminal delta δ moves A(t) by exactly δ·(γλ)^(T−t) with V fixed.
4. **REGIME CORRECTION (2026-07-02, after first validation run):** the n=5 J-read runs were driven by
   `scripts/causal_contribution_r1_pilot.py` with `configs/config-3slot-3seed-suppress-slot-r0c0.json`, which sets
   **gae_lambda=0.95** (NOT the leyline default 0.98), **auto_forward_g1/g2/g3=true** (auto-forward IS the regime
   for this experiment line), entropy anneal 0.15→0.08 over 3000 episodes (effectively ≈0.15 across a 200-episode
   run) with per-head multipliers (op ×3.0), value_coef=0.25. The probe loads this SAME config
   (`--config`, default = that JSON) and overrides only rounds/seed/device/telemetry — regime fidelity by
   construction. γλ = **0.94525**; credit half-life ≈ 12.3 steps.
5. **Telemetry is LOAD-BEARING for the lifecycle (found via lifecycle-stall debugging):** seed gradient stats are
   collected only when telemetry is on (`vectorized_trainer.py:1818`: `collect_gradients = use_telemetry and
   stride`), and the G2 blending gate HARD-FAILS on unmeasured gradient health (KTS-001, `slot.py:_check_g2`).
   A probe run with `use_telemetry=False` therefore silently blocks ALL blending/fossilization — zero
   fossilize-legal steps in 48 episodes vs blending from episode 2 in the reference. The probe pins
   `use_telemetry=True` (as the pilot ran). Also pinned `amp=False/amp_dtype="off"`: the config JSON says
   `amp: true, amp_dtype: auto` but the n=5 RECORDED `amp_enabled: false`; on a bf16-capable local GPU "auto"
   would enable bf16 host training (a quarantined artifact source,
   docs/analysis/2026-06-15-bf16-artifact-quarantine.md) — pinned to the recorded regime. (AMP was empirically
   NOT the stall cause — validate4/5 behaved identically — but the pin keeps the regime faithful.)

## Tier 0a — measured on the logged n=5 control runs (2026-07-02, DONE)

| Quantity | Value |
|---|---|
| Episode length | 150 decision steps (1/host-epoch); 12 envs; 1 PPO update / 12-episode batch (1800 steps) |
| Fossilize rate | 0.207/ep ≈ 2.4 events per update batch |
| r0c0 fossilize step t_f (n=2018) | q10=30, median=74, q90=133 |
| Terminal→t_f GAE factor (γλ)^(150−t_f) at the REGIME γλ=0.94525 | q10=0.0012, **median=0.0139**, q90=0.384; **76.3% of events < 0.1, 46.3% < 0.01**. (An earlier read used the leyline default λ=0.98 → median 0.147; superseded — the run config pins λ=0.95.) Terminal-flush is analytically moribund before Tier 1; the Tier-1 screen formalizes the kill. |
| **Free-vs-forced at fossilize** | **2,475/2,475 free (100%)** — op_entropy median 0.875, all > 0.1. The credit-to-action premise is WELL-POSED. |
| P(FOSSILIZE) when chosen | median 0.150 — ample headroom for a credit to move it |
| Raw per-step reward std (normalizer scale) | ≈ 1.86 (seed 41, n=360k steps) → clip ±10 ⇒ raw-δ clip saturation ≈ 18.6; realistic δ far below |

## The four-link causal chain (why "moves the advantage" is insufficient)

- **L1 reward→advantage:** terminal-flush decays by (γλ)^(150−t_f) (median 0.0139 at the regime γλ=0.94525) and
  divides by the running reward std; retro-write delivers δ full-strength at A(t_f). *Analytic; Tier 1 confirms
  through the production pipeline.*
- **L2 advantage SNR:** the systematic bump competes with σ_A over only ~2.4 events/update; per-update noise
  σ_batch ≈ σ_A/√2.4 ≈ 0.65σ_A. *Tier 0b measures σ_A.*
- **L3 gradient→logit:** shared LSTM trunk with 1,798 other timesteps; entropy bonus pushback; clipping. *Tier 2.*
- **L4 persistence:** the critic learns to predict the credit and erases the advantage; is the transient long
  enough to move the policy first? *Tier 2, tertiary metric.*

## Probe structure (three tiers, hard stop at first analytic FAIL)

- **Tier 0b (in-harness, every batch):** σ_A (pre-norm advantage std over valid steps) + advantage values at
  fossilize steps, from live buffers through the production pipeline. Locks the σ_A-relative constants.
- **Tier 1 (in-harness, every batch, both arms):** paired GAE recompute — inject δ, recompute advantages,
  record realized Δadv(t_f)/σ_A for BOTH routings; GAE-linearity self-check asserts the exact identities.
  - **G2a (terminal-flush) gate:** CLEAR if realized Δadv(t_f)/σ_A ≥ 0.10 at δ*; **FAIL if < 0.05** → no Tier-2
    budget for terminal-flush.
- **Tier 2 (paired short training, retro-write only):** same-seed paired arms (control vs retro-write injection),
  N ≥ 5 seed-pairs, K = 25–40 updates (interim read at 15; minimal-credible fallback K=20).
  - **Primary:** paired Δ P(op=FOSSILIZE | fossilize-legal) — eligibility-conditioned, not the marginal rate.
  - **Secondary:** paired fossilize-logit shift (entropy-independent).
  - **Tertiary:** trajectory of mean Â(t_f) in the injection arm (L4 transient: rise-then-collapse is a specific,
    informative FAIL that routes to candidate B).

## Pre-registered verdict rules

- **PASS:** paired ΔP(fossilize|eligible) > 0 at final K, one-sided paired p < 0.05 across N≥5 pairs, AND the
  effect persists at final K, AND magnitude ≥ the control arm's own natural logit drift over 5 updates
  (self-calibrating floor).
- **FAIL:** p > 0.2 at K, OR clear rise-then-collapse to ~0 (L4 wins). Both routings failing ⇒ GATE 2 FAIL →
  candidate B salvage. **A FAIL at L2/L3 kills candidate B too** (it shares the delivery point) — that would be
  the structural finding "this PPO cannot learn from ~2.4 sparse credited events/update regardless of routing."
- **MARGINAL:** 0.05 < p < 0.2, or a borderline transient → do NOT pass candidate A; escalate to a persistence
  sub-probe (longer K or critic-frozen ablation).
- **Routing verdict structure:** G2a (terminal-flush) and G2b (retro-write) do not share a pass. Terminal-flush
  failing + retro-write passing = GATE 2 PASS **with retro-write as a mandated build requirement** (retro-write is
  a semantically different term: bypasses the reward normalizer — must use `divide_by_std`, no clip, no stat
  update — lands at t_f, and changes the critic's prediction problem).
- **δ\* discipline:** δ* is pinned to the realistic top-up magnitude (order of the existing fossilize-bonus scale;
  default 2.0 raw ≈ 1.1σ_r), NOT tuned for effect size. GAE linearity makes the per-unit-δ effect exact, so Tier-1
  results rescale analytically. Report clip-saturation fraction.

## Probe guardrails (from the design consult)

Same-seed pairing mandatory, all pairs reported; assert `std_floored == False` every update (advantage std floor
0.1 must not bite); report the pre-entropy Tier-1 signal AND the Tier-2 behavioral endpoint separately (a real
signal hidden only by entropy is a design-tunable fail, not a learnability fail); rollback-contaminated envs
excluded from injection and measurement; NO isolation of the fossilize gradient (the shared-trunk swamping is the
question, not an artifact); the probe loads the n=5 regime config verbatim (auto-forward, entropy anneal +
per-head multipliers, λ=0.95) so Tier-2 verdicts generalize to the regime the reward term would ship into.

## Strategic note (expert finding, held for the design record)

Candidate B is a priori superior on L1 — retro-write **is** B's delivery mechanism bolted onto A's Shapley
valuation. WHERE the credit lands (routing) and WHAT its value is (Shapley vs eligibility weighting) are
orthogonal and composable. The single Tier-2 retro-write test therefore transfers to both candidates; the gate
money is spent once.
