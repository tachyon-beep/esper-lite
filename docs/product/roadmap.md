# Roadmap — Esper            Updated: 2026-07-14 (PDR-0074 — commitment defect = fossil MEASUREMENT gap; PDR-0075 refocus PROPOSED, owner-gated, band unchanged pending sign-off)

> Sequencing, WSJF / cost-of-delay, and dated forecasts are produced by
> /axiom-program-management. This file records bets as INTENT, not a delivery
> schedule. Do not compute WSJF here; hand the committed bet over for sequencing.

## Now  (committed, in-flight)
- **Commitment root-cause CANDIDATE → exploration-primitive fix (REFRAMED 2026-07-13, PDR-0069/0070;
  CALIBRATED by the round-8 eval `docs/analysis/2026-07-13-round8-findings-evaluation.md`).**
  The commitment defect the EV epic exists to fix is traced to the **anti-WAIT floor gradient
  dead-zone**: `_apply_floor_to_logits` sits in the differentiable PPO path across all 8 heads, and
  a floor-bound action gets ZERO gradient to its OWN logit — so ~94% of FOSSILIZE / 93-95% of PRUNE
  samples deliver no credit to their own action (both seeds; MECHANISM established, autograd-forced,
  dual-expert numeric-confirmed). The commit reward is strongly LOO-graded (−0.7→+8.2) but that is
  IMMEDIATE reward — the ADVANTAGE-vs-LOO gate is UNMEASURED and directionally ambiguous (round-8),
  NOT passed. As a **REPRIORITIZATION** (you cannot move a zero-gradient logit with a critic/reward arm)
  this demotes the critic (PDR-0066) and cf-shaping/SNR (PDR-0067/0068) lines to secondary; it does NOT
  prove causal SUPERSESSION of the value-target-variance premise (fix unrun; the |H|≥2 minority + trunk
  still train; "SET_ALPHA is the new WAIT" not ruled out — only the λ=0 smoke disambiguates).
  **The dead-zone is WHOLE-HEAD (PDR-0071):
  in 67-80% of HOLDING decisions only one op is above the floor → the output vector is CONSTANT →
  zero gradient to ALL op logits; SET_ALPHA's 0.55 is the analytic cap, not a learned preference.**
  **New lever = a differentiable floor (NOT straight-through) — ONE primitive, λ-sweep {0,0.3,0.6}
  (λ=0 = no-floor necessity smoke; PDR-0071, proposed — owner-gated: core-primitive code change + GPU
  + drl-expert review).** Harness enablers (default-on checkpointing + per-decision advantage/pre-
  floor/per-head-KL logging) land first — those quantities were NOT captured (PDR-0026 recurrence).
  **CO-REQUISITE (PDR-0071): the floor fix is SCOREABLE but NOT SHIPPABLE until the transfer
  settlement (`hindsight_credit`, inert 0.005%) carries signal** — else unfreezing the gradient yields
  LOO-greedy myopic commit collapse; the dead-zone may be an accidental safety property (but the gate
  must be stated in ADVANTAGE terms, and `hindsight_credit`=0.005% root-caused first — it is keyed on the
  non-causal `total_improvement`; round-8). · metric: commit-logit LOO-sensitivity + a LIVE
  advantage-vs-LOO relation (NOT floor-bound-rate→0, a tautology under any mixture); per-head KL activity
  where the head was frozen; WAIT share on multi-choice states; guardrail = terminal accuracy
  contribution + destructive-intervention rate.
  **Round-8 additions (verified; full detail in the eval doc):** the same transform is ALSO a wide-head
  EXPRESSIVENESS clamp (N1 — the 0.12 floor caps preference and survives any gradient fix; telemetry
  REVISED severity DOWN: realized num_valid=7, ceiling ~2× uniform, realized 1.3-1.4× — a mild ceiling,
  not a near-uniform sampler), an `approx_kl`
  dilution source (N2), a monitoring blind-spot (N3 — entropy/collapse health guaranteed green where the
  head is deadest → a TIP scar), and an advantage-standardization contaminant (N4). The λ-sweep must use
  **per-num_valid λ** + a **hard-floor control arm** (N5), and a high-value zero-GPU candidate is an
  OFFLINE GAE advantage reconstruction from the existing `value_estimate` telemetry — IF per-decision
  reward + episode boundaries are also logged (value_estimate confirmed present; the rest UNVERIFIED),
  it would close the gate without GPU.
- **EV-stabilization — joint value-target variance reduction for recurrent
  factored-action PPO** *(commitment-premise SUPERSEDED by PDR-0069; retained for history + the
  still-open cf-earns-keep/SNR sub-questions, now secondary).* Moved Next → Now 2026-07-05 (PDR-0027) after the owner parked
  the Committed-Shapley term. Rationale: advantage noise is the live suspect for
  suppressed commitment; fixing it may organically widen the k≥2 co-fossilization
  channel the A/B found too narrow (1.1% of episodes). **Stage-0 DELIVERED + gate PASSED
  (PDR-0032): `Cov(R_cf,R)/Var(R)` median ~1.02 >> 0.40 on a Stage-2-OFF control run — the
  cf stream dominates value-target variance, so de-shaping is justified. Stage-2 MAJOR-1
  acceptance harness is **built, review-remediated, and tracker-closed in `3669934d`
  (PDR-0044)**. Escrow telescoping correctness is fixed in `89b50a16` (PDR-0047):
  reward-path escrow delta clipping is gone, and Obs V3 schema v2 exposes the stable
  accuracy + escrow-credit state needed for the critic to model the potential. **Gate
  formulas FROZEN + OFF wave LAUNCHED 2026-07-08 (PDR-0048); OFF wave COMPLETE 5/5
  valid + step-3 scalars FROZEN 2026-07-09 (PDR-0050: W=17, δ=0.050311,
  Δparam_max=1527.8; LEG-B demoted to descriptive after both pre-registered
  escalations fired — screen predicate = LEG-A ∧ MECH ∧ G1–G4; device placement ruled
  provenance, harness amended). First ON wave DEGRADED
  (telemetry writer starved by co-tenant review workload — operational, not code;
  PDR-0056) → quarantined; full 5-seed RERUN completed CLEAN 2026-07-11 (5× rc=0,
  zero drops — PDR-0056 reversal trigger never fired). First scoring INVALID on a
  checker-population artifact → owner-ruled Option B validity-envelope correction
  (§11.3, PDR-0058) → rescore of the identical frozen spec: **Stage-2 MAJOR-1 n=5
  screen verdict = REJECT (PDR-0059)** — LEG-A FAIL (median Δ_A ≈ −0.093 < −δ;
  4/5 seeds regress) ∧ MECH fail ∧ G4 fail; G1/G2/G3 pass; LEG-B descriptive
  INCONCLUSIVE (confound downgrade). NO n=10 for HRA objective-A as implemented;
  Stage-0 de-shaping diagnosis untouched. **Penalty-schedule fix + vg Gram
  telemetry LANDED `7ea84dc6` 2026-07-11 (drl APPROVE-WITH-NITS, nits closed,
  PDR-0061); clean 600-round seed-41 OFF diagnostic IN FLIGHT since 2026-07-11
  19:29 on the fixed code (read window rounds 250–600, pre-committed readings
  PDR-0055).** Entropy-floor population audit remains front-of-TIP (PDR-0057).
  EPIC DIRECTION after the diagnostic read = owner DECIDE (TIP / reward-efficiency /
  redesigned HRA with cf-target stabilization / Stage-1 flag).**
  · tracker: esper-lite-f25b71c165 (epic) · metric: EV Stage-0 gate (PASSED);
  Stage-2 MAJOR-1 gate READ 2026-07-11 = REJECT; watch fossilize/ep + k≥2
  frequency as the organic-coverage signal

## Parked (banked, not dead)
- **Committed-Shapley top-up (reward credit-assignment redesign)** — PARKED 2026-07-05
  (owner call, PDR-0027) after the n=5 A/B scored **null-not-informative**
  (coverage-bound; PDR-0026). The instrument is BANKED as validated: criteria (i)/(ii)
  pass, zero deadband violations, guards clear, no farming signature, design priors hit
  within a point (P(pay|k≥2) 44.7% vs 45.4%). `shapley_synergy_scale` stays 0.0 and
  owner-gated. Re-run licensing + τ-placebo preconditions live in PDR-0026/0027
  reversal triggers. · tracker: esper-lite-f22a1d48a7 (closed) · full history:
  PDR-0009 → 0013 → 0018 → 0021 → 0023 → 0025 → 0026

## Next (shaped, decreasing certainty)
- **PPO learning-gate / reward-efficiency statistics** — paired multi-seed lockstep ROI
  verdicts. · tracker: esper-lite-a2abff5ec5
- **Telemetry Improvement Program (TIP)** — make experimental reads self-defending:
  manifest-driven scoring, machine-readable metric registry (population/denominator/
  units/validity-envelope), reusable ValidityReport before interpretation,
  historical-scar regression suite, mechanism-vs-outcome monitor discipline; later
  phases = seed_uid lifecycle identity and per-site host-perception telemetry with
  offline information-ceiling probes gating any Obs V4. Captured 2026-07-09 from an
  owner-drafted plan, adapted (PDR-0049). Positioned AFTER the Stage-2 read; Phase A is
  read-path-only and *technically* freeze-safe, but the owner ruled 2026-07-09: **do
  not start early — finish Stage-2, then pick the next direction** (serial focus). · tracker: esper-lite-c62d4891b0
  (epic) · concept: `docs/plans/concepts/2026-07-09-telemetry-improvement-program.md`
  · metric: every scored verdict passes manifest+validity mechanically; scar-suite
  green in CI; zero glob-scored reads
- **A/B re-run at organic coverage** — only if the EV track raises k≥2 frequency
  materially (PDR-0027 reversal trigger); a ≥10× coverage null would then be
  informative and count against the term.

## Later (directional bets, no order, no dates)
- **Three-phase curriculum scale-up** — University (CIFAR/TinyStories) → Internship
  (real domains) → Masterpiece (1B+ runs); the controller as a persistent, transferable
  "architect".
- **Planned subsystems** — Emrakul (decay/maintenance policy), Narset (slow-timescale
  allocator), Esika (host superstructure at scale). *(designed, not yet first-class.)*
