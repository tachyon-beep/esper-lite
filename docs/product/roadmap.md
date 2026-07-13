# Roadmap — Esper            Updated: 2026-07-14 (PDR-0076 — Now bet REFOCUSED to "make permanence visible", owner-approved; floor/critic/K=1 deferred-but-banked)

> Sequencing, WSJF / cost-of-delay, and dated forecasts are produced by
> /axiom-program-management. This file records bets as INTENT, not a delivery
> schedule. Do not compute WSJF here; hand the committed bet over for sequencing.

## Now  (committed, in-flight)
- **Make permanence VISIBLE — repair the contribution measurement-support discontinuity (REFOCUSED 2026-07-14,
  owner-approved; PDR-0074 diagnosis → PDR-0075/0076 refocus).** The commitment defect the epic exists to fix is a
  single permanent-value MEASUREMENT gap: `counterfactual_contribution` (LOO) is structurally undefined at α=0 birth AND
  at permanence (fossils excluded from the ablation by design, `vectorized_trainer.py:956-975`), and every consumer of
  the `None` inherits a silent `→0` lie — reward (`bounded_attribution` stops firing, realized −113 forfeit), observation
  (`features.py:830` fossil contribution feature = 0), settlement (`hindsight_credit` inert), and the critic (fit to the
  lying observation; its 4-6× overstatement is a SYMPTOM). A `None→0` coercion at a measurement boundary is a silent,
  learned lie: it teaches the agent the unmeasured thing is worthless, and the only reason the product produces any
  fossils is the floor forcing ~15% random commits. **Now bet:** make missingness EXPLICIT in observation + reward +
  settlement — **L1** freeze-don't-zero (carry last-valid LOO with lifecycle scoping + the built-but-bypassed freshness
  channel), **L2** boundary settlement priced to NEUTRALIZE the forfeited stream (≈ −113), not subsidize it — then test
  whether the last-valid pre-permanence LOO is an adequate proxy (**H7**), then re-ask whether commitment learning
  improves. · metric: does a genuinely-contributing seed retain its measured value across FOSSILIZE (no `→0`); guardrail
  = the fix does NOT farm (fossils don't cluster at LOO spikes; mean fossil quality at t+20 ≈ settlement-time LOO) +
  terminal accuracy contribution.
  **Discipline (PDR-0076 conditions): H7 GATES L1 (H7 measures the freshness γ — do not guess it); design against the
  fossil-farm adversary (sell-at-spike / decay-lie / double-count; revenue must ≈ the forfeited stream, cost ~0.12);
  OFFLINE-REPLAY on existing trajectories before any run; DESIGN-don't-land; floor + critic held CONSTANT; the premise
  "under-fossilization is a defect" is UNMEASURED, not wrong.** Two gates: training-legibility (does L1/L2 remove the
  `→0` discontinuity?) vs product-validity (is the last-valid LOO a valid durable-value proxy? — H7 decides
  legibility-adequacy, NOT causality). · tracker: epic esper-lite-f25b71c165; harness esper-lite-7fe21bd091.

## Deferred / blocked behind the instrument (real + banked, NOT the next bet)
- **Anti-WAIT floor dead-zone (PDR-0069/0071, round-8 eval):** autograd-confirmed op-head gradient censor (|H|=1
  whole-head constant, 63/72% of HOLDING) + mild wide-head expressiveness clamp (N1) + `approx_kl` dilution (N2) +
  monitoring blind-spot (N3, a TIP scar) + advantage-standardization contaminant (N4). CONTRAINDICATED as the next
  intervention and COUPLED to the reward (FOSSILIZE floor-forced 94% → fixing the gradient alone drives fossilization
  ~0). Deferred; packaged with the reward, never alone. The ρ-sweep design (per-num_valid λ + hard-floor control) is
  retained for later.
- **Critic:** RETIRED as an independent line (round-11/H6) — a symptom of the observation gap; PDR-0066 ("wrong lever")
  right for a new reason. Remeasure calibration AFTER L1/L2; a residual, if any, returns then.
- **K=1 trust-region (Read B, P8=0):** the configured target-KL early-stop is inoperative at K=1 — a competing,
  reward-independent structural fact. Fix K (→ K>1 or post-step KL gating) as P0 harness, independent of the epic.
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
