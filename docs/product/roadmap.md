# Roadmap — Esper            Updated: 2026-07-05 (PDR-0032)

> Sequencing, WSJF / cost-of-delay, and dated forecasts are produced by
> /axiom-program-management. This file records bets as INTENT, not a delivery
> schedule. Do not compute WSJF here; hand the committed bet over for sequencing.

## Now  (committed, in-flight)
- **EV-stabilization — joint value-target variance reduction for recurrent
  factored-action PPO.** Moved Next → Now 2026-07-05 (PDR-0027) after the owner parked
  the Committed-Shapley term. Rationale: advantage noise is the live suspect for
  suppressed commitment; fixing it may organically widen the k≥2 co-fossilization
  channel the A/B found too narrow (1.1% of episodes). **Stage-0 DELIVERED + gate PASSED
  (PDR-0032): `Cov(R_cf,R)/Var(R)` median ~1.02 >> 0.40 on a Stage-2-OFF control run — the
  cf stream dominates value-target variance, so de-shaping is justified. Next leg = Stage-2
  (HRA cf value head, already built on `feat/ev-stab-stage2-hra`) + its acceptance criteria
  (ev_sum floor / provenance).** · tracker: esper-lite-f25b71c165 (epic) · metric: EV
  Stage-0 gate (PASSED); watch fossilize/ep + k≥2 frequency as the organic-coverage signal

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
- **A/B re-run at organic coverage** — only if the EV track raises k≥2 frequency
  materially (PDR-0027 reversal trigger); a ≥10× coverage null would then be
  informative and count against the term.

## Later (directional bets, no order, no dates)
- **Three-phase curriculum scale-up** — University (CIFAR/TinyStories) → Internship
  (real domains) → Masterpiece (1B+ runs); the controller as a persistent, transferable
  "architect".
- **Planned subsystems** — Emrakul (decay/maintenance policy), Narset (slow-timescale
  allocator), Esika (host superstructure at scale). *(designed, not yet first-class.)*
