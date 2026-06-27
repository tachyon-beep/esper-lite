# Roadmap — Esper            Updated: 2026-06-28 (PDR-0001)

> Sequencing, WSJF / cost-of-delay, and dated forecasts are produced by
> /axiom-program-management. This file records bets as INTENT, not a delivery
> schedule. Do not compute WSJF here; hand the committed bet over for sequencing.

## Now  (committed, in-flight)
- **Reward credit-assignment redesign — Phase 0 (instrument-first)** — the controller's
  *committed* structure carries ~0 committed-per-param counterfactual value (J) on the
  control arm: the reward credits survival/ensemble residency, not committed marginal
  value. Phase 0 makes J a computable post-hoc yardstick and runs the GATE −1 cheap-fix
  falsifier sweep to decide rescale-vs-redesign. Binding constraint = signal/credit, not
  optimizer. · tracker: esper-lite-3d67b09687 (Stage-0 instrument), esper-lite-a221da47ea
  (Phase −1 falsifier) · metric: committed-J / corr(reward,J)

## Next (shaped, decreasing certainty)
- **Credit-assignment redesign itself** — a two-term targeted credit (enabling/synergy
  term + survival-without-contribution penalty), *gated on the (a)/(b) fork resolution*
  (freeloader defect vs LOO-undervalued enabling stem). Hypothesis, not yet committed.
  · metric: corr(reward,J), committed-J
- **Placebo noise-floor harness** — near-inert small-real seed to certify J above noise
  (GATE-0's last open leg). · tracker: esper-lite-3d67b09687
- **EV-stabilization** — joint value-target variance reduction for recurrent
  factored-action PPO. · tracker: esper-lite-f25b71c165 (epic)
- **PPO learning-gate / reward-efficiency statistics** — paired multi-seed lockstep ROI
  verdicts. · tracker: esper-lite-a2abff5ec5

## Later (directional bets, no order, no dates)
- **Three-phase curriculum scale-up** — University (CIFAR/TinyStories) → Internship
  (real domains) → Masterpiece (1B+ runs); the controller as a persistent, transferable
  "architect".
- **Planned subsystems** — Emrakul (decay/maintenance policy), Narset (slow-timescale
  allocator), Esika (host superstructure at scale). *(designed, not yet first-class.)*
