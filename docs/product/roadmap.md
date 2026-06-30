# Roadmap — Esper            Updated: 2026-06-30 (PDR-0002, PDR-0003, PDR-0005)

> Sequencing, WSJF / cost-of-delay, and dated forecasts are produced by
> /axiom-program-management. This file records bets as INTENT, not a delivery
> schedule. Do not compute WSJF here; hand the committed bet over for sequencing.

## Now  (committed, in-flight)
- **Reward credit-assignment redesign — causal resolution of the (a)/(b) fork, measured on J.**
  The cheap-rescale escape hatch is CLOSED (GATE −1 SURVIVE, PDR-0002); the estimand-invariant
  harness is built + GPU-validated + committed (PDR-0003). The "entropy collapse" that looked
  like the blocker was a **MEASUREMENT ARTIFACT** (PDR-0006) — the policy is healthy, no training
  fix needed. The fork's discriminator is **J / acc-per-param, NOT accuracy** (accuracy ties):
  the n=3 pilot leans **(b) r0c0 is an efficiency-enabling stem** (suppressing it ~halves system
  efficiency — the opposite of a freeloader). Active work: the **n=5 J-read** (PDR-0007) — Phase 0
  (analyzer + pre-registered run sheet) DONE; Phase 1 (seeds 44–45, ~13h GPU) awaits owner go.
  · tracker: esper-lite-8190ff1c95 (n=5 J-read), esper-lite-425dcc4ca2 (telemetry hygiene)
  · metric: paired ΔJ/acc-per-param (seed-level), n=5 → n=10

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
