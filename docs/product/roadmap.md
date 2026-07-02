# Roadmap — Esper            Updated: 2026-07-02 (PDR-0013)

> Sequencing, WSJF / cost-of-delay, and dated forecasts are produced by
> /axiom-program-management. This file records bets as INTENT, not a delivery
> schedule. Do not compute WSJF here; hand the committed bet over for sequencing.

## Now  (committed, in-flight)
- **Reward credit-assignment redesign — design the credit term for the enabling stem.** The (a)/(b)
  fork is **RESOLVED**: the n=5 J-read banks **(b)** — r0c0 is an efficiency-enabling stem
  (suppressing it ~halves system param-efficiency, 5/5 seeds, PDR-0009), and the optimizer is
  adequate so the defect is reward-side (PDR-0008). ⇒ the reward must CREDIT the enabling
  contribution the per-step LOO undervalues. **The reward-credit term is BUILT (PDR-0013,
  2026-07-02, owner-signed): default-OFF Committed-Shapley top-up, retro-write delivery,
  specialist-reviewed + reviewer re-pass approved + adversarial code review clean;
  esper-lite-254175df90 CLOSED.** Flag stays shapley_synergy_scale=0.0 everywhere. **Next
  concrete step: the ENABLEMENT gate (esper-lite-f22a1d48a7, owner-gated)** — tau from the
  PIN-E placebo (esper-lite-3d67b09687), F2 scale/cap calibration, hard off-switch-J
  efficiency + fossilize-count gate, entrenchment monitor, ON-run dormancy recheck, then the
  paired ≥5-seed OFF/ON A/B. · tracker: esper-lite-f22a1d48a7 (enablement),
  esper-lite-425dcc4ca2 (telemetry hygiene) · metric: committed-J / corr(reward,J) in the A/B

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
