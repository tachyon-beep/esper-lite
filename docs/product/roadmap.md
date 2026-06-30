# Roadmap — Esper            Updated: 2026-06-30 (PDR-0002, PDR-0003, PDR-0005)

> Sequencing, WSJF / cost-of-delay, and dated forecasts are produced by
> /axiom-program-management. This file records bets as INTENT, not a delivery
> schedule. Do not compute WSJF here; hand the committed bet over for sequencing.

## Now  (committed, in-flight)
- **Reward credit-assignment redesign — causal resolution of the (a)/(b) fork.** The
  controller's *committed* structure carries ~0 committed-per-param counterfactual value (J):
  the reward credits survival/ensemble residency, not committed marginal value. The
  **cheap-rescale escape hatch is CLOSED** (GATE −1 falsifier verdict NO STOP / SURVIVE,
  PDR-0002 — rescale does not fix the defect). Active work is the **owner-gated
  causal-contribution run** (PDR-0003): is the early-conv r0c0 cohort a freeloader (a) or an
  LOO-undervalued enabling stem (b)? — opposite reward fixes. **Immediate prerequisite
  (PDR-0005): fix the pervasive policy entropy collapse** — general + pre-existing (also in
  GATE-1), it blocks any healthy-policy causal read; the morphogenesis is confirmed REAL
  (+7.69pp / n=6), so this is a fixable stability issue, not a reason to abandon. Binding
  constraint = signal/credit, not optimizer. · tracker: esper-lite-3d67b09687 (Stage-0
  instrument) · metric: policy-entropy guardrail (restore) → then committed-J / corr(reward,J)

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
