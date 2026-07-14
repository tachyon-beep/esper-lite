# PDR-0089 — Owner RATIFIES ΔP_req = 0.10 (the materiality constant); round-22 pins registered; pending-visibility fork flagged; threshold-variance bound shown data-conditional; Phase-2 remains owner-pending

Date: 2026-07-14   Status: accepted (owner ratification quoted + same-session clarification)
Follows PDR-0088. Inputs: claude-prime round-22 note + gpt-prime ratification document (owner-relayed) + the owner's ruling.

## The owner's ruling (verbatim, with correction)
> "lets ratify 10 for now, we can always drop it later if we need to - this is locally running so we've got compute"
> …clarified same session: "sorry, I meant using .10 not item 10"

**Ruling: the 0.10 materiality constant is RATIFIED — ΔP_req = 0.10** (power-calc §8 Q1; the number claude-prime's
sort identified as the one genuine preference in the queue: against a floor-forced commit rate of ~0.15, a 0.10
absolute request-probability shift over the interquartile quote range is a large, defensibly conservative bar).
"Drop it later" = the constant is revisable pre-freeze if evidence argues; after freeze it is frozen like everything
else. **Interpretation note (provenance):** the session initially read "10" as item #10 (Phase-2 authorization); the
owner corrected before anything was committed or enacted — no Phase-2 work was started. **Phase-2 B-path
authorization is BACK on the owner queue, un-ruled.**

## Decisions
1. **ΔP_req = 0.10 RATIFIED (owner).** Consequence: the moment the r9 `q_decision`-IQR read lands,
   `Δ_material = 0.10 / IQR_q` is a NUMBER, and the blinded-reassessment certification rule (PDR-0086 §5.4) plus the
   gate bounds become concrete. The 0.10 constant propagates as the WORKING value where gpt-prime's scheme reuses it
   (gate UCB materiality; threshold-variance ΔP bound) — one coherent family, carried as proposed-working-constants
   until the freeze signature (they inherit the ratified number, not independent rulings).
2. **The r9 calibration read DISPATCHES now** (esper-lite-e12e2d1543): its governing constant is ratified, its
   legitimacy needs a citation not a ruling (pre-reg §1.12's frozen text authorizes r9-only calibration reads —
   claude-prime round-22), and both primes agree it precedes the final adversarial pass (bounds-postured gates
   without Δ_material are vacuous-as-written). **Scope EXTENDED:** also measure within-window quote noise near the
   cliff (the σ the threshold-variance surface keys on) — see #5. Pre-registered read spec persisted BEFORE results
   are inspected (gpt item-5 discipline, adopted as good practice).
3. **Round-22 pins REGISTERED** (claude-prime; must not be lost between checkpoints):
   (a) the Phase-2 replay case asserts "exactly one **stage-entry mint**," NOT "no potential increase" — the
   legitimate pre-cap HOLDING climb during the window would false-fire the naive form;
   (b) **layer-scope fence:** the B≡A-at-boundary identity is PBRS-ONLY — false for the contribution layer (B pays
   provisional through the window; A pays nothing post-commit) and the warning layer;
   (c) pendingness-visibility carries an interpretation note (now a flagged fork, #4);
   (d) **sequencing: IQR read BEFORE the final adversarial pass.**
4. **FORK FLAGGED (owner, at or before freeze): pending-visibility.** gpt-prime requires pending status +
   time-to-boundary be visible to the policy/value function (hidden `committed=True` = state-aliasing defect); the
   frozen §2.5.8 pin says obs dim 31 OFF / no new obs dim (minimal-delta); claude-prime's round-22 pin treats
   invisibility as an interpretation note. These conflict; owner resolves. Note obs-v4's FROZEN_AT_FOSSILIZE flag
   machinery exists as a possible middle ground. The Phase-2 REWARD semantics are independent of this choice.
5. **Threshold-variance bound is DATA-CONDITIONAL, not blind (claude-prime's challenge answered with Phase-1
   numbers):** measured windfall ΔP(pay) vs σ→0 — raw EWMA gate: μ=0.9 → +0.019/+0.153/+0.249/+0.340 at
   σ=0.1/0.2/0.3/0.5; μ=0.95 → +0.153 already at σ=0.1. LCB-qualified (z=1): μ=0.9 → +0.002/+0.031/+0.066/+0.110.
   **Under ΔP<0.10, the raw gate fails for in-regime quote noise σ ≳ 0.15–0.2 near the cliff; the LCB variant holds
   to σ ≈ 0.3.** The extended r9 read supplies the in-regime σ; the gate choice is made from data at freeze.
6. **Execution order ADOPTED** (both primes converge): IQR+noise read → [owner rulings incl. Phase-2 authorization]
   → Phase-2 B replay (if authorized; acceptance contract = gpt's ten invariants incl. exactly-one-boundary-mint per
   pin 3a, layer fence per 3b) → Option-A lock-burden quantification → absolute-unit power/UCB finalization → final
   fresh-context adversarial pass → no-peek freeze (owner signs) → GPU decision.
7. **Owner queue after this ruling (unchanged items):** Φ(PENDING) dissolution; gates-as-bounds posture;
   price-don't-fix; UCB posture (point-estimate primary); **Phase-2 B-path authorization**; pending-visibility fork;
   premium signature; Option-A lock; threshold-variance gate choice (post-noise-read); max-n=10 confirmation.

## Reversal trigger
- If pre-freeze evidence shows 0.10 is mis-scaled against the realized request-probability dynamics (e.g., the
  floor-forced base rate shifts materially under K=4/obs-v4) → revisit ΔP_req BEFORE freeze; after freeze it is frozen.
- If the r9 noise read shows within-window σ < ~0.1 near the cliff → the raw EWMA gate is within the bound (LCB
  fallback NOT forced); if σ ≳ 0.2 → the LCB fallback fires by pre-registration.
- If the IQR read yields a degenerate IQR_q → PDR-0086's trigger governs (re-derive materiality in-regime; never a
  relative target).
