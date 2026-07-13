# PDR-0077 — Round-13 review: L2 (A[ewma]) BLOCKED as a farm; L1 lands with a representation correction; a cheaper causal read replaces L3-first

Date: 2026-07-14   Status: accepted (both primes converged; ratifying their verdict, not a novel call)
Related: `docs/analysis/2026-07-14-permanence-visibility-primes-review-pack.md` (round-13 outcome banner), PDR-0074/0076, the drl deliverable. Corrects the drl deliverable's "A[ewma] viable" grade and this session's pack §5.

## Context
The primes reviewed the "make permanence visible" pack. They converged: the pack over-graded L2 A[ewma] "viable" — it passed a STATIC replay (old-policy commit times), not an adversarial POLICY test.

## What does this buy?  (REQUIRED — PDR-0068)
It stops us landing a reward change that would replace a "permanence = 0" lie with a "permanence = whatever the policy can time" farm — the same error class, inverted. Measured: the go/no-go on L2 is now an oracle-timing farm test + a proxy-free product-outcome read, not a static-replay mean.

## The verdict (ratified)
- **L2 A[ewma] = BLOCKED-candidate, NOT viable.** Median net 0 / mean net +5.8 ⇒ the expected value is a right tail a policy gradient selects; the static replay scored the OLD policy's timing. Cost 0.12 vs revenue +5.8 ≈ **48:1 farm**. Unblock gates (all zero-GPU): (1) **oracle-timing replay** (settle at EVERY eligible commit time, out-of-sample seed-41-tune/seed-42-test then reverse); (2) **commitment-hazard null** (`P(spike|FOSSILIZE)` vs `P(spike|all eligible)` — the 45–47% has no meaning without the opportunity-set baseline); (3) the RCT-CATE read below. Redesign toward **removing the policy's control of the settlement instant** — fixed-time/terminal settlement, pre-action lagged quote (`EWMA(c_{≤t−1})`), or a non-cancellable REQUEST_FOSSILIZE confirmation window. Also close the four L2 spec holes (annuity dimensional semantics; NO zero-fallback — FOSSILIZE ineligible without a valid quote, fail-closed; signed-value parity with the provisional stream; exact one-shot-bonus rule so spot LOO can't re-enter).
- **L1 = land the representation, HOLD freeze-forever.** Land the canonical `ContributionState` + an EXPLICIT observation status/mask reaching the policy (a `−1` sentinel alone is not self-describing — a true negative clips to −1). Correct invariant 2: a value frozen at full magnitude with freshness→0 over ~70 epochs is the mirror lie (H7's rising pre-commit slope ⇒ the frozen anchor is biased HIGH); **shrink the value dim toward UNKNOWN as staleness rises**, not toward 0 (original bug) and not frozen-forever (mirror bug). Obs schema v3→v4 ⇒ re-warm/retrain (not a hot patch). **Do NOT run L1 alone** (repairs the observation while the reward still forfeits — the critic gets a truthful state of a still-broken objective).
- **De-shape:** SOFTEN "not survivable" → removes the dominant dense channel (not a drop-in remedy); reward-mass ≠ gradient signal; retained as a diagnostic control. H8's real headline: the host-task signal was never trainable — a finding about the OBJECTIVE.
- **Cheaper causal read before full L3:** the existing pinned-simplex |H|=1 RCT tests whether the pre-commit EWMA quote predicts PROXY-FREE product outcomes (val-acc@{1,5,10,25}, terminal acc, AUC, compute-to-target, destructive events) via `β3 = FOSSILIZE×q_t`, episode-clustered — zero GPU. Plus a **terminal host-level objective** (final acc vs no-morphogenesis control, one scalar/episode) as the anchor to validate the 95% proxy. Full L3 branching likely still needed for shipping; not first.
- **Critic:** LEADING explanation, not proven; re-test calibration after L1/L2. Do not bank "perfectly fit."
- **ESCROW guard:** kept. Hardening (future): replace the `"configured"` string assertion with a concrete validated settlement-strategy before enabling ESCROW in a real fossilize-capable config.
- **K>1:** P0, independent, five rounds open — do it; every run is contaminated until fixed.

## Reversal trigger
- If the oracle-timing replay shows A[ewma]'s max-over-eligible-times farm gain is immaterial out-of-sample AND the hazard-null shows no spike-selection above baseline → A[ewma] un-blocks as an experimental arm.
- If the RCT-CATE `β3` is ≤0 → the pre-commit EWMA quote is not a valid retained-value proxy → L2-on-EWMA is mispriced regardless of the farm test; escalate to L3 / terminal-objective anchoring first.
- If the terminal host-level objective correlates STRONGLY with the shaped per-episode return → the 95% proxy is fine and the "objective needs an anchor" concern (claude-prime §4 / PR14) is wrong.
