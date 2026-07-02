# Reward-credit design — Committed-Shapley synergy top-up (DRAFT)

**Status:** CANDIDATE · default-OFF · **GATE 1 PASSED 2026-07-02** · **GATE 2 (learnability) PASSED 2026-07-02 —
retro-write delivery MANDATED, terminal-flush FAILED** (result: `docs/analysis/2026-07-02-gate2-learnability-result.md`;
NOTE: the "deferred scaffold-credit rail" routing below is WRONG as written — pending_hindsight_credit flushes
next-step and zeroes at reset; build = retro-write into buffer.rewards[env, t_f] pre-GAE via divide_by_std) ·
**reward-function-reviewer: APPROVE_WITH_CHANGES 2026-07-02** (5 build-conditioning changes F1-F6 + enablement
criteria: `docs/analysis/2026-07-02-reward-credit-term-review.md`) · **(b) CONFIRMED (n=5, banked direction)** ·
remaining gate: **owner sign-off** (flag stays shapley_synergy_scale=0.0 throughout).
Advisor-reconciled 2026-07-01 (surfaced the terminal-vs-developmental blind spot); GATE-1 reconcile PENDING.
**Date:** 2026-07-01 · **Source:** ultracode design workflow wc8i117pi (candidate A + adversarial synthesis; candidates B/hindsight + C/efficiency-frontier stalled on infra — not compared, see §Alternatives) · advisor reconcile PENDING
**Decision record:** PDR-0007 (n=5 J-read), PDR-0006 (J-reframe), PDR-0004 (total-system estimand)

## The defect
The reward credits fossilization by a seed's IMMEDIATE per-seed LEAVE-ONE-OUT (LOO) contribution
(`fossilize_contribution_scale` × `seed_contribution`), and applies `fossilize_noncontributing_penalty` (−0.2) below
threshold. **LOO ignores synergy** (phase0 §1.2): an early-converging ENABLING STEM whose value is realized *through*
downstream neighbours scores LOO ≈ 0 or negative (the structure that formed *around* it compensates for its removal) →
it is penalized as a non-contributor. The n=3 causal pilot (n=5 in flight) shows suppressing r0c0 ~halves system
acc-per-param → it is an efficiency-enabling stem (b), not a freeloader (a). The reward must credit it **without**
re-opening the freeloader hole.

## ⚠ Two cheap pre-implementation gates (on the n=5 data — BEFORE building the Shapley apparatus)
Advisor reconcile (2026-07-01) surfaced a load-bearing risk that moves AHEAD of implementation.

**GATE 1 (TOP) — proxy-vs-estimand agreement.** φ(s) measures **terminal co-dependence** ("with r0c1/r0c2 already
formed + trained, does masking r0c0 NOW drop accuracy?"). But the (b)/enabling claim + the ratified SUPPRESS-SLOT
estimand are a **trajectory** claim ("r0c0's early commit ENABLED the efficient path to form at all"). These DIVERGE in
the motivating case: the archetypal enabling stem is **scaffolding** — developmentally essential, **terminally
removable** — for which terminal-φ ≈ 0 (the cured structure stands without it). So the terminal proxy would (a) FAIL to
credit the exact stem it is built for, and (b) actively teach "don't commit scaffolds, the credit goes to the survivor"
— suppressing the decision the +7.7pp-at-good-efficiency depends on. And terminal co-dependence can be HIGH for a
non-enabling entrenched seed (opportunity-cost) — simultaneously blind to real enabling and generous to fake.
**Check (uses the in-flight n=5 data, NO new runs):** per seed 41–45, correlate the **terminal co-dependence proxy** —
the CONTROL arm's LATE-run r0c0 leave-out accuracy drop (`all-committed-ON − r0c0-OFF` from the terminal
COUNTERFACTUAL_MATRIX; the full committed-coalition Shapley φ is a refinement of the same quantity) — against the
**trajectory signal we already trust**: the suppress arm's Δ(acc-per-param) for the same cohort. AGREEMENT → the proxy
is sound, build A. DIVERGENCE (terminal drop ≈ 0 where whole-run suppression shows a large effect = the scaffolding
signature) → the proxy is pointed at the WRONG quantity; no clamp/deadband hardening fixes that → pivot to candidate B
(temporal/hindsight credit), see §Alternatives. (Analyzer: extend `causal_contribution_j_analyze.py` with a terminal
r0c0-LOO column and the per-seed correlation.)

**RESULT (2026-07-02) — GATE 1 PASSES (scaffolding failure mode CLEARED).** Load-bearing finding: terminal r0c0-LOO drop
(control, late 2/3-slot matrices) is large and positive on ALL 5 seeds (14.0/7.6/10.8/9.0/9.8 pp, median 9.8) ⇒ r0c0 is
**terminally load-bearing, NOT the scaffolding archetype** (which would show ≈ 0). It is the ≫0 MAGNITUDE that clears
scaffolding. The proxy is *consistent with* the trajectory effect (Pearson r=+0.92) but that is 5 points sharing the
control arm — suggestive, NOT independent evidence; do not lean on it. ⇒ candidate A's terminal-Shapley proxy is
DEFENSIBLE for r0c0 (not pointed at the wrong quantity).

**MIRROR CHECK (2026-07-02) — over-credit surface not evident, but not fully cleared.** GATE 1 cleared UNDER-crediting
(scaffolding); the mirror tests OVER-crediting (an entrenched-worthless seed drawing large terminal-drop once it is a
training target). Per-slot terminal-drop is **SELECTIVE, not uniform** — it tracks slot value: the minor slot r0c2
(fossil cf ~1.4) shows a small drop (~2.6) while r0c0/r0c1 (cf ~10) show large drops (~8–14). So terminal-Shapley does
NOT indiscriminately credit every committed seed. **Limitation:** the causal config has no KNOWN entrenched-worthless
seed (3 clean slots) to stress the pathological "high-drop-yet-worthless" case; the rigorous test needs the §5.2
freeloader-tail run (seed-identity blocked) or a per-slot suppress read (not run). **Entrenchment/opportunity-cost
stays a MONITORED gate on A** (watch via the suppress harness when enabled), not eliminated.

**STATUS: A is the SURVIVING candidate** (scaffolding cleared; over-credit not evident; survives over B) — NOT yet "the
design." Remaining before build: **GATE 2 (learnability — UNRUN)** + owner + reward-function-reviewer sign-off; reward
stays default-OFF regardless. **Also verify before wiring:** `SEED_FOSSILIZED.counterfactual` reads ~10 for r0c0 (a
joint/ensemble measure, not the ~0 per-seed LOO) — confirm which signal `fossilize_contribution_scale` actually
multiplies, since the "r0c0 is undervalued" premise is about the per-seed LOO, not the joint cf.

**GATE 2 — learnability (first-class, not just success-criterion v).** A once-per-episode TERMINAL credit attributed
through the scaffold rail to a FOSSILIZE decision hundreds of steps earlier is the HARDEST signal for this recurrent PPO
to learn from (sparse terminal reward on a sparse long-past action). **Scope trap:** the "optimizer adequate / defect
reward-side" finding was established for the CURRENT reward's signal density — it does NOT transfer to a terminal-sparse
credit regime. **Check (cheap, before building the Shapley apparatus):** does a synthetic terminal top-up on a fossilize
action measurably move that action's advantage? If the credit cannot propagate, the mechanism is dead regardless of how
correctly φ is computed.

Only if BOTH gates pass does the mechanism below become "the design."

## The mechanism
A **default-OFF, terminal-only, additive** supplement paid once at `epoch == max_epochs`, on top of the untouched
per-step LOO channel and the untouched −0.2 gate. Over the k committed (FOSSILIZED) slots (k ≤ max_seeds = 3), run an
**exact 2^k committed-coalition Shapley** (full factorial — no permutation sampling → zero estimator variance) in the
already-fused terminal val pass:

```
phi(s)    = Σ_{S⊆C\{s}} w(|S|,k) · ( v(S∪{s}) − v(S) )   # forward alpha-mask leave-out over committed set C; v = fused-val acc; acc-POINTS
c_paid(s) = seed_contribution(s) at the FOSSILIZE step    # acc-point proxy for the dense LOO already paid (~0 for the undervalued stem)
gap(s)    = max(0, phi(s) − c_paid(s) − tau)              # tau = PIN-E placebo Shapley-std deadband
raw(s)    = min(shapley_synergy_cap, shapley_synergy_scale · gap(s))
G         = max(0, v(all committed ON) − v(all committed OFF))   # both endpoints are already masks in the 2^k family (free)
top_up(s) = raw(s) · ( min(1, G / Σ raw) if Σ raw > 0 else 0 )   # HARDENING: episode-level efficiency clamp
reward   += top_up(s)                                     # routed through the deferred scaffold-credit rail → attributes to the FOSSILIZE action
```

Why this form: **acc-point units, not per-param** — the reward already prices params via rent/occupancy_rent/
fossilized_rent, so a per-param credit would double-charge (J divides by params precisely because J has no rent term;
the reward does). `c_paid` subtraction makes it a SUPPLEMENT (no double-pay). Payment via the existing scaffold-credit
rail so credit attributes to the fossilize decision, not a free-floating terminal bonus.

## Config changes (all experiment flags, default-OFF, beside `shaped_attribution_clip`)
- `shapley_synergy_scale: float = 0.0` — blend weight; 0.0 ⇒ byte-identical status quo.
- `shapley_synergy_noise_floor: float = 0.0` (tau) — deadband, calibrated from the PIN-E near-inert small-real placebo.
- `shapley_synergy_cap: float = 0.0` — per-seed ceiling on `raw(s)`.
- Assert `k_committed ≤ 3` for the exact-factorial path; refuse sampled permutation Shapley for the committed family.

## Freeloader guard preserved (two types)
1. **Parked / non-committed freeloader** (PIN A's concern): the term pays only FOSSILIZED seeds → a parked seed gets 0
   top_up; its per-step LOO integrand + J are bit-identical to status quo ⇒ trivially ≤ no-op.
2. **Fossilized-useless seed:** null-player axiom → φ = 0 exactly (bit-identical leave-out logits), tau zeroes noise;
   the −0.2 gate still fires at commit ⇒ net −0.2 + 0 ≤ no-op.

The locked J form and PIN A per-seed LOO integrand are NEVER modified — this is a strictly additive terminal
reward-side supplement.

## Goodhart guards
- **Episode-level efficiency clamp (primary):** Σ top_up ≤ G — cannot be paid more synergy credit than the committed
  set actually improved terminal accuracy. (Replaces the candidate's additivity claim, which its own max(0,·) broke.)
  **Scope (advisor):** this bounds the aggregate POT, not the per-seed MISATTRIBUTION that drives decisions — G is
  measured with the same entrenchment-biased terminal ablation as φ. Keep it, but it does NOT bound the entrenchment
  problem (that is GATE 1's job).
- **Null-player + untouched −0.2 gate** does the freeloader discrimination for free.
- **Exact-factorial-only (k≤3):** no sampled-Shapley variance to farm. Asserted, not assumed.
- **Per-seed cap + PIN-E deadband** bound single-seed harvest + the ε-gain/ε-param ratio pathology.
- **Params priced once** (acc-point units + existing rent) ⇒ no double-incentive to fossilize many/large seeds.
- **Anti-commitment-avoidance:** pays only committed seeds → pushes AGAINST park-the-freeloader; monitor fossilize/ep
  (control 0.207±0.006) for the opposite failure (premature over-fossilization to harvest entrenchment).
- **Terminal-only:** single evaluation point → no dense per-step signal, no snapshot-timing gaming.

## Falsifiable success criterion (paired ≥5-seed A/B, term OFF vs ON)
(i) enabling stems (the suppress-confirmed r0c0 cohort) get top_up > 0; (ii) freeloaders ≤ no-op (PIN-E placebo → 0;
parked freeloader unchanged); (iii) acc-per-param RISES (suppress-read / off-switch-J improves) AND the G-clamp binds on
some episodes (proof the ceiling is real); (iv) commitment-avoidance no worse + fossilize/ep no degenerate spike;
(v) learnability — the scaffold rail demonstrably raises the FOSSILIZE-enabling-stem action's advantage (not merely
terminal return). Any of (i)–(v) failing falsifies the term. **HARD PRECONDITION:** the suppress-slot read must have
confirmed (b) for the cohort being credited.

## Does NOT solve / open risks
- **The (a)/(b) fork is a BET.** This term bets the dominant committed case is (b); the forward-mask read is
  per-episode INDISTINGUISHABLE from (a) (both look like "host depends on this fossil at inference"). Ship default-OFF;
  enable ONLY after the n=5 suppress-slot read confirms (b).
- **Entrenchment / opportunity-cost:** the inference-time leave-out credits any load-bearing-at-inference fossil, incl.
  a mediocre seed the host merely entrenched around. The true discriminator (a trajectory re-run without the commit —
  the PDR-0004 estimand) is NOT per-episode computable; the mask is a cheap proxy. Bounded (clamp+cap+deadband), not
  eliminated; monitor via the suppress-slot harness.
- **Double-pay** of long-lived good seeds (c_paid is a single fossilize snapshot); bounded by tau+cap+clamp.
- Does NOT change J/PIN A (J stays synergy-blind — needs the terminal-Shapley cross-check, §1.2); NOT the mechanistic
  per-pair claim (out of scope, PDR-0004); learnability + the scaffold-rail attribution must be tested, not assumed.
- **tau depends on the PIN-E placebo harness (GATE-0 noise-floor leg, still PENDING)** — cannot be responsibly enabled
  until tau is set from a non-degenerate small-real placebo. Also verify the fused-val kernel forces a fossilized slot's
  contribution to zero under an alpha=0 override (else G is mismeasured).

## Alternatives — B is the pre-identified salvage, NOT a speculative re-run
Candidates **B (hindsight/developmental** — an eligibility trace flowing downstream acc-per-param BACK to the
temporally-earlier enabling stem) and **C (efficiency-frontier)** stalled on infra. **Do NOT speculatively re-run the
bake-off** (advisor). B is inherently **developmental/temporal** (not a terminal snapshot), so it is the pre-identified
SALVAGE precisely if GATE 1 (proxy-vs-estimand) fails — the agreement check **subsumes** the comparison: agreement → A
stands, B moot; divergence (scaffolding signature) → B (temporal credit) becomes the design. C (efficiency-frontier)
risks re-triggering commitment-avoidance and is not preferred. A's null-player = freeloader guard is decisive *given*
terminal-φ is the right quantity — which GATE 1 is exactly what tests.
