# PDR-0096 — Re-check r3: my annuity ratio_penalty exclusion REVERSED (symmetric inclusion) — the exclusion would have re-opened ransomware-farming as the annuity revenue line; T_annuity ≡ T_provisional now holds literally. Supersedes PDR-0094 #3 (N-r2a leg).

Date: 2026-07-14   Status: accepted   Supersedes: PDR-0094 decision #3 (the N-r2a transform-consistency leg ONLY; #1/#2/#4/#5 stand)
Input: drl-expert narrow re-check (W1-A/W1-B/§6/N-r2b/N-r2c PASS — settled; one blocking finding on N-r2a, verified before adopting).

## What does this buy?  (REQUIRED — PDR-0068)
It stops a control from being removed by footnote. My rev-3 rationale ("the spot-spike detector has no referent
against a smoothed statistic") was WRONG in a precise way: it closed the timing channel and called the job done,
while the ratio channel — ransomware as a PERSISTENT seed property — stayed wide open and policy-selectable. The
reviewer also named the process failure class exactly: silently dropping the shipping code's anti-ransomware
stance for the annuity is a decision-without-provenance of the kind this project treats as its founding wound.
The fix restores the spec's own words (`T_annuity ≡ T_provisional`) instead of my selective reading of them.

## The exploit (verified against code before banking)
`q_settle` = EWMA of `seed_contribution` — the LOO drop (`contribution.py:943`), NOT the clean counterfactual;
forced by T_annuity ≡ T_provisional. `ratio_penalty` (`:499–509`) is the law's ONLY counterfactual cross-check;
it fires on the ransomware signature (LOO drop ≫ total improvement — the seed holds the ensemble hostage). That
signature is persistent, so a ransomware seed reads as HIGH `q_decision` → the policy can SELECT it to commit →
high `q_settle` at the boundary → with the penalty dropped, an unbounded, unpenalized annuity to horizon, having
paid only the bounded window penalty. The retained bounds protect the wrong axis (`attribution_discount` is inert
for cf ≥ 0; the sqrt-cap bounds against progress, not cf). **No gate catches it:** GATE-INFL watches the BOTTOM
q_decision tercile; ransomware lives in the top. And the estimand corrupts: B−A would measure ransomware-farming,
not the protocol effect.

## Decisions
1. **Symmetric INCLUSION (the reviewer's direction, adopted):** the window keeps the full live law (unchanged);
   the annuity carries the SAME law at boundary-frozen args — `ratio_penalty(c:=q_settle, cf:=cf_B)`, BOTH
   branches (ratio>threshold AND the cf≤safe_threshold ransomware branch), computed once at B, applied per epoch.
   One law, two input regimes. My earlier "no referent" premise is withdrawn: the referent is q_settle vs the
   boundary-frozen clean counterfactual.
2. **Ledger scope cleanup (non-blocking, adopted):** the single-owner ledger owns [R+1, **B**] for the HOLDING
   per-seed row (the seed is HOLDING at B's reward phase; the A-at-boundary comparator pays the HOLDING climb at
   B), with provisional-suppressed-at-B the sole exception — B−A boundary-clean by construction.
3. **Process note banked:** my pre-commitment in the re-review request ("if you can name a gaming path... that
   would flip me") worked as designed — the reviewer named it, I flipped, and the flip is recorded with the
   original wrong rationale intact above. Instrument-corrected-claim tally: eight.
4. Remaining from the narrow re-check: NOTHING — W1-A/W1-B/§6/N-r2b/N-r2c passed and are settled. The reviewer
   verifies the annuity bullet at the pre-commit code review (their offer); the plan is otherwise clear to build
   once that eyeball lands.

## Reversal trigger
- If the frozen-args ratio check proves to misfire on legitimate seeds (e.g., cf_B legitimately small for a
  genuinely contributing seed at boundary — the replay's case matrix must include a legit-low-cf scenario) → the
  cf-smoothing form (windowed cf rather than lifetime-at-B) is the pre-identified refinement; never removal.
- If a future audit finds ANOTHER counterfactual-free payment path into the settlement → same class as this
  finding; the single-owner row inventory and this PDR are the checklist to re-run.
