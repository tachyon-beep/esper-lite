# PDR-0082 — Rounds 16–17: the premium fork RESOLVED by convergence (matched-control, retain the existing flat prior); reward-path audit COMPLETE (PR28 confirmed + two unpredicted findings)

Date: 2026-07-14   Status: accepted (gpt-prime adjudication + claude-prime round-17 concession = converged; audit executed against code at `5ca2a04f`)
Follows PDR-0081. Artifacts: `docs/analysis/2026-07-14-fossilize-reward-path-audit.md` (the audit), pre-registration §ROUND-15 Groups A (resolved) + D (new).

## What does this buy?  (REQUIRED — PDR-0068)
It closes the last genuine design fork (the premium) with an argument better than either prime's original position, and it replaces guesses about the commit-time reward with a code-grounded ledger. Measured: the fork is resolved without adding a tuned term; the audit found the adjudicated number itself was keyed to a mis-read (the "flat 0.5" is actually ≈1.464 of flat payment) — exactly why the audit was a blocker.

## 1. The premium fork — RESOLVED (matched-control)
**Decision: retain the EXISTING flat commitment premium as a matched control across arms; remove every contribution-dependent immediate FOSSILIZE payment; the premium is not an estimate of permanence value and must never be retuned from experimental results.**
- gpt-prime's frame: `fossilize_base_bonus` is already in arm A's shipping config → premium=0 is *itself an intervention* (removal), not the neutral point. Minimal-delta discipline (the F2 "match f=0.15 exactly" logic) applies to reward terms too.
- claude-prime **conceded in round 17**: its round-15 (+0.5–1.0 hedge) and round-16 (0 + data-priced Exp-2 premium) positions are both withdrawn; "pricing a reward term from the measured policy response IS the tuning loop — precisely how the anti-WAIT floor got installed."
- Process note (banked): the correct answer was found by the adjudication layer, not either prime — the second time this arc that the answer was "change nothing that's already controlled."

**Audit amendment (⚠ one-line owner confirm outstanding):** the adjudication said "+0.5" believing the tanh bonus was c-scaled. It is **constant** (`3.0·tanh(1/3) ≈ 0.964`, spot-c-gated not graded) → the existing flat prior = **0.5 + 0.964 ≈ 1.464 × legitimacy**. Matched-control, correctly applied, retains the consolidated ≈1.464; deleting the tanh term would halve the existing prior (an intervention). Recommend: consolidate to a single flat ≈1.464 term; owner confirms.

## 2. The gates (round-16 catch + round-17 refinements) — ACCEPTED
Round-16 caught that the round-15 "raw-logit diagnostic" measured the wrong object: raw logits are gauge (softmax shift-invariance); a marginal Δp misses state-dependent dominance (suppress good-seed commits, inflate junk commits → Δp̄≈0 → gate silent). Respecified: **GATE-DOM** = per-run paired slope of pre-floor `p_FOSS` on `q_decision` (B vs A, one-sided) + top-tercile `z_FOSS − z_SET_ALPHA` margin; **GATE-INFL** = bottom-tercile preference/request non-inflation. Both verdict-changing (fired GATE-DOM → SETTLEMENT-PROTOCOL-SAFE-BUT-DOMINATED → redesign, not retune). Round-17: gates are **one-sided in B** (op head updates only on the |H|≥2 minority → null = weak comfort, fired = strong alarm; fully informative in C); score pre-floor quantities (realized rates are floor-clamped in both arms); add the **trend stratum** (trailing quote slope). Pseudoreplication fix applies to the gates: per-run paired unit; ONE power calc covers P2 + both gates.

## 3. Reward-path audit — COMPLETE; findings (details in the audit doc)
- **PR28 CONFIRMED:** beyond base/`0.1·c`/tanh, the commit instant is c-keyed via the spot-c≥1.0 threshold gate on BOTH flat bonuses, the −0.2 noncontributing branch, and counterfactual-graded penalty branches. The invariant ("no contribution measurement available at/selected by the request instant may enter an immediate payment") **fails four ways as shipped**; compliant form = flat ≈1.464 premium, gate re-keyed to `q_decision`, all c-graded payment moved to the policy-unselectable `q_settle` annuity, premium+maintenance paid/started at the boundary.
- **F5 (unpredicted):** PBRS progress-forfeiture = a dwell-graded commit-time penalty (−0.31 at dwell 5, −0.46 at dwell ≥7; the HOLDING progress potential is forfeited on transition). The settlement window lengthens dwell → G-LEDGER must price the boundary transition's PBRS.
- **F8 (round-17 §3(i) is live TODAY):** a declining seed (lifetime counterfactual ≥0, spot c<1.0 incl. c<0) can commit for −0.2 and stop paying negative carry — the escape hatch exists in the shipping reward, ~10–12% negative-LOO share makes it material. Structural fix (identical sign/clip through the annuity, or mask negative-`q_decision` requests); never gate-only.
- **F7:** corrected fixed ledger ≈ **+1.02** net (not gpt's +0.37, not round-15's net-negative). **F6:** round-15's dominance ledger also missed the stay-branch `holding_warning` (−0.1→−0.3/epoch on good seeds).
- **F-drip:** the 70% drip pool is BASIC_PLUS-only (inactive in SHAPED) — must stay mode-fenced from the annuity.

## 4. Freeze blockers now
(1) the ⚠ premium-arithmetic confirm (≈1.464 consolidated vs 0.5-alone); (2) the paired-run power number (P2+GATE-DOM+GATE-INFL) + blinded reassessment rule + frozen max-n; (3) one more adversarial pass on the integrated doc (rounds 13/14/15/16 each surfaced ≥1 load-bearing flaw; round 17 added three holes). **No GPU.**

## Reversal trigger
- If the owner rejects the consolidation (keeps 0.5-alone) → the flat prior changes across A→B; record it as a deliberate intervention in the estimand statement (B−A then includes a −0.964 premium delta) — do NOT let it pass silently as "matched."
- If the power calc shows the gates cannot be bounded even at the frozen max-n → GATE-DOM demotes to descriptive at screen ONLY with an explicit pre-registered statement that the fatal case is live at screen (claude-prime PR26), and the claim tier inherits the gate.
- If the next adversarial pass finds another load-bearing flaw → the pass repeats; freeze only after a clean pass.
