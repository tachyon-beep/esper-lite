# PDR-0079 — Mechanism RESOLVED (READ 4): the FOSSILIZE benefit is ~50% intrinsic permanence value + ~50% shelter from the floor's own forced-PRUNE → a COUPLED package (repair the floor AND reward permanence)

Date: 2026-07-14   Status: accepted (zero-GPU r9 read; IPCW/g-formula, pre-registered verdict correctly applied, well-supported ESS≈0.9)
Resolves the OPEN mechanism question in **PDR-0078**. Related: PDR-0074 (measurement gap), PDR-0076/0077 (refocus + settlement), round-13.5 primes (the g-formula estimator; the no-survivor-conditioning correction). Read scripts durable in the session scratchpad (`read4_mechanism.py`, `read4_out.txt`).

## What does this buy?  (REQUIRED — PDR-0068)
It tells us which intervention to build. PDR-0078 measured a positive total effect but left OPEN whether it was intrinsic permanence value or shelter from the floor's forced-PRUNE — a fork that pointed at *different* fixes (settlement vs floor). READ 4 resolves it: **both**, ~50/50, so the fixes are co-required and neither ships alone.

## The decomposition (IPCW / no-future-PRUNE g-formula; NOT survivor conditioning)
Controlled regime `g_¬P` = after WAIT, follow the policy but censor the seed's later forced-PRUNE, IPCW-weighted by the KNOWN structural prune propensity (0.15 at pinned decisions). Cohort = first |H|=1 floored decision per lifecycle (collider-safe). `shelter = E[Y^{WAIT,g_¬P}] − E[Y^{WAIT}]`, `residual = E[Y^{FOSS}] − E[Y^{WAIT,g_¬P}]`, `shelter + residual = TE`. Well-supported: ESS/n ≈ 0.9, wmax 3.1–5.1 (NOT the underpowered fallback).

**Terminal final_accuracy (95% episode-clustered CI), per seed:**
- s41: TE +1.56 = **shelter +0.79 [0.53,1.05]** + **residual +0.77 [0.36,1.16]** (51% shelter)
- s42: TE +1.52 = **shelter +0.65 [0.39,0.91]** + **residual +0.87 [0.44,1.29]** (43% shelter)
- Both components significant (CIs exclude 0), both seeds.

**Support-1 (pre-mediator, val@1):** shelter ≈ 0 (NS: +0.09/+0.20), effect is **89–95% RESIDUAL** (+1.64/+1.59, CI far from 0). Forced-PRUNE cannot act at h=1 (median first-prune delay 13 epochs) → the immediate benefit is **intrinsic permanence value, not protection.** Shelter builds over the horizon (5% at h=1 → 72% at h=25).
**Support-2 (descriptive):** **55% of WAIT-arm seeds are later force-PRUNED** (both seeds), median delay 13 epochs (p10=1).

## The call / resolution
1. **BOTH mechanisms are material → repair the floor's forced-PRUNE AND reward permanence are a COUPLED PACKAGE; neither ships alone.**
2. **The intrinsic-permanence value is real and proven** (residual dominant early, significant at terminal) → the non-selectable settlement (PDR-0078 READ-1 invariant) is justified — NOT merely paying to escape a bug.
3. **The floor is BACK ON THE CRITICAL PATH (co-required).** ~Half the terminal FOSSILIZE benefit is escaping the floor's own forced-PRUNE of good seeds → this **strongly motivates** a floor/forced-PRUNE repair. **But soften (advisor):** the shelter ATE measures "don't force-prune THIS good seed," NOT "a floor *repair* recovers +0.7pp" — the un-forced policy's behaviour (does it keep the seed? commit it? the |H|≥2 gradient decides) is UNMEASURED, so the magnitude a repair recovers is a hypothesis for the eventual arm. And the two halves are **not additively bankable** (both are counterfactuals vs the SAME WAIT baseline; under a policy that both commits well and isn't force-pruned they interact) — the finding is "build both, coupled," NOT a +1.5pp forecast. **NEW banked finding (the sharpest one-line case for the coupled repair):** the same anti-WAIT floor that forces the *commits* also forces the *destruction* — **55% of good held seeds are force-pruned within ~13 epochs, at the same 0.15, equally uncontrollable by the policy.** The floor giveth commits and taketh away seeds. This destructive-PRUNE facet is a distinct workstream from the ρ-sweep gradient fix.
4. Sequencing unchanged upstream: mine remaining r9 first; L1(landed) + non-selectable-settlement design + floor forced-PRUNE repair are the coupled bet; K held; GPU held.

## Scope caveats (banked — do not over-read)
- Local ATE on the |H|=1 floor-pinned subpopulation, **commit-now-vs-defer** contrast (READ-3 balance-validated, |SMD|<0.08). K=1/obs-v3/shaped regime only.
- IPCW weights the prune hazard exactly at PINNED decisions (known 0.15; ~95% of prunes are floored); the ~5% deliberate prunes are a minor approximation. `g_¬P` intervenes only on the focal seed's prune.
- Terminal accuracy is episode-level but a valid causal outcome under the validated |H|=1 randomization.

## Reversal trigger
- If a designed floor/forced-PRUNE repair shows the pruned good-seeds would have added little terminal value anyway (the forced-PRUNE was near-benign on this cohort) → the shelter component shrinks and the settlement carries more of the weight.
- If a v4/K=4 rerun does not reproduce the ~50/50 split under the new regime → the decomposition is regime-specific.
- If the non-selectable settlement, once built, does not move fossilization toward higher-quote seeds → the intrinsic-value proxy (EWMA quote) is not actionable and L3 returns.
