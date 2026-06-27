# PDR-0001 — Bootstrap the product workspace from observed state

Date: 2026-06-28   Status: accepted   Author: Claude (agent)   Owner sign-off: yes (grant)
Supersedes: —   Related: vision.md, roadmap.md, metrics.md, current-state.md

## Context
No `docs/product/` workspace existed. The owner invoked `/own-product` to bootstrap
standing ownership for Esper, a morphogenetic-NN research framework. There was no prior
product state to resume; the workspace had to be inferred from observed reality rather
than a remembered history.

## Options considered
1. **Infer the workspace from observed reality** (README, ROADMAP/Constitution, git
   history, filigree tracker, and this session's evidence packet + memory) — pro: honest,
   grounded, cheap; con: audience/secondary and metric targets are partly inferred.
2. Interrogate the owner for vision/strategy from scratch — pro: nothing inferred; con:
   slower, ignores the rich stated direction already in README/ROADMAP.
3. Do nothing / refuse — con: leaves ownership unestablished after an explicit request.

## The call
Option 1. Seeded `vision.md`, `roadmap.md`, `metrics.md`, `current-state.md` from observed
direction. **The authority grant was NOT inferred** — it was proposed via AskUserQuestion
and the owner confirmed the **Default research grant** (autonomous within the active bet
incl. GPU runs + workspace commits; escalate vision/push/release/deprecation/data-deletion;
identity stays tachyon-beep; never push without ask). Grant clause: vision/grant change
escalates — this bootstrap records intent, the owner confirmed the grant live.

## Rationale
The product's purpose is stated plainly in README ("morphogenetic AI… grow capabilities,
don't just train weights") and ROADMAP ("Architectural Engineering → Architectural
Ecology"), so inference is well-grounded for purpose and direction. The substantive prior
decisions (J pinned with PIN A per-seed LOO; GATE thresholds; the reward-redesign
methodology) already live in `docs/analysis/2026-06-25-phase0-objective-and-instrumentation.md`
and are referenced rather than re-derived here.

## Reversal trigger
Reopen if the owner corrects the inferred **audience/secondary** or the **metric targets**
(currently `<owner-set>` placeholders), or on any change to the confirmed authority grant.
Audience and north-star TARGET numbers are the least-grounded fields and should be set by
the owner before the first acceptance decision fires against them.
