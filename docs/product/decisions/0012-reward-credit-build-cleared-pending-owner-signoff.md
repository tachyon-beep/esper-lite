# PDR-0012 — Reward-credit term cleared for build (reviewer APPROVE_WITH_CHANGES); BUILD AWAITS OWNER SIGN-OFF

Date: 2026-07-02   Status: **proposed** (the build start is owner-gated; the reviewer-gate completion itself was within grant)   Author: Claude (agent)
Supersedes: —   Related: PDR-0010, PDR-0011, docs/analysis/2026-07-02-reward-credit-term-review.md, esper-lite-254175df90

## Context
The issue's gate order was: GATE 2 → reward-function-reviewer → owner sign-off → build. GATE 2 passed
(PDR-0011); the owed signal verification closed (fossilize_contribution_scale multiplies the per-seed LOO —
premise intact). The reward-function-reviewer was dispatched on the design + full evidence package.

## The call (what is being proposed to the owner)
**Build the Committed-Shapley synergy top-up, default-OFF (shapley_synergy_scale=0.0), with retro-write delivery
and the reviewer's five build-conditioning changes:**
1. True no-op at scale=0 — gate the ENTIRE terminal 2^k apparatus behind scale>0; RNG/determinism-neutral.
2. Double-pay reconciliation with the two live-but-dormant sibling channels (per-step synergy_bonus; retroactive
   scaffold hindsight credit) + resolve the "synergy" naming collision.
3. Scale-space safety: per-seed cap AND a normalized-space clamp/std-floor — divide_by_std is the ONLY unclipped
   reward channel; small early-run std amplifies the credit 2.5–5×.
4. Broadened kernel verification (alpha=0 zeroing over ALL coalitions; v() returns percentage points) and c_paid
   recomputed from the terminal same-time standalone baseline (not the fossilize-time snapshot).
5. The mandated GAE retro-write unit test vs hand-computed advantages.

Reviewer verdict: **APPROVE_WITH_CHANGES — "safe to BUILD default-OFF and present for owner sign-off."**
Enablement criteria are banked separately and remain a LATER gate: scale/cap/tau calibration (tau still blocked
on the GATE-0 PIN-E placebo), a HARD off-switch-J efficiency + fossilize-count gate (per-step rent ~0.3/episode
is not a counterweight to a 1–10 terminal lump), the entrenchment monitor, and an ON-run dormancy recheck of the
sibling channels.

## Rationale
Every gate that is not the owner's has passed with pre-registered or reviewed evidence. The flag stays 0.0
throughout the build; enabling the term is a separate, further-gated decision.

## Reversal trigger
Kill the build (pivot to candidate B's eligibility weighting on the same retro-write delivery) if the build-time
verifications fail: the alpha=0/units assertions (F6), the terminal standalone baseline making gap(s) ≤ tau for
the r0c0 cohort (the synergy premise dissolving), or the sibling channels proving non-reconcilable without
reward-behavior change. Enablement reversal triggers are defined at the enablement gate, not here.

## Awaiting owner
Sign-off to START the build as scoped above. Until then esper-lite-254175df90 stays in_progress/blocked-on-owner.
