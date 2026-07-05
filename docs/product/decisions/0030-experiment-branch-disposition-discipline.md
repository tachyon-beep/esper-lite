# PDR-0030 — Standing discipline: force an explicit merge/fork/abandon disposition at the end of every experiment

Date: 2026-07-05   Status: accepted (owner directive, in-session)
Author: Claude (agent)   Related: PDR-0028 (the branch-divergence reconciliation that
motivated this), the 2026-07-05 branch consolidation (4 → 2 live branches).

## Context

The EV/Shapley work had drifted into FOUR divergent live branches (Shapley/product,
ev-stab EV code, `0.3.0`, main) with two diverging copies of `partition.py`/
`reward_variance.py` and the Stage-2 head in both squashed and granular form — the
"multiple live codebases" hazard. Consolidating it was non-trivial and error-prone.
The owner's response: prevent the accumulation at the source.

## Options

- **(a) Force a disposition call at each experiment's end** — every experiment ends
  with exactly one of {merge to main, fork as an enduring line, abandon}.
- **(b) Leave branch hygiene ad hoc** — status quo; branches accumulate until a
  painful consolidation.

## Call

**(a). STANDING RULE (owner directive).** At the END of every experiment, make a HARD,
EXPLICIT disposition call — exactly one of:
1. **Merge to `main`** — the DEFAULT / preferred outcome for keep-worthy work.
2. **Fork as a separate enduring line of effort** — legitimate when the work is a
   genuinely independent, still-active line; the owner prefers to avoid a proliferation
   of these, so choose it with explicit justification, not as a default parking spot.
3. **Abandon** — retire the branch (only when the experiment is genuinely done/dead).

This folds into the **CHECKPOINT** step of the ownership loop: a checkpoint is not
complete until the experiment's branch has a stated disposition. The disposition CALL
is the agent's to recommend; the merge/push EXECUTION remains owner-authorized (no push
without the owner's word; identity stays tachyon-beep).

## Rationale

An explicit disposition at each experiment's end keeps the tree consolidated on `main`
and avoids the divergence tax (conflicting single-source files, squashed-vs-granular
duplication) that a deferred consolidation pays.

**Intent (owner clarification, 2026-07-05):** the rule exists to stop a dozen
experimental projects accumulating with INCOMPLETE DATA and no known disposition — it is
NOT a mandate to prematurely retire work we still need. The requirement is that each
experiment reaches a KNOWN, data-complete disposition rather than drifting in limbo; a
well-justified enduring line is itself a valid disposition. Merge-to-main is the default
because it is the cheapest way to keep data and code from being orphaned, not because
longevity is bad.

## Reversal triggers

- If the discipline creates material friction (e.g. forces premature merges of genuinely
  independent long-lived research lines) → the owner revisits the default and may sanction
  a small set of named enduring branches.

## Current open disposition

- `feat/ev-stab-stage2-hra` (the active EV delivery branch) → disposition = **merge to
  main**, DEFERRED until Stage-0 completes + OFF-leg byte-identity + drl/pytorch review
  pass (PDR-0028). It is the one sanctioned open branch until then.
