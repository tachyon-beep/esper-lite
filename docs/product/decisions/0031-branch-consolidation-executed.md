# PDR-0031 — Branch consolidation executed (owner-authorized); ev-stab disposition = merge-to-main, deferred

Date: 2026-07-05   Status: accepted (owner-authorized in-session: "do the rebase and
the branch unification work, then push to remote … as much as possible onto a common
main")
Author: Claude (agent)   Related: PDR-0028 (flagged branch unification as owner-gated),
PDR-0030 (the disposition discipline this executes under). Un-gates PDR-0028's escalation.

## Context

PDR-0028 recorded branch unification (feat/ev-stab-stage2-hra → mainline, the "0.3.0
collapse") as OWNER-GATED / flag-only. The owner then explicitly authorized executing it
and pushing, motivated by not wanting to juggle multiple diverging live codebases (four
at the time: the Shapley/product branch, ev-stab, 0.3.0, and main). `git merge-tree`
dry-runs split the work into a safe half and a risky half.

## Options

- **(a) Big-bang merge** both feature and ev-stab onto main, resolve all conflicts at once.
- **(b) Sequence by risk** — bank the clean `feature → main` (1 trivial doc conflict)
  first for a clean 2-branch checkpoint, then reconcile ev-stab (the ~9 real conflicts,
  incl. the two diverging `partition.py`/`reward_variance.py`) separately, off-main.

## Call

**(b), executed and pushed.** On `origin/main` now:
- `fd0246bc` — product workspace (5 artifacts + PDRs) rebased onto origin/main + pushed.
- `f3758236` — `feat/phase-minus1-scale-falsifier` (Shapley/Phase-0 + product + squashed
  Stage-2 `575d2560`) merged to main (only conflict: `PLAN_TRACKER.md`, resolved by
  union) + pushed. Smoke-tested green (imports + 26 reward/telemetry tests).
- Subsequent doc commits (current-state, PDR-0030, CLAUDE.md orientation) pushed.

**Live branches: 4 → 2.** `feat/phase-minus1-scale-falsifier` is now fully contained in
main (redundant). `feat/ev-stab-stage2-hra` remains the ONE active branch.

**ev-stab disposition = MERGE-TO-MAIN, DEFERRED** (a *known* disposition per PDR-0030):
gated on Stage-0 completion + OFF-leg byte-identity + drl/pytorch review (CLAUDE.md). Its
reconciliation is a per-hunk merge (Shapley-TOPUP → main; Stage-0 SoA → ev-stab; Stage-2
→ ev-stab's newer post-`fab280c0`; `partition.py` = main superset; `reward_variance.py` =
UNION per-step + per-return), done off-main on a scratch branch, verified, then fast-merged.

Backup tags `backup-preconsolidation/*` retained for all pre-merge tips.

## Rationale

Sequencing by risk banked the 76-commit safe consolidation immediately (major reduction
in divergence) while leaving the correctness-critical core-training-path merge as a
focused, review-gated task — not a rushed big-bang onto canonical main. Deferring the
ev-stab merge (rather than forcing it) is the correct disposition because it is active
WIP (Stage-0 incomplete) with a KNOWN destination.

## Reversal triggers

- If the ev-stab reconciliation reveals the squashed (`575d2560`) and granular Stage-2
  diverged behaviourally (not just in commit form) → RCA before merging; the OFF-leg
  byte-identity test is the detector.
- If Stage-0 stalls indefinitely on ev-stab → re-open the disposition (continue, or
  stand-alone-fork per PDR-0030's reproducibility clause), don't let it drift as an orphan.
