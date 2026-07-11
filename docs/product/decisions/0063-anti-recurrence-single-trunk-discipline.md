# PDR-0063 — Anti-recurrence: single-trunk discipline to prevent workspace/code forks

Date: 2026-07-12
Status: proposed (working-practice change; owner adoption makes it binding — flagged)
Related: PDR-0062 (the fork this exists to prevent recurring)

## Context

The branch-vs-main divergence PDR-0062 resolves was not a one-off. Root cause: two
workstreams edited the same core training-path files AND both maintained the product
workspace in parallel — this lineage on `feat/ev-stab-stage2-hra`, other sessions /
Codex landing on `main` via PRs (the `575d2560` HRA squash, the reward-redesign /
Committed-Shapley / PIN-E track). The lines drifted for ~2.5 weeks and produced
duplicated implementations (HRA, Stage-0 gate, entropy fix) and two product workspaces.
The divergence was organizational, not technical.

## Options considered

- **A — Do nothing.** Rejected: the fork cost a full salvage audit and a
  merge-reconciliation exercise; unmanaged it recurs.
- **B — Single-trunk discipline (CHOSEN, proposed for owner adoption).**
- **C — Heavy branch-governance tooling / CODEOWNERS gates.** Deferred: premature for a
  single-maintainer research repo; revisit only if B proves insufficient.

## The proposed practice

1. **One trunk (`main`); short-lived branches merged back promptly.** No long-running
   parallel feature line accumulating weeks of divergence on the core training path.
2. **Product workspace owned in one place** — `docs/product/` lives on the trunk;
   experiment branches do not fork the workspace (the append-only PDR ledger is the
   continuity spine and must not diverge).
3. **Core-training-path files** (ppo_agent, rollout_buffer, advantages, contribution,
   reward_variance, leyline/telemetry) are the highest-conflict surface — land changes
   to them on the trunk quickly rather than staging them on a divergent branch.

## Why this is secondary

Getting the PDR-0062 reconciliation right is the immediate priority; this practice
prevents the *next* fork. It changes working practice, not the product vision or the
authority grant, so it is recorded as `proposed` and becomes binding on owner adoption.

## Reversal trigger

If single-trunk discipline measurably slows delivery or a genuine need for a
long-running parallel line arises (e.g. a multi-week incompatible experiment), revisit
with option C (lightweight branch-governance) rather than reverting to unmanaged forks.
