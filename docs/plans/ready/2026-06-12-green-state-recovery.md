# Green State Recovery Program

```yaml
# Plan Metadata
id: green-state-recovery-2026-06-12
title: Green State Recovery Program
type: in-progress
created: 2026-06-12
updated: 2026-06-12
owner: Codex

urgency: critical
value: Restore Esper to a mergeable, verified baseline by resolving the long-running env-refactor PR, then draining or closing stale PRs against that baseline.

complexity: M
risk: high
risk_notes: >
  PR #52 is a large baseline reset touching reward semantics, PPO update accounting,
  telemetry/tooling, and project workflow files. The risk is accepting an unstable
  branch or discarding months of useful work without evidence.

depends_on: []
soft_depends: []
blocks:
  - reward-efficiency
  - phase3-tinystories
  - dependency-security-drain

status_notes: >
  Active recovery program. PR #52 is the critical path. No other PR should land
  until #52 is either made green and accepted as the new baseline, or explicitly
  rejected with a smaller replacement path.
percent_complete: 20

reviewed_by:
  - reviewer: python-engineering
    date: 2026-06-12
    verdict: approved-with-changes
    notes: >
      Scope is constrained to stabilization and verification. Code changes must
      remain minimal until PR #52 is accepted or rejected. Broad refactors and
      product-workspace bootstrapping wait until after the baseline is green.
```

## Objective

Return the project to a steady state:

1. A known-good baseline is selected.
2. Required CI and local validation gates pass.
3. PR #52 is either merged as the new baseline or replaced by an explicit smaller path.
4. Remaining open PRs are classified, closed, merged, or retargeted against the selected baseline.
5. The plan tracker and status notes reflect current reality.

## Current Evidence

- Current branch: `env-refactor`, PR #52.
- PR #52 is the large unlanded baseline-reset candidate.
- Previous GitHub CI for PR #52 failed in `lint` on `src/esper/simic/training/policy_group.py`.
- Local targeted verification after the stabilization patch passed:
  - `uv run ruff check src/ tests/`
  - `uv run ruff check scripts/`
  - `uv run python scripts/lint_leyline_types.py`
  - `uv run python scripts/lint_defensive_patterns.py`
  - `uv run python scripts/lint_gpu_sync.py`
  - `MYPYPATH=src uv run mypy -p esper`
  - `PYTHONPATH=src uv run pytest tests/simic -q`
- The working tree also contains unrelated dirty skill/config files. These must not be reverted or silently included in the PR #52 stabilization commit.

## Definition Of Green

PR #52 is locally green only when all commands below pass from the PR branch:

```bash
uv run ruff check src/ tests/
uv run ruff check scripts/
uv run python scripts/lint_leyline_types.py
uv run python scripts/lint_defensive_patterns.py
uv run python scripts/lint_gpu_sync.py
MYPYPATH=src uv run mypy -p esper
PYTHONPATH=src uv run pytest tests/simic -q
```

PR #52 is remotely green only when GitHub Actions for the PR has no failing required checks. Skipped optional/nightly jobs are acceptable only if the workflow intentionally skips them for that event.

The project is steady only when:

- `main` contains the selected baseline.
- `main` or the merge commit has a passing required CI run.
- Open PRs have an explicit disposition.
- No known task-scope defects are hidden as scratch observations.

## Workstreams

### A. Stabilize PR #52

Owner: primary agent.

Scope:

- Commit only the stabilization changes needed to make PR #52 pass its green gate.
- Do not include unrelated dirty files.
- Do not broaden the PR with product workspace bootstrapping or dependency queue work.

Known patch:

- `src/esper/simic/training/policy_group.py` — module docstring must precede `from __future__ import annotations`.
- `leyline_boundaries.yaml` — expiry dates must match current mainline policy so the branch validates against current time.

Acceptance:

- All local green commands pass.
- Stabilization commit contains only intended files.
- Commit is pushed to `env-refactor`.

### B. Verify `origin/main`

Owner: subagent.

Scope:

- Use an isolated worktree or clean clone.
- Do not touch the dirty PR #52 worktree.
- Run the same green gate against current `origin/main`.

Acceptance:

- Report each command as pass/fail/blocked.
- Classify whether any failure is independent of PR #52.

### C. Triage PR #52 GitHub CI

Owner: CI subagent after Workstream A is pushed.

Scope:

- Inspect the new PR #52 workflow run.
- Fetch failing logs if any.
- Classify failures as PR regression, inherited baseline issue, or CI infrastructure issue.

Acceptance:

- Every failing job has a first failing command and a recommended next action.
- No code changes are made during triage unless explicitly dispatched.

### D. Open PR Queue Disposition

Owner: subagent.

Scope:

- Read current open PR metadata.
- Do not merge, close, or modify PRs.
- Classify each PR relative to the PR #52 baseline decision.

Acceptance:

- Each open PR is classified as:
  - `baseline-critical`
  - `keep-after-baseline`
  - `close-obsolete-after-baseline`
  - `security-priority-after-baseline`
  - `human-decision`

### E. Land Baseline

Owner: primary agent.

Scope:

- If local and remote evidence prove PR #52 is green, merge it.
- If the PR cannot be made green without broad new work, stop merging and record the replacement path.
- Any merge must preserve history and avoid destructive git operations.

Acceptance:

- `main` has the selected baseline.
- Required checks pass on the resulting merge or post-merge main run.

### F. Drain Or Retarget Remaining PRs

Owner: primary agent with subagents as needed.

Scope:

- Handle security PRs first after the baseline is green.
- Close obsolete dependency PRs that are superseded by the new lockfile or baseline.
- Retarget/recreate small useful PRs that still apply.

Acceptance:

- Open PR list no longer contains stale ambiguous PRs.
- Remaining open PRs have a clear next action.

## Execution Log

- 2026-06-12: Program opened. PR #52 identified as critical path. Sidecar subagents dispatched for `origin/main` baseline verification and open PR classification.

