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
value: Restore Esper to a mergeable, verified baseline and then drain the critical correctness backlog against that baseline.

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
  PR #52 was made green, accepted, and merged as the new baseline on 2026-06-12.
  Main CI passed on merge commit cdff9c43. The initial P0 Filigree bug drain is
  complete for the six critical reward, gradient, resume, and attribution
  defects identified in this recovery program.
percent_complete: 90

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
5. Critical Filigree P0 bugs that can silently corrupt training, telemetry, or resume behavior are fixed or explicitly reclassified.
6. The plan tracker and status notes reflect current reality.

## Current Evidence

- PR #52 (`env-refactor`) was merged into `main` on 2026-06-12.
- Merge commit: `cdff9c43847efdab65ca30faba79deade278bc1e`.
- PR #52 verification run `27410877590` passed:
  - `lint`
  - `typecheck`
  - `property-tests`
  - `unit-and-integration-tests`
  - `e2e-smoke-tests`
- Main post-merge Test Suite run `27411344212` passed on merge commit `cdff9c43`.
- The six initial P0 Filigree bugs were fixed and closed:
  - `esper-lite-41841f` BASIC_PLUS drip state silently lost when `return_components=False`
  - `esper-lite-7078b7` NaN gradient norms bypass exploding detection
  - `esper-lite-52ee59` resume path loads checkpoint twice to GPU
  - `esper-lite-b765c2` missing slot config validation on resume
  - `esper-lite-30e631` pair attribution order mismatch
  - `esper-lite-102ff8` FP16-scaled gradients collected before unscale
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

PR #52 was remotely green when GitHub Actions for the PR had no failing required checks. Skipped optional/nightly jobs are acceptable only if the workflow intentionally skips them for that event.

The project is steady only when:

- `main` contains the selected baseline.
- `main` or the merge commit has a passing required CI run.
- Critical P0 issues are fixed, closed, or explicitly reclassified with evidence.
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

Status:

- Completed: PR #52 merged to `main` at `cdff9c43`.
- Completed: post-merge main CI run `27411344212` passed.

### F. Drain Or Retarget Remaining PRs

Owner: primary agent with subagents as needed.

Scope:

- Handle security PRs first after the baseline is green.
- Close obsolete dependency PRs that are superseded by the new lockfile or baseline.
- Retarget/recreate small useful PRs that still apply.

Acceptance:

- Open PR list no longer contains stale ambiguous PRs.
- Remaining open PRs have a clear next action.

Status:

- Pending user action:
  - `#71` Add Knowledge Base in `jules/` — `human-decision`; likely keep only if still desired after baseline.
  - `#69`, `#65`, `#64` cleanup/comment PRs — `close-obsolete-after-baseline` unless still relevant after recovery branch lands.
  - `#70` SQL injection fix — `security-priority-after-baseline`; inspect and rebase/land next.
  - `#45`-`#63` Dependabot PRs and draft `#51` dependency consolidation — `security-priority-after-baseline` for security updates, otherwise retarget/consolidate after recovery branch.

### G. Drain Critical Filigree P0 Bugs

Owner: primary agent with specialist subagents as needed.

Scope:

- Fix P0 bugs that silently corrupt reward accounting, gradient health,
  checkpoint resume, or slot attribution.
- Use atomic Filigree `start-work --advance` before editing.
- Add focused regression tests for each corrected invariant.
- Run the local gate appropriate to the touched subsystem before commit.

Acceptance:

- Each claimed P0 has a comment summarizing the fix and verification.
- Each fixed P0 is closed in Filigree after code is committed.
- Full repository gates are rerun after a coherent batch of P0 fixes.

Status:

- Completed: initial six P0 issues fixed, verified, and closed.
- Completed: broad local gates passed where environment permits.
- Blocked by local CUDA/data fetch: full `tests/simic` includes CUDA CIFAR iterator smoke files that attempted SSL dataset access and timed out locally.

Final local evidence:

```bash
uv run ruff check src/ tests/
uv run ruff check scripts/
uv run python scripts/lint_leyline_types.py
uv run python scripts/lint_defensive_patterns.py
uv run python scripts/lint_gpu_sync.py
MYPYPATH=src uv run mypy -p esper
PYTHONPATH=src uv run pytest tests/simic --ignore=tests/simic/test_data_opt.py --ignore=tests/simic/test_record_stream_fix.py --ignore=tests/simic/training/test_dual_ab.py -q
```

The excluded files are CUDA/data-dependent smoke tests:

- `tests/simic/test_data_opt.py`
- `tests/simic/test_record_stream_fix.py`
- `tests/simic/training/test_dual_ab.py`

## Execution Log

- 2026-06-12: Program opened. PR #52 identified as critical path. Sidecar subagents dispatched for `origin/main` baseline verification and open PR classification.
- 2026-06-12: PR #52 made green and merged as baseline at `cdff9c43`.
- 2026-06-12: Recovery program moved to P0 Filigree bug drainage while post-merge main CI runs.
- 2026-06-12: Main post-merge Test Suite `27411344212` passed.
- 2026-06-12: Closed initial six P0 bugs: `esper-lite-41841f`, `esper-lite-7078b7`, `esper-lite-52ee59`, `esper-lite-b765c2`, `esper-lite-30e631`, and `esper-lite-102ff8`.
- 2026-06-12: Final local gates passed except CUDA/data-dependent smoke files blocked by local SSL dataset fetch.
