# Green State Recovery Program

```yaml
# Plan Metadata
id: green-state-recovery-2026-06-12
title: Green State Recovery Program
type: in-progress
created: 2026-06-12
updated: 2026-06-13
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
  Main CI passed on merge commit cdff9c43. Recovery PR #72 landed the initial
  P0 Filigree bug drain and project Filigree install removal. Follow-up PRs
  #78 and #79 merged telemetry and training-control correctness fixes. PRs
  #80, #81, and #82 merged the first three P2 batches. The repository is green
  on CI, but not yet steady. The P2 GPU-sync batch is locally fixed and
  verified against focused tests, custom guardrails, type, lint, full pytest,
  and Wardline gates.
percent_complete: 86

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
- Recovery PR #72 (`codex/steady-state-recovery`) was merged into `main` on
  2026-06-12 at merge commit `514e04a6d7235eba0ef61a147477f7768448730f`.
- PR #72 verification run `27414100263` passed:
  - `lint`
  - `typecheck`
  - `property-tests`
  - `unit-and-integration-tests`
  - `e2e-smoke-tests`
- PR #78 (`codex/p1-telemetry-correctness`) was merged into `main` on
  2026-06-12 at merge commit `9cd6284cdcbe266ddaefec3925e7620222d16960`.
- PR #78 verification run `27418134962` passed:
  - `lint`
  - `typecheck`
  - `property-tests`
  - `unit-and-integration-tests`
  - `e2e-smoke-tests`
- PR #79 (`codex/p1-training-correctness`) was merged into `main` on
  2026-06-12 at merge commit `e66517dee8abeabb8e6855b09d0efee1915a68e3`.
- PR #79 verification run `27420206496` passed:
  - `lint`
  - `typecheck`
  - `property-tests`
  - `unit-and-integration-tests`
  - `e2e-smoke-tests`
- Local full-suite verification after PR #79: `uv run pytest` passed
  `4681 passed, 10 skipped, 69 deselected`.
- Local P2 contract batch verification on 2026-06-13 passed:
  - `uv run pytest tests/simic/test_ppo_checkpoint.py tests/simic/agent/test_ppo_finiteness_gate.py tests/simic/agent/test_ppo_metrics_contract.py -q`
  - `uv run python scripts/lint_defensive_patterns.py`
  - `uv run python scripts/lint_leyline_types.py`
  - `uv run python scripts/lint_gpu_sync.py`
  - `uv run ruff check src/ tests/`
  - `MYPYPATH=src uv run mypy -p esper`
  - `uv run pytest` (`4686 passed, 10 skipped, 69 deselected`)
  - `wardline scan . --fail-on ERROR` (`0 active`)
- PR #80 (`codex/p2-steady-state-drain`) was merged into `main` on
  2026-06-12 at merge commit `6676449bd972ea219613f027a376a36f3f4612d9`.
- PR #80 verification run passed:
  - `lint`
  - `typecheck`
  - `property-tests`
  - `unit-and-integration-tests`
  - `e2e-smoke-tests`
- The first three P2 contract bugs were fixed and closed:
  - `esper-lite-aa2a27` PPO checkpoint metadata declared fields not emitted
  - `esper-lite-dcb298` PPO update metrics contract included stale keys
  - `esper-lite-860e79` all-epochs-skipped updates emitted zeroed metrics
- Local second P2 action/reward contract batch verification on 2026-06-13 passed:
  - `uv run pytest tests/simic/training/handlers/test_alpha_handler.py tests/simic/training/handlers/test_prune_handler.py tests/simic/test_reward_telemetry.py -q`
  - `uv run python scripts/lint_defensive_patterns.py`
  - `uv run python scripts/lint_leyline_types.py`
  - `uv run python scripts/lint_gpu_sync.py`
  - `uv run ruff check src/ tests/`
  - `MYPYPATH=src uv run mypy -p esper`
  - `uv run pytest` (`4690 passed, 10 skipped, 69 deselected`)
  - `wardline scan . --fail-on ERROR` (`0 active`)
- PR #81 (`codex/p2-action-reward-contracts`) was merged into `main` on
  2026-06-12 at merge commit `32ddc39df5018a70c68e5c3603d586759cb527bf`.
- PR #81 verification run `27423441338` passed:
  - `lint`
  - `typecheck`
  - `property-tests`
  - `unit-and-integration-tests`
  - `e2e-smoke-tests`
- The second P2 action/reward contract bugs were fixed and closed:
  - `esper-lite-642150` alpha target handling dropped sigmoid steepness variants
  - `esper-lite-c8d465` reward telemetry deserialized string `"False"` as true
- Local P2 counterfactual telemetry batch verification on 2026-06-13 passed:
  - `uv run pytest tests/simic/attribution/test_counterfactual.py -q`
  - `uv run python scripts/lint_defensive_patterns.py`
  - `uv run python scripts/lint_leyline_types.py`
  - `uv run python scripts/lint_gpu_sync.py`
  - `uv run ruff check src/ tests/`
  - `MYPYPATH=src uv run mypy -p esper`
  - `uv run pytest` (`4694 passed, 10 skipped, 69 deselected`)
  - `wardline scan . --fail-on ERROR` (`0 active`)
- PR #82 (`codex/p2-counterfactual-telemetry`) was merged into `main` on
  2026-06-12 at merge commit `68398d4a447f2051eb423e3ebc07cc85802125ef`.
- PR #82 verification run `27424591784` passed:
  - `lint`
  - `typecheck`
  - `property-tests`
  - `unit-and-integration-tests`
  - `e2e-smoke-tests`
- The P2 counterfactual telemetry bugs were fixed and closed:
  - `esper-lite-65d044` counterfactual matrices persisted compute time as zero
  - `esper-lite-d9edae` simple ablation treated missing data as neutral baseline
- Local P2 GPU-sync batch verification on 2026-06-13 passed:
  - `uv run pytest tests/simic/agent/test_rollout_buffer_unit.py tests/telemetry/test_environment_metrics.py -q`
  - `uv run python scripts/lint_gpu_sync.py`
  - `uv run python scripts/lint_leyline_types.py`
  - `uv run python scripts/lint_defensive_patterns.py`
  - `uv run ruff check src/ tests/`
  - `MYPYPATH=src uv run mypy -p esper`
  - `uv run pytest` (`4697 passed, 10 skipped, 69 deselected`)
  - `wardline scan . --fail-on ERROR` (`0 active`)
- Filigree was removed from the UV tool install on 2026-06-13:
  `uv tool uninstall filigree` removed `filigree`, `filigree-dashboard`,
  `filigree-mcp`, `filigree-scanner-claude`, and `filigree-scanner-codex`.
  `uv tool list` now retains only Legis, Loomweave, Loomweave plugins, and
  Wardline from the local standard tooling set.
- Filigree on 2026-06-13 reports `13 ready`, `0 blocked`, and `2 wip`
  after claiming `esper-lite-1eeab1` and `esper-lite-52d4c3`.
- Loomweave MCP is visible, but the active MCP server still reports no
  `.weft/loomweave/loomweave.db` even after a worktree analysis pass. The
  available MCP session needs a reconnect before graph queries can be used.
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
- P1 correctness issues found during recovery are fixed, closed, or explicitly
  reclassified with evidence.
- Remaining P2/P3 issues are either fixed and closed, bundled into a documented
  follow-up release plan, or intentionally reclassified with evidence.
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
- Completed: recovery PR #72 merged into `main` at `514e04a6`.
- Completed: P1 stability batch 1 commit `b4b8f2d0` fixed or verified and closed:
  - `esper-lite-18eb2f`
  - `esper-lite-2fcc87`
  - `esper-lite-5f7f67`
  - `esper-lite-1bbfb2`
  - `esper-lite-c82e50`
  - `esper-lite-ee44b1`
- Blocked by local CUDA/data fetch: full `tests/simic` includes CUDA CIFAR iterator smoke files that attempted SSL dataset access and timed out locally.

### H. Drain P1 Stability Batch 1

Owner: primary agent with read-only specialist subagents.

Scope:

- Fix or close as stale the first six claimed P1 issues:
  - `esper-lite-18eb2f` entropy floor schedule inversion
  - `esper-lite-2fcc87` epoch-0 KL no-step accounting
  - `esper-lite-5f7f67` zero-availability entropy floor penalty
  - `esper-lite-1bbfb2` zero-mask KL weighting dilution
  - `esper-lite-c82e50` `rollout_total_steps` after buffer clear
  - `esper-lite-ee44b1` non-finite gradient drift poisoning
- Keep the branch limited to PPO math, PPO coordinator, gradient EMA, tests,
  and plan/tracker updates.

Acceptance:

- Every claimed issue has focused regression evidence or a stale-issue closure
  comment referencing current tests.
- Branch gates pass locally.
- GitHub PR checks pass before merge.

Status:

- Locally complete on branch `codex/p1-stability-batch-1`; pending PR and GitHub checks.

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

### I. Drain P1/P2 Contract Bugs

Owner: primary agent with specialist subagents as needed.

Scope:

- Continue from the current Filigree ready queue after PR #79.
- Prefer coherent batches that share verification gates and code ownership:
  checkpoint/resume contracts, telemetry schema contracts, PPO/tensor
  performance contracts, and action-handler parameter contracts.
- Keep each batch small enough to review and merge independently.
- Do not mix dependency-security work with training-correctness fixes unless a
  dependency update is directly required by a fix.

Acceptance:

- Use `filigree start-work <id> --advance --assignee Codex` before editing.
- Every claimed issue has a focused regression test or an explicit stale-issue
  closure comment backed by current source/tests.
- Full local static gates pass for any batch that changes shared contracts:

```bash
uv run python scripts/lint_defensive_patterns.py
uv run python scripts/lint_leyline_types.py
uv run python scripts/lint_gpu_sync.py
uv run ruff check src/ tests/
MYPYPATH=src uv run mypy -p esper
uv run pytest
```

- GitHub PR checks pass before merge.
- Filigree issues move `fixing -> verifying -> closed` only after merged code
  or explicit current-state verification.

Current queue snapshot, 2026-06-13:

- P2 ready bugs: 19
- P3 ready bugs: 2
- Blocked: 0
- WIP: 0

Current local batch:

1. `esper-lite-aa2a27` checkpoint load fallback/legacy filtering.
2. `esper-lite-dcb298` PPO update metrics TypedDict drift.
3. `esper-lite-860e79` finiteness gate failure list type preservation.

These all affect typed training/checkpoint/telemetry contracts and should be
verified together with checkpoint, PPO metric, defensive-pattern, type, and
full default pytest gates.

Status: fixed and locally verified on branch `codex/p2-steady-state-drain`.
Tracker closure remains pending because the Filigree UV tool install was
removed and no writable Filigree MCP tool is currently exposed in this session.

## Execution Log

- 2026-06-12: Program opened. PR #52 identified as critical path. Sidecar subagents dispatched for `origin/main` baseline verification and open PR classification.
- 2026-06-12: PR #52 made green and merged as baseline at `cdff9c43`.
- 2026-06-12: Recovery program moved to P0 Filigree bug drainage while post-merge main CI runs.
- 2026-06-12: Main post-merge Test Suite `27411344212` passed.
- 2026-06-12: Closed initial six P0 bugs: `esper-lite-41841f`, `esper-lite-7078b7`, `esper-lite-52ee59`, `esper-lite-b765c2`, `esper-lite-30e631`, and `esper-lite-102ff8`.
- 2026-06-12: Final local gates passed except CUDA/data-dependent smoke files blocked by local SSL dataset fetch.
- 2026-06-12: PR #78 merged telemetry correctness fixes and closed
  `esper-lite-0aa641`, `esper-lite-2ac173`, and `esper-lite-d612b3`.
- 2026-06-12: PR #79 merged training-control correctness fixes and closed
  `esper-lite-afaf1a`, `esper-lite-df2f30`, and `esper-lite-6cc6b6`.
- 2026-06-13: Recovery plan refreshed for the remaining P2/P3 contract drain.
- 2026-06-13: Removed Filigree from the UV tool install; Legis, Loomweave, and
  Wardline remain installed as UV tools.
- 2026-06-13: Locally fixed and verified P2 contract batch
  `esper-lite-aa2a27`, `esper-lite-dcb298`, and `esper-lite-860e79`; tracker
  closure is pending a writable Filigree surface.
