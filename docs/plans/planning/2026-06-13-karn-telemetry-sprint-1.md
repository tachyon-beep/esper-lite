# Karn Telemetry Quality Sprint 1 Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restore `main` to a clean, low-noise baseline and make Karn's current CI/build failures reproducible, fixed, or explicitly scoped for Sprint 2.

**Architecture:** Sprint 1 is a stabilization slice of the Karn Telemetry Quality Strategic Arc. It avoids broad refactors and focuses on dependency-drain landing, Sanctum deterministic state transitions, dependency PR cleanup, and Overwatch contract-drift inventory.

**Tech Stack:** Python 3.11, uv, pytest, mypy, ruff, Textual, FastAPI/MCP analytics, Vite/Vue/TypeScript, npm.

**Prerequisites:**
- Work from a clean branch based on current `main` or the active dependency-drain PR branch.
- Read `CLAUDE.md`, `README.md`, and `ROADMAP.md`.
- Preserve user changes and avoid destructive git operations without approval.
- Use Loomweave before broad call/reference edits.

```yaml
# Plan Metadata
id: karn-telemetry-sprint-1
title: Karn Telemetry Quality Sprint 1
type: planning
created: 2026-06-13
updated: 2026-06-13
owner: Codex

urgency: high
value: Convert the current post-recovery state into a green, low-noise baseline and establish Karn as the next upgrade package.

complexity: M
risk: medium
risk_notes: The sprint touches CI, dependency lockfiles, Karn TUI state, stale PR disposal, and web contract diagnostics. Scope must stay narrow to avoid destabilizing the newly recovered baseline.

depends_on:
  - dependency-drain
soft_depends:
  - karn-telemetry-quality-arc
blocks:
  - karn-telemetry-sprint-2
  - reward-efficiency

status_notes: Draft sprint plan. Needs specialist review before promotion to ready because it includes code changes in Karn and test/CI behavior.
percent_complete: 0

reviewed_by: []
```

---

## Sprint Outcome

At the end of Sprint 1, the repo should be back on `main`, green, and quiet enough to start a real Karn quality package. Dependency security patch PRs should be consolidated or closed. The Sanctum toggle failure should be deterministic and fixed. Overwatch build breakage should have exact source mapping, not a vague "contracts drifted" note.

## Task 1: Land or Replace Dependency Drain

**Files:**
- Modify only if CI requires it: `src/esper/karn/sanctum/app.py`
- Existing PR branch: `codex/dependency-drain`
- Existing PR: `https://github.com/tachyon-beep/esper-lite/pull/91`

**Steps:**

1. Confirm current PR #91 checks.
   - Run: `gh pr view 91 --json mergeStateStatus,statusCheckRollup,headRefOid,url`
   - Success: lint, typecheck, property, unit/integration, and e2e are pass or intentionally skipped by workflow rules.

2. Reproduce any failing unit job locally under Python 3.11 before changing code.
   - Run: `uv run --python 3.11 pytest -m "not integration and not stress and not property" -x --cov=src --cov-report=json`
   - Success: the local command either passes or fails on the same first test as CI.

3. If the failure is the Sanctum policy-group toggle, fix the state transition rather than adding sleeps or test-only timing hacks.
   - Primary file: `src/esper/karn/sanctum/app.py`
   - Test first: `uv run --python 3.11 pytest tests/karn/sanctum/test_app_integration.py::test_sanctum_app_shows_multiple_tamiyo_widgets -q`
   - Broader test: `uv run --python 3.11 pytest tests/karn/sanctum/test_app_integration.py tests/karn/sanctum/test_app.py -q`

4. Commit and push only the necessary fix.
   - Commit style: `fix(karn): ...` or `chore(deps): ...`
   - Success: PR #91 re-runs.

5. Merge PR #91 once checks are green.
   - Success: `main` contains the consolidated patch dependency updates and CI passes on the merge.

**Definition of Done:**
- [ ] PR #91 or successor is green.
- [ ] Dependency lock updates are merged through one reviewed PR.
- [ ] No unrelated files are included.
- [ ] `main` is checked out locally after merge.

## Task 2: Close Superseded Dependency PRs

**PRs:**
- Superseded by dependency drain: #45, #46, #47, #48, #50, #53, #54, #55, #56, #57, #59, #60, #63
- Stale consolidated draft: #51
- Explicitly deferred: #61 Vite direct bump, #62 Transformers 5.0.0rc3

**Steps:**

1. Confirm the merged dependency drain contains each patch dependency update.
   - Run: `git show --stat main`
   - Run: `gh pr view 91 --json mergedAt,mergeCommit,url`

2. Close each superseded PR with a short reason.
   - Message for superseded PRs: `Superseded by the consolidated dependency-drain PR, which landed this patch in one green lockfile update.`
   - Message for #61: `Deferred: direct Vite bump needs a separate Overwatch web build repair because current npm build exposes contract/tooling drift.`
   - Message for #62: `Deferred: Transformers 5.0.0rc3 is a prerelease major update and should not be included in the security patch drain.`

3. Delete only the remote branches for PRs that are closed and no longer needed.
   - Do not delete branches with unassessed unique code.

**Definition of Done:**
- [ ] Superseded dependency PRs are closed.
- [ ] Deferred dependency PRs have an explicit reason.
- [ ] Remote branch list no longer contains stale dependency branches that were closed as superseded.

## Task 3: Produce Karn Contract Drift Inventory

**Files:**
- Inspect: `src/esper/karn/overwatch/web/src/types/sanctum.ts`
- Inspect: `src/esper/karn/overwatch/web/package.json`
- Inspect: `src/esper/karn/overwatch/web/package-lock.json`
- Inspect likely generators under `src/esper/karn/overwatch/` and `scripts/`
- Output: `docs/analysis/2026-06-13-overwatch-contract-drift.md`

**Steps:**

1. Run the web install/build from the web workspace.
   - Run: `cd src/esper/karn/overwatch/web && npm ci --legacy-peer-deps && npm run build`
   - Success: either build passes or the exact missing symbols/errors are captured.

2. Find the Python source of each missing TypeScript contract.
   - Use Loomweave first for entity discovery when available.
   - Fallback: `rg -n "ShapleySnapshot|InfrastructureMetrics|GradientQualityMetrics|ValueFunctionMetrics" src tests scripts`

3. Classify each missing type:
   - `generate`: Python contract exists and TS generation missed it.
   - `rename`: Python contract exists under a new name and the web consumer is stale.
   - `delete`: Web consumer references a removed concept.
   - `design`: Contract is genuinely missing and needs Sprint 2 design.

4. Write the inventory doc with exact file paths and recommended Sprint 2 action.

**Definition of Done:**
- [ ] Build command output is captured in the inventory.
- [ ] Every missing TS type has a classification.
- [ ] Sprint 2 has a concrete repair queue rather than a generic build-failure note.

## Task 4: Verify Mainline Branch Hygiene

**Files:**
- No code files expected.

**Steps:**

1. Fetch and prune.
   - Run: `git fetch --prune origin`

2. List remaining open PRs.
   - Run: `gh pr list --state open --limit 100`

3. List unmerged remote branches.
   - Run: `git branch -r --no-merged origin/main`

4. Assess each remaining branch:
   - `merge`: active, green, relevant content.
   - `defer`: meaningful but unsafe for current sprint.
   - `delete`: merged, stale, superseded, or generated noise.

5. Do not delete unassessed unique branches.

**Definition of Done:**
- [ ] Open PR list is short and intentional.
- [ ] Remaining unmerged branches have a disposition.
- [ ] Stale/superseded branches are deleted.
- [ ] Unknown branches are documented rather than destroyed.

## Task 5: Sprint Closeout Evidence

**Files:**
- Modify: `docs/coord/PLAN_TRACKER.md`
- Optionally move this plan to `docs/plans/completed/` after execution.

**Validation commands:**

```bash
uv run --python 3.11 pytest tests/karn -q
uv run --python 3.11 pytest -m "not integration and not stress and not property" -x --cov=src --cov-report=json
uv run ruff check src/ tests/
MYPYPATH=src uv run mypy -p esper
```

For Overwatch:

```bash
cd src/esper/karn/overwatch/web
npm ci --legacy-peer-deps
npm run build
```

**Definition of Done:**
- [ ] Tracker reflects the strategic arc and Sprint 1 status.
- [ ] Commands run are recorded with pass/fail state.
- [ ] Any failing command has a linked follow-up in Sprint 2 scope.
- [ ] `main` is the local checkout when the sprint closes.
