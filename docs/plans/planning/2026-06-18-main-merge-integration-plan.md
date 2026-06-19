# 0.1.1 → main Merge Integration — Executable Plan

```yaml
# Plan Metadata
id: main-merge-integration
title: 0.1.1 → main Merge Integration
type: ready
created: 2026-06-18
updated: 2026-06-18  # revised per post-P0.1 hardening-sprint defect report (B2,B6-B12,W1-W5,W13,W14)
owner: Claude

# Prioritization
urgency: high
value: >
  Lands ~46 commits of accumulated work (vectorized-trainer refactor, multi-epoch
  recurrent PPO PR1/PR2, P0-1 op-independent V(s) baseline, P0-2/P0-3, Tier-0
  profiler, leyline contracts move) onto main as a clean fast-forward, ships the
  checkpoint-break migration story, and folds the Dependabot bump pass into the
  same CI window.

# Constraints
complexity: L
risk: high
risk_notes: >
  Mechanically a fast-forward (zero conflict surface, conflict probability zero by
  construction), so there is no MERGE-ALGORITHM risk. The residual risk is
  operational: (a) the 46-commit unit must pass CI on main's triggers including
  nightly-only slow/stress paths first exercised post-push; (b) the
  VALUE_HEAD_SCHEMA_VERSION=2 checkpoint break rejects all pre-v2 PPO checkpoints
  with no remap; (c) EV-keyed health gates / anomaly rules / Karn views will
  false-alarm on K>1 runs unless Sprint item 1's gate-fix co-lands; (d) a
  push to main between now and the merge would break the FF assumption.

# Dependencies
depends_on:
  - ev-telemetry-robustness        # Sprint item 1 gate-fix MUST co-land — HARD constraint (Locked decision 5).
                                   # Promoted from soft_depends: the spec frames this as a hard "must co-land"
                                   # gate; a soft dependency would (wrongly) tell tooling the FF can proceed
                                   # without it. The run_confounders view CORRECTLY proof-blocks any emitted
                                   # VALUE_COLLAPSE_DETECTED row (views.py:649) — the fix is UPSTREAM (prevent the
                                   # artefactual emission), NOT view-layer suppression, so K>1 runs on main do not
                                   # emit a spurious collapse the moment the new V(s) estimator lands.
soft_depends: []

blocks: []

# Status
status_notes: >
  READY AFTER EV TELEMETRY ROBUSTNESS LANDS. This plan is NOT unconditionally
  ready to execute: it hard-depends on the EV-telemetry-robustness fix being
  STRUCTURALLY present on 0.1.1. Current 0.1.1 HEAD still carries the OLD EV
  branch (`var_returns > 1e-8 → torch.tensor(0.0)` at ppo_agent.py:624) and does
  NOT yet contain the planned ev_return_variance_floor / ev_low_return_variance
  artifacts. The EV structural check (Step 1.5) is the FIRST blocking execution
  gate — execution MUST NOT begin until it passes. Topology verified at spec time
  (merge-base == main HEAD == f7f1aece). Execution order: EV-structural gate →
  schema/checkpoint confirmation → all five CI lanes locally → PR → FF push →
  dependency bumps last.
percent_complete: 0

# Expert Review (REQUIRED before promotion to ready)
reviewed_by:
  - reviewer: axiom-python-engineering
    date: 2026-06-18
    verdict: approved
    notes: >
      Lockfile-only dependency strategy and the no-remap checkpoint contract align
      with the No-Legacy policy. Confirm the uv.lock delta is dependency-version-only
      (no pyproject.toml pin edits) before the bump commit lands.
  - reviewer: plan-review-systems
    date: 2026-06-18
    verdict: approved-with-changes
    notes: >
      FF topology and zero-conflict claim are sound. Required: (1) every git
      operation that could touch main HEAD must re-verify the merge-base
      immediately before running (a push to main invalidates the FF); (2) the
      dependency bump must be the LAST commit and independently revertable; (3) no
      git reset --hard / push --force on main without explicit user permission
      (CLAUDE.md Git Safety) — rollback is forward-revert only; (4) the EV co-landing
      gate must be a STRUCTURAL code-artifact check run pre-push, not a commit-message
      grep (existing commits 177a53aa / 6a27b8e3 already false-positive on an
      'ev|explained_variance' grep).
```

## Global execution assumptions

- **All commands assume `cwd = /home/john/esper-lite`** (the repo root). Hardcoded absolute paths below encode this assumption for a single human/agent executor on this machine. If executed elsewhere, substitute `$(git rev-parse --show-toplevel)` for `/home/john/esper-lite` throughout. This is a human-executed release plan; the explicit paths are intentional, not a defect.
- **Co-author trailer (the assistant's own commit convention — NOT a CLAUDE.md rule; CLAUDE.md contains no trailer requirement):** every commit this plan authors (only Step 8c) MUST end with the exact trailer:
  ```
  Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
  ```
  This is inlined here because the bump commit may be authored in a different context (a dependency-bump pass, not necessarily an active Claude Code session), so the trailer must be unambiguous regardless of who executes the step.

## Source spec

`docs/superpowers/specs/2026-06-18-main-merge-integration-design.md` (status: ready). All locked decisions, the FF reframe, the divergence inventory, and the acceptance criteria originate there; this plan is the executable, step-gated realization.

## Branch

- **Source:** `0.1.1` (HEAD carries the 46-commit lead; already pushed to origin).
- **Target:** `main`.
- **Topology:** `main` HEAD *is* the merge-base — this is a fast-forward, not a three-way merge. **No working branch is created**; the integration *is* the fast-forward of `main` to `0.1.1`. The dependency-bump pass lands as new commits *after* the FF, on `0.1.1` (which becomes `== main` post-FF) so they ride the same push.

## Discipline: Gated execution (PRECONDITION → ACTION → VERIFY → ROLLBACK)

This is a release-engineering plan, not a code-authoring plan: **no functional code is written by this plan** — the merge transports already-committed, already-tested work; the only authored artifact is the trailing lockfile-bump commit (Step 8c). The TDD RED→GREEN discipline is adapted to merge execution:

- **PRECONDITION (the "RED-equivalent"):** a verification command whose *expected output is stated*. If the actual output differs, **STOP** — the assumption underpinning the step is false and the plan must be re-assessed. This is the analogue of "expected RED": a failing precondition is a real signal, not a nuisance.
- **ACTION:** the exact command(s) to run.
- **VERIFY (the "GREEN"):** a post-condition command with stated expected output.
- **ROLLBACK:** the forward-only recovery path (per CLAUDE.md Git Safety — never `--hard`/`--force` on `main` without explicit user permission).

**Order is load-bearing:** the **EV-telemetry-robustness structural gate (Step 0) runs FIRST** — this plan hard-depends on that fix being present on `0.1.1`, and execution must not begin if it is absent (the current `0.1.1` HEAD still carries the old EV branch at `ppo_agent.py:624`). Then schema/checkpoint confirmation (so a stranded-checkpoint discovery aborts before anything lands), then all five CI lanes locally, then the PR, then the FF push, then the dependency bumps LAST (so a flaky transitive cannot block or be confused with the functional merge). Step 6 (pre-push) re-confirms the EV gate immediately before the only mutating push.

## Verified code anchors (from the source spec; authoritative for the checkpoint-break comms)

| Anchor | Location | Role |
|--------|----------|------|
| `VALUE_HEAD_SCHEMA_VERSION = 2` (definition) | `src/esper/leyline/__init__.py:125` | The single break; does not exist on main |
| `VALUE_HEAD_SCHEMA_VERSION` (export) | `src/esper/leyline/__init__.py:877` | Public surface (`__all__`) |
| Import into agent | `src/esper/simic/agent/ppo_agent.py:51` | Consumer |
| Written at save | `src/esper/simic/agent/ppo_agent.py:1540` | `'value_head_schema_version': VALUE_HEAD_SCHEMA_VERSION` |
| Required-field read at load (inside `try`) | `src/esper/simic/agent/ppo_agent.py:1656` | Reads `checkpoint['value_head_schema_version']`; a genuine pre-v2 checkpoint lacks this key |
| **KeyError path (the REAL path for genuine pre-v2 checkpoints)** | `except KeyError` at `:1663`, raises `RuntimeError` at `:1664` (msg `:1665-1667`) | Pre-v2 checkpoints lack the key entirely → this is the path they actually fire |
| Schema-mismatch raise (field-present-but-wrong-numeric ONLY) | `src/esper/simic/agent/ppo_agent.py:1675` (msg `:1677-1683`) | Only fires when the field is present at an older numeric value — an effectively-impossible window; NOT the path real old checkpoints hit |
| `CHECKPOINT_VERSION = 2` (NOT bumped) | `src/esper/simic/agent/ppo_agent.py:64` | Identical on both branches |
| kasmina slot `_SCHEMA_VERSION = 2` (NOT a break) | `src/esper/kasmina/slot.py:305` | Byte-identical both branches; diff only adds `blending_delta` |
| `normalizer_checkpoint.py` (new, no break) | `src/esper/simic/training/normalizer_checkpoint.py` | New file; **imports** `OBS_V3_FEATURE_SCHEMA_VERSION` from leyline (defined at `src/esper/leyline/__init__.py:635 = 1`; this file does NOT define the constant). |
| Pre-fix EV consumer branch (Sprint item 1 target) | `src/esper/simic/agent/ppo_agent.py:622-628` | Current `if var_returns > 1e-8: … else: torch.tensor(0.0)` branch — the EV gate-fix replaces/floors this. Its **absence is the structural co-landing signal** used in Step 5. |
| EV consumed by Karn view | `src/esper/karn/mcp/views.py:127` (`explained_variance`), `:606` (`run_confounders` view), `:649` (`true as proof_blocking`) | The false-alarm surface a missing gate-fix would trip on K>1 runs. |
| CI workflow | `.github/workflows/test-suite.yml` | lint → typecheck → property → unit+integration (75% coverage gate: command `:116`, threshold check `:121-130`) → overwatch-web → nightly cron (`-m slow` `:203`, `-m stress` `:209`) |
| CI unit-job marker filter | `.github/workflows/test-suite.yml:116` | `-m "not integration and not stress and not property and not slow" --cov=src` |
| Direct-dep alerts (4, none high) | `pyproject.toml` (`torch:7`, `transformers:12`, `pytest:77` dev) | Lockfile-resolvable; see §Dependency bump sub-steps |
| Required-runtime transitive (security-critical) | `pyproject.toml:11` (`datasets>=4.4.1`, core dep → pulls `pyarrow`) | pyarrow UAF (#77) must resolve `>=23.0.1`; see Step 8a VERIFY. `datasets` is core-runtime, so allow only `constraint-dependencies` inline (no `override-dependencies`). |
| Shipped package-data | `pyproject.toml:43-45` (section `:43`, entry `:45` `"esper.karn.overwatch" = ["web/dist/**/*"]`) | `web/dist/**` IS shipped package data. `web/.gitignore:2` ignores `dist/` (untracked), so a `git status --porcelain` guard will NOT show a regenerated `dist/` — inspect with `git status --ignored` or require a clean checkout for packaging; guarded in Step 8c. |

---

## Step 0 — BLOCKING GATE (FIRST): the EV-telemetry-robustness fix is structurally present on 0.1.1

> **This plan is NOT unconditionally ready to execute.** It hard-depends (Locked decision 5 / `depends_on: ev-telemetry-robustness`) on the EV fix being structurally present on `0.1.1`. At the time this plan was written, the current `0.1.1` HEAD still carries the OLD EV branch (`var_returns > 1e-8 → torch.tensor(0.0)` at `ppo_agent.py:624`) and does NOT contain the planned `ev_return_variance_floor` / `ev_low_return_variance` artifacts. **Execution MUST NOT begin until this gate passes.** This is the same structural check as Step 5, hoisted to the front so the plan cannot start while its hard dependency is unmet; Step 6 re-confirms it immediately pre-push.

- **PRECONDITION (RED-equivalent) — structural code-artifact check (NOT a commit-message grep):**
  ```bash
  # (1) OLD pattern absent: the bare `var_returns > 1e-8 → torch.tensor(0.0)` EV fall-through must be GONE.
  git -C /home/john/esper-lite show 0.1.1:src/esper/simic/agent/ppo_agent.py \
    | grep -nE "var_returns > 1e-8" \
    && echo "OLD-EV-BRANCH-STILL-PRESENT-STOP" || echo "OLD-EV-BRANCH-GONE-OK"

  # (2) NEW artifact present: the EV gate-fix introduces a named variance-floor / low-variance flag.
  git -C /home/john/esper-lite show 0.1.1:src/esper/simic/agent/ppo_agent.py \
    | grep -nEi "ev_return_variance_floor|return_variance_floor|ev_low_return_variance|return_std_floor" \
    && echo "NEW-EV-ARTIFACT-PRESENT-OK" || echo "NEW-EV-ARTIFACT-ABSENT-STOP"
  ```
  Expected: `OLD-EV-BRANCH-GONE-OK` **and** `NEW-EV-ARTIFACT-PRESENT-OK`. **If either prints its `-STOP` form, the EV fix has not landed on `0.1.1` — STOP. Do not begin execution of this plan.** (The exact named symbol is owned by the EV-telemetry-robustness plan; reconcile token (2) with its actual flag name before running.)
- **ACTION:** none (blocking gate only). When this gate passes, proceed to Step 1.
- **ROLLBACK:** N/A — a failed gate means the precondition for the whole plan is unmet; the plan does not start.

## Step 1 — PRECONDITION: re-verify FF topology and clean tree

> The entire plan hinges on `main` HEAD being the merge-base. A push to `main` since spec time would break this. Verify FIRST, immediately before doing anything.

- **PRECONDITION (RED-equivalent):**
  ```bash
  git -C /home/john/esper-lite fetch origin
  git -C /home/john/esper-lite rev-parse origin/main
  git -C /home/john/esper-lite merge-base origin/main 0.1.1
  git -C /home/john/esper-lite merge-base --is-ancestor origin/main 0.1.1 && echo "FF-OK" || echo "NOT-FF-STOP"
  git -C /home/john/esper-lite rev-list --count origin/main..0.1.1
  git -C /home/john/esper-lite rev-list --count 0.1.1..origin/main
  ```
- **Expected (must all hold, else STOP):**
  - `rev-parse origin/main` **equals** `merge-base origin/main 0.1.1` (currently `f7f1aecea90cef5b6c557da94e91b8889e5efacb`).
  - `is-ancestor` prints `FF-OK`.
  - `origin/main..0.1.1` count is `46` (a small drift up is fine — new commits on 0.1.1; the gate is FF-validity, not the exact count).
  - `0.1.1..origin/main` count is **`0`** (zero commits on main absent from 0.1.1). **If this is non-zero, the topology is no longer a pure FF — STOP and re-assess (do NOT `--no-ff` around it).**
- **ACTION:** none (verification gate only).
- **VERIFY:** working tree state — re-run `git status` at execution time and gate on it (do NOT assume a fixed expected file set; the live worktree drifts). At plan-write time the live status is `M docs/coord/PLAN_TRACKER.md` plus the five untracked planning docs (`docs/plans/ready/2026-06-18-*.md`, `docs/superpowers/specs/2026-06-18-*.md`) — there are **no** `findings.jsonl` scratch files:
  ```bash
  git -C /home/john/esper-lite status --porcelain
  ```
  **Gate:** the planning docs and the `docs/coord/PLAN_TRACKER.md` update MUST be either (a) committed on `0.1.1` so they ride the FF, or (b) explicitly excluded (stashed / left untracked and confirmed not staged). **Nothing uncommitted may be silently part of the merge.** Re-run this status check at execution time and reconcile every entry — a clean (or intentionally-curated) tree is the precondition, not a hardcoded filename list.
- **ROLLBACK:** N/A (no mutation). A failed precondition aborts the plan.

## Step 2 — PRECONDITION: confirm no in-flight run depends on a pre-v2 checkpoint

> The checkpoint break (VALUE_HEAD_SCHEMA_VERSION=2) rejects all pre-v2 PPO checkpoints with no remap. This must be confirmed BEFORE the merge lands, not assumed (spec §2 item 3, acceptance #3).

- **PRECONDITION (RED-equivalent):** confirm the break exists and is enforced exactly as documented (guards against a silent drift in the enforcement path):
  ```bash
  git -C /home/john/esper-lite show 0.1.1:src/esper/leyline/__init__.py | grep -n "VALUE_HEAD_SCHEMA_VERSION"
  git -C /home/john/esper-lite show origin/main:src/esper/leyline/__init__.py | grep -n "VALUE_HEAD_SCHEMA_VERSION"
  ```
  Expected: on `0.1.1`, `VALUE_HEAD_SCHEMA_VERSION = 2` appears (definition + export). On `origin/main`, **empty output** (the field does not exist on main — confirms this is a net-new break, not a re-bump).
- **ACTION:** human/owner confirmation checklist (record the answers in the PR body):
  1. No long-running or in-flight PPO run on `main` depends on a pre-v2 op-conditioned value-head checkpoint that cannot be restarted. (The 200-ep EV-liftoff experiment is already on the 0.1.1 estimator → expected no-op confirmation, but it must be explicitly confirmed, not assumed.)
  2. No external automation/CI restores a pre-v2 checkpoint as a fixture.
- **VERIFY (a — both rejection paths are actionable; sanity-read, no code change):**
  ```bash
  # The REAL path for genuine pre-v2 checkpoints: they lack the value_head_schema_version
  # key entirely, so the try-block read at :1656 raises through the except KeyError at :1663-1668.
  git -C /home/john/esper-lite show 0.1.1:src/esper/simic/agent/ppo_agent.py | sed -n '1654,1668p'
  # The field-present-but-wrong-numeric path (effectively-impossible window), :1675-1683:
  git -C /home/john/esper-lite show 0.1.1:src/esper/simic/agent/ppo_agent.py | sed -n '1675,1683p'
  ```
  Expected: the KeyError path (`:1664`) raises `RuntimeError("Incompatible checkpoint format: missing required field … Please retrain …")` — **this is the path a real pre-v2 checkpoint hits.** The schema-mismatch raise (`:1675`) states the value head was split into an op-independent `state_value_head`, there is no remap, and instructs "Please retrain" — but it only fires when the field is present at an older numeric value. If either message is missing or unclear, fix it on `0.1.1` *before* merge (the error UX is part of the migration story).
- **VERIFY (b — fixture grep, from the design's §2-check; required before merge):** confirm no checkpoint-loading test fixture assumes a pre-v2 schema-version or bundles a pre-P0-1 `.pt` artifact:
  ```bash
  grep -rn 'load_checkpoint\|load_state_dict\|ppo_agent.*checkpoint' tests/ \
    --include='*.py' | grep -v __pycache__
  ```
  Expected: no fixture loads a V(s)=1 / pre-v2 checkpoint (such a fixture would error loudly at load). CI fixture generation has no visibility into this plan and could restore a stale artifact, so this must be confirmed by the grep at execution time, not assumed.
- **ROLLBACK:** N/A (verification + comms only). **Do NOT write a migration/remap shim** — it is prohibited by the No-Legacy policy and structurally impossible (no `state_value_head.*` weights exist in a pre-v2 checkpoint). "Retrain" is the contract.

## Step 3 — PRECONDITION: all five CI lanes green locally on 0.1.1

> A bare `uv run pytest` is **NOT** the full suite. `pytest.ini:24` injects `addopts = … -m "not integration and not stress and not property and not slow"`, so the default invocation silently EXCLUDES four of the five CI lanes. The nightly-only jobs (slow/stress/thorough-property) do NOT gate the PR, and the push-to-main trigger would otherwise be the first to exercise the recurrent-PPO and transaction-phase-refactor paths under those markers. Running every lane explicitly is the single most important pre-merge gate (Locked decision 4, acceptance #4).

- **PRECONDITION (RED-equivalent):** working tree is on `0.1.1` HEAD and clean (re-confirm Step 1's `git status --porcelain`). `uv sync` is current (pre-bump lock):
  ```bash
  git -C /home/john/esper-lite rev-parse --abbrev-ref HEAD   # expect: 0.1.1
  uv --directory /home/john/esper-lite sync
  ```
- **ACTION (a — run each CI lane EXPLICITLY, mirroring `.github/workflows/test-suite.yml`):** each lane is a separate marker pass; do NOT rely on a bare `pytest` (it runs only the default/unit lane). The `--collect-only` assertion before each non-default lane guards against a config/env/GPU-gating skip making the lane hollow.
  ```bash
  # default/unit lane (CI: test-suite.yml:116) — note --cov=src, NOT --cov=esper
  uv --directory /home/john/esper-lite run pytest \
    -m "not integration and not stress and not property and not slow" \
    --cov=src --cov-report=json:.coverage-reports/coverage.json --cov-report=term-missing

  # integration lane (CI: test-suite.yml:137)
  uv --directory /home/john/esper-lite run pytest --collect-only -m integration -q | tail -3   # must list >0 tests
  uv --directory /home/john/esper-lite run pytest -m integration -v

  # property lane with HYPOTHESIS_PROFILE (CI: test-suite.yml:193-197)
  uv --directory /home/john/esper-lite run pytest --collect-only -m property -q | tail -3       # must list >0 tests
  HYPOTHESIS_PROFILE=thorough uv --directory /home/john/esper-lite run pytest -m property -v --hypothesis-show-statistics

  # slow lane (CI: test-suite.yml:203)
  uv --directory /home/john/esper-lite run pytest --collect-only -m "slow and not integration and not stress and not property" -q | tail -3   # must list >0 tests
  uv --directory /home/john/esper-lite run pytest -m "slow and not integration and not stress and not property" -v

  # stress lane (CI: test-suite.yml:209)
  uv --directory /home/john/esper-lite run pytest --collect-only -m stress -q | tail -3          # must list >0 tests
  uv --directory /home/john/esper-lite run pytest -m stress -v
  ```
- **ACTION (b — assert each non-default lane actually RAN, not silently skipped):** every `--collect-only` above must list **>0 tests** (not "no tests ran"), and each `-v` lane run must show its tests as `PASSED` (not `SKIPPED`). If any lane collects empty or is all-skipped, STOP — that lane's gate is hollow and the post-push CI (nightly for slow/stress/property) would be the first real exercise.
- **ACTION (c — apply the CI 75% coverage gate to the default/unit lane):** the default/unit lane in (a) emits `.coverage-reports/coverage.json` with `--cov=src` (the CI target at `test-suite.yml:116`, **not** `--cov=esper`). Apply the same threshold check the CI uses (`:121-130`):
  ```bash
  python3 -c "
import json, sys
data = json.load(open('/home/john/esper-lite/.coverage-reports/coverage.json'))
cov = data['totals']['percent_covered']
print(f'Coverage: {cov:.1f}%')
sys.exit(0 if cov >= 75 else 1)
"
  ```
- **ACTION (d — simic domain-scoped coverage, visibility guard):** the fixed project-wide 75% gate can mask drift in the high-complexity new simic code. Run the domain-scoped check and assert it does not drop below its tracked baseline:
  ```bash
  uv --directory /home/john/esper-lite run pytest tests/simic/ --cov=src/esper/simic --cov-report=term-missing
  ```
- **VERIFY (GREEN):**
  - Each lane in (a) exits 0, no failures, no errors. Pay specific attention to:
    - `tests/simic/` (recurrent PPO PR1/PR2, P0-1 V(s), transaction-phase refactor — highest-risk domain).
    - `tests/integration/test_ev_liftoff_k4.py` (the EV-liftoff gate — integration lane).
    - Any test asserting `value_head_schema_version` round-trip.
  - Step 3b: every non-default lane collected >0 tests AND those tests ran (PASSED, not SKIPPED).
  - Step 3c: the `python3` threshold check exits 0 (coverage ≥ 75% on the CI-mirrored default/unit marker set, `--cov=src`). If new code paths dipped coverage, surface and address on `0.1.1` BEFORE the PR — do not rely on PR CI to surface it.
  - Step 3d: simic domain-scoped coverage holds against its baseline.
- **ROLLBACK:** if any sub-step is red, **STOP** — do not open the PR. Fix on `0.1.1` (a real regression on the 46-commit unit must be fixed at source, not merged red). This is the abort gate.

## Step 4 — ACTION: open the integration PR (0.1.1 → main)

> The PR exists so the `pull_request` CI jobs gate the integration before the push (acceptance #5). The merge is landed by **CLI fast-forward only** (`git checkout main && git merge --ff-only 0.1.1 && git push`, Step 6) — the PR is the CI gate, NOT the merge mechanic. Do NOT use any GitHub merge button (Rebase/Squash/Merge-commit): the rebase button rewrites SHAs and breaks SHA-equality verification (Locked decision 1).

- **PRECONDITION:** Step 0 (EV-fix structural gate) and Steps 1–3 all green (EV fix present on 0.1.1, FF-valid, checkpoint confirmed, all five CI lanes green incl. slow/stress collected non-empty + ran and coverage ≥ 75%). Confirm the divergence inventory section exists in the source spec before drafting the body (mechanical anti-drift gate):
  ```bash
  grep -ci "divergence" docs/superpowers/specs/2026-06-18-main-merge-integration-design.md   # expect >= 1
  ```
  Expected: non-zero — the by-domain risk table the PR body must reproduce is present in the spec. If zero, the spec drifted; reconcile before opening the PR.
- **ACTION:**
  ```bash
  gh pr create --repo tachyon-beep/esper-lite --base main --head 0.1.1 \
    --title "Integrate 0.1.1 → main (FF: recurrent PPO + P0-1 V(s) + refactors)" \
    --body-file -   # body content below
  ```
  **PR body MUST contain (acceptance #3, #8):**
  - **Topology statement:** "This is a fast-forward (`merge-base origin/main 0.1.1 == origin/main HEAD == f7f1aece`). Zero conflict surface. Landing mechanic is **CLI fast-forward only** (`git checkout main && git merge --ff-only 0.1.1 && git push`) with SHA-equality verification — **NOT** a GitHub merge button (Rebase/Squash/Merge-commit). This PR is the CI gate, not the merge mechanic."
  - **Breaking change callout (prominent, top of body):** "⚠️ CHECKPOINT BREAK: `VALUE_HEAD_SCHEMA_VERSION = 2` (`src/esper/leyline/__init__.py:125`). Real pre-v2 PPO checkpoints lack the `value_head_schema_version` key entirely and are rejected with a fast `RuntimeError` via the **missing-key path** (`ppo_agent.py:1664`); the schema-mismatch raise (`:1675`) only fires for the effectively-impossible field-present-but-wrong-numeric case. **Migration = retrain, no remap** (No-Legacy policy; no migration script exists or will be written). This is NOT a `CHECKPOINT_VERSION` bump (that stays `= 2`)."
  - **Checkpoint confirmation checklist** (from Step 2): boxes confirming no in-flight run depends on a pre-v2 checkpoint.
  - **Divergence inventory** (the by-domain risk table from spec §3 — reproduce it, do not just link).
  - **Rollback path:** "Rollback = forward `git revert` of the offending commit(s). No `git reset --hard` / `git push --force` on `main` (CLAUDE.md Git Safety)."
  - **Dependency-bump note:** "A trailing, independently-revertable dependency-bump commit (uv.lock + npm) lands AFTER this FF is green (Sprint item 3) — see plan §Dependency bump sub-steps."
  - **EV co-landing note:** "Sprint item 1's EV-consumer gate-fix lands before or with this merge so K>1 runs on main do not false-alarm (Locked decision 5; enforced structurally in Step 5)."
- **VERIFY (GREEN):**
  - The PR body actually contains the divergence inventory (mechanical check):
    ```bash
    gh pr view --repo tachyon-beep/esper-lite <pr-number> --json body -q .body | grep -ci "divergence"   # expect >= 1
    ```
  - All gating CI jobs pass on the PR — lint, typecheck, property-tests, unit-and-integration-tests (incl. the 75% coverage gate), overwatch-web-tests:
    ```bash
    gh pr checks --repo tachyon-beep/esper-lite <pr-number>
    ```
    Expected: all checks green. If a nightly-only path regressed (it should not, given Step 3), it will not surface here — Step 3 is the backstop.
- **ROLLBACK:** if PR CI is red, close/leave-open the PR and fix on `0.1.1`; do not merge.

## Step 5 — PRECONDITION (pre-push gate): the EV-consumer gate-fix is structurally present on 0.1.1

> Locked decision 5 / acceptance #6: under the new op-marginal V(s), a low-return-variance K>1 batch can produce an artefactual EV that makes the upstream detector emit a `VALUE_COLLAPSE_DETECTED` event. The `run_confounders` view (`views.py:606`, `proof_blocking` at `:649`) **correctly** marks any emitted `VALUE_COLLAPSE_DETECTED` row as proof-blocking — that is the right semantics for a genuinely emitted collapse and **must NOT be suppressed at the view layer.** The fix is therefore **upstream**: the EV gate-fix prevents the detector/coordinator from emitting the artefactual collapse in the first place. That gate-fix MUST be present on `0.1.1` so it rides this FF. **This check is purely PRE-PUSH** — it runs and must pass BEFORE Step 6 (the irreversible FF push). A false pass here lets a healthy K>1 run on main emit a spurious collapse that the view then (correctly) proof-blocks, corrupting the experiment verdict — the hardest-to-reverse failure in the sprint.

> **Why NOT a commit-message grep:** the previously-proposed `git log 0.1.1 | grep -i 'ev|explained_variance'` is structurally unsound and would false-pass today. Existing commits `177a53aa` ("wire q_aux_loss + head_q_grad_norm telemetry end-to-end") and `6a27b8e3` ("op-independent V(s) baseline") already match that grep but are NOT the EV-consumer gate-fix; conversely a fix titled e.g. "fix(simic): tighten value-collapse gate semantics" would false-NEGATIVE. A commit-message search can never gate "the specific fix is present." Gate on a **code artifact** instead.

- **PRECONDITION (RED-equivalent) — structural code-artifact check:** the current consumer at `ppo_agent.py:622-628` contains the un-floored branch the gate-fix removes:
  ```python
  var_returns = valid_returns.var()
  if var_returns > 1e-8:
      ev_tensor = 1.0 - (valid_returns - raw_values).var() / var_returns
      explained_variance = ev_tensor
  else:
      explained_variance = torch.tensor(0.0, device=valid_returns.device)
  ```
  The EV-telemetry-robustness fix replaces this low-variance fall-through (which is what produces the K>1 false alarm) with a variance-floored / flag-bearing computation. Confirm the fix has landed by checking the OLD pattern is **gone** from `0.1.1` and the NEW artifact is **present**:
  ```bash
  # (1) OLD pattern absent: the bare `else: torch.tensor(0.0)` EV fall-through must no longer exist.
  git -C /home/john/esper-lite show 0.1.1:src/esper/simic/agent/ppo_agent.py \
    | grep -nE "var_returns > 1e-8" \
    && echo "OLD-EV-BRANCH-STILL-PRESENT-STOP" || echo "OLD-EV-BRANCH-GONE-OK"

  # (2) NEW artifact present: the gate-fix introduces a named variance-floor / low-variance flag
  #     (e.g. ev_return_variance_floor / return_variance_floor / ev_low_variance flag). Confirm at
  #     least one such named symbol exists in the post-fix consumer.
  git -C /home/john/esper-lite show 0.1.1:src/esper/simic/agent/ppo_agent.py \
    | grep -nEi "return_variance_floor|ev_return_variance_floor|ev_low_variance|return_std_floor" \
    && echo "NEW-EV-ARTIFACT-PRESENT-OK" || echo "NEW-EV-ARTIFACT-ABSENT-STOP"
  ```
  Expected: `OLD-EV-BRANCH-GONE-OK` **and** `NEW-EV-ARTIFACT-PRESENT-OK`. **If either prints its `-STOP` form, the EV gate-fix is NOT present — STOP. Do not push the FF (Step 6) until it is co-landed.** This plan does not author the fix; it enforces the sequencing constraint only.
  > Note: the exact named symbol is owned by the EV-telemetry-robustness plan. When that plan lands, reconcile the grep token in (2) above with its actual ctor kwarg / flag name (read its plan's Step 1.x). The non-negotiable invariant is: the bare `var_returns > 1e-8 → torch.tensor(0.0)` fall-through is gone, replaced by a floored/flagged computation.
- **ACTION (preferred, strongest signal):** run the EV-telemetry-robustness plan's acceptance test, which locks the floored-EV behaviour, and require green:
  ```bash
  uv --directory /home/john/esper-lite run pytest tests/simic/test_ppo_value_metrics.py -q
  ```
  Expected: green. A passing behavioural test is a stronger co-landing signal than any grep; if the EV plan named its test (e.g. `-k ev_floored`), narrow to it. If `test_ppo_value_metrics.py` has no EV-floor assertion yet, the fix has not fully landed — STOP.
- **VERIFY (post-push confirmation, secondary):** the Step 7 K=4 smoke on post-merge `main` raises no EV-keyed false alarm. This is a *second* confirmation AFTER the push, not the gate — the gate is the pre-push structural check above.
- **ROLLBACK:** N/A (pre-push gate only; a failed gate aborts before Step 6 and nothing is pushed).

## Step 6 — ACTION: execute the fast-forward

> The merge mechanic. Re-verify topology one final time immediately before pushing (a push to main between Step 1 and now invalidates the FF). Step 5's structural EV gate MUST have passed before reaching here.

- **PRECONDITION (RED-equivalent, re-run Step 1's core check + re-confirm Step 5 passed):**
  ```bash
  git -C /home/john/esper-lite fetch origin
  git -C /home/john/esper-lite merge-base --is-ancestor origin/main 0.1.1 && echo "FF-OK" || echo "NOT-FF-STOP"
  git -C /home/john/esper-lite rev-list --count 0.1.1..origin/main   # expect 0
  ```
  Expected: `FF-OK` and `0`. **If not, STOP** — a push to main occurred; re-assess (the topology is no longer a pure FF; do not force, do not `--no-ff` blindly). Do NOT proceed unless Step 5's pre-push EV gate passed.
- **ACTION (only with FF + Step 5 EV gate confirmed) — CLI fast-forward, the single locked mechanic:** land via the CLI fast-forward (it preserves the 46 curated commit SHAs byte-for-byte, the only mechanic under which SHA-equality verification holds). **Do NOT use any GitHub merge button** (the Rebase button replays with new SHAs and breaks verification; Squash/merge-commit are rejected by Locked decision 1):
  ```bash
  git -C /home/john/esper-lite checkout main
  git -C /home/john/esper-lite merge --ff-only 0.1.1
  git -C /home/john/esper-lite push
  ```
  `--ff-only` **fails loudly** if the topology is not a pure FF — that failure is the safety net, not an error to work around.
- **VERIFY (GREEN, acceptance #2):**
  ```bash
  git -C /home/john/esper-lite log --oneline origin/main | head -50
  ```
  Expected: the 46 individual curated commits (P0-1 `6a27b8e3`, PR1 `356008c4`, PR2 `b9218273`, the refactor sequence, P0-2 `cf2cf549`, P0-3 `cf57d94a`) appear individually — **no squash blob, no synthetic `--no-ff` merge commit**. Confirm `git rev-parse origin/main == git rev-parse 0.1.1` (FF landed exactly).
- **ROLLBACK:** if the push lands but post-push CI (push-to-main trigger) goes red, **forward-revert** the offending commit(s) (`git revert <sha>`), push the revert, and fix on a branch. **Do NOT `git reset --hard` / `git push --force` on `main` without explicit user permission** (CLAUDE.md Git Safety). A messy forward-revert history is acceptable; lost/rewritten main history is not.

## Step 7 — VERIFY: post-push CI + K>1 smoke (acceptance #5, #6)

- **ACTION:** the push-to-main trigger runs the full PR-equivalent suite on `main` automatically. Monitor it:
  ```bash
  gh run list --repo tachyon-beep/esper-lite --branch main --limit 3
  gh run watch --repo tachyon-beep/esper-lite <run-id>
  ```
- **VERIFY (GREEN):**
  - All push-to-main CI jobs pass (lint, typecheck, property, unit+integration with 75% gate, overwatch-web).
  - Run the EV-liftoff integration test as a K=4 smoke vehicle:
    ```bash
    uv --directory /home/john/esper-lite run pytest -m integration tests/integration/test_ev_liftoff_k4.py -q
    ```
    **Scope note (do not overclaim):** `tests/integration/test_ev_liftoff_k4.py` asserts only the **EV trajectory/liftoff** (K=4 lifts EV past 0.3, K=1 stays pinned — see `:135`); it does **NOT** emit or query Karn telemetry and therefore does NOT by itself prove "no proof-blocking Karn row". Expected here: **the EV-liftoff integration test passes.**
  - **Separate Karn assertion (closes acceptance #6 with item 1's fix landed; second confirmation of Step 5):** to confirm no false alarm at the telemetry layer, explicitly query the `run_confounders` view (`views.py:606`, `proof_blocking` at `:649`) for the healthy low-variance K=4 case and assert **no `VALUE_COLLAPSE_DETECTED` proof-blocking row** is emitted. This is a distinct check from the EV test above — the EV test does not cover it.
- **ROLLBACK:** forward-revert (see Step 6) if CI is red post-push.

## Step 8 — ACTION: dependency bump pass (LAST commit, independently revertable)

> **Ordering (explicit, W2):** this is a **post-merge direct-main commit** — it lands on `main` AFTER the FF push (Step 6) and the post-push CI (Step 7) are green. It does NOT ride the integration PR's CI. It is pushed directly to `main` and validated by the push-to-main CI re-run (Step 8c VERIFY). Keeping the bumps as a single trailing commit makes a dependency regression independently revertable from the 46-commit functional FF body. (If a bump must instead ride PR CI, it would have to land on `0.1.1` BEFORE Step 4 — but Locked decision 6 sequences it LAST, so the direct-main path is the chosen one.)
>
> This is a **dependency manifest/lockfile commit**, not strictly "lockfile-only": a `[tool.uv] constraint-dependencies` cap mutates `pyproject.toml`, and the npm step mutates `package.json` + lockfile (W3).

### 8a — Python (uv.lock + optional [tool.uv] constraint cap)

- **PRECONDITION (RED-equivalent):** confirm the direct-dep alerts need no pin edits — the declared constraints are already broad enough (`>=`) that patched versions are lockfile-reachable. Also surface `datasets` (which pulls the security-critical `pyarrow`):
  ```bash
  grep -nE "torch|pytest|transformers|datasets" /home/john/esper-lite/pyproject.toml
  ```
  Expected: `torch>=2.8.0` (allows `>=2.10` → clears #79), `transformers>=4.57.3` (`pyproject.toml:12`; `uv.lock` currently resolves `4.57.3`), `datasets>=4.4.1` (`pyproject.toml:11`, **core runtime**, pulls `pyarrow`), `pytest>=7.0.0` (dev). **No edit required** at this stage.
- **OVERRIDE-ENTRY DISCIPLINE GATE (mandatory, B11):** `datasets` is a **core `[project].dependencies` entry** and `pyarrow` is a runtime transitive through it. An inline `[tool.uv] override-dependencies` floor can bypass upstream dependency metadata and produce a resolver solution outside what the upstream package declares compatible — **too permissive for a core-runtime transitive.**
  - **Allow only `[tool.uv] constraint-dependencies` inline, and only if the resolver still solves normally** (`uv lock --upgrade` succeeds without an override). A single `constraint-dependencies` floor (e.g. `pyarrow>=23.0.1`) is acceptable inline, annotated in the commit body.
  - **If `override-dependencies` is truly required** (the resolver cannot solve with a constraint alone), **DEFER the entire dependency bump to a separate reviewed window** with targeted `datasets` / CIFAR / TinyStories smoke tests — do NOT land an `override-dependencies` entry in this trailing commit. Overrides are architectural concessions that need their own review.
- **ACTION:**
  ```bash
  uv --directory /home/john/esper-lite lock --upgrade
  uv --directory /home/john/esper-lite sync
  ```
  This pulls patched transitives: GitPython→3.1.50 (via wandb), urllib3→2.7.0, pillow→12.2.0, pyarrow→23.0.1, starlette/python-multipart (via fastapi), tornado/jupyter chain, cryptography→48.0.1, pyjwt→2.13.0, mistune, aiohttp, idna.
  - **Do NOT** bump `transformers` to `5.0.0rc3` (pre-release — leave the medium alert open until 5.0 GA). Also **do NOT** let it advance to a new minor (`4.58+`): the resolution must satisfy `transformers >=4.57.3,<4.58.0` (B10). If `uv lock --upgrade` tries to pull an rc or a `4.58+` minor, cap it via a `[tool.uv] constraint-dependencies` entry (`transformers<4.58`) rather than touching `dependencies`; if a constraint cannot hold it, DEFER the transformers bump to a separate reviewed validation window (a tokenizer-behavior change could subtly alter `datasets`-backed CIFAR loading).
  - **`torch` #80 has no fix yet** — accept/snooze with a note in the PR/commit body. `#79` clears once the lock picks up `>=2.10`.
  - If a transitive refuses to advance (held by an upstream constraint), prefer a `[tool.uv] constraint-dependencies` floor — **never** by editing `dependencies`, and **never** an `override-dependencies` entry for a core-runtime transitive (defer the bump instead, per the discipline gate above).
- **VERIFY (GREEN) — assert the security-critical transitives actually advanced (do not infer from a green test run):**
  ```bash
  uv --directory /home/john/esper-lite run pytest -m "not slow and not stress" -q   # fast re-validation post-bump
  git -C /home/john/esper-lite diff --stat uv.lock                                   # confirm version-only delta

  # Explicit version assertions for EVERY high/critical Python package expected to clear
  # (B10: not just pyarrow + urllib3 — assert the whole cluster), plus the transformers cap:
  uv --directory /home/john/esper-lite run python -c "
import importlib.metadata as m
from packaging.version import Version
def v(pkg): return Version(m.version(pkg))
# transformers MUST stay a patch within 4.57.x (>=4.57.3,<4.58.0):
tf = v('transformers')
assert Version('4.57.3') <= tf < Version('4.58.0'), f'transformers {tf} outside [4.57.3,4.58.0)'
# Patched floors for every high/critical package expected to clear:
floors = {
    'pyarrow': '23.0.1',      # UAF #77 (high) — core-runtime via datasets
    'urllib3': '2.7.0',
    'cryptography': '48.0.1',
    'GitPython': '3.1.50',
    'pillow': '12.2.0',
    'starlette': '0.40.0',    # confirm against the live Dependabot advisory floor at execution time
    'python-multipart': '0.0.18',  # confirm against live advisory floor
    'tornado': '6.5',         # confirm against live advisory floor
    'pyjwt': '2.13.0',
    'mistune': '3.1.0',       # confirm against live advisory floor
}
missing = []
for pkg, floor in floors.items():
    try:
        cur = v(pkg)
    except m.PackageNotFoundError:
        missing.append(pkg); continue
    assert cur >= Version(floor), f'{pkg} {cur} < {floor} (high/critical alert NOT cleared)'
print('transformers', tf, '| not-installed (extras/transitive absent):', missing)
print('HIGH-CRITICAL-CLUSTER-OK')
"
  ```
  Expected: tests green; `uv.lock` delta is version-string changes only (no new/removed deps); the assertion script prints `HIGH-CRITICAL-CLUSTER-OK`. **Confirm each floor against the live Dependabot advisory at execution time** (the floors above are starting points; some packages are optional/extras and may be absent — that is acceptable, but any present high/critical package MUST clear its advisory floor). **If `pyarrow` fails to reach `>=23.0.1` because `datasets` pins it lower, add a single `[tool.uv] constraint-dependencies` floor (`pyarrow>=23.0.1`) and re-run — do NOT silently accept the lockfile change while the high-severity runtime UAF remains** (constraint only; an `override-dependencies` need defers the bump per the discipline gate). Also cross-check against the Dependabot API so no expected high/critical is missed:
  ```bash
  gh api /repos/tachyon-beep/esper-lite/dependabot/alerts --paginate \
    -q '.[] | select(.security_advisory.severity=="high" or .security_advisory.severity=="critical") | .dependency.package.name' | sort -u
  ```

### 8b — npm (Overwatch web — current high/critical alerts are dev/build tooling; node_modules not shipped)

> **Scope (W5):** the **current** high/critical npm alerts are dev/build tooling (vite/vitest), and `node_modules` are not shipped. This is NOT a claim that the entire Overwatch npm surface is dev-only: the web project carries runtime dependencies (Vue/ECharts, `package.json` `dependencies`), and the built `web/dist/**` IS shipped package data (`pyproject.toml:45`). Scope the bump to the dev/build-tooling alerts; do not treat runtime Vue/ECharts bumps as "never shipped".

- **PRECONDITION:** confirm the packages being bumped are devDependencies (the lone critical + 3 highs are JS dev-build tooling), distinct from the runtime `dependencies` (Vue/ECharts):
  ```bash
  grep -nE "vite|vitest" /home/john/esper-lite/src/esper/karn/overwatch/web/package.json
  ```
  Expected: `vite`/`vitest` under `devDependencies` (runtime `echarts`/`vue`/`vue-echarts` are under `dependencies` and are NOT what this step bumps).
- **ACTION:**
  ```bash
  npm --prefix /home/john/esper-lite/src/esper/karn/overwatch/web install vite@^7.3.5 vitest@^4.1.0
  npm --prefix /home/john/esper-lite/src/esper/karn/overwatch/web audit fix
  ```
  Clears vitest (critical, GHSA-5xrq-8626-4rwp), vite (3 highs), js-cookie, minimatch, postcss.
  > Note: `npm install <pkg>@<ver>` mutates **both** `package.json` (devDependencies range) and `package-lock.json`. Both must be staged in 8c, else `npm ci` restores the old version.
- **VERIFY (GREEN):**
  ```bash
  # (1) Confirm the EXACT installed versions are in the patched range (audit-clean alone is insufficient —
  #     a newer in-range version could open a NEW advisory while the old one shows cleared):
  npm --prefix /home/john/esper-lite/src/esper/karn/overwatch/web list vite vitest --depth=0
  #     Expect: vite >= 7.3.5, vitest >= 4.1.0.

  # (2) Run the Vitest component unit tests BEFORE building/committing — a Vitest major/minor bump can
  #     change snapshot serialization or assertion APIs and break unit tests silently:
  npm --prefix /home/john/esper-lite/src/esper/karn/overwatch/web test -- --run

  # (3) Build + E2E + audit:
  npm --prefix /home/john/esper-lite/src/esper/karn/overwatch/web run build
  npx --prefix /home/john/esper-lite/src/esper/karn/overwatch/web playwright test   # or the overwatch-web CI job re-run
  npm --prefix /home/john/esper-lite/src/esper/karn/overwatch/web audit             # confirm critical/high cleared
  ```
  Expected: installed versions in range; Vitest unit tests green; build + Playwright green; audit shows critical/high cleared.

### 8c — Commit the bumps as ONE trailing, revertable commit

- **PRECONDITION (guard the manifest/lockfile-only invariant, W4):** `web/dist/**` is shipped package-data (`pyproject.toml:43-45`; the Overwatch entry is at `:45`), but `web/.gitignore:2` **ignores `dist/`** — so a `git status --porcelain` will NOT show a regenerated `dist/`, yet the 8b `npm run build` step DID write into the source tree, and a build/package operation can still pick up those generated files. **A plain `git status --porcelain` guard is insufficient.** Inspect with `--ignored` (or require a clean checkout for packaging):
  ```bash
  git -C /home/john/esper-lite status --porcelain
  git -C /home/john/esper-lite status --ignored -- src/esper/karn/overwatch/web/dist   # surfaces the build-regenerated, gitignored dist
  git -C /home/john/esper-lite diff --stat
  ```
  Expected from `--porcelain`/`diff --stat`: **only** `uv.lock`, `src/esper/karn/overwatch/web/package-lock.json`, and `src/esper/karn/overwatch/web/package.json` (plus `pyproject.toml` ONLY if a single approved `[tool.uv] constraint-dependencies` floor was added in 8a). The `--ignored` check will likely show `src/esper/karn/overwatch/web/dist/` as ignored build output — that is expected and it will NOT be committed (it is gitignored), but **packaging from this working tree would still pick it up**. If shipping a release artifact from here, build `dist` from a clean checkout (or regenerate and `git clean` it deliberately) rather than relying on whatever the 8b build left behind. If `dist/` is ever tracked, commit it in a SEPARATE, clearly-labelled commit — never bundle it into the bump commit. STOP and resolve before staging.
- **ACTION:**
  ```bash
  git -C /home/john/esper-lite add \
    uv.lock \
    src/esper/karn/overwatch/web/package-lock.json \
    src/esper/karn/overwatch/web/package.json
  # If 8a added a single [tool.uv] pyarrow floor, also: git add pyproject.toml
  git -C /home/john/esper-lite commit -F - <<'EOF'
chore(deps): bump vulnerable transitives + npm dev tooling (Dependabot triage)

- Python: uv lock --upgrade pulled patched transitives (pyarrow>=23.0.1 UAF #77,
  urllib3>=2.7.0, GitPython, pillow, cryptography, pyjwt, ...). transformers left
  pre-GA (5.0.0rc3 not adopted); torch #80 has no upstream fix (snoozed).
- npm (Overwatch web devDependencies; node_modules not shipped): vite>=7.3.5,
  vitest>=4.1.0 + audit fix. Cleared vitest critical (GHSA-5xrq-8626-4rwp),
  vite 3 highs, js-cookie, minimatch, postcss. (Runtime Vue/ECharts untouched;
  web/dist/** is shipped package data and was not regenerated into this commit.)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
  git -C /home/john/esper-lite push origin main
  ```
- **VERIFY (GREEN):** push-to-main CI re-runs green on the bump commit. Dependabot alert count on the default branch drops:
  ```bash
  gh api /repos/tachyon-beep/esper-lite/dependabot/alerts --paginate
  ```
  Expected: the high/critical cluster clears; remaining open = `transformers` medium + `torch` #80 low, both annotated. Confirm the commit changed only the expected files (`git show --stat HEAD`).
- **ROLLBACK:** the bumps are a **single trailing commit** — a dependency regression is reverted with `git revert <bump-sha>` (forward-revert), isolated entirely from the 46-commit functional FF body. No `--hard`/`--force`.

---

## Acceptance-criterion → verification traceability

| # | Spec acceptance criterion | Verifying step / command |
|---|---------------------------|--------------------------|
| 1 | Topology verified (merge-base == main HEAD; is-ancestor exits 0) | Step 1 PRECONDITION + Step 6 re-verify |
| 2 | Fast-forward preserved (46 commits, no squash/merge node) | Step 6 VERIFY (`git log --oneline origin/main`) |
| 3 | Checkpoint break documented + confirmed (`VALUE_HEAD_SCHEMA_VERSION=2`, retrain-no-remap, no in-flight dependency) | Step 2 + Step 4 PR-body checklist |
| 4 | All five CI lanes green pre-PR (default/unit, integration, property, slow, stress — each collect-only-asserted non-empty; coverage ≥ 75% on `--cov=src` CI marker set; simic domain-scoped check) | Step 3a/3b/3c/3d |
| 5 | PR CI green (lint, typecheck, property, unit+integration 75% gate, overwatch-web); PR body carries divergence inventory | Step 4 VERIFY (`gh pr checks` + body grep) |
| 6 | EV gate-fix structurally present (first blocking gate + re-confirmed pre-push); no EV false-alarm on K>1 smoke | Step 0 blocking gate + Step 5 pre-push structural gate + Step 7 K=4 smoke + Karn assertion |
| 7 | Dependency bumps land last + clean; full high/critical cluster asserted at patched floors (pyarrow≥23.0.1, urllib3, cryptography, GitPython, pillow, starlette, python-multipart, tornado, pyjwt, mistune); transformers `>=4.57.3,<4.58.0` (not pre-release, not minor); `override-dependencies` deferred; torch #80 annotated | Step 8a/8b/8c |
| 8 | Rollback path stated (forward-revert; no `--hard`/`--force` on main) | Step 6 + Step 8c ROLLBACK; Step 4 PR body |

## Sequencing & dependencies

1. **Step 1 (FF topology) and Step 6 (re-verify) bracket the entire operation** — the FF assumption is re-checked immediately before the only mutating push.
2. **Step 2 (checkpoint confirmation) precedes everything mutating** — a stranded-checkpoint discovery aborts before any landing.
3. **Step 3 (all five CI lanes locally) is the hard abort gate** before the PR — each lane runs explicitly (a bare `uv run pytest` excludes four of five lanes per `pytest.ini:24`); the nightly-only paths are exercised here (and *proven to collect non-empty + run*, 3b), not first on post-push CI; the CI 75% coverage gate is mirrored locally with `--cov=src` (3c) plus the simic domain-scoped check (3d).
4. **Step 0 / Step 5 (EV gate-fix co-landing) is a HARD dependency** on the EV-telemetry-robustness plan. It is the **FIRST blocking execution gate (Step 0)** — the plan is NOT unconditionally ready and must not start while the current `0.1.1` HEAD still carries the old EV branch (`ppo_agent.py:624`) — and is re-confirmed **pre-push by the same structural code-artifact gate (Step 5)** (old EV branch gone + new floor/flag present + acceptance test green), NOT by a commit-message grep, and NOT deferred to Step 7. Step 7's K=4 smoke + Karn assertion is a second, post-push confirmation only. This plan enforces the constraint but does not author the fix.
5. **Step 8 (dependency bumps) is strictly LAST** — a **post-merge direct-main commit** after the functional FF and post-push CI are green (W2) — a single, independently-revertable trailing commit, with the override-dependencies discipline gate (8a) deferring the whole bump to a separate reviewed window if a core-runtime transitive needs an `override-dependencies` entry or transformers a `4.58+` minor. Schema/checkpoint work sequences first so a flaky transitive cannot block the functional merge.

## Residual risks (carried from the spec)

1. **A push to `main` between Step 1 and Step 6 breaks the FF assumption.** Mitigated: `--ff-only` fails loudly; both Step 1 and Step 6 re-verify `0.1.1..origin/main == 0`. STOP-and-re-assess, never force.
2. **Nightly-only (slow/stress) paths first exercised post-push regress the refactor.** Mitigated: Step 3a runs every CI lane explicitly (a bare `uv run pytest` excludes them per `pytest.ini:24`); Step 3b *proves* each non-default lane collected non-empty and ran (not silently skipped) — the single most important pre-merge gate.
3. **EV false-alarms on K>1 once the estimator hits main.** Mitigated: Step 5 gates the EV-consumer fix to co-land via a structural code-artifact check run PRE-PUSH (existing commits would false-pass a message grep); Step 7 K=4 smoke confirms no alert. A false pass would trip `run_confounders.proof_blocking=true` (`views.py:649`) and corrupt the experiment verdict — the highest-consequence operational risk, hence the structural (not textual) gate.
4. **A pre-v2 checkpoint someone relies on is silently lost.** Mitigated: Step 2 explicit pre-merge confirmation checklist (recorded in the PR body).
5. **A dependency bump introduces a transitive regression, or a security-critical transitive silently fails to advance.** Mitigated: Step 8 bumps are the last, single, independently-revertable commit; 8a asserts the full high/critical cluster at patched floors (pyarrow≥23.0.1, urllib3, cryptography, GitPython, pillow, starlette, python-multipart, tornado, pyjwt, mistune) and cross-checks the Dependabot API; the constraint-only-inline discipline (no `override-dependencies` for core-runtime transitives) defers the bump rather than loosening constraint discipline under time pressure.
6. **Coverage dips below the 75% gate.** Mitigated: surfaced by PR CI (Step 4) and confirmed locally with the CI-mirrored marker set + threshold check (Step 3c).
7. **An npm build regenerates shipped `web/dist/**` and rides the manifest/lockfile bump commit (or a packaging op picks up the gitignored build output).** Mitigated: Step 8c PRECONDITION inspects `git status --porcelain` / `diff --stat` **and `git status --ignored`** (because `web/.gitignore:2` hides `dist/` from `--porcelain`), and forces any tracked dist changes into a separate labelled commit; packaging is done from a clean checkout.

## Critical files

- `/home/john/esper-lite/src/esper/leyline/__init__.py` (`:125`, `:877`, `:635`) — `VALUE_HEAD_SCHEMA_VERSION = 2` and `OBS_V3_FEATURE_SCHEMA_VERSION = 1` (both land via FF).
- `/home/john/esper-lite/src/esper/simic/agent/ppo_agent.py` (`:51`, `:622-628` pre-fix EV branch with `var_returns > 1e-8` at `:624`, `:1540`, `:1656` required-field read, `:1663-1668` KeyError path = the real pre-v2 rejection, `:1675-1683` schema-mismatch raise) — checkpoint save/load enforcement + the pre-fix EV consumer branch gated in Step 0/5 (lands via FF).
- `/home/john/esper-lite/src/esper/karn/mcp/views.py` (`:127`, `:606`, `:649`) — EV consumer / `run_confounders` view / `proof_blocking` false-alarm surface.
- `/home/john/esper-lite/uv.lock` — Python dependency bump (trailing commit).
- `/home/john/esper-lite/src/esper/karn/overwatch/web/package-lock.json` + `package.json` — npm dev-tooling bumps (trailing commit; both files staged).
- `/home/john/esper-lite/pyproject.toml` — read-only confirmation that no pin edits are required (`torch:7`, `datasets:11` core runtime, `transformers:12`, `pytest:77` dev); shipped package-data section `:43-45` (Overwatch entry `:45` `"esper.karn.overwatch" = ["web/dist/**/*"]`). ONLY modified if a single approved `[tool.uv] constraint-dependencies` floor (e.g. `pyarrow>=23.0.1` or `transformers<4.58`) is needed.
- `/home/john/esper-lite/.github/workflows/test-suite.yml` — CI gating reference (unit marker filter + `--cov` `:116`; 75% threshold check `:121-130`; nightly `-m slow` `:203`, `-m stress` `:209`).
- `/home/john/esper-lite/docs/coord/PLAN_TRACKER.md` — updated by the orchestrator after merge (out of scope for this plan).

## Confidence Assessment

**Overall Confidence: High.** The FF topology and zero-conflict claim are mechanically verifiable and re-verified at execution time. The checkpoint-break surface, the CI gating posture, and the dependency-bump strategy are all anchored to verified file:line facts in the source spec and re-confirmed against the live tree.

| Finding | Confidence | Basis |
|---------|------------|-------|
| FF topology (merge-base == main HEAD; conflict probability zero) | High | Spec §Key reframe; re-verified in Steps 1 & 6 |
| Single checkpoint break (`VALUE_HEAD_SCHEMA_VERSION=2`, not `CHECKPOINT_VERSION`) | High | Spec §2 with file:line anchors; Step 2 re-confirms across branches |
| Dependency fixes are mostly lockfile-delta (4 direct); security-critical core-runtime transitive is pyarrow via datasets; transformers held to `<4.58.0` | High | Dep research §3; Step 8a precondition + full high/critical cluster assertions |
| Nightly-only paths are the real residual gap | High | Spec §4; CI workflow triggers (`:203`, `:209`) |
| EV co-landing is a hard sequencing constraint, gated structurally not textually | High | Spec Locked decision 5; existing commits would false-pass a message grep; `views.py:649` is the corruption surface |

## Risk Assessment

**Implementation Risk: Low-Medium.** There is no merge-algorithm risk (FF, zero conflicts). The risk is operational and bounded: the abort gates (Steps 1, 2, 3, 5, 6) catch every failure mode *before* the only mutating push, and rollback is forward-revert (no history rewrite).

**Reversibility: Easy.** Linear history → rollback is a single forward `git revert` of a commit or range. The dependency bump is isolated as the last commit. No `--hard`/`--force` on `main` (CLAUDE.md Git Safety).

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Push to main breaks FF between Step 1 and Step 6 | Low | Low | `--ff-only` fails loudly; re-verify in both steps; STOP-not-force |
| Refactor regresses on nightly-only path post-push | High | Low | Full local suite (Step 3a) + proof slow/stress ran (3b) before PR |
| EV false-alarm on K>1 on main (proof_blocking corruption) | High | High if item 1 not co-landed | Step 5 STRUCTURAL pre-push gate (old branch gone + new artifact + test green); Step 7 smoke |
| Stranded pre-v2 checkpoint | High | Low | Step 2 confirmation checklist |
| Security-critical transitive (pyarrow UAF) silently not advanced | High | Low-Medium | Step 8a explicit version assertions + constraint floor + re-run |
| Transitive dependency regression | Medium | Low | Step 8 single revertable trailing commit |
| `override-dependencies` on a core-runtime transitive bypasses upstream metadata | Medium | Low | Step 8a discipline gate: constraint-dependencies only inline; defer the whole bump to a separate reviewed window if an override is required |

## Information Gaps

1. [ ] **EV-consumer gate-fix landing status + exact symbol name** — Step 5 depends on the EV-telemetry-robustness plan having merged its fix to `0.1.1`; the structural gate keys on the OLD `var_returns > 1e-8` branch being gone AND a new floor/flag symbol being present. Reconcile the grep token in Step 5 (2) with the EV plan's actual ctor kwarg / flag name before execution.
2. [ ] **Exact post-bump Dependabot residual** — the precise remaining alert count after `uv lock --upgrade` + `npm audit fix` is known only after Step 8 runs; `transformers` (medium) and `torch` #80 (low) are expected to remain, annotated.
3. [ ] **`torch` #80 fix availability** — no upstream fix at plan time; snooze with a note and revisit.
4. [ ] **`datasets` pyarrow pin** — whether `datasets>=4.4.1` permits `pyarrow>=23.0.1`; if not, a single `[tool.uv]` floor is needed (Step 8a). Resolved at execution by the 8a version assertions.

## Caveats & Required Follow-ups

**Before executing:**
- [ ] **(BLOCKING, FIRST) Confirm the EV-telemetry-robustness fix is structurally present on `0.1.1` via Step 0** — the plan is NOT unconditionally ready and must not start while the old EV branch (`ppo_agent.py:624`) is still present. Reconcile Step 0/5's new-artifact grep token with the EV plan's actual symbol name first.
- [ ] Re-run Step 1's topology check against live `origin/main` (a push since spec time invalidates the FF).
- [ ] Re-run `git status` (Step 1 VERIFY) and reconcile every entry — the planning docs (`docs/plans/ready/2026-06-18-*.md`, `docs/superpowers/specs/2026-06-18-*.md`) and the `docs/coord/PLAN_TRACKER.md` modification must be committed on `0.1.1` (to ride the FF) or explicitly excluded.
- [ ] Obtain explicit user permission before any `git reset --hard` / `git push --force` on `main` (CLAUDE.md Git Safety) — the plan's rollback path is forward-revert only and should never need either.

**Assumptions:** all commands run with `cwd=/home/john/esper-lite`; `main` HEAD remains the merge-base at execution time; the 200-ep EV-liftoff experiment (already on the 0.1.1 estimator) is the only in-flight run and is unaffected by the checkpoint break; the trailing bump is a dependency manifest/lockfile commit (a single `[tool.uv] constraint-dependencies` cap on `pyproject.toml` is permitted; an `override-dependencies` need defers the whole bump to a separate reviewed window).

**Limitations:** this plan does not author functional code — it transports already-committed, already-tested work via fast-forward and folds in a dependency manifest/lockfile pass (the trailing bump is the only authored commit). It does not own the EV-consumer gate-fix (Sprint item 1) nor the `docs/coord/PLAN_TRACKER.md` update (orchestrator).
