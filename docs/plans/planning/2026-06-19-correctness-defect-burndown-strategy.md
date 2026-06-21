# Correctness Defect Burndown Strategy

> **For Claude:** This is a planning-stage strategy, not an execution ticket. Use
> `superpowers:executing-plans` only after a child work package is promoted to
> `ready`. Do not run the reward-efficiency exam as product evidence until this
> strategy's lower gates are green.

**Goal:** Build a multi-month, evidence-led program that finds, tests, repairs,
and closes correctness defects across Esper without confusing instrumentation
failures, mechanics failures, algorithm failures, and theory failures.

**Architecture:** Use the existing correctness proof ladder as the spine:
tracker intake -> evidence integrity -> precision/replay -> mechanics -> oracle
sandbox -> PPO learning -> reward-efficiency verdict. Each defect is repaired
through a red regression, a minimal fix, proof-grade verification, and Filigree
closure evidence.

**Tech Stack:** Python 3.11, PyTorch, pytest, Hypothesis, Karn/DuckDB telemetry,
Nissa event payloads, Filigree, Loomweave, Wardline/Legis/Warpline as needed.

**Prerequisites:**
- Read `CLAUDE.md`, `README.md`, `ROADMAP.md`, `docs/coord/PLAN_TRACKER.md`,
  and the active plans listed under "Related Plans".
- Re-run `filigree session-context` and `git status --short` before each
  execution session.
- Refresh or treat Loomweave as orientation-only if `index_diff_get` reports
  drift.
- Do not introduce compatibility shims, defensive bug-hiding patterns, or
  silent fallback evidence. Missing proof evidence must block.

---

```yaml
# Plan Metadata
id: correctness-defect-burndown
title: Correctness Defect Burndown Strategy
type: planning
created: 2026-06-19
updated: 2026-06-19
owner: Codex

urgency: high
value: Establishes the operating model and first two-week package for repairing correctness defects without overclaiming from noisy PPO runs.

complexity: XL
risk: high
risk_notes: The main risk is mixing defect intake, proof instrumentation, mechanics repair, and theory evaluation in one loose effort. This plan keeps those gates separate and fail-closed.

depends_on:
  - correctness-proof-strategy
soft_depends:
  - morphogenesis-governor-integrity
  - ppo-stability-oracle-sandbox
  - proof-baseline-controls
  - post-p01-hardening-sprint
blocks:
  - reward-efficiency
  - counterfactual-oracle
  - phase3-tinystories

status_notes: Drafted from live repo/tracker orientation on 2026-06-19 and updated after four read-only specialist reviews. Package A was executed on 2026-06-19: stale recurrent PPO P1 closed, observation queue empty, June 18 defect report reconciled, sprint work bound to Filigree IDs, and ready-folder hygiene applied. Package B begins with P-EV-RECAL and EV telemetry robustness.
percent_complete: 15

reviewed_by:
  - reviewer: axiom-python-engineering
    date: 2026-06-19
    verdict: approved-with-changes
    notes: Reconcile Filigree/tracker/ready-folder state before execution; demote or annotate blocked-ready plans; add defect-report reconciliation before merge/dependency work.
  - reviewer: yzmir-pytorch-engineering
    date: 2026-06-19
    verdict: approved-with-changes
    notes: Clarify checkpoint semantics, add exhaustive config/RNG replay gates, guard recurrent hidden-state contracts, and validate CPU plus CUDA precision lanes.
  - reviewer: yzmir-deep-rl
    date: 2026-06-19
    verdict: approved-with-changes
    notes: Treat SIMPLIFIED as diagnostic-only for structural economy; use same-seed proof-baseline lockstep for reward claims; add multi-seed statistics before reward-efficiency verdicts.
  - reviewer: quality-engineering
    date: 2026-06-19
    verdict: approved-with-changes
    notes: Add a proof-critical test lane, flake-rate tracking, EV preflight required-column/raw-event fallback, CUDA/perf manual gates, and explicit metric tracking.
```

## Related Plans

- `docs/plans/planning/2026-06-15-correctness-proof-strategy.md`
- `docs/plans/planning/2026-06-15-ppo-stability-oracle-sandbox.md`
- `docs/plans/planning/2026-06-13-morphogenesis-governor-integrity.md`
- `docs/plans/planning/2026-06-15-proof-baseline-controls.md`
- `docs/plans/completed/2026-06-18-post-p01-hardening-sprint.md`
- `docs/plans/completed/2026-06-18-ev-telemetry-robustness-plan.md`
- `docs/plans/completed/2026-06-18-main-merge-integration-plan.md`

## Live Intake Snapshot

Captured 2026-06-19 from `/home/john/esper-lite`:

- Branch/HEAD: `0.3.0` at `f8089677` (`origin/main`, `main`).
- Dirty worktree before this plan: `uv.lock` modified. This plan does not own
  that file.
- Filigree: 2 ready, 0 blocked.
  - P1 `esper-lite-6682b3faea` is still `triage`, but commit `31bf8cb7` is
    present in history and `uv run pytest tests/simic/training/test_batch_bootstrap.py -q`
    passed `6 passed`. First action is tracker reconciliation, not another fix.
  - P4 `esper-lite-88a71d93f9` is the non-startable `Future` release shell.
- Filigree analyzer bridge: `filigree finding list --kind defect` reports no
  defect findings. Session context reports 33525 unbridged telemetry/info
  findings, not defect-signal.
- Pending observation: `esper-lite-obs-c8de6a4b7b` references
  `tests/telemetry/test_reward_metrics.py::test_reward_health_data_all_fields`;
  the current node moved under `TestRewardHealthDataIntegration`, and
  `uv run pytest tests/telemetry/test_reward_metrics.py -q` passed `25 passed`.
  First action is dismiss or promote based on current evidence.
- Loomweave: available, but stale. `project_status_get` reported 11198
  entities, 124 findings, SEIs populated, and index commit `b1935d39d` while
  current commit is `f8089677`. Use Loomweave for orientation only until
  re-analyzed.
- Loomweave open defect findings: 7. Three are high-entropy secret detections
  (`proof_baselines.py`, `test_proof_baselines.py`, `.env`); four are Pyright
  analysis infrastructure timeouts/restarts. These are security/tooling hygiene
  intake, not current PPO correctness evidence.
- Ready-folder hygiene: `docs/plans/README.md` defines `ready/` as
  implementation-ready, but current review found some ready artifacts carrying
  blocked-ready language or completed work. This is process debt that can cause
  the wrong work to be started.
- Defect-report reconciliation: the 2026-06-18 post-P0-1 defect report listed
  P1 findings in q-aux finiteness, Leyline guardrail state, Sanctum crash
  propagation, and q-head telemetry. Some are fixed in current source, but the
  plan/tracker state must explicitly say fixed commit, not-applicable target
  branch, or open Filigree issue.

## Peer Review Synthesis

The reviewers agreed on these corrections to the initial strategy:

1. **Reconciliation is Workstream 0.** Do not begin a new correctness sprint
   while Filigree, `PLAN_TRACKER.md`, ready-folder state, and defect reports
   disagree about what is fixed or startable.
2. **EV Step 0 must be executable.** If `ppo_updates` lacks robust-signal
   columns needed for calibration, the plan must either add those columns first
   or query `raw_events` JSON directly. Add a preflight that fails if the chosen
   query is unavailable.
3. **Proof-critical tests need a named lane.** Default pytest excludes property,
   integration, stress, and slow tests. Add a `proof_gate` marker or equivalent
   PR lane for proof packet, oracle sandbox, governor rollback, EV gate, and
   baseline-control tests.
4. **CUDA is a correctness gate where behavior can diverge.** CPU-only checks
   are not enough for BF16 AMP, recurrent hidden state, compile, or K=4/K=8
   throughput regressions. Keep CUDA gates bounded and explicit.
5. **Checkpoint semantics must be explicit.** Current checkpoints are not exact
   replay boundaries unless RNG and data-iterator cursor state are persisted.
   Either classify them as coarse continuation or add exact-replay state and a
   straight-run versus save/resume equivalence test.
6. **SIMPLIFIED is diagnostic-only for structural economy claims.** Code excludes
   it from blueprint-economy evidence because it omits structural rent. It can
   be a stability/learnability cohort, not the product proof mode.
7. **Reward claims use proof-baseline lockstep.** Sequential dual A/B is useful
   operationally, but final reward claims require paired same-seed lockstep
   proof-baseline controls.
8. **Statistics are required, not optional.** Before final reward-efficiency
   claims, require paired seeds across cohorts, all-seed reporting, confidence
   intervals/effect sizes, train/eval separation, declared eval mode, and
   preregistered `CONTINUE` / `REVISE_ALGORITHM` / `STOP_THEORY` thresholds.

## Defect Taxonomy

### 0. Intake And Tracker Truth

Purpose: prevent stale issues, stale observations, and stale indexes from
polluting the correctness queue.

Defects in scope:
- Ready Filigree items whose code has already landed but closure evidence is
  missing.
- Observations that are stale, duplicated, or actually in-scope task defects.
- Loomweave/Filigree findings that are not bridged or are mislabeled.
- Documentation/tracker divergence around what is fixed.

Acceptance:
- Each open correctness defect has one Filigree issue, owner, priority, current
  status, and reproduction/verification command.
- Stale observations are dismissed with evidence, or promoted to real issues.
- No in-scope defect is hidden in a 14-day observation.

### 1. Evidence Integrity And Proof Packet Contracts

Primary surfaces:
- `scripts/proof_packet.py`
- `src/esper/leyline/telemetry.py`
- `src/esper/karn/store.py`
- `src/esper/karn/mcp/views.py`
- `tests/scripts/test_proof_packet.py`
- `tests/karn/`

Defects in scope:
- Missing or malformed telemetry interpreted as evidence.
- `BLOCKED_*` proof states turning into product verdicts.
- Absent precision, baseline, fixed-schedule, static-final, or lockstep evidence
  passing silently.
- Karn/Nissa schema drift.

Acceptance:
- Every proof verdict has a fixture.
- Missing evidence fails closed into the right `BLOCKED_*` verdict.
- Packet outputs cite only precision-proven, outcome-bearing telemetry.

### 2. Recurrent PPO Mechanics

Primary surfaces:
- `src/esper/simic/agent/rollout_buffer.py`
- `src/esper/simic/agent/ppo_agent.py`
- `src/esper/simic/agent/ppo_update.py`
- `src/esper/simic/training/vectorized_trainer.py`
- `src/esper/simic/training/ppo_coordinator.py`
- `src/esper/tamiyo/networks/factored_lstm.py`
- `tests/simic/agent/`
- `tests/simic/training/`
- `tests/simic/properties/test_ppo_invariants.py`

Defects in scope:
- Hidden state batch shape/order mismatch.
- Hidden reset at the wrong transaction point.
- GAE bootstrap using the wrong value scale, precision, mask, or env subset.
- Advantage re-normalization corruption across multi-update recurrent PPO.
- Forced steps contributing actor gradients when the policy had no agency.
- Finiteness gates that continue silently or halt without auditable evidence.

Acceptance:
- Desynced env truncation, rollback, and full-batch lockstep paths have focused
  tests.
- Recurrent state invariants cover shape, device, dtype, env order, duplicate
  env selection, and reset timing.
- PPO updates emit enough robust metrics to distinguish value-fit failure,
  ratio failure, advantage collapse, and no-agency saturation.

### 3. Morphogenesis Mechanics And Governor Integrity

Primary surfaces:
- `src/esper/simic/training/action_execution.py`
- `src/esper/simic/training/handlers/`
- `src/esper/tolaria/governor.py`
- `src/esper/kasmina/slot.py`
- `src/esper/kasmina/blueprints/registry.py`
- `tests/simic/training/test_governor_integration.py`
- `tests/tolaria/`
- `tests/kasmina/`

Defects in scope:
- Rollback snapshots taken before current state is known clean.
- Rollback envs executing stale sampled lifecycle mutations in the same step.
- Duplicate lifecycle execution semantics between production paths and handlers.
- Missing Tolaria pre-flight authority over lifecycle mutation.
- Malformed blueprint or registry data reaching germination.

Acceptance:
- A panic epoch cannot replace the last-good snapshot.
- Rollback prevents normal lifecycle mutation for the same env/step.
- Every lifecycle op family has allow and deny pre-flight tests.
- Production has one authoritative mutation path.

### 4. Observation, Reward, And Credit Truthfulness

Primary surfaces:
- `src/esper/tamiyo/policy/features.py`
- `src/esper/simic/rewards/contribution.py`
- `src/esper/simic/rewards/rewards.py`
- `src/esper/simic/training/action_execution.py`
- `src/esper/simic/telemetry/emitters.py`
- `tests/simic/rewards/`
- `tests/simic/properties/`
- `tests/tamiyo/`

Defects in scope:
- Missing gradient/counterfactual evidence encoded as healthy evidence.
- Host drift leaking into the counterfactual contribution channel.
- Reward telemetry claiming action success before the action outcome is known.
- Simplified reward being used as structural-economy proof without rent semantics
  or an explicit diagnostic label.
- Reward hacking, timing gaming, auto-prune gaming, and forced-action credit.

Acceptance:
- Missing evidence is a distinct sentinel, neutral value plus freshness flag, or
  hard failure, never fabricated health.
- Reward mode semantics are tested under property/golden tests.
- Proof-facing reward telemetry agrees with action outcome telemetry.

### 5. Determinism, Precision, And Replay

Primary surfaces:
- `src/esper/simic/training/proof_baselines.py`
- `src/esper/simic/training/static_final_replay.py`
- `src/esper/simic/training/normalizer_checkpoint.py`
- `src/esper/simic/control/`
- `src/esper/leyline/proof_baselines.py`
- `tests/simic/training/test_proof_baselines.py`
- `tests/simic/training/test_static_final_replay.py`
- `tests/simic/control/`

Defects in scope:
- Missing seed/schedule/provenance on proof baselines.
- Static-final source/replay mismatch.
- Normalizer checkpoint schema mismatch.
- Checkpoints that resume with defaulted or missing train-affecting config.
- Checkpoints that are treated as exact replay while omitting RNG or data cursor
  state.
- Precision mismatch between rollout, bootstrap, and PPO update.
- CPU/GPU divergence not covered by smoke tests.

Acceptance:
- Proof runs record seed, schedule hash, precision, static-final manifest, and
  normalizer schema.
- Static-final replay fails closed on source/replay mismatch.
- Checkpoint documentation and tests define whether resume is coarse
  continuation or exact replay.
- Every rollout/update/telemetry-affecting constructor field is either saved and
  checked on load, or intentionally excluded with a hard error if missing.
- At least one CPU-safe and one CUDA-capable smoke lane cover recurrent PPO
  mechanics before long experiments.

### 6. Operator-Facing Telemetry And UI Truth

Primary surfaces:
- `src/esper/karn/sanctum/`
- `src/esper/karn/overwatch/`
- `src/esper/karn/mcp/views.py`
- `src/esper/nissa/`
- `tests/karn/`
- `src/esper/karn/overwatch/web/`

Defects in scope:
- UI panels or MCP views presenting diagnostics as proof.
- EV thresholding artifacts after op-independent `V(s)`.
- Web/TUI generated types drifting from payloads.
- Telemetry ingestion swallowing malformed payloads.

Acceptance:
- Proof-blocking Karn rows can be joined back to robust PPO metrics.
- Display-only thresholds are labeled display-only.
- Generated web types are fresh and tested when payload contracts change.

## Operating Model

1. Intake every candidate from Filigree, failed tests, proof packet verdicts,
   Loomweave findings, Wardline/Legis gates, and live run telemetry.
2. Classify it into exactly one primary taxonomy lane plus any secondary labels.
3. Reproduce or prove the failure. If it cannot be reproduced, write the
   smallest characterization test or mark the issue blocked with the missing
   evidence.
4. Write the red regression first. For proof-facing defects, add a packet-level
   fixture or Karn view assertion in the same package.
5. Implement the smallest fix that removes the root cause.
6. Run the lane-specific command, the neighboring suite, and the relevant CI
   guardrails.
7. For proof-critical tests, run at least one repeat pass or flake check before
   closing; any quarantine needs owner, issue, expiry, and approval.
8. Add a Filigree comment with reproduction, fix commit, verification commands,
   and residual risk. Close only after the current branch contains the fix.
9. Re-run `filigree session-context`; do not leave stale ready issues or
   in-scope observations behind.

## First Two-Week Package

### Package A: Reconcile Intake, Plans, And Defect Reports

**Files/tools:**
- Filigree CLI/MCP
- `docs/coord/PLAN_TRACKER.md`
- `docs/plans/README.md`
- `docs/plans/completed/2026-06-18-post-p01-hardening-sprint-defect-report.md`
- `docs/plans/planning/2026-06-19-correctness-defect-burndown-strategy.md`

**Steps:**
1. Verify `esper-lite-6682b3faea` on current branch:
   `uv run pytest tests/simic/training/test_batch_bootstrap.py -q`.
2. If still green, add a Filigree comment with commit `31bf8cb7`, current HEAD,
   and test output, then close it.
3. Verify observation `esper-lite-obs-c8de6a4b7b` with:
   `uv run pytest tests/telemetry/test_reward_metrics.py -q`.
4. Dismiss the observation if still green, or promote it if a current failure
   reappears.
5. Re-run `filigree session-context` and record the new open-count baseline.
6. Reconcile each 2026-06-18 defect-report P1:
   - fixed commit plus focused test,
   - explicit not-applicable branch note, or
   - new/open Filigree issue.
7. Create or link Filigree issues for the post-P0-1 sprint umbrella, EV
   robustness, main merge, dependency triage, and `P-EV-RECAL`.
8. Audit `docs/plans/ready/`: move completed artifacts to `completed/`, demote
   blocked artifacts to `planning/`, or document a strict blocked-ready
   convention in the tracker.

**Definition of Done:**
- No stale P1 issue remains ready.
- No stale observation remains pending.
- The correctness queue has a trustworthy starting count.
- The ready folder contains only truly startable plans or explicitly marked
  blocked-ready exceptions.

**2026-06-19 outcome:** Executed. Live Filigree now has `P-EV-RECAL`
(`esper-lite-26e96f0578`) and dependency triage (`esper-lite-d289d208ac`) as
ready P1 work; EV robustness (`esper-lite-a20b180e26`) and main merge
(`esper-lite-569292a32b`) are correctly dependency-blocked. The remaining
Sanctum pre-ready crash behavior was tracked and later closed as
`esper-lite-440748cb34`.

### Package B: Make EV Preflight And Consumers Executable

**Files:**
- `docs/plans/completed/2026-06-18-ev-telemetry-robustness-plan.md`
- `src/esper/karn/mcp/views.py`
- `src/esper/simic/telemetry/anomaly_detector.py`
- `src/esper/simic/training/ppo_coordinator.py`
- `src/esper/karn/sanctum/aggregator.py`
- `src/esper/karn/sanctum/widgets/tamiyo/ppo_losses_panel.py`
- `src/esper/karn/sanctum/widgets/tamiyo/critic_calibration_panel.py`
- `tests/simic/telemetry/test_anomaly_detector.py`
- `tests/karn/mcp/test_views.py`
- `tests/karn/sanctum/`

**Steps:**
1. Add an EV Step 0 preflight: assert required `ppo_updates` columns exist.
   P-EV-RECAL resolved this by adding `bellman_error` and
   `v_return_correlation` to `ppo_updates`; `raw_events` remains an internal
   evidence source, not the public calibration query path.
2. Add a telemetry-producing smoke that proves `run_confounders` is clean for
   known low-return-variance artifact batches.
3. Remove or isolate defaulted robust-signal parameters from live anomaly paths.
   Live paths should direct-index mandatory metrics; defaults belong only in
   explicit old-event/test fixture builders.
4. Audit UI/status consumers so raw EV is diagnostic-only when low-return
   variance is flagged. Health/status must key on robust value diagnostics plus
   explicit missing/low-variance states.
5. Record the empirical EV floor calibration, or block the sprint.

**Definition of Done:**
- EV calibration is executable on current telemetry.
- A false EV denominator artifact cannot block a proof verdict.
- A genuine value collapse still blocks quickly with robust evidence.

**2026-06-19 P-EV-RECAL outcome:** `esper-lite-26e96f0578` selected the
`ppo_updates` path and added a fail-loud preflight query that blocks when a run
has no PPO updates or when `return_std`, `value_loss`, `bellman_error`, or
`v_return_correlation` are absent. This removes the query ambiguity for
`esper-lite-a20b180e26`. Live evidence on `telemetry_2026-06-16_160350`:
`preflight_status=ok`, `updates=10`, and `missing_required_rows=0`. The
remaining Package B work is the downstream EV robustness implementation and
consumer audit.

### Package C: Checkpoint And Recurrent PPO Mechanics Harness

**Files:**
- `src/esper/simic/training/vectorized_trainer.py`
- `src/esper/simic/training/action_execution.py`
- `src/esper/simic/agent/rollout_buffer.py`
- `src/esper/simic/agent/ppo_agent.py`
- `src/esper/simic/training/vectorized.py`
- `src/esper/simic/training/config.py`
- `tests/simic/training/test_batch_bootstrap.py`
- `tests/simic/agent/test_rollout_buffer_unit.py`
- `tests/simic/test_ppo_checkpoint.py`
- `tests/simic/test_anchor_reference_pass.py`
- `tests/simic/properties/test_ppo_invariants.py`

**Steps:**
1. Convert the subset-bootstrap regression into a broader desync harness:
   strict subset, full batch, env order, duplicate env selection, rollback reset,
   CPU device, CUDA device when available.
2. Add property tests for GAE terminal/truncation combinations over desynced
   env step counts.
3. Add an invariant test that forced/proof-controlled steps remain in value/GAE
   unrolls but are excluded from actor loss.
4. Verify the current P0-1 value-normalizer and EV metrics cannot silently pass
   non-finite return statistics.
5. Define checkpoint semantics:
   - coarse continuation, or
   - exact replay with Python/NumPy/Torch/CUDA RNG state and data iterator cursor.
6. Add exhaustive checkpoint config roundtrip tests with non-default
   `probability_floor`, EV floor, recurrent settings, and value-normalizer
   state. Missing train-affecting fields must fail fast.
7. Add straight-run versus save/resume equivalence if exact replay is chosen.
8. Add a guard test that corrupts per-step telemetry hidden state and proves PPO
   loss is unchanged while telemetry changes.

**Definition of Done:**
- The next recurrent/vectorized state bug should fail in the harness before it
  reaches a long PPO run.
- Resume semantics are explicit and tested.

### Package D: Promote Oracle Sandbox To Ready

**Files:**
- Modify: `docs/plans/planning/2026-06-15-ppo-stability-oracle-sandbox.md`
- Likely create later: `src/esper/simic/training/oracle_sandbox.py`
- Test later: `tests/simic/training/test_oracle_sandbox.py`
- Packet tests later: `tests/scripts/test_proof_packet.py`
- Existing smoke reference: `tests/integration/test_scripted_policy_runner.py`

**Steps:**
1. Reality-check current scripted runner and proof packet surfaces.
2. Move helper logic from the integration smoke into source without direct
   slot-state forcing.
3. Specify a deterministic oracle schedule that uses normal action masks,
   lifecycle gates, governor checks, reward, Nissa telemetry, and Karn packet
   ingestion.
4. Define failure fixtures: invalid schedule, missing oracle evidence, malformed
   lifecycle trace, missing outcome, and mechanics confounder.
5. Add exact commands for a CI-safe oracle run and packet verdict.
6. Send to `drl-expert`, `pytorch-expert`, `quality-engineering`, and
   `python-engineering` for sign-off before promotion to `ready`.

**Definition of Done:**
- The oracle sandbox plan is implementation-ready and does not bypass the
  mechanics it is intended to prove.
- Reward-efficiency remains blocked on this package.

### Package E: Governor Integrity Task 1 And 2

**Files:**
- `docs/plans/planning/2026-06-13-morphogenesis-governor-integrity.md`
- `src/esper/simic/training/vectorized_trainer.py`
- `src/esper/simic/training/action_execution.py`
- `src/esper/tamiyo/policy/features.py`
- `src/esper/kasmina/slot.py`
- `src/esper/kasmina/blueprints/registry.py`
- `tests/simic/training/test_governor_integration.py`
- `tests/tamiyo/`
- `tests/kasmina/`

**Steps:**
1. Promote Task 1 rollback ordering/halt semantics into a ready child package
   with red tests and exact commands.
2. Promote Task 2 observation truthfulness/blueprint contracts into a ready
   child package with red tests and exact commands.
3. Keep Tasks 3-6 as dependent packages until Tasks 1-2 land.

**Definition of Done:**
- Rollback-ordering and observation-truthfulness packages can be implemented
  independently and reviewed by DRL, PyTorch, Python, and quality reviewers.

### Package F: Proof Lane, CUDA Lane, And Statistics Contract

**Files:**
- `pytest.ini`
- `.github/workflows/test-suite.yml`
- `scripts/proof_packet.py`
- `docs/plans/planning/2026-06-15-proof-baseline-controls.md`
- `src/esper/simic/training/proof_baselines.py`
- `src/esper/simic/training/dual_ab.py`
- `src/esper/simic/telemetry/anomaly_detector.py`
- `src/esper/simic/training/ppo_coordinator.py`
- `src/esper/simic/agent/ppo_agent.py`
- `src/esper/karn/mcp/views.py`
- `src/esper/karn/sanctum/widgets/reward_health.py`
- `tests/scripts/test_proof_packet.py`
- `tests/simic/telemetry/test_anomaly_detector.py`
- `tests/karn/mcp/test_views.py`
- `tests/karn/sanctum/test_reward_health.py`

**Steps:**
1. Add a `proof_gate` marker or equivalent named PR lane for proof packet,
   oracle sandbox, governor rollback, EV gate, and baseline-control tests.
2. Promote the merge-window flake quarantine rule into the normal triage loop:
   owner, issue, expiry, approval, repeat-run evidence, no indefinite `xfail`.
3. Define CUDA/manual gates for BF16 AMP, K=4/K=8 phase-profiler trend,
   finite gradients for all heads, compile on/off smoke, and `.item()`/`.cpu()`
   sync regression checks.
4. Audit every proof-blocking Karn row back to the PPO/update event that
   supplies robust context.
5. Verify EV remains diagnostic-only where planned, and robust signals own
   value-collapse decisions.
6. Add or update packet fixtures for `BLOCKED_INSTRUMENTATION`,
   `BLOCKED_PRECISION`, `BLOCKED_MATH`, and `BLOCKED_MECHANICS`.
7. Rewrite reward-efficiency protocol:
   - `SIMPLIFIED` is diagnostic/stability-only for structural economy.
   - Product reward claims use structural-rent modes.
   - Same-seed proof-baseline lockstep is required for reward claims.
   - Sequential dual A/B is not final proof evidence.
8. Define statistical acceptance: at least 5 paired seeds, preferably 10; all
   seeds reported; confidence intervals/effect sizes; train/eval separation;
   deterministic vs stochastic eval mode declared; preregistered verdict
   thresholds.

**Definition of Done:**
- Proof-critical tests have a named lane and flake policy.
- CUDA-sensitive behavior has a bounded validation path.
- Reward-efficiency packets block or label evidence preliminary when statistical
  criteria are missing.

## Verification Matrix

Run focused commands per package, then the shared guardrails:

```bash
uv run pytest tests/simic/training/test_batch_bootstrap.py -q
uv run pytest tests/telemetry/test_reward_metrics.py -q
PYTHONPATH=src uv run pytest tests/scripts/test_proof_packet.py -q
# After Package F creates the named proof lane:
PYTHONPATH=src uv run pytest -m proof_gate -v -x
PYTHONPATH=src uv run pytest tests/simic/training tests/simic/agent -q
HYPOTHESIS_PROFILE=ci PYTHONPATH=src uv run pytest -m property -v -x --hypothesis-show-statistics
uv run python scripts/lint_leyline_types.py
uv run python scripts/lint_defensive_patterns.py
uv run python scripts/lint_gpu_sync.py
uv run ruff check src/ tests/
MYPYPATH=src uv run mypy -p esper
```

If a package touches telemetry ingestion, file/network input, CLI config, or
other trust boundaries, also run:

```bash
wardline scan . --fail-on ERROR
```

Use the bounded Wardline project config. If the scan warns about missing or
unbounded config, stop and repair the config rather than launching an ambiguous
root scan.

## Promotion Rules

- `planning` -> `ready`: exact red tests, exact files, exact commands, and
  specialist review are present.
- `ready` -> `in-progress`: Filigree issue is claimed atomically with
  `start-work` or `start-next-work --advance`.
- `in-progress` -> `completed`: code is committed, tests/guardrails are run,
  proof packet or lane-specific evidence is attached, and Filigree is closed.
- A package that discovers a new in-scope defect either expands scope explicitly
  or files a real dependent issue. It must not hide the defect as an
  observation.

## Open Questions

- Should the correctness ledger live entirely in Filigree, or should the repo
  also carry a generated markdown snapshot for release notes?
- Should Loomweave open secret findings be handled in this burndown or moved to
  a separate security hygiene package?
- How much CUDA capacity should be reserved for nightly correctness rehearsals
  versus development-time focused checks?
