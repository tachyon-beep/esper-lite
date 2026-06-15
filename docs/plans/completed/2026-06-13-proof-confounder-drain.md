# Proof Confounder Drain Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the bugs and missing telemetry that can dampen, distort, or hide Esper's already-observed morphogenetic signal, then produce a proof packet that explains how much of the missing effect size was confounder-driven.

**Architecture:** Treat this as a signal-recovery and experimental-validity package, not a dashboard feature package. First make the evidence surface fail closed on confounders, then add the missing learnability and counterfactual freshness signals, then repair reward-accounting confounders, and only then rerun the reward-efficiency exam with acceptance criteria that separate "the theory is weak" from "the theory is being obscured by bad measurement or reward accounting."

**Tech Stack:** Python 3.11, PyTorch, pytest, Hypothesis, DuckDB, Karn MCP views, Sanctum, Overwatch, uv, Ruff, mypy.

**Prerequisites:**
- Start from a clean worktree or a new worktree after the current `overwatch-refurb` UX changes land.
- Read `CLAUDE.md`, `README.md`, `ROADMAP.md`, `docs/coord/PLAN_TRACKER.md`, and this plan.
- Run Loomweave freshness check before broad call/reference edits: `loomweave analyze /home/john/esper-lite` if `project_status_get` reports stale.
- Required specialist reviews before promotion to `ready`: DRL, PyTorch, Python engineering, and QA.
- Do not use defensive bug-hiding access patterns. Missing required telemetry fields should fail tests, not silently default.

---

```yaml
# Plan Metadata
id: proof-confounder-drain
title: Proof Confounder Drain
type: completed
created: 2026-06-13
updated: 2026-06-13
owner: Codex

urgency: high
value: Recover and quantify the real morphogenetic signal suggested by prior runs by removing measurement and training confounders before spending more time on Phase 3 or larger architectures.

complexity: L
risk: high
risk_notes: Crosses Simic reward/training, Leyline telemetry contracts, Karn evidence surfaces, and experiment execution. The main risk is accidentally adding another metric that looks authoritative but is not causally tied to the training state.

depends_on:
  - karn-telemetry-quality-arc
soft_depends:
  - telemetry-domain-sep
  - post-fossilization-drip-reward-impl
blocks:
  - reward-efficiency verdict
  - phase3-tinystories validation
  - counterfactual-oracle

status_notes: Implementation complete on `confounder-drain`. Baseline captured; run-level confounder ledger, action-head learnability telemetry, stale counterfactual fail-closed behavior, BASIC_PLUS drip measurement closure, and proof packet generator are implemented and verified. Short dual-A/B rehearsal produced a blocked packet due to value-collapse and gradient-anomaly confounders, so the expensive reward-efficiency exam remains deferred.
percent_complete: 100

reviewed_by:
  - Task 3 learnability read-only reviewer
  - Task 4 counterfactual freshness read-only reviewer
  - Task 5 reward-accounting read-only reviewer
```

## Prior Evidence and Product Bet

Prior results have strongly suggested that the underlying theory is sound, but the effect size has been smaller and less stable than expected. This plan assumes that means the next product risk is not "there is no signal"; it is that confounders are compressing the signal, misattributing credit, or hiding why promising runs fail to convert into reliable ROI.

The bet is not "ship more telemetry." The bet is:

> If Esper's metrics are made confounder-aware and reward accounting is closed against known gaming paths, then a controlled reward-efficiency exam will show a clearer accuracy-per-parameter signal and identify any remaining bottleneck by name.

The package succeeds if it produces one of three honest outcomes:

- **Continue:** simplified or another candidate reward mode beats the control on final accuracy and accuracy ROI, with no blocking confounders.
- **Revise:** the signal is present but still damped, and the proof packet names the next confounder class to fix.
- **Stop:** the controlled run shows no ROI advantage and no instrumentation blocker plausibly explains the failure.

## Execution Outcome

The package reached the **Revise** outcome for the proof infrastructure rather than a
reward-efficiency verdict. The short dual-A/B rehearsal generated
`docs/analysis/2026-06-13-proof-rehearsal.md`, and the packet is correctly
`BLOCKED` by value-collapse and gradient-anomaly confounders. That means the real
100-round reward-efficiency exam should not be run as proof yet.

Useful outside-review guidance from Gemini and DeepSeek aligns with the next
package: prove Kasmina/Tolaria lifecycle mechanics with an oracle or hardcoded
heuristic before asking PPO to be the verdict, and add a cheap mathematical
micro-sandbox for Tamiyo/Simic controller learning. Those are follow-on work, not
changes to this package's implemented acceptance gates.

## Non-Goals

- Do not implement Phase 3 TinyStories features in this package.
- Do not start Counterfactual Oracle; this package may unblock it, but does not build it.
- Do not redesign PPO or Tamiyo policy architecture.
- Do not make Overwatch pretty except where a UI state is required to prevent misreading the experiment.
- Do not default new reward modes on for existing runs unless an acceptance decision explicitly says to.

## Acceptance Criteria

1. A proof run cannot be marked valid if any cohort has `NUMERICAL_INSTABILITY_DETECTED`, ratio explosion/collapse, value collapse, stale counterfactual contribution, missing baseline counterfactual, or an all-zero learnable fraction for a required action head.
2. PPO update telemetry distinguishes:
   - no gradient because the head is not learnable this batch,
   - missing gradient because the head was not in the graph,
   - non-finite gradient because training is unstable.
3. Counterfactual contribution telemetry carries enough freshness and coverage data to tell whether rewards and fossilization gates used current causal evidence.
4. Reward-accounting tests prevent fossilize-at-peak, wait-in-holding, and post-fossil degradation from being counted as wins without accountability.
5. The reward-efficiency proof packet contains commands, run IDs, cohort configs, confounder ledger, accuracy ROI, lifecycle efficiency, and a continue/revise/stop verdict.
6. If the signal remains weaker than expected, the packet identifies the dominant loss channel: learnability, counterfactual freshness, numerical stability, reward accounting, or capacity/rent economics.

## Task 1: Preflight and Current-State Baseline

**Files:**
- Create: `docs/analysis/2026-06-13-proof-confounder-baseline.md`
- Inspect: `docs/coord/PLAN_TRACKER.md`
- Inspect: `docs/bugs/investigations/CRITICAL-telemetry-freeze.md`
- Inspect: `docs/bugs/investigations/CRITICAL-op-value-mismatch.md`
- Inspect: `src/esper/simic/training/dual_ab.py`
- Inspect: `src/esper/scripts/train.py`

**Steps:**
1. Confirm the UX agent's edits have landed or move this work to a separate clean worktree.
2. Run the current focused health checks:
   ```bash
   git status --short --branch
   uv run --python 3.11 pytest tests/tamiyo/networks/test_op_value_consistency.py tests/simic/training/test_bootstrap_consistency.py -q
   uv run --python 3.11 pytest tests/karn -q
   uv run python scripts/lint_defensive_patterns.py
   ```
3. Confirm the current valid task names from source:
   ```bash
   PYTHONPATH=src uv run python - <<'PY'
   from esper.runtime.tasks import VALID_TASKS
   print(sorted(VALID_TASKS))
   PY
   ```
4. Write `docs/analysis/2026-06-13-proof-confounder-baseline.md` with:
   - branch and commit,
   - dirty files observed before work,
   - Filigree state,
   - valid task names,
   - test command results,
   - known concurrent UX changes.

**Definition of Done:**
- Baseline document exists and names every pre-existing dirty file.
- Current task names are source-verified; do not rely on stale `cifar_blind` wording.
- No code has been changed in this task.

## Task 2: Run Confounder Ledger

**Files:**
- Modify: `src/esper/leyline/telemetry.py`
- Modify: `src/esper/karn/collector.py`
- Modify: `src/esper/karn/store.py`
- Modify: `src/esper/karn/mcp/views.py`
- Modify after UX lands: `src/esper/karn/sanctum/aggregator.py`
- Test: `tests/leyline/test_anomaly_payload.py`
- Test: `tests/karn/test_ingest.py`
- Test: `tests/karn/mcp/test_views.py`
- Test: `tests/karn/sanctum/test_aggregator.py`

**Behavior:**
- Preserve `AnomalyDetectedPayload` as the typed anomaly contract.
- Add a run-level confounder ledger in Karn that records anomaly type, group, env, episode, batch, detail, and whether the anomaly blocks proof validity.
- Add or extend a DuckDB view, for example `run_confounders`, that derives blocking confounders from `raw_events` anomaly payloads.
- Make Sanctum/Overwatch consume the same ledger instead of independently guessing from text.

**Test first:**
```bash
uv run --python 3.11 pytest tests/karn/mcp/test_views.py::test_run_confounders_view_surfaces_numerical_instability -q
```

Expected first failure:
```text
FAILED ... run_confounders view is missing
```

**Implementation notes:**
- Use direct typed field access for required payload fields.
- Do not parse anomaly details out of `message`.
- Treat the following as proof-blocking by default: numerical instability, ratio explosion, ratio collapse, value collapse, gradient anomaly, gradient pathology.

**Verification:**
```bash
uv run --python 3.11 pytest tests/leyline/test_anomaly_payload.py tests/karn/test_ingest.py tests/karn/mcp/test_views.py tests/karn/sanctum/test_aggregator.py -q
```

**Definition of Done:**
- A proof consumer can query blocking confounders by run and group.
- Existing anomaly payload tests still pass.
- A cohort with numerical instability cannot be presented as a clean winner.

## Task 3: Action-Head Learnability Telemetry

**Files:**
- Modify: `src/esper/simic/agent/ppo_update.py`
- Modify: `src/esper/simic/agent/ppo_metrics.py`
- Modify: `src/esper/leyline/telemetry.py`
- Modify: `src/esper/simic/telemetry/emitters.py`
- Modify: `src/esper/karn/mcp/views.py`
- Test: `tests/simic/agent/test_ppo_entropy_floor.py`
- Test: `tests/simic/agent/test_ppo_ratio_metrics.py`
- Test: `tests/karn/mcp/test_views.py`
- Test: `tests/integration/test_q_values_telemetry.py`

**Behavior:**
- Emit per-head `learnable_fraction`: fraction of timesteps where that head is causally relevant and has more than one valid action.
- Emit per-head gradient state, not just a scalar: `finite`, `missing`, `nonfinite`, or `not_learnable`.
- Preserve existing entropy and conditional entropy fields.
- Surface the metrics in Karn MCP views next to head entropy and head gradient norms.

**Test first:**
```bash
uv run --python 3.11 pytest tests/simic/agent/test_ppo_entropy_floor.py::test_zero_availability_head_reports_not_learnable -q
```

Expected first failure:
```text
FAILED ... key 'head_slot_learnable_fraction' not found
```

**Implementation notes:**
- Reuse the existing availability and causal masks already used for entropy-floor and ratio calculations.
- Do not classify a zero gradient as unhealthy when `learnable_fraction == 0.0`.
- Do classify `grad is None` as missing when `learnable_fraction > 0.0`.

**Verification:**
```bash
uv run --python 3.11 pytest tests/simic/agent/test_ppo_entropy_floor.py tests/simic/agent/test_ppo_ratio_metrics.py tests/karn/mcp/test_views.py tests/integration/test_q_values_telemetry.py -q
```

**Definition of Done:**
- The telemetry-freeze historical ambiguity is closed: zeros are explainable as not learnable, missing, or unstable.
- Karn can answer "was this head capable of learning during this batch?"

## Task 4: Counterfactual Freshness and Coverage Gate

**Files:**
- Modify: `src/esper/simic/attribution/counterfactual.py`
- Modify: `src/esper/simic/training/parallel_env_state.py`
- Modify: `src/esper/simic/training/counterfactual_eval.py`
- Modify: `src/esper/simic/telemetry/emitters.py`
- Modify: `src/esper/kasmina/slot.py` only if fossilization gate metadata needs a typed freshness reason.
- Modify: `src/esper/karn/mcp/views.py`
- Test: `tests/simic/properties/test_counterfactual_attribution_properties.py`
- Test: `tests/simic/training/test_contribution_propagation.py`
- Test: `tests/telemetry/test_tele_counterfactual.py`
- Test: `tests/karn/test_counterfactual_telemetry.py`

**Behavior:**
- Record counterfactual coverage per env/slot:
  - baseline/all-disabled measured,
  - all-enabled measured,
  - solo-on measured or ablation-estimated,
  - pair terms measured when claimed.
- Emit freshness metadata: epoch measured, epochs since measurement, and whether reward/fossilization consumed a fresh contribution.
- Fail closed when a reward or fossilization gate needs counterfactual truth but only stale or missing data is available.

**Test first:**
```bash
uv run --python 3.11 pytest tests/simic/training/test_contribution_propagation.py::test_stale_counterfactual_is_not_marked_fresh -q
```

Expected first failure:
```text
FAILED ... stale counterfactual freshness field is missing
```

**Implementation notes:**
- Keep expensive Shapley logic bounded; this task is about freshness and coverage, not perfect attribution.
- Use existing `epochs_since_counterfactual` state where possible.
- If a field is required to claim freshness, access it directly in tests and code.

**Verification:**
```bash
uv run --python 3.11 pytest tests/simic/properties/test_counterfactual_attribution_properties.py tests/simic/training/test_contribution_propagation.py tests/telemetry/test_tele_counterfactual.py tests/karn/test_counterfactual_telemetry.py -q
```

**Definition of Done:**
- Proof runs can distinguish "counterfactual showed no contribution" from "counterfactual was never measured or stale."
- Fossilization and reward decisions that rely on causal contribution expose the measurement age.

## Task 5: Reward-Accounting Confounder Closure

**Files:**
- Modify: `src/esper/simic/rewards/contribution.py`
- Modify: `src/esper/leyline/reward_config.py`
- Modify if needed: `src/esper/simic/training/action_execution.py`
- Modify if needed: `src/esper/simic/training/parallel_env_state.py`
- Test: `tests/simic/rewards/test_anti_timing_gaming.py`
- Test: `tests/simic/rewards/shaped/test_shaped_attribution.py`
- Test: `tests/simic/rewards/test_capacity_economics.py`
- Test: `tests/simic/training/test_epoch_runner.py`
- Reference: `docs/plans/ready/2026-01-12-post-fossilization-drip-reward-impl.md`

**Behavior:**
- Close the known proof confounders where a seed can look good at peak and then degrade without accountability.
- Finish or consciously defer `BASIC_PLUS` drip integration behind an explicit reward mode. Do not change the default reward mode in this task.
- Ensure reward components expose immediate fossilize bonus, ongoing fossilized maintenance rent, drip credit/penalty if enabled, and stale-counterfactual refusal.

**Test first:**
```bash
uv run --python 3.11 pytest tests/simic/rewards/test_anti_timing_gaming.py::test_fossilized_seed_regression_is_accounted_after_peak -q
```

Expected first failure:
```text
FAILED ... post-fossilization accountability field is missing
```

**Implementation notes:**
- If `BASIC_PLUS` is too large for this package, write the explicit deferral in this plan and leave a `TODO: [FUTURE FUNCTIONALITY]` at the narrowest code location required by `CLAUDE.md`.
- Keep `BASIC` unchanged.
- Keep `SHAPED` and `SIMPLIFIED` experiment-comparable; do not mix experimental reward semantics into the proof run without naming the mode.

**Verification:**
```bash
uv run --python 3.11 pytest tests/simic/rewards/test_anti_timing_gaming.py tests/simic/rewards/shaped/test_shaped_attribution.py tests/simic/rewards/test_capacity_economics.py tests/simic/training/test_epoch_runner.py -q
```

**Definition of Done:**
- A peak-only seed cannot be counted as a clean success if it later harms final accuracy or rent-adjusted ROI.
- Reward telemetry explains the accountability path used.

## Task 6: Proof Packet Generator

**Files:**
- Create: `scripts/proof_packet.py`
- Modify: `src/esper/karn/mcp/reports.py`
- Modify: `src/esper/karn/mcp/views.py`
- Test: `tests/karn/mcp/test_views.py`
- Test: `tests/karn/mcp/test_server.py`
- Test: `tests/scripts/test_proof_packet.py`

**Behavior:**
- Generate a Markdown proof packet from a telemetry directory and run IDs.
- Include:
  - cohort config and task,
  - run health and blocking confounders,
  - final accuracy,
  - accuracy ROI,
  - parameter growth/growth ratio,
  - lifecycle efficiency,
  - learnable fractions,
  - counterfactual freshness/coverage,
  - continue/revise/stop verdict.

**Test first:**
```bash
uv run --python 3.11 pytest tests/scripts/test_proof_packet.py::test_proof_packet_blocks_verdict_when_confounders_present -q
```

Expected first failure:
```text
FAILED ... script not found or verdict field missing
```

**Implementation notes:**
- Read telemetry via Karn/MCP DuckDB views rather than duplicating JSON parsing.
- A packet with blocking confounders must say `verdict: invalid`, not choose a winner.
- A packet with no blocking confounders but weak ROI must say `verdict: revise` or `stop`.

**Verification:**
```bash
uv run --python 3.11 pytest tests/karn/mcp/test_views.py tests/karn/mcp/test_server.py tests/scripts/test_proof_packet.py -q
```

**Definition of Done:**
- One command can produce a reviewable proof packet from a telemetry run.
- Invalid runs are visibly invalid, not quietly ranked.

## Task 7: Execute the Proof Exam

**Files:**
- Create: `docs/analysis/2026-06-13-reward-efficiency-proof-packet.md`
- Update after verdict: `docs/coord/PLAN_TRACKER.md`

**Steps:**
1. Run a short rehearsal to prove commands, telemetry, and proof-packet generation:
   ```bash
   PYTHONPATH=src uv run python -m esper.scripts.train ppo \
     --task cifar_impaired \
     --dual-ab shaped-vs-simplified \
     --rounds 2 \
     --envs 2 \
     --episode-length 25 \
     --telemetry-dir telemetry/proof-rehearsal
   ```
2. Generate the rehearsal proof packet:
   ```bash
   PYTHONPATH=src uv run python scripts/proof_packet.py \
     --telemetry-dir telemetry/proof-rehearsal \
     --output docs/analysis/2026-06-13-proof-rehearsal.md \
     --proof-profile generic
   ```
3. If rehearsal has no proof-blocking instrumentation defects, run the real exam. Confirm task and env count based on available hardware before execution:
   ```bash
   PYTHONPATH=src uv run python -m esper.scripts.train ppo \
     --task cifar_impaired \
     --dual-ab shaped-vs-simplified \
     --rounds 100 \
     --envs 8 \
     --episode-length 150 \
     --telemetry-dir telemetry/reward-efficiency-2026-06-13
   ```
4. Generate the final packet:
   ```bash
   PYTHONPATH=src uv run python scripts/proof_packet.py \
     --telemetry-dir telemetry/reward-efficiency-2026-06-13 \
     --output docs/analysis/2026-06-13-reward-efficiency-proof-packet.md \
     --proof-profile reward-efficiency
   ```

**Definition of Done:**
- The proof packet contains a clear verdict:
  - `continue`: simplified or another candidate beats control on final accuracy and accuracy ROI with no blocking confounders.
  - `revise`: concept remains plausible but reward or telemetry is still confounded.
  - `stop`: the controlled run shows no ROI advantage and no instrumentation blocker explains the failure.
- `PLAN_TRACKER.md` is updated with the verdict and next action.

## Final Verification Gate

Run before promoting this package to complete:

```bash
uv run --python 3.11 pytest tests/simic tests/karn tests/telemetry tests/leyline -q
HYPOTHESIS_PROFILE=ci PYTHONPATH=src uv run pytest -m property -v -x --hypothesis-show-statistics
uv run ruff check src/ tests/ scripts/
MYPYPATH=src uv run mypy -p esper
uv run python scripts/lint_leyline_types.py
uv run python scripts/lint_defensive_patterns.py
uv run python scripts/lint_gpu_sync.py
```

If external input handling, telemetry ingestion, HTTP, SQL, or file parsing changes in the implementation, also run:

```bash
wardline scan . --fail-on ERROR
```

## Review Requirements Before Ready

- **DRL review:** reward accounting, proof criteria, and whether the exam design can actually answer the product question.
- **PyTorch review:** learnability metrics, GPU sync cost, counterfactual evaluation safety, and numerical-instability gates.
- **Python engineering review:** module boundaries, typed contracts, and no defensive bug-hiding patterns.
- **QA review:** verification/validation split, proof-packet adequacy, and regression coverage.
