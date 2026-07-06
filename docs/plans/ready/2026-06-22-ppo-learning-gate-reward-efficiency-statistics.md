# PPO Learning Gate / Reward-Efficiency Statistics

```yaml
id: ppo-learning-gate-reward-efficiency-statistics
title: PPO Learning Gate / Reward-Efficiency Statistics
type: ready
created: 2026-06-22
updated: 2026-06-22
owner: Codex

urgency: high
value: |
  Make reward-efficiency proof evidence decision-grade. The package prevents
  missing PPO telemetry, missing precision provenance, sequential A/B bias, or
  single-seed variance from being mistaken for reward-theory evidence.

complexity: L
risk: high
risk_notes: |
  Crosses Simic PPO training, Leyline telemetry contracts, Karn proof views,
  proof-packet verdict logic, test lanes, and CUDA/manual validation. The main
  risk is overclaiming reward-efficiency from incomplete evidence; mitigation is
  fail-closed packet gates before the long run.

depends_on:
  - correctness-proof-strategy
  - morphogenesis-governor-integrity
  - ppo-stability-oracle-sandbox
  - proof-baseline-controls

soft_depends: []

blocks:
  - reward-efficiency verdict
  - counterfactual-oracle
  - emrakul-phase1

status_notes: |
  Promoted to Filigree as work package esper-lite-a2abff5ec5. First startable
  child is esper-lite-441fbc6810. Do not start the long reward-efficiency exam
  until instrumentation, Karn traceability, ROI semantics, statistics, proof
  lane, and CUDA/manual contracts are closed.
percent_complete: 0

reviewed_by:
  - reviewer: deep-rl
    date: 2026-06-22
    verdict: approved-with-changes
    notes: PPO learnability and multi-seed evidence are the next blockers; ordinary sequential dual-ab is smoke-only, not proof-grade.
  - reviewer: training-optimization
    date: 2026-06-22
    verdict: approved-with-changes
    notes: Fix precision provenance and short PPO preflight before long runs; gate on Karn public views and explicit seed/provenance metadata.
  - reviewer: telemetry-proof
    date: 2026-06-22
    verdict: approved-with-changes
    notes: Packet must fail closed on missing PPO rows, missing robust value fields, and current best-ROI verdict semantics.
  - reviewer: quality-engineering
    date: 2026-06-22
    verdict: approved-with-changes
    notes: Add a named proof lane, statistics contract, CUDA/manual lane, exact child issues, and closeout commands.
```

## Filigree Package

Parent package: `esper-lite-a2abff5ec5` (`PPO Learning Gate / Reward-Efficiency Statistics`)

Children:

1. `esper-lite-441fbc6810` - Fix PPO proof provenance and missing-evidence blockers
2. `esper-lite-570d98c451` - Add Karn PPO traceability evidence section
3. `esper-lite-9678c6d05a` - Reconcile reward-efficiency ROI verdict semantics
4. `esper-lite-e70a09590e` - Add multi-seed lockstep statistics contract
5. `esper-lite-c3cb5338c4` - Add CI-safe PPO learnability proof lane
6. `esper-lite-9f30f62993` - Define CUDA and manual validation contract
7. `esper-lite-2b077204e9` - Run reward-efficiency rehearsal and publish verdict packet

Dependency shape:

```text
441fbc6810
  -> 570d98c451
      -> 9678c6d05a -> e70a09590e
      -> c3cb5338c4 -> 9f30f62993
e70a09590e + 9f30f62993
  -> 2b077204e9
```

## Package Stance

The delivered oracle sandbox proves deterministic mechanics through real masks,
lifecycle gates, governor checks, action handlers, telemetry, Karn ingestion,
and the `oracle-sandbox` proof-packet profile. It does not prove PPO can learn
the reward economy on `cifar_impaired`, and it does not make single-seed
reward-efficiency runs decision-grade.

This package therefore starts with fail-closed evidence contracts. The long
reward-efficiency rehearsal is the final child, not the first task.

## Code Anchors

- `src/esper/simic/training/vectorized.py` - PPO training path and `TRAINING_STARTED` emission.
- `src/esper/leyline/telemetry.py` - typed telemetry payload contract, including precision fields.
- `src/esper/karn/mcp/views.py` - public proof surface for `runs`, `ppo_updates`, `episode_outcomes`, and `run_confounders`.
- `scripts/proof_packet.py` - typed proof verdict ladder and reward-efficiency packet profile.
- `src/esper/simic/training/proof_baselines.py` - blueprint-health control cohorts and lockstep reward A/B controls.
- `src/esper/simic/training/dual_ab.py` - smoke-only sequential A/B runner; not proof-grade for final reward-efficiency evidence.

## Child Acceptance

### 1. PPO Provenance and Missing Evidence

Fix the first blocker before adding new long-run surfaces.

Acceptance:

- Outcome-bearing PPO runs with zero `PPO_UPDATE_COMPLETED` rows return `BLOCKED_INSTRUMENTATION`.
- Missing resolved `amp_enabled` or `amp_dtype` returns `BLOCKED_PRECISION`.
- Missing required PPO fields returns `BLOCKED_INSTRUMENTATION`; present measured zero values remain valid evidence.
- Focused tests cover the precision, zero-row, and missing-field failures.

### 2. Karn PPO Traceability

Karn public views are the proof surface. Do not use raw telemetry fallback to
make proof claims.

Acceptance:

- `ppo_updates` exposes proof-critical status including nonfinite gradient
  counts, skipped/update counts, robust value fields, return variance, and
  low-return-variance counts.
- The proof packet includes a PPO evidence section with per-run/group update
  count, missing-field count, skipped/nonfinite count, low-variance count, and
  finite/range validation.
- Skipped, nonfinite, malformed, or incomplete evidence blocks before ROI.

### 3. Reward-Efficiency ROI Semantics

The final verdict must match the roadmap:

- Accuracy ROI is baseline-subtracted and parameter-normalized, not simply
  `final_accuracy / param_ratio`.
- Simplified must beat Shaped in ROI on matched budget/seed cohorts.
- Simplified must beat Control in final accuracy.
- Diagnostic-only or missing cohorts block or revise; they cannot pass through
  unrelated Shaped/default rows.

### 4. Multi-Seed Statistics Contract

Single-seed PPO curves are not proof-grade.

Acceptance:

- At least five paired base seeds are required for a decision-grade
  reward-efficiency verdict.
- Control, Shaped, Simplified, and any Sparse stretch cohort share base seeds,
  budget, task, slots, evaluation mode, precision, compile mode, and telemetry
  provenance.
- Packet reports per-seed ROI, mean/std or confidence interval, median/IQR,
  sample-efficiency curves or AUC, and thresholds.
- Single-seed, mismatched-seed, missing budget, missing eval-mode, or missing
  sample-efficiency evidence blocks or marks the packet preliminary.

### 5. CI-Safe PPO Learnability Proof Lane

Add or document a named proof lane before package closeout.

Initial proof lane command:

```bash
PYTHONPATH=src uv run pytest \
  tests/scripts/test_proof_packet.py \
  tests/simic/training/test_oracle_sandbox.py \
  tests/simic/training/test_oracle_sandbox_acceptance.py \
  tests/simic/training/test_proof_baselines.py \
  tests/simic/training/test_static_final_replay.py \
  tests/simic/test_vectorized_correctness.py \
  tests/karn/mcp/test_views.py \
  tests/simic/training/test_governor_integration.py \
  -q
```

Acceptance:

- A short real PPO update fixture or equivalent tight fixture populates `runs`,
  `ppo_updates`, `episode_outcomes`, `run_confounders`, and packet evidence.
- Entropy collapse, KL/clip/ratio failure, advantage-scale failure, value-health
  failure, and missing per-head signal fixtures block loudly.
- The lane stays CPU-safe unless the child explicitly documents a CUDA/manual
  requirement.

### 6. CUDA and Manual Validation Contract

Hardware-sensitive evidence must be explicit.

Acceptance:

- Document exact CUDA/manual commands for BF16/AMP, K=4/K=8, compile on/off,
  recurrent state, and GPU-sync-sensitive behavior.
- Missing CUDA evidence is recorded as manual evidence unavailable, not clean
  proof.
- `scripts/lint_gpu_sync.py` remains in the closeout stack.

### 7. Reward-Efficiency Rehearsal and Verdict Packet

Run only after the previous children are closed.

Expected packet shape:

```bash
PYTHONPATH=src uv run python scripts/proof_packet.py \
  --telemetry-dir telemetry/reward-efficiency-YYYY-MM-DD \
  --output docs/analysis/reward-efficiency-proof-packet.md \
  --proof-profile reward-efficiency
```

Acceptance:

- The telemetry dir, task, slots, seed matrix, env count, episode length,
  precision, compile mode, and proof profile are explicit in closeout evidence.
- Packet emits one of `CONTINUE`, `REVISE_ALGORITHM`, `STOP_THEORY`, or a
  specific `BLOCKED_*` verdict.
- A `BLOCKED_*` packet is reported as a blocker, not as reward-theory evidence.

## Closeout Verification Stack

Run the proof-lane command above plus:

```bash
uv run pytest -m "not integration and not stress and not property and not slow" -x --cov=src --cov-report=json
PYTHONPATH=src uv run pytest -m integration -v -x
HYPOTHESIS_PROFILE=ci PYTHONPATH=src uv run pytest -m property -v -x --hypothesis-show-statistics
uv run python scripts/lint_leyline_types.py
uv run python scripts/lint_defensive_patterns.py
uv run python scripts/lint_gpu_sync.py
uv run ruff check src/ tests/
MYPYPATH=src uv run mypy -p esper
wardline scan . --fail-on ERROR
git diff --check
```

Slow/stress/CUDA runs are explicit package evidence or nightly validation, not
implicit per-child prerequisites.

## Next Action

Start the first child atomically:

```bash
filigree start-work esper-lite-441fbc6810 --assignee <agent> --actor <agent>
```

Do not run the long reward-efficiency exam until `esper-lite-2b077204e9` is
startable.
