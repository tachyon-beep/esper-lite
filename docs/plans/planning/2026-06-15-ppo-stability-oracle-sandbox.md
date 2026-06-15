# PPO Stability / Oracle Sandbox Plan

> **For Claude:** Do not use PPO reward-efficiency failures as theory evidence until this sandbox can drive the same lifecycle mechanics with a deterministic oracle policy.

**Goal:** Prove Kasmina/Tolaria/Simic lifecycle mechanics and reward accounting independently of PPO learning. If the oracle policy cannot produce the expected lifecycle trace and packet verdict, the fault is mechanics or math, not PPO or theory.

**Architecture:** Extract the existing scripted lifecycle smoke test into a small proof-grade oracle runner. The runner applies a declared factored-action schedule under the normal action mask, lifecycle gate, governor, reward, Nissa telemetry, and Karn packet surfaces. It must not bypass the mechanics it is proving.

**Tech Stack:** Python 3.11, PyTorch CPU-first micro fixture, pytest, Karn telemetry views, `scripts/proof_packet.py`.

```yaml
# Plan Metadata
id: ppo-stability-oracle-sandbox
title: PPO Stability / Oracle Sandbox
type: planning
created: 2026-06-15
updated: 2026-06-15
owner: Codex

urgency: high
value: Separates PPO learning failure from lifecycle-mechanics failure before the expensive reward-efficiency exam.

complexity: M
risk: high
risk_notes: The main risk is building a fake oracle path that bypasses action masks, lifecycle gates, governor rollback, or telemetry contracts.

depends_on:
  - correctness-proof-strategy
soft_depends:
  - morphogenesis-governor-integrity
  - proof-baseline-controls
blocks:
  - reward-efficiency
  - counterfactual-oracle

status_notes: Planning artifact created. Current repo has `tests/integration/test_scripted_policy_runner.py`, but it is a smoke test only: it does not emit proof telemetry or produce a packet. `scripts/proof_packet.py` is also still PPO-update-learnability oriented, so an oracle proof profile or oracle telemetry bridge is required.
percent_complete: 20

reviewed_by:
  - reviewer: drl-expert
    date: 2026-06-15
    verdict: needs-revision
    notes: Requires formal review before promotion to ready; current notes are Codex-derived from deep-RL evaluation guidance.
  - reviewer: pytorch-expert
    date: 2026-06-15
    verdict: needs-revision
    notes: Requires formal review before promotion to ready; checkpoint and determinism boundaries must be checked.
  - reviewer: quality-engineering
    date: 2026-06-15
    verdict: needs-revision
    notes: Requires executable acceptance tests, not prose-only approval.
```

## Current Implemented Surface

- `tests/integration/test_scripted_policy_runner.py` proves a hand-authored factored action sequence can pass masks and lifecycle gates on a minimal host.
- `apply_proof_baseline_action_controls()` can force WAIT-only controls, and refuses fake unsupported controls.
- `scripts/proof_packet.py --proof-profile reward-efficiency` now makes final-exam packet generation require precision provenance and blueprint-health baselines.

## Missing Executable Machinery

1. A source module, likely `src/esper/simic/training/oracle_sandbox.py`, that runs a deterministic lifecycle schedule through the real action execution path.
2. A tiny deterministic task/host fixture suitable for CI that completes in seconds and produces stable expected outcomes.
3. Oracle telemetry that Karn can ingest:
   - `TRAINING_STARTED` with precision provenance and `proof_profile=oracle_sandbox`.
   - lifecycle mutation events with seed/slot identity.
   - episode outcomes with valid ROI fields.
   - an oracle-policy evidence event or a packet profile that does not require PPO learnability telemetry.
4. Packet tests showing the oracle sandbox can reach a product verdict only when the lifecycle trace, outcome contract, and confounder ledger are clean.

## Implementation Tasks

1. Move the scripted action application helper out of the integration test into a source module, preserving the test as a consumer.
2. Add a deterministic oracle schedule format: germinate, advance to training, blend/hold, set alpha, fossilize or prune.
3. Run the schedule through masks, lifecycle gates, governor checks, and action handlers; do not mutate slots directly except through existing public lifecycle APIs.
4. Emit proof-grade telemetry into a temp telemetry directory.
5. Add `proof_profile="oracle-sandbox"` support to the proof packet or add a distinct oracle packet command. It must fail closed on malformed telemetry, missing outcomes, invalid lifecycle trace, and mechanics confounders.
6. Add tests:
   - oracle schedule succeeds on expected trace,
   - invalid schedule is blocked by masks/gates,
   - missing oracle evidence cannot produce a product verdict,
   - a clean oracle packet reaches `CONTINUE` or `REVISE_ALGORITHM`.

## Definition of Done

- One CI-safe command runs the oracle sandbox and emits Karn-ingestable telemetry.
- One packet command produces a typed verdict for the oracle sandbox.
- The sandbox fails if action masks, lifecycle gates, rollback/governor checks, or telemetry contracts are bypassed.
- A passing sandbox is required before any PPO learning failure is interpreted as algorithmic rather than mechanical.
