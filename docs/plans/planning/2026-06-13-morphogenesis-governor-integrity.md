# Morphogenesis Governor Integrity Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task after the plan is reviewed and promoted to ready.

**Goal:** Remove the highest-risk confounders in Esper's morphogenetic control loop so the next proof run measures lifecycle behavior rather than rollback ordering, contaminated observations, or non-auditable mutation events.

**Architecture:** Tamiyo proposes actions, Tolaria adjudicates unsafe mutation boundaries, Kasmina applies approved mutations, Simic receives typed reward/penalty signals, and Karn/Nissa receive causal evidence. This package does not prove reward efficiency; it makes the proof surface credible enough to run.

**Tech Stack:** Python 3.11, PyTorch, uv, pytest, Hypothesis, Simic PPO training loop, Tamiyo observation extraction, Kasmina seed slots/blueprints, Tolaria governor, Leyline telemetry contracts.

**Prerequisites:**
- Read `CLAUDE.md`, `README.md`, `ROADMAP.md`, and `docs/arch-analysis-2026-06-13-0836/01-kasmina-tolaria-blueprint-health.md`.
- Use Loomweave before broad call/reference edits.
- Do not introduce defensive bug-hiding patterns or compatibility shims.
- Run Wardline if edits touch telemetry ingestion, file/network input, or other trust boundaries.

```yaml
# Plan Metadata
id: morphogenesis-governor-integrity
title: Morphogenesis Governor Integrity
type: planning
created: 2026-06-13
updated: 2026-06-13
owner: Codex

urgency: high
value: Make morphology proof runs interpretable by fixing rollback ordering, observation truthfulness, blueprint contracts, and the first causal proposal/verdict/mutation evidence path.

complexity: L
risk: high
risk_notes: Crosses Simic action execution, Tamiyo policy features, Kasmina germination contracts, Tolaria rollback/governor authority, and telemetry. The main risk is accidentally changing policy behavior while trying to make evidence fail closed.

depends_on:
  - proof-confounder-drain
soft_depends:
  - karn-telemetry-quality-arc
  - telemetry-domain-sep
blocks:
  - ppo-stability-oracle-sandbox
  - reward-efficiency
  - proof-baseline-controls
  - counterfactual-oracle

status_notes: Drafted from the 2026-06-13 Kasmina/Tolaria/Blueprint health report. Needs DRL, PyTorch, Python architecture, and training-stability review before promotion to ready.
percent_complete: 0

reviewed_by: []
```

---

## Outcome

At closeout, a rollback env cannot execute a stale sampled lifecycle mutation in the same step, snapshots cannot bless already-divergent state, active-slot missing telemetry cannot look healthy, counterfactual absence cannot leak host drift into the policy value channel, malformed blueprints fail at germination, and morphology events have a minimal stable identity that can be followed across proposal, decision, mutation, reward, and rollback telemetry.

## Task 1: Fix Rollback Ordering and Halt Semantics

**Primary files:**
- `src/esper/simic/training/vectorized_trainer.py`
- `src/esper/simic/training/action_execution.py`
- `tests/simic/training/test_governor_integration.py`
- New focused tests under `tests/simic/training/` as needed

**Steps:**

1. Move Tolaria vital-sign checks before periodic and fossilization-triggered snapshots.
2. Snapshot only after the current validation state is known clean.
3. When an env rolls back inside `execute_actions()`, emit a typed terminal rollback outcome and skip normal lifecycle mutation for that env in the current step.
4. Ensure the next policy decision observes post-rollback state.
5. Preserve existing CUDA stream contracts around snapshot and rollback.

**Definition of Done:**
- [ ] A regression test proves a panic epoch does not replace the last-good snapshot.
- [ ] A regression test proves rollback prevents germinate/prune/alpha/advance mutation in the same action step.
- [ ] Existing governor integration tests pass.

## Task 2: Fix Observation Truthfulness and Blueprint Contracts

**Primary files:**
- `src/esper/tamiyo/policy/features.py`
- `src/esper/kasmina/slot.py`
- `src/esper/kasmina/blueprints/registry.py`
- `tests/tamiyo/policy/`
- `tests/kasmina/` or existing blueprint contract suites

**Steps:**

1. For active slots, stop converting missing telemetry into healthy gradient features. If a bootstrap exception is legitimate, expose an explicit missing/freshness feature or fail closed at the call site.
2. Remove the fallback from missing `counterfactual_contribution` to `improvement_since_stage_start` in Tamiyo observations. Use neutral contribution plus explicit freshness/missing signals.
3. Make Kasmina blending gates fail closed when telemetry required for safety is missing outside the allowed bootstrap window.
4. Reject non-tensor blueprint smoke-test outputs immediately before shape validation.
5. Reject duplicate blueprint registrations and unknown blueprint IDs loudly rather than encoding them as sentinel values.

**Definition of Done:**
- [ ] Active-slot missing telemetry cannot yield `gradient_health == 1.0` by default.
- [ ] Missing counterfactual contribution cannot enter the contribution value channel through host drift.
- [ ] Malformed blueprints returning tuple/list/object fail at germination.
- [ ] Duplicate registry keys and unknown blueprint IDs have direct tests.

## Task 3: Add Minimal Tolaria Pre-Flight Proposal/Verdict Boundary

**Primary files:**
- `src/esper/tolaria/`
- `src/esper/simic/training/action_execution.py`
- `src/esper/leyline/` for shared proposal/verdict contracts if needed
- `tests/tolaria/`
- `tests/simic/training/`

**Steps:**

1. Define typed lifecycle mutation proposals with operation, slot, blueprint/action indices, alpha schedule, current health envelope, budget/rent state, cooldown state, and event identity.
2. Add a Tolaria-owned pre-flight API that returns an allow/deny verdict and reason.
3. Route Simic lifecycle mutation through the verdict before calling Kasmina mutation APIs.
4. Keep the first implementation conservative: it can begin with existing masks and health checks, but the authority boundary must be real and testable.
5. Emit denial telemetry that distinguishes invalid proposal, safety veto, cooldown veto, and budget/rent veto.

**Definition of Done:**
- [ ] Germinate, prune, alpha target, and advance pass through the pre-flight API.
- [ ] Denied proposals do not mutate Kasmina state.
- [ ] Verdict reason is visible in action outcome telemetry.
- [ ] Tests cover at least one allowed and one denied proposal per lifecycle op family.

## Task 4: Add Minimal Deterministic Morphogenesis Event Identity

**Primary files:**
- `src/esper/leyline/telemetry.py`
- `src/esper/simic/training/action_execution.py`
- `src/esper/kasmina/slot.py`
- `src/esper/simic/telemetry/emitters.py`
- Karn views/tests if new fields are exposed

**Steps:**

1. Define a stable morphology event identity that ties together proposal, governor verdict, mutation commit, post-event watch, rollback/fossilization, reward mode, and policy action indices.
2. Capture enough RNG identity for growth debugging: explicit generator, seed, or pre/post RNG state for blueprint creation and shape probes.
3. Add observation hash or equivalent deterministic observation fingerprint at proposal time.
4. Ensure event identity survives into reward and telemetry records without relying on generic UUID-only event IDs.
5. Keep full replay out of scope; this task establishes the audit spine that replay can later use.

**Definition of Done:**
- [ ] A single germination event can be traced from proposal through mutation and reward telemetry.
- [ ] Rollback/fossilization events link back to the proposal or mutation they bless/revert.
- [ ] Tests verify event IDs are stable under fixed seed/action inputs.

## Task 5: Collapse Duplicate Lifecycle Execution Semantics

**Primary files:**
- `src/esper/simic/training/action_execution.py`
- `src/esper/simic/training/handlers/`
- Handler tests and production action-execution tests

**Steps:**

1. Decide whether the handler registry becomes the production mutation executor or is deleted.
2. Do not keep two sources of lifecycle semantics.
3. If wiring handlers in, route them through the Tolaria pre-flight verdict and preserve action outcome telemetry.
4. If deleting handlers, remove exports and tests that preserve unused abstraction behavior.

**Definition of Done:**
- [ ] Production has one authoritative lifecycle execution path.
- [ ] No tests assert behavior for an unused duplicate executor.
- [ ] No compatibility adapter remains.

## Task 6: Repair Proof-Facing Reward and Decision Telemetry

**Primary files:**
- `src/esper/simic/rewards/reward_telemetry.py`
- `src/esper/simic/telemetry/emitters.py`
- `src/esper/simic/training/vectorized_trainer.py`
- `src/esper/simic/rewards/contribution.py`
- Karn MCP/view tests if surfaced

**Steps:**

1. Remove or delay `RewardComponentsTelemetry.action_success` until actual action outcome is known.
2. Ensure reward components cannot claim action success before execution.
3. Compute real per-head decision entropy for proof-facing telemetry, or explicitly mark it unavailable so UI/proof packets do not treat placeholders as evidence.
4. Either add structural rent to `SIMPLIFIED` when it is used for blueprint health claims, or label it as a diagnostic mode that cannot prove structural economy.

**Definition of Done:**
- [ ] Reward telemetry cannot disagree with action outcome telemetry about success.
- [ ] Proof packets cannot treat unavailable entropy as real per-decision entropy.
- [ ] `SIMPLIFIED` reward semantics are explicit in tests and docs.

## Validation Commands

Focused commands:

```bash
PYTHONPATH=src uv run pytest tests/simic/training/test_governor_integration.py -q
PYTHONPATH=src uv run pytest tests/tamiyo/policy tests/tamiyo/properties/test_feature_extraction_properties.py -q
PYTHONPATH=src uv run pytest tests/kasmina tests/tolaria -q
PYTHONPATH=src uv run pytest tests/simic/telemetry tests/simic/rewards -q
```

Guardrails:

```bash
uv run python scripts/lint_defensive_patterns.py
uv run python scripts/lint_gpu_sync.py
uv run ruff check src/ tests/
MYPYPATH=src uv run mypy -p esper
```

Proof rehearsal after fixes:

```bash
PYTHONPATH=src uv run python -m esper.scripts.train ppo --task cifar_impaired --dual-ab shaped-vs-simplified --rounds 2 --envs 2 --episode-length 25
PYTHONPATH=src uv run python scripts/proof_packet.py <run-dir>
```

## Out of Scope

- Full deterministic replay infrastructure.
- Counterfactual oracle training.
- Blueprint compiler implementation.
- TinyStories validation runs.
- Long reward-efficiency A/B execution.
- New dashboard features unrelated to the new event/verdict fields.
