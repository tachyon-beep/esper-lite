# Simic2: Vectorized PPO Maintainability Refactor Roadmap

**Goal:** Make Simic’s vectorized PPO training and adjacent modules easier to modify safely without changing behavior, performance characteristics, or telemetry semantics.

**Primary pain point:** `src/esper/simic/training/vectorized.py` mixes responsibilities (data iteration, per-env model training, fused counterfactual evaluation, action interpretation/execution, reward shaping, PPO updates, anomaly handling, telemetry, checkpointing) and hides most logic inside nested closures.

This roadmap defines phases that keep changes shippable and reviewable.

---

## Scope

**In scope**
- Mechanical refactor and module decomposition of:
  - `src/esper/simic/training/vectorized.py`
  - `src/esper/simic/training/parallel_env_state.py` (only as needed to clarify invariants)
  - `src/esper/simic/rewards/rewards.py`
  - `src/esper/simic/agent/ppo.py`
- Typing/contract improvements that eliminate dict-key drift and reduce reliance on optional-key patterns.

**Out of scope**
- Algorithm changes (PPO math, reward design changes, action mask semantics changes)
- New telemetry features (unless required to preserve existing signals after refactor)
- Throughput optimization work beyond “don’t regress”

---

## Phase 0: Baseline + refactor safety rails (docs + tests)

Workspace: `docs/plans/planning/simic2/phases/phase0_baseline_and_tests/README.md`

**Deliverables**
- Baseline “counts” and sanity checks (telemetry events, import graph, test/lint).
- Unit tests for pure helper logic we plan to extract (annealing step calc, action validity rules).

**Why**
- A refactor that changes behavior is hard to detect in this subsystem because most signals are emergent (training curves). Baselines give us fast regressions.

---

## Phase 1: Split `vectorized.py` into trainer + extracted modules (no behavior change)

Workspace: `docs/plans/planning/simic2/phases/phase1_vectorized_modular_split/README.md`

**Deliverables**
- Introduce `VectorizedPPOTrainer` and move the main loop out of `train_ppo_vectorized`.
- Extract coherent subsystems into separate modules:
  - env creation/reset and slot wiring
  - batch ops (train + fused val)
  - counterfactual evaluator
  - action interpretation/execution
- Keep `train_ppo_vectorized(...)` signature and semantics stable as the public entrypoint.

**Guardrails**
- Preserve inverted control flow (batch-first iteration).
- Preserve CUDA stream semantics and synchronization points.
- No additional per-env/per-head device syncs in the hot path.

---

## Phase 2: Typed boundaries (reduce dict surfaces, make dependencies explicit)

Workspace: `docs/plans/planning/simic2/phases/phase2_typed_contracts_and_api/README.md`

**Deliverables**
- Introduce dataclasses for:
  - action decoding/validation (`ActionSpec`, `ActionOutcome`)
  - reward inputs (`RewardInputs`) so reward plumbing is not a 20+ argument list
  - per-batch summary (`BatchSummary`) instead of ad-hoc dict merging
- Replace “optional dict keys” patterns in internal code with typed objects or required keys.

**Decision point**
- Keep the external API stable (preferred) vs change entrypoints to be config-object-only. If we change, do it in one PR with all call sites updated (no dual path).

---

## Phase 3: Decompose other large Simic modules (rewards + agent)

Workspace: `docs/plans/planning/simic2/phases/phase3_simic_module_split/README.md`

**Deliverables**
- Split `rewards/rewards.py` by family/shaping/telemetry boundaries.
- Split `agent/ppo.py` into “agent surface + checkpointing” vs “update internals + metrics”.
- Preserve import surfaces via `__init__.py` re-exports *only if* they don’t become compatibility shims; prefer updating call sites and deleting old names.

---

## Promotion criteria (from planning → ready)

Promote a phase to `docs/plans/ready/` when:
- the phase is implementable as a single PR series (or a small PR stack),
- all call sites are enumerated,
- validation commands and acceptance criteria are explicit.

