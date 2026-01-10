# Simic2 Planning Footprint (Maintainability Refactor Sprawl Map)

This document defines the planning footprint for refactoring Simic’s PPO training implementation without changing behavior.

Non-negotiables from `CLAUDE.md` and `ROADMAP.md` apply:
- **No legacy/backcompat**: contract changes update *all* call sites; no dual paths or compatibility shims.
- **No bug-hiding defensive patterns**: fail fast on contract violations; do not add `.get()`, `getattr()`, `hasattr()`, silent exception handling.
- **GPU-first inverted control flow**: preserve the batch-first, stream-parallel structure in `src/esper/simic/training/vectorized.py`.
- **Sensors match capabilities**: refactors must not drop telemetry signals or make them harder to emit.
- **Metaphors**: organism for domains; botanical only for the seed lifecycle.

---

## 1) Planning artifact set (what lives in this folder)

**A. Roadmap (strategy)**
- Canonical roadmap: `docs/plans/planning/simic2/simic_vectorized_refactor_roadmap.md`

**B. Phase plans (execution)**
- Each phase folder is a working checklist; when it becomes implementable, promote a single plan into `docs/plans/ready/`.

**C. Blast-radius tracking**
- For each phase, explicitly list:
  - files modified/moved
  - API/contract changes (if any) and downstream call sites
  - validation commands and “done means” criteria

---

## 2) Target steady-state layout (direction, not a hard commitment)

We want `train_ppo_vectorized()` to be a small orchestration entrypoint, delegating into a few testable modules.

Proposed direction for `src/esper/simic/training/`:

```
src/esper/simic/training/
  vectorized.py                  # public entrypoint + minimal glue
  vectorized_trainer.py          # VectorizedPPOTrainer (main loop)
  vectorized_types.py            # dataclasses for context/results
  env_factory.py                 # create/reset ParallelEnvState + model wiring
  batch_ops.py                   # train/val batch kernels (stream-safe)
  counterfactual_eval.py         # fused validation/ablation evaluator
  action_execution.py            # decode/validate/execute actions per env
```

Proposed direction for large Simic modules:

```
src/esper/simic/agent/
  ppo_agent.py                   # PPOAgent orchestration + checkpoint surface
  ppo_update.py                  # update() internals + loss/ratio/value metrics
  rollout_buffer.py              # stays (already isolated)
  types.py                       # metrics types (prefer dataclasses over dict)

src/esper/simic/rewards/
  contribution.py                # contribution-primary reward family
  loss_primary.py                # loss-primary reward family
  shaping.py                     # PBRS + intervention costs
  reward_telemetry.py            # stays (typed telemetry)
```

Notes:
- Keep public import surfaces stable until we explicitly decide otherwise (and then change all call sites in one PR).
- Prefer “move then fix imports” over duplicating code.

---

## 3) Cross-domain impact matrix (what changes where)

This effort is Simic-centered, but refactors will touch other organs via contracts.

### Simic (Evolution) — primary scope
Primary files:
- `src/esper/simic/training/vectorized.py`
- `src/esper/simic/training/parallel_env_state.py`
- `src/esper/simic/telemetry/emitters.py`
- `src/esper/simic/agent/ppo.py`
- `src/esper/simic/rewards/rewards.py`

### Tamiyo (Brain) — observation/features and action masks
Potential touchpoints (should remain behavior-identical):
- `src/esper/tamiyo/policy/features.py` (Obs V3 feature assembly is used by vectorized training)
- `src/esper/tamiyo/policy/action_masks.py` (mask semantics are part of Simic’s action contract)

### Leyline (DNA) — shared contracts
Expected to remain stable, but any typing hardening may motivate follow-up:
- `src/esper/leyline/telemetry.py` (payload/event schemas)
- `src/esper/leyline/factored_actions.py` (head naming/order contracts)

### Tolaria / runtime (Metabolism) — training loop integration
Call sites and contracts:
- `src/esper/scripts/train.py` (CLI entry into `train_ppo_vectorized`)
- `src/esper/runtime/*` and `src/esper/tolaria/*` (task specs and model creation)

### Karn / Nissa — telemetry consumers
Refactors must preserve event semantics and payload shapes:
- Sanctum/Overwatch and MCP views (stringified event types, payload field names)

---

## 4) PR sizing recommendation (no backcompat means “end-to-end per slice”)

**PR0 (docs-only)**
- Add/iterate this planning workspace.

**PR1 (Phase 1)**
- Split `vectorized.py` into a trainer + extracted modules while preserving the public entrypoint.
- Add small, pure tests for extracted helper logic (no GPU required).

**PR2 (Phase 2)**
- Introduce typed context/results objects (`RewardInputs`, action/result dataclasses).
- Remove nested closures; make dependencies explicit.

**PR3 (Phase 3)**
- Split `rewards/rewards.py` by family/shaping.
- Split `agent/ppo.py` by responsibility (agent surface vs update internals).

---

## 5) “Done means” (maintainability acceptance criteria)

- `src/esper/simic/training/vectorized.py` is < ~500 LOC and contains no nested helper functions.
- The main loop lives in a dedicated object (`VectorizedPPOTrainer`) with explicit dependencies.
- Pure helper logic has unit tests (mask/action decoding, anneal step calculation, config validation).
- Import cycles are reduced (fewer lazy imports solely to avoid runtime import loops).
- Telemetry event emission and payload shapes are unchanged (or changed deliberately with an explicit blast-radius list and updated consumers).
- Throughput does not regress in the common case (CIFAR on 1–2 GPUs, inverted control flow preserved).

