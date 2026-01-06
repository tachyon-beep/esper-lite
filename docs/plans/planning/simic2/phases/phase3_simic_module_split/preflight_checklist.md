# Phase 3 Preflight Checklist (Simic Module Split)

## Objective
Prepare Phase 3 execution by locking scope, guardrails, and validation so the
Simic rewards and PPO agent splits land without behavior drift, new cycles, or
mega-module re-growth.

## Scope summary (Phase 3)
- Rewards split: break up `src/esper/simic/rewards/rewards.py`.
- PPO agent split: break up `src/esper/simic/agent/ppo.py`.
- Enforce directional imports and remove dead compatibility surfaces.

## Pre-Phase Activities (entry gates)
- [x] Phase 2 acceptance criteria marked complete (typed boundaries + telemetry parity).
- [x] Full test suite passes on the refactor branch.
- [x] Lint and types are clean (`uv run ruff check src/ tests/`, `uv run mypy src/`).
- [x] Throughput baseline captured for a short PPO run; no regression vs Phase 2 baseline.
- [x] Throughput baseline method + numbers recorded (avoid ambiguous comparisons).
- [x] Telemetry xfails (TELE-600–603) tracked as separate work and not in Phase 3 scope.

## Planning and risk-reduction activities

### A) Rewards split plan (single source of truth)
- [x] Decide module names and locations:
  - `shaping.py` owns stage potentials and PBRS utilities.
  - `contribution.py` owns contribution-primary reward computation (pure).
  - `loss_primary.py` owns loss-primary reward computation (pure).
  - `rewards.py` is a thin dispatcher OR removed entirely with call sites updated.
- [x] Confirm there will be **one** definition site for `STAGE_POTENTIALS` / PBRS potentials.
- [x] Define the canonical import path (no dead names, no compatibility shims).
- [x] Define the *only* supported exports in `simic/rewards/__init__.py` (or dispatcher).
- [x] Forbid external imports of internal reward modules (`contribution`, `shaping`, etc.).
- [x] List the functions to delete after extraction (no legacy wrappers retained).

### B) PPO agent split plan (orchestration vs math)
- [x] Decide module names and locations:
  - `ppo_agent.py`: PPOAgent surface, checkpointing, buffer orchestration.
  - `ppo_update.py`: update math only (losses, ratios, clipping, KL stop).
  - `ppo_metrics.py`: metrics dataclasses + conversion from tensors to CPU scalars.
- [ ] Confirm `ppo_update.py` has **no telemetry imports**.
- [ ] Define a typed `PPOUpdateResult` (or equivalent) returned by update math.
- [ ] Centralize `.item()/.cpu()/.tolist()` usage in `ppo_metrics.py` only.
- [ ] Enforce serialization boundary:
  - `ppo_update.py` returns tensors/booleans only (no Python floats/lists).
  - `ppo_metrics.py` is the only tensor → Python conversion site.
  - `ppo_agent.py` consumes typed results without further conversions.

### C) Import directionality and cycle prevention
- [x] Document allowed import directions:
  - training → agent/rewards/telemetry
  - agent → buffer/policy
  - rewards → leyline + reward types
  - telemetry may import leyline types and telemetry payload types
  - telemetry may import `ppo_metrics` dataclasses only if CPU-only and whitelisted
  - telemetry must not import training or `ppo_update`
- [x] Add a small import-direction test (or extend existing import isolation tests)
      to assert the above edges and reject cycles.
- [x] Add a targeted "no cycles" check:
  - `simic/agent/ppo_update.py` must not import `simic/telemetry/*`
  - `simic/telemetry/*` must not import `simic/agent/ppo_update.py`
  - `simic/rewards/*` must not import `simic/training/*`
- [x] Scan for any new back-edges before coding (rg/graph audit).

### D) Design spikes (timeboxed)
- [x] PBRS extraction spike: move stage potentials to a single module and update imports.
- [x] PPO metrics spike: create dataclasses + conversion logic, ensure call sites unchanged.
- [ ] PPO update spike: move loss/ratio math into `ppo_update.py`, return typed result.
- [ ] After each spike, run a short PPO baseline + telemetry ordering test.

## Risk reduction: correctness
- [ ] Preserve invariants:
  - `HEAD_NAMES` ordering and head tensor shapes
  - `SlotConfig` ordering invariants
  - Rollout buffer shapes and indexing semantics
- [x] Ensure reward function outputs are unchanged for all modes.
- [x] Ensure PPO update results are unchanged (losses, KL early stop, finiteness gates).
- [x] Add golden tests:
  - fixed reward input fixtures across modes (outputs stable)
  - fixed PPO buffer fixtures (losses, KL, finiteness gates stable)
- [x] Add one meta-test to assert a single `STAGE_POTENTIALS` definition site.

## Risk reduction: telemetry and contracts
- [x] Telemetry payload keys and types unchanged (decision metrics + PPO updates).
- [x] Event ordering unchanged (batch tail ordering test passes).
- [ ] Telemetry code does not import PPO update internals after split.

## Risk reduction: performance and torch.compile
- [ ] No new `.cpu()`/`.item()` calls in hot paths outside `ppo_metrics.py`.
- [ ] No per-step/per-env allocations added in the PPO update loop.
- [ ] Typed objects do not cross into `torch.compile` regions.
- [x] Throughput baseline within tolerance (no measurable regression).

## Risk reduction: multiprocessing / pickling
- [ ] Typed update/metrics objects are not captured by DataLoader workers.
- [ ] If any cross-process usage appears, add a targeted pickling test.

## Implementation planning artifacts
- [x] Per-file change list (new files + removed code) with owners.
- [x] Call-site update list (agent imports, reward imports, test updates).
- [x] Deletion list for old symbols once new modules land.
- [x] Test plan:
  - targeted unit tests for rewards split and PPO update
  - existing PPO/regression suites
  - import-direction test
  - telemetry decision metrics tests

## Acceptance criteria (Phase 3 done means)
- [ ] Rewards split complete; no mega-module mixing contribution/loss/PBRS/dispatch.
- [ ] PPO agent split complete; `ppo_update.py` is math-only, `ppo_agent.py` is orchestration.
- [ ] Single source of PBRS potentials enforced.
- [ ] Directional imports respected; no new cycles.
- [ ] Telemetry payloads/order unchanged.
- [ ] Full suite + ruff + mypy pass.
- [ ] Throughput baseline within tolerance.

## Notes
- No compatibility shims or legacy wrappers. Update call sites and delete old names.
- Telemetry xfails remain a separate body of work until Simic refactor closure.

## Execution notes (Phase 3 preflight)
- Full test suite: `UV_CACHE_DIR=.uv-cache uv run pytest`
  - 4247 passed, 36 skipped, 69 deselected, 4 xfailed
- Lint: `UV_CACHE_DIR=.uv-cache uv run ruff check src/ tests/` (clean)
- Types: `UV_CACHE_DIR=.uv-cache uv run mypy src/` (clean, 166 files)
- Throughput baseline run:
  - Command: `PYTHONPATH=src UV_CACHE_DIR=.uv-cache uv run python -m esper.scripts.train ppo --preset cifar_baseline --task cifar_baseline --rounds 1 --envs 1 --episode-length 5 --telemetry-dir telemetry --device cpu --devices cpu --num-workers 0`
  - Run dir: `telemetry/telemetry_2026-01-06_205739`
  - Total episode time (TRAINING_STARTED → last EPOCH_COMPLETED): 39.277398s
  - Episodes/sec: 0.025460
  - Per-epoch deltas: [7.500104s, 7.575927s, 7.99248s, 7.513749s], avg 7.645565s
  - Phase 2 baseline for comparison: 44.754748s total, 0.022344 episodes/sec
- PBRS spike validation run:
  - Command: `PYTHONPATH=src UV_CACHE_DIR=.uv-cache uv run python -m esper.scripts.train ppo --preset cifar_baseline --task cifar_baseline --rounds 1 --envs 1 --episode-length 5 --telemetry-dir telemetry --device cpu --devices cpu --num-workers 0`
  - Run dir: `telemetry/telemetry_2026-01-06_212121`
- Telemetry ordering test: `UV_CACHE_DIR=.uv-cache uv run pytest tests/simic/telemetry/test_emitters.py::test_batch_tail_event_order_is_stable`
- PPO metrics spike validation run:
  - Command: `PYTHONPATH=src UV_CACHE_DIR=.uv-cache uv run python -m esper.scripts.train ppo --preset cifar_baseline --task cifar_baseline --rounds 1 --envs 1 --episode-length 5 --telemetry-dir telemetry --device cpu --devices cpu --num-workers 0`
  - Run dir: `telemetry/telemetry_2026-01-06_213352`
  - Tests: `UV_CACHE_DIR=.uv-cache uv run pytest tests/simic/test_ppo_update_golden.py`
  - Telemetry ordering test: `UV_CACHE_DIR=.uv-cache uv run pytest tests/simic/telemetry/test_emitters.py::test_batch_tail_event_order_is_stable`

## Planning artifacts
### Per-file change list (planned)
- `src/esper/simic/rewards/shaping.py` (owner: simic-rewards) - PBRS potentials + shaping utilities
- `src/esper/simic/rewards/contribution.py` (owner: simic-rewards) - contribution reward math
- `src/esper/simic/rewards/loss_primary.py` (owner: simic-rewards) - loss-primary reward math
- `src/esper/simic/rewards/rewards.py` (owner: simic-rewards) - thin dispatcher only
- `src/esper/simic/agent/ppo_agent.py` (owner: simic-agent) - PPOAgent orchestration
- `src/esper/simic/agent/ppo_update.py` (owner: simic-agent) - update math + typed result
- `src/esper/simic/agent/ppo_metrics.py` (owner: simic-agent) - metrics dataclasses + CPU conversions
- `src/esper/simic/agent/ppo.py` (owner: simic-agent) - removed or reduced to dispatcher (no shims)

### Call-site update list (planned)
- Update agent imports to `simic/agent/ppo_agent.py` if `ppo.py` is removed.
- Update reward imports to `simic/rewards/__init__.py` or dispatcher (no direct imports of internal modules).
- Update tests referencing `simic.rewards.rewards` to the canonical import path.

### Deletion list (planned)
- Remove `rewards.py` implementations moved into `contribution.py`, `loss_primary.py`, `shaping.py`.
- Remove any dead wrappers that only forward to the new modules.
- Remove `ppo.py` legacy helpers once `ppo_agent.py`/`ppo_update.py` are canonical.

### Test plan (planned)
- Rewards golden tests (added) + reward mode/unit tests
- PPO update golden test (added) + PPO regression suite
- Import-direction guardrail test (added)
- Telemetry decision metrics + batch tail ordering tests (existing)
