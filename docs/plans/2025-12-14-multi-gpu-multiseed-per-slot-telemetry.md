# Multi‑GPU + Multi‑Seed Hardening Plan (2025‑12‑14)

## Context

We just landed two major capabilities:

1. **Research‑grade telemetry** (Karn/Nissa pipeline)
2. **Multi‑seed per environment** (3 slots per env: `early`, `mid`, `late`)

This plan is the shortest path to a credible “sign‑off” that:

- Multi‑seed behavior is correct end‑to‑end (Kasmina ⇄ Tamiyo ⇄ Simic ⇄ Tolaria)
- Vectorized PPO can run across multiple GPUs reliably
- PPO gets **per‑slot telemetry** in its observation space (original design intent)

## Definitions (so “multi‑GPU” is unambiguous)

This plan treats “multi‑GPU” as **two distinct capabilities**:

1. **Multi‑GPU env parallelism (P0/P1):** one Python process runs `n_envs` environments, distributing envs across `--devices` (e.g. `cuda:0 cuda:1`) and overlapping work with CUDA streams (`src/esper/simic/vectorized.py`).  
   - This is the fastest path to multi‑GPU throughput and the one we can ship without major rework.
2. **True DDP model training (P2, optional):** one environment’s host model trains with `torch.distributed` across GPUs (DDP).  
   - This is *not* currently safe due to dynamic seed parameter creation + counterfactual (`force_alpha`) semantics.

## Current Status (what looks correct vs what blocks sign‑off)

### Looks correct (ready to build on)

- **Kasmina multi‑slot execution**: `MorphogeneticModel.forward()` applies all active slots (`src/esper/kasmina/host.py`).
- **Vectorized PPO multi‑slot mechanics**:
  - Per‑slot optimizers (no dynamic param‑group surgery) and per‑slot lifecycle stepping (`src/esper/simic/vectorized.py`).
  - Per‑slot counterfactual validation path exists (alpha‑0 ablations) and is used for reward attribution.
- **Multi‑GPU guardrails**:
  - Fail‑fast CUDA index validation in vectorized PPO.
  - Model factory checks invalid CUDA device requests early (`src/esper/tolaria/environment.py`).

### Blocks sign‑off today

1. **Per‑slot telemetry is not in PPO observations yet**
   - `signals_to_features()` appends telemetry for only “first active seed” (`src/esper/simic/ppo.py`).
2. **Tamiyo multi‑seed is a partial pass**
   - `SignalTracker` collapses seed summary fields to `active_seeds[0]` (`src/esper/tamiyo/tracker.py`).
   - Heuristic baseline is effectively **single‑slot** (`src/esper/simic/training.py:run_heuristic_episode` selects `target_slot = slots[0]`).
3. **True DDP is not currently safe**
   - `SeedSlot.force_alpha()` mutates module state and is explicitly “not DDP safe” (`src/esper/kasmina/slot.py`).
   - Gate consensus uses `all_reduce` and can deadlock if ranks diverge in call order (`src/esper/kasmina/slot.py:_sync_gate_decision`).

## Non‑Negotiables (No Legacy / No Redundancy)

- **No backwards compatibility / no shims.** If a new observation schema, telemetry path, or training entrypoint replaces an old one, we delete the old code and update all call sites and tests in the same PR.
- **No redundant systems.** When a subsystem is subsumed (e.g., “single‑seed” codepaths after per‑slot telemetry lands), we remove the redundant path rather than keeping it “just in case”.
- **Breaking changes are expected.** Example: PPO checkpoint incompatibility due to observation shape changes is acceptable; we do not add compatibility loaders.

## Risk / Complexity Scale (used below)

- **Complexity**
  - **Low:** 1–2 files, localized change, minimal cross‑domain coupling
  - **Medium:** touches multiple subsystems or changes a shared contract, requires coordinated test updates
  - **High:** distributed/discrete‑event correctness risks, broad coupling, hard to validate without real hardware
- **Risk**
  - **Low:** failure is obvious + tests cover it well
  - **Medium:** likely to regress behavior or break configs; testable with moderate effort
  - **High:** subtle correctness/perf issues; hard to validate locally (CUDA/DDP); high blast radius

---

# Deliverables

After executing this plan:

1. PPO observation space includes **per‑slot telemetry** (`early/mid/late`), deterministic ordering, zero‑padding for empty slots.
2. Tamiyo (SignalTracker + heuristic baseline) is **multi‑slot aware**:
   - correct `available_slots`
   - lifecycle stepping for *all* enabled slots
   - germination selects a slot explicitly (deterministic policy)
3. Multi‑GPU env parallelism is “sign‑off ready” with a runnable smoke test on real hardware.
4. DDP readiness is either:
   - explicitly deferred with clear blockers + TODOs, **or**
   - implemented in a minimal, safe “DDP v0” mode (only if we choose to pay the complexity).

---

# Work Plan (small PRs, no major rework)

## PR0 — Add a GPU smoke test harness (sign‑off gate for multi‑GPU)

### Why
We need a single command that an operator can run on a 2+ GPU machine to validate the “multi‑GPU env parallelism” path end‑to‑end (streams, transfers, optimizers, reward, telemetry).

### Risk / Complexity
- **Complexity:** Low
- **Risk:** Low
- **Main risks**
  - Flaky/slow GPU tests in CI if not properly skipped or scoped.
  - A “smoke test” that doesn’t actually exercise both GPUs (false confidence).
- **Mitigations**
  - Hard‑skip when CUDA not available and when `<2` GPUs.
  - Keep runtime tiny (≤1–2 episodes, small batches) and assert device assignment is non‑degenerate (e.g., envs map to both `cuda:0` and `cuda:1`).

### Changes
- Add a pytest that **skips cleanly** when CUDA is unavailable:
  - `tests/cuda/test_vectorized_multi_gpu_smoke.py`
  - Preconditions:
    - `torch.cuda.is_available()`
    - `torch.cuda.device_count() >= 2`
  - Run a tiny PPO job (e.g., 1 batch / 1–2 episodes) with `--devices cuda:0 cuda:1` and assert it completes.
- Add a documented smoke command in `README.md` (or `docs/plans/...` “How to validate” section if we prefer not to touch README yet).

### Acceptance criteria
- On a 2‑GPU machine: the smoke test completes without error and visibly uses both devices (operator can confirm via `nvidia-smi`).
- On CPU‑only machines: test is skipped, not failed.

---

## PR1 — Implement **per‑slot telemetry** in the PPO observation space

### Why
Multi‑seed is only learnable if the policy can “see” each slot’s local health/progress. Today telemetry is effectively single‑seed.

### Risk / Complexity
- **Complexity:** Medium
- **Risk:** Medium (can become High if telemetry freshness isn’t handled)
- **Main risks**
  - Observation shape change ripples through: PPO network input, rollout buffer, obs normalizer, tests, and checkpoints.
  - “Per‑slot telemetry” can be incorrect or stale if we only refresh telemetry for TRAINING/BLENDING slots (e.g., PROBATIONARY/FOSSILIZED slots might show outdated `stage/alpha/accuracy` if `SeedState.sync_telemetry()` isn’t called).
  - Determinism: slot iteration must be fixed (`early/mid/late`), not dict order.
- **Mitigations**
  - Update all dimension assertions/tests in the same PR; no compatibility shims.
  - Ensure telemetry snapshots are refreshed for **all active enabled slots** each epoch (even if gradient telemetry is unavailable, update the non‑gradient fields and keep gradient fields at “last known” or explicit zeros).
  - Add a targeted unit/integration test that checks telemetry slices align to the correct slot (not “first active seed”).

### Implementation decisions (keep it simple)
- Observation layout stays **fixed‑size**:
  - `MULTISLOT_FEATURE_SIZE` is already hardcoded for `early/mid/late` (50 dims).
  - Telemetry becomes `3 * SeedTelemetry.feature_dim()` (30 dims).
  - New `state_dim = 80` when telemetry is enabled.
- Ordering is deterministic: `[early telemetry][mid telemetry][late telemetry]`.
- Empty slot telemetry is zero‑padded.

### Changes
- `src/esper/simic/ppo.py:signals_to_features()`
  - Replace “first active seed telemetry” with per‑slot telemetry concatenation.
  - Ensure it never depends on dict iteration ordering.
- `src/esper/simic/vectorized.py:train_ppo_vectorized()`
  - Update `state_dim` computation accordingly.
  - Ensure observation normalizer shapes and PPOAgent input sizes match.
- Tests
  - Update any tests asserting `MULTISLOT_FEATURE_SIZE + SeedTelemetry.feature_dim()` (expected now: `+ 3*feature_dim()`).
  - Add a focused test that asserts:
    - the telemetry slice for an active slot is non‑zero
    - other slots remain zero when inactive

### Checkpoints / Compatibility
- **Breaking change:** PPO checkpoints that encode obs normalizer statistics will not be compatible across this observation shape change.
- Per repo policy, we do **not** add compatibility shims; we update call sites and tests only.

### Acceptance criteria
- PPO rollout collection runs with `use_telemetry=True` and produces tensors of the new dimension.
- Tests covering observation sizing pass.

---

## PR2 — Fix the Tamiyo multi‑seed “partial pass” (tracker + heuristic baseline)

### Why
We need the heuristic baseline and signal summaries to reflect the real multi‑slot world; otherwise comparisons and debugging are misleading.

### Risk / Complexity
- **Complexity:** Medium
- **Risk:** Medium
- **Main risks**
  - Heuristic behavior changes (by design) once it can operate across multiple slots; some tests may implicitly assume single‑slot behavior.
  - Tracker “summary seed” choice can create confusing logs if the selection rule isn’t stable and well‑documented.
  - Mapping `target_seed_id` → slot can be ambiguous if there’s ever a bug producing duplicate IDs.
- **Mitigations**
  - Make selection rules deterministic and document tie‑breakers in code + plan.
  - Assert seed IDs are unique within an env; fail fast if not.
  - Update/extend heuristic integration tests to cover multi‑slot lifecycle stepping and slot selection.

### Changes (minimal, no redesign)

1. `src/esper/tamiyo/tracker.py:SignalTracker.update()`
   - Stop collapsing seed summary fields to `active_seeds[0]`.
   - Deterministic rule for seed summaries (pick one, document it):
     - Prefer the seed with the **highest stage**, tie‑break by highest `alpha`, tie‑break by most negative `counterfactual_contribution` (safety).
   - Keep the `TrainingSignals` schema stable unless we’re willing to update every consumer in one PR.

2. `src/esper/simic/training.py:run_heuristic_episode()`
   - Make heuristic training operate across **all enabled slots**:
     - wire telemetry for all slots
     - compute `available_slots` as “count of empty enabled slots”
     - call `step_epoch()` for every enabled slot each epoch
   - Germination slot selection (since heuristic actions don’t name a slot):
     - choose the first empty enabled slot (`early → mid → late`) or round‑robin.
   - Cull/Fossilize:
     - resolve `decision.target_seed_id` → slot (search active slots) and apply the op to that slot.

### Acceptance criteria
- `PYTHONPATH=src python -m esper.scripts.train heuristic --task cifar10 --episodes 1 --slots early mid late` can create/manage seeds in more than one slot over an episode.
- No single‑slot assumptions remain in the heuristic runtime path.

---

## PR3 — Multi‑GPU env parallelism hardening (no DDP yet)

### Why
We want to be confident that `--devices cuda:0 cuda:1 ...` is robust and fails early when misconfigured.

### Risk / Complexity
- **Complexity:** Low
- **Risk:** Medium (hardware‑dependent validation)
- **Main risks**
  - Some bugs only reproduce on CUDA (stream ordering, device placement, non_blocking transfers).
  - Improved validation could reject previously “working by accident” configurations (intended, but can surprise operators).
- **Mitigations**
  - Keep PR0 smoke test as the sign‑off gate for CUDA behavior.
  - Make error messages actionable (requested device vs available device count).

### Changes
- Improve device ergonomics + safety (only where it’s low‑risk):
  - Validate that all `--devices` entries are real CUDA indices at startup (already mostly present; tighten if needed).
  - Improve mapping diagnostics: print env→device assignment counts (helps operators catch accidental single‑GPU runs).
- Ensure stream / device usage stays explicit:
  - Any `torch.cuda.default_stream(...)` calls use `torch.device(...)` (no implicit string assumptions).

### Acceptance criteria
- PR0 smoke test passes on 2+ GPUs.
- Manual run works:
  - `PYTHONPATH=src python -m esper.scripts.train ppo --task cifar10 --episodes 2 --n-envs 4 --devices cuda:0 cuda:1`

---

## PR4 — (Optional) DDP “v0” design decision + blockers (keep it honest)

### Why
The architecture report correctly flags that **true DDP** is not currently safe. We should either:
- explicitly defer DDP with clear TODOs and a tracked plan, or
- implement a constrained DDP v0 with strong invariants.

### Risk / Complexity
- **Complexity:** High
- **Risk:** High
- **Main risks**
  - **Deadlocks** from divergent collective call ordering (`_sync_gate_decision`).
  - **Incorrectness** from dynamic parameter sets (seed germination/cull) under DDP.
  - **Counterfactual eval** is not DDP‑safe (`force_alpha` mutates state).
  - Hard to validate without multi‑GPU + distributed test harness; “seems fine” is not evidence.
- **Mitigations**
  - Treat as a discrete decision point: implement only if we accept the complexity; otherwise defer explicitly.
  - If implemented, enforce strict invariants: rank0 action broadcast, fixed slot ordering for all collectives, no rank‑local lifecycle divergence.
  - Prefer “functional counterfactual forward” over mutating `force_alpha`, or disable counterfactual in DDP v0 with a clear TODO (but only if we accept the research trade‑off).

### DDP blockers to resolve (minimum set)
- **Dynamic parameters:** seeds are created/removed at runtime (DDP hates changing parameter sets).
- **Counterfactual path:** `force_alpha()` mutates module state and is explicitly unsafe under DDP.
- **Collective call ordering:** `_sync_gate_decision()` requires identical all‑reduce call ordering across ranks.

### If we choose to implement DDP v0 (still “no major rework”)
Constrain scope aggressively:
- DDP is supported only for **single env training** (one model replica across ranks), not multi‑env PPO.
- Actions are decided on rank0 and **broadcast** to all ranks; all ranks execute identical lifecycle ops.
- Replace `force_alpha()` usage in counterfactual evaluation with one of:
  - a functional “counterfactual forward” path (no mutation), or
  - rank0‑only counterfactual eval with barriers and no gradients (slower, but safe).
- Move gate synchronization out of per‑slot conditional paths:
  - execute “sync points” in a fixed slot order (`early/mid/late`) every epoch so collectives can’t diverge.

If we *don’t* choose DDP now:
- Add explicit TODOs where DDP is blocked and keep DDP as a separate plan.

### Acceptance criteria
- A written decision (ship DDP v0 vs defer) and an explicit list of invariants either way.

---

## PR5 — (Opportunistic) AMP for throughput (cheap win, helps multi‑GPU)

### Why
AMP is a low‑effort throughput boost and synergizes with multi‑GPU env parallelism.

### Risk / Complexity
- **Complexity:** Medium
- **Risk:** Medium
- **Main risks**
  - Mixed precision can destabilize PPO (NaNs/Inf) and distort gradient‑based telemetry if not handled carefully.
  - Interaction with custom masking/log_prob math and value loss scaling (common FP16 footguns).
- **Mitigations**
  - Gate AMP behind an explicit flag/config; default remains stable FP32 unless enabled.
  - Use `GradScaler` (FP16) and add anomaly detection hooks around non‑finite losses/metrics.
  - Validate on a tiny CUDA run before longer jobs.

### Changes
- Wrap forward/loss/backward in `torch.amp.autocast('cuda', dtype=...)` where appropriate.
- Add `GradScaler` for stability (if FP16).

### Acceptance criteria
- AMP path runs without NaNs on a short PPO job.
- Guarded so CPU runs are unaffected (no compatibility shims; just conditional CUDA execution).

---

# Sign‑Off Checklist (what we require before calling it “done”)

## Multi‑Seed
- PPO and heuristic both step lifecycle across all enabled slots each epoch.
- Germination can fill multiple slots in a single episode.
- Reward/param accounting reflects **active seed params** correctly (no fossilized double‑counting).

## Per‑Slot Telemetry
- PPO obs includes 3× telemetry vectors in a deterministic order.
- Empty slots are zero‑padded.
- Telemetry slices correspond to the correct slot (not “first active seed”).

## Multi‑GPU Env Parallelism
- Smoke test passes on 2+ GPUs.
- No cross‑device tensor errors (all env tensors live on their assigned device; policy tensors on policy device).

## DDP (only if we choose to ship it)
- No DDP deadlocks in gate synchronization.
- Counterfactual evaluation is DDP‑safe (functional or rank‑coordinated).
- Seed parameter sets are consistent across ranks (no dynamic divergence).
