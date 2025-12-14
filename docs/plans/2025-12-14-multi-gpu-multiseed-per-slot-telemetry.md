# Multi‑GPU + Multi‑Seed Hardening Plan (2025‑12‑14)

## Context

We just landed two major capabilities:

1. **Research‑grade telemetry** (Karn/Nissa pipeline)
2. **Multi‑seed per environment** (3 slots per env: `early`, `mid`, `late`)

This plan is the shortest path to a credible “sign‑off” that:

- Multi‑seed behavior is correct end‑to‑end (Kasmina ⇄ Tamiyo ⇄ Simic ⇄ Tolaria)
- Vectorized PPO can run across multiple GPUs reliably
- PPO gets **per‑slot telemetry** in its observation space (original design intent)

## Alignment with Workstream‑001 (Master Plan)

This document is a **detailed supplement** to the canonical master plan:
`docs/plans/workstream-001-multiseed-telemetry-remediation-package.md`.

Treat the master plan as the authoritative execution order; do not execute this plan independently or out of order.

**Marker mapping (this doc → master plan):**
- PR0 ↔ Workstream‑001 Step 5
- PR1 ↔ Workstream‑001 Step 6
- PR2 ↔ Workstream‑001 Step 7
- PR3 ↔ Workstream‑001 Step 8
- PR5 (optional AMP) ↔ Workstream‑001 Step 14
- DDP v0 is explicitly deferred in Workstream‑001 (see “Explicit Deferrals” below)

## Prerequisites (Workstream‑001 Phase 1)

This plan assumes Steps 1–4 in the master plan are already complete, especially:
- Step 1: delete `max_seeds_per_slot` everywhere (no lying knobs)
- Step 3: telemetry/report fields exist on `SeedStateReport` / `SeedMetrics` for Simic to consume
- Step 4: dead‑code sweep is done so we don’t build on redundant APIs

## Definitions (so “multi‑GPU” is unambiguous)

This workstream ships **single‑process multi‑GPU env parallelism**. “Multi‑GPU” can mean two distinct capabilities:

1. **Multi‑GPU env parallelism (P0/P1):** one Python process runs `n_envs` environments, distributing envs across `--devices` (e.g. `cuda:0 cuda:1`) and overlapping work with CUDA streams (`src/esper/simic/vectorized.py`).  
   - This is the fastest path to multi‑GPU throughput and the one we can ship without major rework.
2. **True DDP model training (deferred; out of scope for Workstream‑001):** one environment’s host model trains with `torch.distributed` across GPUs (DDP).  
   - This is *not* currently safe due to dynamic seed parameter creation + counterfactual (`force_alpha`) semantics, and is explicitly deferred in the master plan.

## Current Status (what looks correct vs what blocks sign‑off)

### Looks correct (ready to build on)

- **Kasmina multi‑slot execution**: `MorphogeneticModel.forward()` applies all active slots (`src/esper/kasmina/host.py`).
- **Vectorized PPO multi‑slot mechanics**:
  - Per‑slot optimizers (no dynamic param‑group surgery) and per‑slot lifecycle stepping (`src/esper/simic/vectorized.py`).
  - Per‑slot counterfactual validation path exists (alpha‑0 ablations) and is used for reward attribution.
- **Multi‑GPU guardrails**:
  - Fail‑fast CUDA index validation in vectorized PPO.
  - Model factory checks invalid CUDA device requests early (`src/esper/tolaria/environment.py`).

### Blocks sign‑off today (Workstream‑001 scope)

1. **Per‑slot telemetry is not in PPO observations yet**
   - `signals_to_features()` appends telemetry for only “first active seed” (`src/esper/simic/ppo.py`).
2. **Tamiyo multi‑seed is a partial pass**
   - `SignalTracker` collapses seed summary fields to `active_seeds[0]` (`src/esper/tamiyo/tracker.py`).
   - Heuristic baseline is effectively **single‑slot** (`src/esper/simic/training.py:run_heuristic_episode` selects `target_slot = slots[0]`).

### Out of scope / deferred risks (tracked, not sign‑off blockers for Workstream‑001)

- **True DDP is not currently safe**
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
4. Simic consumes slot state/telemetry via **reports as the single source of truth**, and redundant “internal state readers” are deleted (no parallel representations).

---

# Work Plan (small PRs, no major rework)

## PR0 — GPU smoke test harness (Workstream‑001 Step 5)

### Why
We need a single command that an operator can run on a 2+ GPU machine to validate the “multi‑GPU env parallelism” path end‑to‑end (streams, transfers, optimizers, reward, telemetry).

### Risk / Complexity
- **Complexity:** Low
- **Risk:** Low
- **Main risks**
  - Flaky/slow GPU tests in CI if not properly skipped or scoped.
  - A “smoke test” that doesn’t actually exercise both GPUs (false confidence).
  - Non‑hermetic test behavior (dataset download / network) makes validation unreliable.
- **Mitigations**
  - Hard‑skip when CUDA not available and when `<2` GPUs.
  - Keep runtime tiny (≤1–2 episodes, small batches) and assert device assignment is non‑degenerate (e.g., envs map to both `cuda:0` and `cuda:1`).
  - Use mock/synthetic datasets (or pre‑seeded local datasets) so the smoke test never depends on network access.

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

## PR1 — Per‑slot telemetry & report wiring in PPO observations (Workstream‑001 Step 6)

### Why
Multi‑seed is only learnable if the policy can “see” each slot’s local health/progress. Today telemetry is effectively single‑seed.

### Risk / Complexity
- **Complexity:** Medium
- **Risk:** Medium (can become High if telemetry freshness isn’t handled)
- **Main risks**
  - Observation shape change ripples through: PPO network input, rollout buffer, obs normalizer, tests, and checkpoints.
  - “Per‑slot telemetry” can be incorrect or stale if we only refresh telemetry for TRAINING/BLENDING slots (e.g., PROBATIONARY/FOSSILIZED slots might show outdated `stage/alpha/accuracy` if `SeedState.sync_telemetry()` isn’t called).
  - Two sources of truth (raw internal state vs reports) can drift unless we delete the redundant path.
  - Determinism: slot iteration must be fixed (`early/mid/late`), not dict order.
- **Mitigations**
  - Update all dimension assertions/tests in the same PR; no compatibility shims.
  - Ensure telemetry snapshots are refreshed for **all active enabled slots** each epoch (even if gradient telemetry is unavailable, update the non‑gradient fields and keep gradient fields at “last known” or explicit zeros).
  - Add a targeted unit/integration test that checks telemetry slices align to the correct slot (not “first active seed”).
  - Consume **only** `SeedStateReport` / slot reports in Simic (features + masks) and delete the old internal‑state reader APIs.

### Implementation decisions (keep it simple)
- Observation layout stays **fixed‑size**:
  - `MULTISLOT_FEATURE_SIZE` is already hardcoded for `early/mid/late` (50 dims).
  - Telemetry becomes `3 * SeedTelemetry.feature_dim()` (30 dims).
  - New `state_dim = 80` when telemetry is enabled.
- Slot reports must carry the per‑slot telemetry snapshot needed for PPO observations.
  - Prefer adding an explicit `SeedTelemetry` (or equivalent typed telemetry field) onto `SeedStateReport` in Workstream‑001 Step 3 so Step 6 can remain “reports‑only”.
- Ordering is deterministic: `[early telemetry][mid telemetry][late telemetry]`.
- Empty slot telemetry is zero‑padded.

### Changes
- `src/esper/kasmina/host.py`
  - Implement `MorphogeneticModel.get_slot_reports()` (canonical Kasmina → Simic/Tamiyo snapshot API).
  - Delete `MorphogeneticModel.get_slot_states()` (redundant internal‑state exposure).
- `src/esper/simic/action_masks.py`
  - Update `build_slot_states(...)` to build masking inputs from `SeedStateReport` (slot reports), not internal model state.
  - Delete any redundant “read internal seed slot state” paths once report wiring is in place.
- `src/esper/simic/ppo.py:signals_to_features()`
  - Replace “first active seed telemetry” with deterministic per‑slot telemetry concatenation.
  - Consume `SeedStateReport` (via `model.get_slot_reports()`), not internal slot state.
- `src/esper/simic/vectorized.py:train_ppo_vectorized()`
  - Update `state_dim` computation accordingly.
  - Ensure observation normalizer shapes and PPOAgent input sizes match.
  - Build one per‑epoch **slot report snapshot** (after metrics/counterfactual/telemetry sync) and use it for both masking and feature extraction.
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
- Action masking behaves correctly when slots are empty/disabled (no implicit reads of internal state).

---

## PR2 — Tamiyo multi‑slot fix (Workstream‑001 Step 7)

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

## PR3 — Multi‑GPU env parallelism hardening (Workstream‑001 Step 8)

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
- Ensure the vectorized training loop is the canonical Step‑8 engine (inverted control flow + `SharedBatchIterator` + CUDA streams).
  - Delete any redundant/older training paths in the same PR (no parallel engines).
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

## PR5 — (Optional) AMP for throughput (Workstream‑001 Step 14)

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

## Explicit Deferrals (Workstream‑001 Alignment)

- **DDP v0:** Out of scope for Workstream‑001 and explicitly deferred in the master plan.
  - Track only as a risk register: dynamic parameters (germinate/cull), `force_alpha` state mutation, and collective call ordering (`_sync_gate_decision`).

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

## DDP (explicitly deferred)
- No partial DDP implementation is introduced in this workstream (avoid half‑wired distributed paths).
