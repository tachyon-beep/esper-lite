# 📜 Esper-Lite Architecture Convergence Plan (Super-Sprint)

**Version:** 1.2 (Hardened)
**Status:** 🟢 Ready for Execution
**Objective:** Merge three parallel modernization plans (Simic Cleanup, Kasmina Wiring, Multi-GPU Hardening) into a single, conflict-free execution sequence.

## 🛑 Global Guardrails (Read First)

1. **Linear Order is Critical:** Do not reorder phases. **Phase 2** builds the engine; **Phase 3** upgrades the brain.
2. **The "Lying Knob" Rule:** The parameter `max_seeds_per_slot` is dead. Delete it immediately.
3. **One-Way Door:** No legacy code. Delete old paths immediately upon replacing them.
4. **Incremental Verification:** Run the "Verification" command after *every* step. Do not batch verify.

---

## 🏗️ Phase 1: The Foundation (Clean the Deck)

**Goal:** Fix bugs, establish contracts, and remove dead weight.

### [ ] Step 1: Kill `max_seeds_per_slot` (Simic PR0)

* **Target:** `src/esper/scripts/train.py`, `src/esper/simic/vectorized.py`, `src/esper/leyline/__init__.py`
* **Action:** Delete `max_seeds_per_slot` from CLI, signatures, and constants.
* **Verification:** `rg "max_seeds_per_slot"` returns 0 hits.

### [ ] Step 2: Fix CNN Segmented Forward (Kasmina PR0)

* **Target:** `src/esper/kasmina/host.py`
* **Action:** Update `forward_to_segment`/`forward_from_segment` in `CNNHost` to apply registered injection-point slots.
* **Verification:** New test `test_cnn_segment_consistency` passes.

### [ ] Step 3: Establish Telemetry Contracts (Kasmina PR1)

* **Target:** `src/esper/leyline/reports.py`, `src/esper/kasmina/slot.py`
* **Action:** Update `SeedStateReport` to include `counterfactual_contribution`, `seed_gradient_norm_ratio`, `seed_param_count`.
* **Verification:** Unit test confirms fields populate during `to_leyline()`.

### [ ] Step 4: Dead Code Sweep & Blueprint Hardening (Kasmina PR3 + PR5)

* **Target:** `src/esper/kasmina/`
* **Action:**
  * Delete `BlueprintRegistry.reset()`, `MorphogeneticModel.count_seeds_in_slot()`, `SeedState.record_epoch()`.
  * Add validation to `TransformerBlueprint` (assert `dim % n_head == 0`).
* **Verification:** `ruff check` passes; **`uv run pytest tests/kasmina` passes**.

---

## ⚙️ Phase 2: Runtime Architecture (The Engine)

**Goal:** Build the physical Multi-GPU / Multi-Slot engine.

### [ ] Step 5: GPU Smoke Test Harness (Plan B PR0)

* **Target:** `tests/cuda/test_vectorized_multi_gpu_smoke.py`
* **Action:** Create a test that runs a tiny PPO job on 2 GPUs (skips gracefully if < 2 GPUs).
* **Verification:** Test passes on multi-GPU node (or skips on CPU).

### [ ] Step 6: Per-Slot Telemetry & Report Wiring (Plan B PR1 + Kasmina PR2)

* **Target:** `src/esper/simic/ppo.py`, `src/esper/simic/vectorized.py`, `src/esper/simic/action_masks.py`
* **Action:**
  * Update `signals_to_features` to consume `SeedStateReport`.
  * **Contract:** Enforce deterministic `[Early, Mid, Late]` order based on enabled slots. Zero-pad empty slots.
  * Expand observation space: `base + (3 * telemetry)`.
* **Verification:** PPO runs with `use_telemetry=True`; tensor shapes match 80 dims.

### [ ] Step 7: Tamiyo Multi-Slot Fix (Plan B PR2)

* **Target:** `src/esper/tamiyo/tracker.py`, `src/esper/simic/training.py`
* **Action:** Update `SignalTracker` and `run_heuristic_episode` to manage *all* enabled slots, not just the first one.
* **Verification:** Heuristic baseline runs with `--slots early mid late`.

### [ ] Step 8: Multi-GPU Parallelism (Plan B PR3)

* **Target:** `src/esper/simic/vectorized.py`
* **8a (Mapping):** Validate `--devices`. Print env→device map. Ensure `to(device)` calls are explicit.
* **8b (Loop Logic):** Rewrite loop to **Batch-First** (Inverted Control Flow) using `SharedBatchIterator`.
* **8c (Concurrency):** Add `cuda.Stream` per env. Wrap inner ops in `stream_ctx`.
* **Verification:** Smoke test (Step 5) passes; manual run confirms GPU utilization.

---

## 🧠 Phase 3: Algorithm Upgrade (The Brain)

**Goal:** Upgrade PPO logic *within* the new Phase 2 training loop.

### [ ] Step 9: Multiple PPO Updates per Batch (Simic PR1)

* **Target:** `src/esper/simic/vectorized.py`
* **Action:** Loop `agent.update()` $N$ times. Fix entropy annealing to count steps.
* **Verification:** Logs show multiple policy updates per batch.

### [ ] Step 10: Telemetry Escalation (Simic PR2)

* **Target:** `src/esper/simic/vectorized.py`
* **Action:** Wire `telemetry_config.escalate_temporarily()` to `AnomalyDetector`.
* **Verification:** Anomaly triggers DEBUG logs.

### [ ] Step 11: Masked Categorical Safety (Simic PR4)

* **Target:** `src/esper/simic/tamiyo_network.py`
* **Action:** Replace distribution with `MaskedCategorical`. Raise error on all-false masks. Use FP16-safe mask value (`-1e4` or similar).
* **Verification:** Unit test with all-false mask triggers error.

### [ ] Step 12: Ratio Diagnostics (Simic PR5)

* **Target:** `src/esper/simic/ppo.py`, `src/esper/simic/vectorized.py`
* **Action:** Return `RatioExplosionDiagnostic` metrics when ratio > threshold.
* **Verification:** Force a ratio explosion (mock) and see diagnostic in logs.

### [ ] Step 13: Reward Families (Simic PR6)

* **Target:** `src/esper/simic/vectorized.py`, `src/esper/simic/rewards.py`
* **Action:** Add `--reward-family {contribution, loss}`. Wire `compute_loss_reward`.
* **Verification:** Training runs with `--reward-family loss`.

---

## 🔒 Phase 4: Consolidation (Lock Down)

**Goal:** Finalize the interface and clean up.

### [ ] Step 14: Opportunistic AMP (Plan B PR5)

* **Target:** `src/esper/simic/vectorized.py`
* **Action:** Wrap forward/loss/backward in `torch.amp.autocast`. Add `GradScaler`.
* **Verification:** PPO runs without NaNs; G2 gate metrics remain stable.

### [ ] Step 15: Real `TrainingConfig` (Simic PR7)

* **Target:** `src/esper/config.py`, `src/esper/scripts/train.py`
* **Action:** Make `TrainingConfig` the single source of truth. Map all new flags.
* **Verification:** `train.py` initializes solely from `TrainingConfig`.

### [ ] Step 16: Final Cleanup (Simic PR8)

* **Target:** Global
* **Action:** Delete unused imports, `normalize_observation`, and temporary code.
* **Verification:** `ruff check` is clean. `pytest` passes.
