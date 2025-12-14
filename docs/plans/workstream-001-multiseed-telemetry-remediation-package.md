# 📜 Esper-Lite Architecture Convergence Plan (Super-Sprint)

**Workstream:** 001 (Multi-seed + Telemetry + Remediation Package)
**Version:** 1.7
**Status:** 🟢 Ready for Execution
**Objective:** Merge three parallel modernization plans (Simic Cleanup, Kasmina Wiring, Multi-GPU Hardening) into a single, conflict-free execution sequence.

## 🛑 Global Guardrails (Read First)

1. **Linear Order is Critical:** Do not reorder phases. **Phase 2** builds the engine; **Phase 3** upgrades the brain.
2. **The "Lying Knob" Rule:** The parameter `max_seeds_per_slot` is dead. Delete it immediately.
3. **One-Way Door:** No legacy code. Delete old paths immediately upon replacing them.
4. **Incremental Verification:** Run the step’s **Verification** command(s) after *every* step. Do not batch verify.
5. **One PR Per Step:** Each step is a single PR. Don’t bundle steps; don’t start a later step until the prior step is merged.
6. **Archives Stay Historical:** Don’t “fix up” `docs/plans/archive/` or `docs/specifications/archive/` as part of this workstream.

## Source Docs

This document implements the combined plans from:

- docs/plans/2025-12-14-kasmina-wire-or-delete.md
- docs/plans/2025-12-14-multi-gpu-multiseed-per-slot-telemetry.md
- docs/plans/2025-12-14-simic-activate-or-delete.md

**Hierarchy / precedence**

- `ROADMAP.md` + `CLAUDE.md` guardrails are absolute.
- This file is the **canonical execution order** for Workstream 001.
- The source docs above provide rationale and detail; if they conflict with this file, update them (do not “support both”).

**Marker mapping (source doc → this plan’s steps)**

- **Simic:** PR0 ↔ Step 1, PR3 ↔ Step 1b, PR1 ↔ Step 9, PR2 ↔ Step 10, PR4 ↔ Step 11, PR5 ↔ Step 12, PR6 ↔ Step 13, PR7a/PR7b ↔ Step 15, PR8 ↔ Step 16
- **Kasmina:** PR0 ↔ Step 2, PR1 ↔ Step 3, PR2 ↔ Step 6, PR3/PR5 ↔ Step 4
- **Multi‑GPU plan:** PR0 ↔ Step 5, PR1 ↔ Step 6, PR2 ↔ Step 7, PR3 ↔ Step 8, PR5 ↔ Step 14

> Note: “PR#” labels are **planned PR-sized units of work** in the source docs, not necessarily your hosting provider’s PR numbers.

## ✅ Definition of Done (Workstream 001)

- No “lying knobs”: `max_seeds_per_slot` is deleted from code and current docs/specs.
- PPO observations include deterministic **per-slot telemetry** (Early/Mid/Late, zero‑padded), and the observation dim is **80** when telemetry is enabled.
- Tamiyo (tracker + heuristic baseline) handles **all enabled slots**, not just the first.
- Single-process multi‑GPU env parallelism has a smoke test that **passes on 2+ GPUs** (and skips gracefully otherwise).
- Simic config surface is **honest**: no no‑op flags, strict JSON config loading, and contract tests prevent drift.
- CI-parity checks pass: `ruff check src/ tests/`, `uv run pytest`, `mypy src/`.

## Risk / Complexity Summary (per step)

Scale:
- **Complexity:** S / M / L / XL (engineering surface area)
- **Risk:** Low / Med / High (runtime correctness + training dynamics + integration/hardware)

| Step | Scope | Complexity | Risk | Primary risk driver |
|---|---|---:|---:|---|
| 1 | Delete `max_seeds_per_slot` everywhere | S | Low | Pure deletion + doc/spec sync |
| 1b | Delete dead dual-gradient collector API | S | Low | Pure deletion |
| 2 | Fix CNN segmented forward (slot injection points) | M | Med | Subtle correctness; needs unit test |
| 3 | Add report/metrics fields to Leyline contracts | M | Med | Shared contract change across domains |
| 4 | Kasmina dead-code sweep + blueprint validation | S–M | Low | Mostly deletions + asserts |
| 5 | Multi‑GPU smoke test harness | S | Low | Avoid flaky/non-hermetic GPU test |
| 6 | Per-slot telemetry wiring + obs dim 80 | L | High | Cross-domain schema change + ordering/zero‑pad rules |
| 7 | Tamiyo multi-slot fix | M | Med | Behavior change in baseline/tracker |
| 8 | Multi‑GPU parallelism (batch-first + streams) | XL | High | Hardware-dependent perf/correctness |
| 9 | Multiple PPO updates per batch | M–L | Med | Training semantics (anneal/normalizer) |
| 10 | Telemetry auto-escalation on anomalies | S–M | Low–Med | Over-triggering / log noise |
| 11 | MaskedCategorical safety | L | High | Core sampling/log_prob/entropy behavior |
| 12 | Ratio explosion diagnostics | M | Med | Metrics aggregation/serialization |
| 13 | Reward families (contribution vs loss) | M–L | Med | Reward scaling/sign semantics |
| 14 | Opportunistic AMP | M | Med | FP16 stability/NaNs |
| 15 | Real `TrainingConfig` (prune + config-first CLI) | M–L | Med | User-facing breakage + drift risk |
| 16 | Final cleanup sweep | S–M | Low–Med | Mechanical deletions across repo |

---

## 🏗️ Phase 1: The Foundation (Clean the Deck)

**Goal:** Fix bugs, establish contracts, and remove dead weight.

### [x] Step 1: Kill `max_seeds_per_slot` (Simic PR0)

- **Pre-flight checklist:**
  - [ ] Start from a clean branch at the current `main` head; confirm no other unmerged work touches seed-budget flags/CLI.
  - [ ] Inventory all hits (including docs/specs) with `rg -n "max_seeds_per_slot|max-seeds-per-slot|DEFAULT_MAX_SEEDS_PER_SLOT" src/ tests/ README.md ROADMAP.md docs/specifications/`.
  - [ ] Confirm you will delete the knob end-to-end (CLI → config → training signature → docs/specs); no compatibility paths.
  - [ ] (Recommended) Run a baseline sanity subset: `uv run pytest -q tests/test_simic_vectorized.py tests/integration/test_vectorized_factored.py`.
- **Target:** `src/esper/scripts/train.py`, `src/esper/simic/vectorized.py`, `src/esper/leyline/__init__.py`, `README.md`, `ROADMAP.md`, `docs/specifications/kasmina.md`
- **Action:** Delete `max_seeds_per_slot` / `--max-seeds-per-slot` from CLI, signatures, constants, and current docs/specs.
- **Verification:** `rg -n "max_seeds_per_slot|max-seeds-per-slot|DEFAULT_MAX_SEEDS_PER_SLOT" src/ tests/ README.md ROADMAP.md docs/specifications/` returns 0 hits.

### [ ] Step 1b: Delete Dead Gradient Collector (Simic PR3)

- **Pre-flight checklist:**
  - [ ] Confirm Step 1 is merged (this step should not be stacked on unrelated churn).
  - [ ] Inventory call sites with `rg -n "collect_dual_gradients_async" src/ tests/`.
  - [ ] Confirm there is no remaining “dual gradients” path needed for current PPO/vectorized flow.
  - [ ] Run a baseline sanity subset: `uv run pytest -q tests/test_simic_gradient_collector.py tests/test_simic_vectorized.py`.
- **Target:** `src/esper/simic/gradient_collector.py`, `src/esper/simic/vectorized.py`
- **Action:**
  - Delete `collect_dual_gradients_async` (unused API).
  - Remove unused imports of this function in `vectorized.py`.
- **Verification:** `rg "collect_dual_gradients_async"` returns 0 hits.

### [x] Step 2: Fix CNN Segmented Forward (Kasmina PR0)

- **Pre-flight checklist:**
  - [ ] Confirm Steps 1–1b are merged.
  - [ ] Identify intended injection-point behavior for segmented execution (what segments must apply which slots).
  - [ ] Locate the segment APIs with `rg -n "forward_to_segment|forward_from_segment|segment" src/esper/kasmina/host.py`.
  - [ ] Run a baseline host/model subset: `uv run pytest -q tests/test_host.py tests/test_morphogenetic_model.py`.
- **Target:** `src/esper/kasmina/host.py`
- **Action:** Update `forward_to_segment`/`forward_from_segment` in `CNNHost` to apply registered injection-point slots.
- **Verification:** New test `test_cnn_segment_consistency` passes.

### [ ] Step 3: Establish Telemetry Contracts (Kasmina PR1)

- **Pre-flight checklist:**
  - [ ] Confirm Step 2 is merged.
  - [ ] Review current Leyline report types: `src/esper/leyline/reports.py` (identify the exact contract consumers will use in Simic).
  - [ ] Identify the production boundary where Kasmina emits reports (e.g., `to_leyline()`), and confirm it can populate the new fields without reading private state from Simic.
  - [ ] Run a baseline telemetry/slot subset: `uv run pytest -q tests/test_kasmina_telemetry.py tests/test_seed_slot.py`.
- **Target:** `src/esper/leyline/reports.py`, `src/esper/kasmina/slot.py`
- **Action:** Update `SeedStateReport` / `SeedMetrics` to include `counterfactual_contribution`, `seed_gradient_norm_ratio`, `seed_param_count`, `host_param_count`.
- **Verification:** Unit test confirms fields populate during `to_leyline()`.

### [ ] Step 4: Dead Code Sweep & Blueprint Hardening (Kasmina PR3 + PR5)

- **Pre-flight checklist:**
  - [ ] Confirm Step 3 is merged.
  - [ ] Inventory each deletion candidate with `rg -n "<symbol_name>" src/esper/kasmina/ tests/kasmina` and confirm there are no runtime call sites.
  - [ ] Run baseline `uv run pytest tests/kasmina` to ensure you start from green before deletions.
  - [ ] Confirm you will delete code fully (no “deprecated” stubs, no compatibility wrappers).
- **Target:** `src/esper/kasmina/`
- **Action:**
  - Delete `BlueprintRegistry.reset()`, `MorphogeneticModel.count_seeds_in_slot()`, `SeedState.increment_epoch()`, `SeedState.record_epoch()`.
  - Remove unused `temperature` param from `SeedSlot.start_blending(...)`.
  - Simplify `override` imports in `host.py` (always use `typing_extensions`).
  - Add validation to `TransformerBlueprint` (assert `dim % n_head == 0`).
- **Verification:** `ruff check` passes; **`uv run pytest tests/kasmina` passes**.

---

## ⚙️ Phase 2: Runtime Architecture (The Engine)

**Goal:** Build the physical Multi-GPU / Multi-Slot engine.

### [ ] Step 5: GPU Smoke Test Harness (Multi‑GPU PR0)

- **Pre-flight checklist:**
  - [ ] Confirm Steps 1–4 are merged (no stacking on moving contracts).
  - [ ] Decide marking strategy under `--strict-markers`: either (a) no marker + fast skip when `<2` GPUs, or (b) add a `cuda` marker to `pytest.ini` and use it consistently.
  - [ ] Confirm a hermetic dataset path exists for the test (no attempted downloads): use synthetic data via existing `mock=True` dataset support or explicit test monkeypatching; do not rely on “download fails then fallback”.
  - [ ] On a target node, confirm CUDA visibility: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"`.
  - [ ] Define the minimal runtime budget (tiny episodes/envs) so this is a true smoke test, not a perf run.
- **Target:** `tests/cuda/test_vectorized_multi_gpu_smoke.py`
- **Action:**
  - Create a test that runs a tiny PPO job on 2 GPUs (skips gracefully if < 2 GPUs).
  - **Hermetic rule:** must not trigger dataset download or depend on network. Use a synthetic dataset or a local-only cached dataset path; fail fast with a clear message if it would download.
  - Assert envs actually map to both GPUs (avoid false-pass “all envs landed on cuda:0”).
- **Verification:** Test passes on multi-GPU node (or skips on CPU) and demonstrates both GPUs are exercised.

### [ ] Step 6: Per-Slot Telemetry & Report Wiring (Multi‑GPU PR1 + Kasmina PR2)

- **Pre-flight checklist:**
  - [ ] Confirm Step 5 is merged (or at least the repo has a known-good CUDA gate before touching the hot path).
  - [ ] Confirm Step 3 contract fields are present on `SeedStateReport` / `SeedMetrics` and can be produced by Kasmina without Simic introspection.
  - [ ] Inventory all internal-state reads you plan to delete (e.g., `get_slot_states`, direct `seed_slots[..].state` reads used for policy obs/masks).
  - [ ] Identify every place that assumes the old observation dim; pre-list the tests/docs that will change with `rg -n "MULTISLOT_FEATURE_SIZE|feature_dim\\(|60\\-dim|state_dim" src/ tests/ README.md docs/specifications/`.
  - [ ] Decide and document the single canonical slot ordering function (`ordered_slots(...)`) and list every consumer that must use it (features + masks at minimum).
  - [ ] Run a baseline Simic subset: `uv run pytest -q tests/test_simic_vectorized.py tests/test_simic_features.py tests/test_observation_normalization.py tests/integration/test_multislot_pipeline.py`.
- **Target:** `src/esper/simic/ppo.py`, `src/esper/simic/vectorized.py`, `src/esper/simic/action_masks.py`, `src/esper/kasmina/host.py`, `README.md`, `docs/specifications/simic.md`
- **Action:**
  - **Kasmina API:** Implement `MorphogeneticModel.get_slot_reports()` and delete `get_slot_states()`.
  - Update `signals_to_features` to consume `SeedStateReport`.
  - Update `build_slot_states` in `action_masks.py` to read from `SeedStateReport` (no internal-state reads).
  - **Contract:** Enforce deterministic `[Early, Mid, Late]` order based on enabled slots. Zero-pad empty slots.
  - **Ordering function:** one function owns slot ordering (e.g., `ordered_slots(enabled_slots) -> tuple[str, ...]`) and both masking + feature extraction use it; add a unit test that detects regressions in ordering.
  - **Dynamic vs fixed:** slot report APIs remain dynamic (dict keyed by slot); only PPO feature concatenation is fixed-size (3-slot) for now.
  - Expand observation space: `base(50) + (3 * telemetry(10)) = 80`.
  - Update docs/specs to reflect 80-dim observations when telemetry is enabled (and remove “60-dim” language).
- **Verification:** PPO runs with `use_telemetry=True`; tensor shapes match 80 dims.

### [ ] Step 7: Tamiyo Multi-Slot Fix (Multi‑GPU PR2)

- **Pre-flight checklist:**
  - [ ] Confirm Step 6 is merged (Tamiyo should consume the new “truth” sources, not old internal state).
  - [ ] Inventory all single-slot assumptions in Tamiyo + heuristic path: `rg -n "slots\\[0\\]|active_seeds\\[0\\]|first_slot|target_slot" src/esper/tamiyo src/esper/simic/training.py`.
  - [ ] Decide deterministic multi-slot tie-breakers for any “summary” signals (document them in code; avoid implicit dict order).
  - [ ] Run a baseline Tamiyo/heuristic subset: `uv run pytest -q tests/tamiyo tests/test_stabilization_tracking.py tests/integration/test_multislot_pipeline.py`.
- **Target:** `src/esper/tamiyo/tracker.py`, `src/esper/simic/training.py`
- **Action:** Update `SignalTracker` and `run_heuristic_episode` to manage *all* enabled slots, not just the first one.
- **Verification:** Heuristic baseline runs with `--slots early mid late`.

### [ ] Step 8: Multi-GPU Parallelism (Multi‑GPU PR3)

- **Pre-flight checklist:**
  - [ ] Confirm Steps 5–7 are merged.
  - [ ] Ensure you have access to a 2+ GPU node and Step 5 passes there (or at least can be run after the change).
  - [ ] Capture a 1‑GPU baseline (throughput + p95 env step time + mapping printout) using the same hermetic dataset/config you will use for Step 8 comparisons.
  - [ ] Inventory device placement + default-stream assumptions: `rg -n "\\.to\\(|default_stream\\(|cuda\\.Stream\\(" src/esper/simic/vectorized.py src/esper/utils/data.py`.
  - [ ] Define what you will record for 8a/8b/8c done checks (paste the exact command + the observed numbers into the PR description).
- **Target:** `src/esper/simic/vectorized.py`
- **8a (Mapping):** Validate `--devices`. Print env→device map. Ensure `to(device)` calls are explicit.
  - Done check: mapping shows >0 envs assigned to each requested device.
- **8b (Loop Logic):** Rewrite loop to **Batch-First** (Inverted Control Flow) using `SharedBatchIterator`.
  - Done check: throughput does not regress vs the pre-8b baseline on 1 GPU (operator measurement is fine).
- **8c (Concurrency):** Add `cuda.Stream` per env. Wrap inner ops in `stream_ctx`.
  - Done check: p95 env step time improves or GPU util becomes smoother (sawtooth reduced), vs pre-8c baseline.
- **Verification:** Smoke test (Step 5) passes; Step 8 done checks recorded (even if “operator eyeball”).

---

## 🧠 Phase 3: Algorithm Upgrade (The Brain)

**Goal:** Upgrade PPO logic *within* the new Phase 2 training loop.

### [ ] Step 9: Multiple PPO Updates per Batch (Simic PR1)

- **Pre-flight checklist:**
  - [ ] Confirm Step 8 is merged (this change depends on the canonical engine loop).
  - [ ] Identify current annealing + normalizer update sites in `src/esper/simic/vectorized.py` (list exact lines/sections you will modify).
  - [ ] Decide how you will enforce/observe the “normalizer updates once per rollout batch” invariant (test, counter, or log).
  - [ ] Run a baseline PPO subset: `uv run pytest -q tests/simic/test_ppo.py tests/test_observation_normalization.py tests/test_recurrent_vectorized.py`.
- **Target:** `src/esper/simic/vectorized.py`
- **Action:**
  - Loop `agent.update()` $N$ times. Fix entropy annealing to count steps.
  - **Normalizer invariant:** `RunningMeanStd` updates exactly once per rollout batch (not per PPO update step).
- **Verification:** Logs show multiple policy updates per batch; normalizer update invariant holds.

### [ ] Step 10: Telemetry Escalation (Simic PR2)

- **Pre-flight checklist:**
  - [ ] Confirm Step 9 is merged.
  - [ ] Identify the anomaly surface you will listen to (`AnomalyDetector`) and the escalation mechanism (`telemetry_config.escalate_temporarily()`).
  - [ ] Decide how you will test escalation deterministically (unit test with a fake detector event, or forced anomaly injection).
  - [ ] Run a baseline subset: `uv run pytest -q tests/test_simic_vectorized.py tests/test_nissa_analytics.py`.
- **Target:** `src/esper/simic/vectorized.py`
- **Action:** Wire `telemetry_config.escalate_temporarily()` to `AnomalyDetector`.
- **Verification:** Anomaly triggers DEBUG logs.

### [ ] Step 11: Masked Categorical Safety (Simic PR4)

- **Pre-flight checklist:**
  - [ ] Confirm Step 10 is merged.
  - [ ] Inventory how masks are produced and consumed (shape per head, dtype, device) before changing distribution semantics.
  - [ ] Write down the exact entropy semantics you will implement (valid-actions only; optional normalization by `log(n_valid)` → `[0,1]`) and where the normalization lives.
  - [ ] Identify all existing distribution/log_prob usages that could silently break (e.g., advantage per head, KL diagnostics).
  - [ ] Run a baseline subset: `uv run pytest -q tests/simic/test_ppo.py tests/test_simic_vectorized.py tests/test_recurrent_vectorized.py`.
- **Target:** `src/esper/simic/tamiyo_network.py`
- **Action:**
  - Replace distribution with `MaskedCategorical`. Raise error on all-false masks. Use FP16-safe mask value (`-1e4` or similar).
  - Specify entropy semantics explicitly: entropy computed over valid actions only, and (if normalized) normalized by `log(n_valid)` to land in `[0, 1]`.
- **Verification:** Unit test with all-false mask triggers error.

### [ ] Step 12: Ratio Diagnostics (Simic PR5)

- **Pre-flight checklist:**
  - [ ] Confirm Step 11 is merged.
  - [ ] Identify where ratio is computed and where diagnostics are emitted; ensure you can thread structured data to telemetry without adding a second parallel reporting system.
  - [ ] Decide the test strategy: mock ratio explosion deterministically (don’t rely on RL stochasticity).
  - [ ] Run a baseline subset: `uv run pytest -q tests/simic/test_ppo.py tests/test_simic_vectorized.py`.
- **Target:** `src/esper/simic/ppo.py`, `src/esper/simic/vectorized.py`
- **Action:** Return `RatioExplosionDiagnostic` metrics when ratio > threshold.
- **Verification:** Force a ratio explosion (mock) and see diagnostic in logs.

### [ ] Step 13: Reward Families (Simic PR6)

- **Pre-flight checklist:**
  - [ ] Confirm Step 12 is merged.
  - [ ] Inventory current reward mode plumbing (`reward_mode`, `ContributionRewardConfig`, `compute_reward`) and decide the minimal, non-redundant API surface after adding `--reward-family`.
  - [ ] Identify any tests that assume a specific reward family/scale and list them before changing semantics.
  - [ ] Run a baseline rewards subset: `uv run pytest -q tests/test_simic_rewards.py tests/test_loss_primary_rewards.py`.
- **Target:** `src/esper/simic/vectorized.py`, `src/esper/simic/rewards.py`
- **Action:** Add `--reward-family {contribution, loss}`. Wire `compute_loss_reward`.
- **Verification:** Training runs with `--reward-family loss`.

---

## 🔒 Phase 4: Consolidation (Lock Down)

**Goal:** Finalize the interface and clean up.

### [ ] Step 14: Opportunistic AMP (Multi‑GPU PR5)

- **Pre-flight checklist:**
  - [ ] Confirm Step 13 is merged and FP32 training is stable (no existing NaN/inf issues).
  - [ ] Identify the minimal scope for autocast + scaler (forward/loss/backward only) and where you will explicitly keep FP32 (e.g., advantage normalization if needed).
  - [ ] Decide the AMP gate metric(s) you will check (e.g., no NaNs + G2 gate rates/accuracy trend remain stable vs FP32).
  - [ ] Run a baseline subset: `uv run pytest -q tests/test_recurrent_vectorized.py tests/simic/test_ppo.py`.
- **Target:** `src/esper/simic/vectorized.py`
- **Action:**
  - Gate AMP behind a single explicit knob (e.g., `--amp` / `TrainingConfig.amp`), default **off**.
  - Wrap forward/loss/backward in `torch.amp.autocast` and add `GradScaler` for FP16 stability.
- **Verification:** PPO runs without NaNs; G2 gate metrics remain stable.

### [ ] Step 15: Real `TrainingConfig` (Simic PR7a + PR7b)

- **Pre-flight checklist:**
  - [ ] Confirm Step 14 is merged.
  - [ ] Inventory all `TrainingConfig` fields + all PPO CLI flags and mark each as: (a) real effect and kept, or (b) deleted; no “paper surface”.
  - [ ] Confirm there is no backwards-compat path: unknown config keys must hard-fail; removed CLI flags must be removed from docs/scripts/tests in the same PR.
  - [ ] Identify all call sites of `train_ppo_vectorized(...)` and confirm the config → kwargs surface is the only route (no parallel hidden defaults).
  - [ ] Run a baseline config subset: `uv run pytest -q tests/test_simic_config.py tests/simic/test_config.py`.
- **Targets:** `src/esper/simic/config.py`, `src/esper/scripts/train.py`, `README.md`, `scripts/train_cifar.sh`, `scripts/train_tinystories.sh`, `tests/test_simic_config.py`, `tests/simic/test_config.py`
- **Action (PR7a — make config honest + add guardrails):**
  - Delete paper surface from `TrainingConfig`:
    - Remove YAML/serialization claims.
    - Remove `for_imagenet()` (no ImageNet task exists in `src/esper/runtime/tasks.py`).
  - Prune config fields so every field is either passed into `train_ppo_vectorized()` (real effect) or deleted (no no-op knobs).
  - Add strict JSON loading for the config surface:
    - `TrainingConfig.from_json_path(path)` uses stdlib `json`.
    - Unknown keys hard-fail (no aliasing / no compatibility keys).
  - Add drift-proofing contract tests:
    - `TrainingConfig().to_train_kwargs().keys()` ⊆ `inspect.signature(train_ppo_vectorized).parameters`.
- **Action (PR7b — cut over PPO CLI to `TrainingConfig` as the single source of truth):**
  - In `esper.scripts.train ppo`:
    - Add `--preset {cifar10,cifar10_deep,tinystories}` and `--config-json PATH`.
    - Delete redundant/no-op PPO hyperparameter flags (including the dead `--update-every` knob).
    - Keep runtime wiring flags (e.g., `--task`, `--device`, `--devices`, `--amp`, dataloader perf knobs, telemetry outputs/TUI), but have them flow through the config surface (no parallel “hidden defaults”).
    - Print the effective config at start (summary or JSON) for reproducibility.
  - Update `README.md` CLI reference + `scripts/train_*.sh` to match the config-first launch path.
- **Verification:**
  - `esper.scripts.train ppo --help` no longer advertises removed/no-op flags (e.g. `--update-every`).
  - Unknown keys in `--config-json` fail fast.
  - Contract tests prevent silent drift between config ⇄ training signature.

### [ ] Step 16: Final Cleanup (Simic PR8)

- **Pre-flight checklist:**
  - [ ] Confirm Step 15 is merged.
  - [ ] Run a baseline `ruff check src/ tests/` and `mypy src/` to see the exact starting deltas (avoid “cleanup creep”).
  - [ ] Inventory “temporary” helpers you intend to delete with `rg -n "TODO:|temporary|normalize_observation" src/ tests/`.
  - [ ] Confirm the cleanup is purely reductive (no behavior changes beyond removing dead code/unused imports).
- **Target:** Global
- **Action:** Delete unused imports, `normalize_observation`, and temporary code.
- **Verification:** `ruff check` is clean. `pytest` passes. `mypy src/` passes.

---

## 🚫 Explicit Deferrals (Out of Scope for Workstream 001)

- **DDP v0 (true distributed training):** deferred; dynamic seed parameters + counterfactual (`force_alpha`) are not DDP-safe yet.
- **Kasmina `fast_mode` knob:** deferred; only revisit after per-slot telemetry schema is stable and we have real throughput data.

---

## Appendix A — External Review Feedback (Verbatim, Incorporated)

The feedback below is **already incorporated** in this plan (notably Steps 5/6/8/9/11 and their pre-flight checklists). It’s kept here as a permanent record to prevent drift.

```text
This bundle is in really good shape. It aligns with my earlier sequencing almost 1:1, and the v1.4 edits fix the main practical risk I flagged (Step 8 being too broad). You’ve also done the most important meta-thing: you’ve made one doc the canonical execution order and demoted the other docs to “rationale”. That prevents drift.

Here’s how it lines up with my thoughts, plus the few tweaks I’d make before you hit “go”.

## What’s excellent and matches my intent

* Phase 1 is “semantics + contracts + deletions”. Correct.
* CNN segmented forward fix is early. Correctness before performance.
* You added Step 1b (delete dead gradient collector) early. Great. It reduces later refactor noise in vectorized.py.
* Step 4 now includes explicit pytest tests/kasmina. This addresses the only real worry with early deletions.
* Step 6 merges report wiring + per-slot telemetry + masks. This is exactly the hotspot consolidation I’d recommend.
* Step 8 split into 8a/8b/8c. Perfect. That makes Step 8 reviewable and bisectable.
* AMP moved to Phase 4 and has an explicit “G2 gate metrics remain stable” check. That’s the right placement and the right acceptance criterion.
* TrainingConfig split PR7a/PR7b with signature contract tests is very strong. This is how you prevent “paper surface” reappearing.

Net: it’s coherent, it’s ruthless, and it minimises churn in the three hot files (vectorized.py, ppo.py, action_masks.py).

## The only places I’d tighten (small adjustments)

### 1) Step 5 smoke test needs to be hermetic

Right now it says “runs a tiny PPO job on 2 GPUs”. The common failure mode is it triggers dataset download or long data prep, so it becomes flaky.

I’d add one explicit rule to Step 5:

* Use a synthetic dataset or a cached/local-only dataset path, and fail with a clear message if it would download.

Also add an assertion that envs actually map to both GPUs (avoid a false pass).

### 2) Step 6: observation layout contract should be encoded in code, not just prose

You’ve stated “deterministic [Early, Mid, Late] order based on enabled slots”. Good.

I’d make the acceptance criteria mention:

* A single function owns slot ordering (e.g. ordered_slots(enabled_slots) -> tuple[str, ...]) and both masking and feature extraction use it.
* Add a unit test: slot order stays [early, mid, late] even if dict insertion order changes.

This prevents subtle regressions later when someone “cleans up” dicts.

### 3) Step 8: add one measurement-driven “done” check per substep

You’ve got “manual run confirms GPU utilisation”, which is good, but I’d specify what you want to see:

* 8a: print env→device mapping counts, confirm >0 envs on each device
* 8b: throughput does not regress versus pre-8b baseline on 1 GPU
* 8c: p95 env step time improves or GPU util becomes smoother (sawtooth reduced)

Even if these are operator eyeballs, writing the criterion prevents “it shipped but it’s slower”.

### 4) Step 9 (multi-updates per batch): add the normaliser invariant explicitly

You’ve got “fix entropy annealing to count steps” but the subtle one is the observation normaliser update once per rollout batch.

I’d add a bullet to Step 9 acceptance criteria:

* RunningMeanStd updates exactly once per rollout batch, not per PPO update step.

This is important enough to bake into the plan (and tests).

### 5) Step 11 (MaskedCategorical): explicitly specify entropy semantics

You already say “correct entropy under masking”. I’d specify:

* Entropy is computed over valid actions only, and (if you normalise) it is normalised by log(n_valid) so it lands in [0,1].

It’s easy to get this wrong and then your entropy coefficients become meaningless.

## Potential internal contradiction: fixed-size PPO obs vs “dynamic slots”

This is the only conceptual mismatch, but it’s not a blocker.

* PPO obs: fixed 3-slot layout for now (80 dims)
* UI / future Web: should be dynamic and render any slot keys

That’s fine, but I’d ensure Step 6 doesn’t accidentally bake “3 slots forever” into the report API. The report API should be dynamic; only the PPO feature concatenation is fixed.

## Does anything in this bundle accidentally violate earlier ground rules?

No. In fact, the bundle enforces them harder:

* lying knobs removed
* no legacy retained
* incremental verification
* explicit deferrals (DDP, fast_mode)

The only thing to watch is Step 4 deletions happening before Step 6 wiring. Your added “pytest tests/kasmina” makes that safe if you stick to the “rg confirms zero call sites” rule.

## My suggested final execution order

Honestly, execute exactly as written. It matches the safe order:

1. delete nonsense + dead APIs
2. fix semantics
3. establish contracts
4. delete kasmina dead code + harden
5. add multi-GPU tripwire test
6. wire reports + per-slot obs + masks
7. make tamiyo/heuristic multi-slot honest
8. build engine in 3 substeps
   9–13 upgrade algorithmic behaviours
   14 AMP
   15 config consolidation
   16 cleanup

Confidence (WEP): highly likely this plan minimises conflict/rework and gives you clean “sign-off” gates at the right times (tests + smoke harness + deterministic obs contract). The risks that remain are mostly the ones you can’t eliminate on paper: CUDA stream subtlety and RL training dynamics after MaskedCategorical, but your sequencing makes those problems measurable rather than mysterious.
```
