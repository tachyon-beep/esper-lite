# Karn PPO Traceability Evidence Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a Karn-backed PPO traceability evidence section to reward-efficiency proof packets so update counts, missing fields, skipped/nonfinite evidence, robust value signals, and low-return-variance counts are visible and block before ROI verdict math when malformed.

**Architecture:** Keep Karn public views as the proof surface. Extend `ppo_updates` with typed fields already present in `PPOUpdatePayload`, add an aggregate `ppo_traceability_evidence` view over `runs`, `episode_outcomes`, `ppo_updates`, and `batch_stats`, update the MCP view catalog, then make `scripts/proof_packet.py` render and gate from that view. Do not add new PPO algorithm behavior.

**Tech Stack:** Python 3.11, DuckDB SQL views, pytest, existing Karn telemetry JSONL fixtures.

**Prerequisites:**
- Active Filigree issue: `esper-lite-570d98c451`.
- Branch: `feat/ev-stab-stage2-hra`.
- Current code checkpoint: `a1c6fcca`.

---

### Task 1: Karn View Contract

**Files:**
- Modify: `src/esper/karn/mcp/views.py`
- Modify: `src/esper/karn/mcp/server.py`
- Test: `tests/karn/mcp/test_views.py`

**Step 1: Write failing tests**

Add tests proving:

1. `VIEW_DEFINITIONS` includes `ppo_traceability_evidence`.
2. `ppo_updates` exposes `inf_grad_count`, `update_skipped`, `skipped`, `ppo_updates_count`, `ev_low_return_variance_count`, `return_variance`, and `value_target_scale`.
3. `ppo_traceability_evidence` reports, per `(run_dir, group_id)`, `update_count`, `valid_outcome_count`, `missing_required_update_count`, `invalid_required_update_count`, `skipped_update_count`, `nonfinite_update_count`, `nan_grad_update_count`, `inf_grad_update_count`, `low_return_variance_update_count`, `low_return_variance_epoch_count`, and robust value ranges.
4. A run with outcome evidence but no PPO update row still appears with `update_count = 0`.
5. `VIEW_CATALOG` advertises the new evidence view as a public proof surface.

Run:

```bash
uv run pytest tests/karn/mcp/test_views.py::test_view_definitions_exist tests/karn/mcp/test_views.py::test_ppo_updates_exposes_proof_traceability_fields tests/karn/mcp/test_views.py::test_ppo_traceability_evidence_summarizes_update_status tests/karn/mcp/test_views.py::test_ppo_traceability_evidence_keeps_outcome_group_with_zero_updates -q
```

Expected RED: missing view/column errors.

**Step 2: Implement minimal view changes**

- In `ppo_updates`, extract:
  - `json_extract(data, '$.inf_grad_count')::INTEGER as inf_grad_count`
  - `json_extract(data, '$.update_skipped')::BOOLEAN as update_skipped`
  - `json_extract(data, '$.skipped')::BOOLEAN as skipped`
  - `json_extract(data, '$.ppo_updates_count')::INTEGER as ppo_updates_count`
  - `json_extract(data, '$.ev_low_return_variance_count')::INTEGER as ev_low_return_variance_count`
  - `json_extract(data, '$.return_variance')::DOUBLE as return_variance`
  - `json_extract(data, '$.value_target_scale')::DOUBLE as value_target_scale`
- Add `ppo_traceability_evidence` after `episode_outcomes`, using cohorts from `runs`, `episode_outcomes`, `ppo_updates`, and `batch_stats`.
- Add a concise `VIEW_CATALOG` entry for `ppo_traceability_evidence`.
- Use DuckDB `isfinite(...)` checks for malformed numeric evidence.
- Treat raw `explained_variance` as diagnostic-only; robust fields are `value_loss`, `bellman_error`, `v_return_correlation`, `value_nrmse`, `return_std`, `ev_low_return_variance`, `ev_return_variance`, and `ev_low_return_variance_count`.

**Definition of Done:**
- RED tests fail for the expected missing view/column reasons.
- View tests pass.
- No defensive Python patterns are introduced.

---

### Task 2: Proof Packet Evidence Section and Blocking

**Files:**
- Modify: `scripts/proof_packet.py`
- Test: `tests/scripts/test_proof_packet.py`

**Step 1: Write failing tests**

Add tests proving:

1. A healthy generic packet renders `## PPO Traceability Evidence` with update counts, missing-field count zero, skipped count zero, nonfinite count zero, NaN/Inf counts zero, and low-return-variance counts.
2. A skipped PPO batch from `batch_stats.skipped_update = true` blocks with `BLOCKED_MECHANICS` before ROI when update evidence is otherwise present.
3. A nonfinite PPO update field blocks with `BLOCKED_INSTRUMENTATION` before ROI.
4. Missing robust value fields block and name the missing fields.
5. Bad raw `explained_variance` alone remains diagnostic when robust value fields are proof-grade.

Run:

```bash
uv run pytest tests/scripts/test_proof_packet.py::test_proof_packet_renders_ppo_traceability_evidence tests/scripts/test_proof_packet.py::test_proof_packet_blocks_skipped_ppo_update_before_roi tests/scripts/test_proof_packet.py::test_proof_packet_blocks_nonfinite_ppo_traceability_before_roi tests/scripts/test_proof_packet.py::test_proof_packet_blocks_missing_robust_value_traceability -q
```

Expected RED: missing section and/or missing blockers.

**Step 2: Implement packet changes**

- Replace the narrow learnability query with a PPO instrumentation query that uses `ppo_traceability_evidence` plus row-level missing-field details from `ppo_updates`.
- Require robust value fields for reward-efficiency and generic PPO proof packets.
- Add a dedicated `## PPO Traceability Evidence` section before `## Learnability Gate`.
- Preserve oracle-sandbox exemption from PPO traceability requirements.
- Keep low-return-variance counts diagnostic: render them, but do not block only because `ev_low_return_variance` is true.
- Block missing or nonfinite required public PPO evidence as `BLOCKED_INSTRUMENTATION`.
- Block observed impossible PPO values as `BLOCKED_MECHANICS`, including skipped updates, positive `nan_grad_count`/`inf_grad_count`, gradient state `nonfinite`, negative counts, `kl_divergence < 0`, clip fractions outside `[0, 1]`, `ratio_min <= 0`, `ratio_max < ratio_min`, and `joint_ratio_max <= 0`.

**Definition of Done:**
- RED tests fail for the expected missing behavior.
- Packet tests pass.
- Existing packet verdict ordering remains fail-closed before ROI.

---

### Task 3: Review and Verification

**Files:**
- Verify touched code and tests only first, then broaden.

**Commands:**

```bash
uv run pytest tests/karn/mcp/test_views.py tests/scripts/test_proof_packet.py -q
uv run ruff check src/esper/karn/mcp/views.py src/esper/karn/mcp/server.py scripts/proof_packet.py tests/karn/mcp/test_views.py tests/scripts/test_proof_packet.py
git diff --check
wardline scan . --fail-on ERROR
```

Run full default pytest before closing the issue if the focused suites are clean:

```bash
uv run pytest -q
```

**Definition of Done:**
- Specialist review findings are addressed or explicitly dispositioned.
- Focused and broad verification pass, except any clearly pre-existing unrelated guardrail failures.
- Commit code with a conventional commit.
- Close `esper-lite-570d98c451` with the commit anchor.
- Checkpoint `docs/product/current-state.md` and append a PDR if the next product critical path moves.
