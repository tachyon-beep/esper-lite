# Telemetry Overwatch Phase 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ship low-risk telemetry improvements so UI/dashboards stay truthful and debuggable.

**Architecture:** Wire richer telemetry at emission points (Simic vectorized/heuristic loops, Kasmina lifecycle, Tolaria governor, CLI wiring) with minimal runtime overhead. Add guarded flags, richer payloads, and missing context fields; keep changes behind config to avoid regressions.

**Tech Stack:** Python 3.11, PyTorch, argparse CLI, pytest.

---

### Task 1: Telemetry guardrails + lifecycle-only mode

**Files:**
- Modify: `scripts/train.py` (CLI flag wiring)
- Modify: `src/esper/simic/vectorized.py`
- Modify: `src/esper/simic/training.py`
- Modify: `src/esper/kasmina/slot.py`
- Tests: `tests/scripts/test_train.py` (add), `tests/simic/test_vectorized.py` (extend)

**Step 1: Write failing test** (CLI flag and lifecycle-only behavior)
```python
# tests/scripts/test_train.py
def test_telemetry_lifecycle_only_flag_wired(monkeypatch):
    import esper.scripts.train as train
    parser = train.build_parser()  # or however parser is built
    args = parser.parse_args(["heuristic", "--telemetry-lifecycle-only"])
    assert args.telemetry_lifecycle_only is True
```
Add vectorized guard test:
```python
# tests/simic/test_vectorized.py
def test_lifecycle_only_keeps_slot_telemetry(mocker):
    from esper.simic import vectorized
    slot = mocker.Mock(fast_mode=False, on_telemetry=None)
    env_state = mocker.Mock(model=mocker.Mock(seed_slots={"mid": slot}))
    vectorized._apply_slot_telemetry(env_state, telemetry_enabled=False, lifecycle_only=True)
    assert slot.fast_mode is False
    assert slot.on_telemetry is not None
```

**Step 2: Run tests to see failures**
- `pytest tests/scripts/test_train.py -q`
- `pytest tests/simic/test_vectorized.py::test_lifecycle_only_keeps_slot_telemetry -q`

**Step 3: Implement**
- Add `--telemetry-lifecycle-only` flag default False in `scripts/train.py`; plumb to training entrypoints.
- In `vectorized.py`/`training.py`, accept lifecycle-only flag; when true, keep slot telemetry active even if main telemetry off; emit single warning `TelemetryEvent` when telemetry disabled.
- In `kasmina/slot.py`, respect lifecycle-only by skipping fast_mode short-circuit when flag is set.

**Step 4: Re-run tests**
- `pytest tests/scripts/test_train.py tests/simic/test_vectorized.py -q`

**Step 5: Commit**
```bash
git add scripts/train.py src/esper/simic/vectorized.py src/esper/simic/training.py src/esper/kasmina/slot.py tests/scripts/test_train.py tests/simic/test_vectorized.py
git commit -m "feat(telemetry): add lifecycle-only mode and telemetry warning"
```

---

### Task 2: Env context enforcement (env_id/device on events)

**Files:**
- Modify: `src/esper/simic/vectorized.py`
- Modify: `src/esper/simic/training.py`
- Modify: `src/esper/tolaria/governor.py`
- Tests: `tests/simic/test_vectorized.py`, `tests/tolaria/test_governor.py` (add)

**Step 1: Write failing test**
```python
# tests/tolaria/test_governor.py
def test_governor_rollback_includes_env_and_device(mocker):
    from esper.tolaria.governor import TolariaGovernor
    model = mocker.Mock(parameters=lambda: iter([mocker.Mock(device="cuda:0")]))
    gov = TolariaGovernor(model)
    gov.last_good_state = {"w": mocker.Mock(to=lambda *a, **k: mocker.Mock())}
    # force panic path to emit event
    hub = mocker.patch("esper.tolaria.governor.get_hub")
    hub.return_value = mocker.Mock()
    gov.execute_rollback()
    event = hub.return_value.emit.call_args[0][0]
    assert event.data["device"] == "cuda:0"
```

**Step 2: Run failing tests**
- `pytest tests/tolaria/test_governor.py::test_governor_rollback_includes_env_and_device -q`

**Step 3: Implement**
- Add helper `_with_env_id(event, env_idx, device)` in vectorized/heuristic telemetry callbacks; inject `env_id` and `device` into all lifecycle/rollback events.
- In `TolariaGovernor.execute_rollback`, include `env_id` (default 0 or passed) and `device` in `TelemetryEvent.data`.

**Step 4: Re-run tests**
- `pytest tests/tolaria/test_governor.py tests/simic/test_vectorized.py -q`

**Step 5: Commit**
```bash
git add src/esper/simic/vectorized.py src/esper/simic/training.py src/esper/tolaria/governor.py tests/tolaria/test_governor.py
git commit -m "feat(telemetry): enforce env_id/device on lifecycle and rollback events"
```

---

### Task 3: Lifecycle payloads (alpha + epochs)

**Files:**
- Modify: `src/esper/kasmina/slot.py`
- Modify: `src/esper/simic/vectorized.py`
- Modify: `src/esper/simic/training.py`
- Tests: `tests/kasmina/test_slot_telemetry.py` (add)

**Step 1: Write failing test**
```python
# tests/kasmina/test_slot_telemetry.py
def test_lifecycle_events_include_alpha_and_epochs(mocker):
    from esper.kasmina.slot import SeedSlot
    slot = SeedSlot(slot_id="mid", channels=64, device="cpu", task_config=None, on_telemetry=mocker.Mock())
    slot.state = mocker.Mock(seed_id="s1", stage=mocker.Mock(name="TRAINING"), alpha=0.3, metrics=mocker.Mock(epochs_in_current_stage=2))
    slot._emit_telemetry(event_type=mocker.Mock(), data={"alpha": 0.3, "inner_epoch": 5, "global_epoch": 12})
    emitted = slot.on_telemetry.call_args[0][0]
    assert emitted.data["alpha"] == 0.3
    assert emitted.data["inner_epoch"] == 5
    assert emitted.data["global_epoch"] == 12
```

**Step 2: Run failing test**
- `pytest tests/kasmina/test_slot_telemetry.py::test_lifecycle_events_include_alpha_and_epochs -q`

**Step 3: Implement**
- Extend lifecycle event payloads to include `alpha`, `inner_epoch`, `global_epoch`; thread epoch counters from training loops into `_emit_telemetry` calls.
- In vectorized/heuristic loops, maintain `inner_epoch` (episode epoch) and `global_epoch` (batch counter) and pass to slot emitters.

**Step 4: Re-run tests**
- `pytest tests/kasmina/test_slot_telemetry.py -q`

**Step 5: Commit**
```bash
git add src/esper/kasmina/slot.py src/esper/simic/vectorized.py src/esper/simic/training.py tests/kasmina/test_slot_telemetry.py
git commit -m "feat(telemetry): add alpha and epoch context to lifecycle events"
```

---

### Task 4: Per-step last-action detail

**Files:**
- Modify: `src/esper/simic/vectorized.py`
- Tests: `tests/simic/test_vectorized.py` (add case)

**Step 1: Write failing test**
```python
# tests/simic/test_vectorized.py
def test_last_action_event_emitted(mocker):
    from esper.simic import vectorized
    hub = mocker.patch("esper.simic.vectorized.get_hub")
    hub.return_value = mocker.Mock()
    event_data = vectorized._emit_last_action(
        env_id=0,
        epoch=3,
        factored_action=mocker.Mock(op=mocker.Mock(name="GERMINATE"), slot_id="mid", blueprint_id="conv", blend_id="sigmoid"),
        masked={"op": False, "slot": False, "blueprint": False, "blend": True},
        success=True,
    )
    emitted = hub.return_value.emit.call_args[0][0]
    assert emitted.data["slot_id"] == "mid"
    assert emitted.data["blend_masked"] is True
```

**Step 2: Run failing test**
- `pytest tests/simic/test_vectorized.py::test_last_action_event_emitted -q`

**Step 3: Implement**
- Add helper `_emit_last_action` that builds and emits a `TelemetryEvent` (new type or reuse `ANALYTICS_SNAPSHOT` payload) with op/slot/blueprint/blend IDs, per-head masked flags, action_success, env_id, epoch.
- Call helper immediately after action execution per env.

**Step 4: Re-run tests**
- `pytest tests/simic/test_vectorized.py -q`

**Step 5: Commit**
```bash
git add src/esper/simic/vectorized.py tests/simic/test_vectorized.py
git commit -m "feat(telemetry): emit per-step last-action detail"
```

---

### Task 5: PPO vitals (lr, grad_norm surrogate, update_time_ms)

**Files:**
- Modify: `src/esper/simic/vectorized.py`
- Tests: `tests/simic/test_vectorized.py` (add)

**Step 1: Write failing test**
```python
def test_ppo_update_event_includes_vitals(mocker):
    from esper.simic import vectorized
    hub = mocker.patch("esper.simic.vectorized.get_hub")
    hub.return_value = mocker.Mock()
    metrics = {"policy_loss": 0.1}
    vectorized._emit_ppo_update_event(
        hub=hub.return_value,
        metrics=metrics,
        episodes_completed=5,
        batch_idx=0,
        epoch=3,
        optimizer=mocker.Mock(param_groups=[{"lr": 0.0003}]),
        grad_norm=1.23,
        update_time_ms=12.5,
    )
    data = hub.return_value.emit.call_args[0][0].data
    assert data["lr"] == 0.0003
    assert data["grad_norm"] == 1.23
    assert data["update_time_ms"] == 12.5
```

**Step 2: Run failing test**
- `pytest tests/simic/test_vectorized.py::test_ppo_update_event_includes_vitals -q`

**Step 3: Implement**
- Time PPO update loop; compute grad norm surrogate over policy params; pull lr from optimizer param_groups.
- Attach `lr`, `grad_norm`, `update_time_ms` to `PPO_UPDATE_COMPLETED` event data.

**Step 4: Re-run tests**
- `pytest tests/simic/test_vectorized.py -q`

**Step 5: Commit**
```bash
git add src/esper/simic/vectorized.py tests/simic/test_vectorized.py
git commit -m "feat(telemetry): add PPO vitals (lr, grad_norm, update time)"
```

---

### Task 6: Action distribution summary per batch

**Files:**
- Modify: `src/esper/simic/vectorized.py`
- Tests: `tests/simic/test_vectorized.py` (add)

**Step 1: Write failing test**
```python
def test_action_distribution_snapshot(mocker):
    from esper.simic import vectorized
    hub = mocker.patch("esper.simic.vectorized.get_hub")
    hub.return_value = mocker.Mock()
    vectorized._emit_action_distribution(
        hub=hub.return_value,
        batch_idx=1,
        episodes_completed=4,
        action_counts={"WAIT": 3, "GERMINATE": 1},
        success_counts={"WAIT": 3, "GERMINATE": 1},
    )
    data = hub.return_value.emit.call_args[0][0].data
    assert data["action_counts"]["WAIT"] == 3
```

**Step 2: Run failing test**
- `pytest tests/simic/test_vectorized.py::test_action_distribution_snapshot -q`

**Step 3: Implement**
- Add `_emit_action_distribution` helper emitting `ANALYTICS_SNAPSHOT` (or new event) with action_counts and success_counts per batch (or every N via modulus).
- Call after batch loop using existing counters.

**Step 4: Re-run tests**
- `pytest tests/simic/test_vectorized.py -q`

**Step 5: Commit**
```bash
git add src/esper/simic/vectorized.py tests/simic/test_vectorized.py
git commit -m "feat(telemetry): emit action distribution summaries"
```

---

### Task 7: Counterfactual coverage + fallback baseline

**Files:**
- Modify: `src/esper/simic/vectorized.py`
- Tests: `tests/simic/test_vectorized.py` (add)

**Step 1: Write failing test**
```python
def test_counterfactual_unavailable_event(mocker):
    from esper.simic import vectorized
    hub = mocker.patch("esper.simic.vectorized.get_hub")
    hub.return_value = mocker.Mock()
    vectorized._emit_cf_unavailable(hub.return_value, env_id=0, slot_id="mid", reason="missing_baseline")
    data = hub.return_value.emit.call_args[0][0].data
    assert data["available"] is False
    assert data["reason"] == "missing_baseline"
```

**Step 2: Run failing test**
- `pytest tests/simic/test_vectorized.py::test_counterfactual_unavailable_event -q`

**Step 3: Implement**
- Emit marker when counterfactual baseline missing (reuse `COUNTERFACTUAL_COMPUTED` with `available=False` or new type).
- Add config flag to run cheap per-slot alpha=0 eval on final epoch for small models; emit resulting `COUNTERFACTUAL_COMPUTED`.

**Step 4: Re-run tests**
- `pytest tests/simic/test_vectorized.py -q`

**Step 5: Commit**
```bash
git add src/esper/simic/vectorized.py tests/simic/test_vectorized.py
git commit -m "feat(telemetry): mark missing counterfactuals and add final-epoch baseline option"
```

---

### Task 8: Per-env throughput metrics (step time/fps/dataloader wait)

**Files:**
- Modify: `src/esper/simic/vectorized.py`
- Tests: `tests/simic/test_vectorized.py` (add)

**Step 1: Write failing test**
```python
def test_throughput_metrics_emitted(mocker):
    from esper.simic import vectorized
    hub = mocker.patch("esper.simic.vectorized.get_hub")
    hub.return_value = mocker.Mock()
    vectorized._emit_throughput(
        hub=hub.return_value,
        env_id=0,
        batch_idx=1,
        episodes_completed=4,
        step_time_ms=5.0,
        dataloader_wait_ms=2.0,
    )
    data = hub.return_value.emit.call_args[0][0].data
    assert data["step_time_ms"] == 5.0
    assert data["dataloader_wait_ms"] == 2.0
```

**Step 2: Run failing test**
- `pytest tests/simic/test_vectorized.py::test_throughput_metrics_emitted -q`

**Step 3: Implement**
- Wrap env step and dataloader fetch with `time.perf_counter()`, compute per-env step_time_ms, fps (optional), dataloader_wait_ms; emit per batch in `ANALYTICS_SNAPSHOT`.

**Step 4: Re-run tests**
- `pytest tests/simic/test_vectorized.py -q`

**Step 5: Commit**
```bash
git add src/esper/simic/vectorized.py tests/simic/test_vectorized.py
git commit -m "feat(telemetry): emit per-env throughput metrics"
```

---

### Task 9 (Optional): Reward summary at ops-normal

**Files:**
- Modify: `src/esper/simic/vectorized.py`
- Tests: `tests/simic/test_vectorized.py`

**Step 1: Write failing test**
```python
def test_reward_summary_emitted(mocker):
    from esper.simic import vectorized
    hub = mocker.patch("esper.simic.vectorized.get_hub")
    hub.return_value = mocker.Mock()
    vectorized._emit_reward_summary(
        hub=hub.return_value,
        env_id=0,
        batch_idx=1,
        summary={"bounded_attribution": 0.4, "compute_rent": -0.1, "total_reward": 0.3},
    )
    data = hub.return_value.emit.call_args[0][0].data
    assert data["summary"]["total_reward"] == 0.3
```

**Step 2: Run failing test**
- `pytest tests/simic/test_vectorized.py::test_reward_summary_emitted -q`

**Step 3: Implement**
- Aggregate reward components per batch when telemetry level >= NORMAL; emit compact summary via `ANALYTICS_SNAPSHOT`.

**Step 4: Re-run tests**
- `pytest tests/simic/test_vectorized.py -q`

**Step 5: Commit**
```bash
git add src/esper/simic/vectorized.py tests/simic/test_vectorized.py
git commit -m "feat(telemetry): emit reward summaries at ops-normal"
```

---

### Task 10 (Optional): Per-head/mask hit rates

**Files:**
- Modify: `src/esper/simic/vectorized.py`
- Tests: `tests/simic/test_vectorized.py`

**Step 1: Write failing test**
```python
def test_mask_hit_rates_emitted(mocker):
    from esper.simic import vectorized
    hub = mocker.patch("esper.simic.vectorized.get_hub")
    hub.return_value = mocker.Mock()
    vectorized._emit_mask_hit_rates(
        hub=hub.return_value,
        batch_idx=1,
        episodes_completed=4,
        mask_hits={"op": 10},
        mask_total={"op": 12},
    )
    data = hub.return_value.emit.call_args[0][0].data
    assert data["mask_hits"]["op"] == 10
    assert data["mask_total"]["op"] == 12
```

**Step 2: Run failing test**
- `pytest tests/simic/test_vectorized.py::test_mask_hit_rates_emitted -q`

**Step 3: Implement**
- Track mask applied/allowed counts per head during action sampling on device; sync to CPU once per batch; emit aggregates in PPO update or analytics snapshot.

**Step 4: Re-run tests**
- `pytest tests/simic/test_vectorized.py -q`

**Step 5: Commit**
```bash
git add src/esper/simic/vectorized.py tests/simic/test_vectorized.py
git commit -m "feat(telemetry): emit per-head mask hit rates"
```

---

## Final Verification
- Run focused tests added above plus smoke suite: `pytest -m "not slow" -q`
- Ensure telemetry runs without regressions in a dry run: `PYTHONPATH=src uv run python -m esper.scripts.train heuristic --task cifar10 --episodes 1 --telemetry-level normal --telemetry-lifecycle-only`

## Execution Handoff
Plan complete. Two execution options:
1) **Subagent-Driven (this session)** — use superpowers:subagent-driven-development to execute tasks sequentially with code review between tasks.
2) **Parallel Session** — new session using superpowers:executing-plans to run the plan in batches with checkpoints.
