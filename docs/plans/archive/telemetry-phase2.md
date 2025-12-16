# Telemetry Overwatch Phase 2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the remaining “Overwatch truthfulness” telemetry gaps (gate visibility + slot health) and add moderate diagnostic coverage without destabilizing throughput.

**Architecture:** Emit explicit gate-evaluation events from Kasmina (seed lifecycle), enrich lifecycle payloads with per-slot health fields, and complete throughput telemetry with an explicit `fps` field. Keep emissions low-cost and gated behind existing telemetry controls.

**Tech Stack:** Python 3.11+, PyTorch, pytest, ruff, uv, argparse.

---

## Scope (Phase 2 inclusions)

### Must-Ship (UI truthfulness gaps)
- **Gate visibility:** Emit gate-evaluated events (gate id, pass/fail, reasons) so UIs can show why a seed is blocked.
- **Slot health fields:** Include `gradient_health/vanish/explode`, `isolation_violations`, `seed_gradient_norm_ratio` in lifecycle telemetry payloads.
- **Throughput completeness:** Add `fps` to per-env throughput telemetry.

### Moderate additions (optional if time stays reasonable)
- **DiagnosticTracker cadence hook:** Optionally run Nissa `DiagnosticTracker` every K epochs and emit a snapshot event for replay/UI.

### Already delivered in Phase 1 (do not re-implement)
- Counterfactual-unavailable markers + reason codes.
- PPO vitals (`lr`, `grad_norm`, `update_time_ms`) + action distribution summary.
- Per-head mask hit rates (head-level) and ops-normal reward summary.

---

## Acceptance

- A failed stage advance produces a gate-evaluated telemetry event with the failed checks/reasons.
- Seed lifecycle events include the slot-health fields when available (and never lie by emitting bogus defaults).
- Throughput telemetry includes `fps` (and remains stable under zero/near-zero timings).
- Tests cover gate emission (pass + fail), health-field inclusion, and `fps` on throughput events.

---

### Task 1: Add `SEED_GATE_EVALUATED` event type (Leyline contract)

**Files:**
- Modify: `src/esper/leyline/telemetry.py`
- Test: `tests/integration/test_telemetry_event_formatters.py`

**Step 1: Write the failing test**

```python
# tests/integration/test_telemetry_event_formatters.py
def test_seed_gate_evaluated_exists() -> None:
    from esper.leyline import TelemetryEventType

    event_type = TelemetryEventType.SEED_GATE_EVALUATED
    assert event_type.name == "SEED_GATE_EVALUATED"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/integration/test_telemetry_event_formatters.py::test_seed_gate_evaluated_exists -q`
Expected: FAIL with `AttributeError: SEED_GATE_EVALUATED`

**Step 3: Implement the minimal contract change**

Add to `TelemetryEventType`:

```python
# src/esper/leyline/telemetry.py
SEED_GATE_EVALUATED = auto()
```

**Step 4: Re-run test**

Run: `uv run pytest tests/integration/test_telemetry_event_formatters.py::test_seed_gate_evaluated_exists -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/leyline/telemetry.py tests/integration/test_telemetry_event_formatters.py
git commit -m "feat(telemetry): add SEED_GATE_EVALUATED event type"
```

---

### Task 2: Emit gate-evaluated events from Kasmina `SeedSlot`

**Intent:** A gate check can fail without any stage transition. Emit a truthful event anyway.

**Files:**
- Modify: `src/esper/kasmina/slot.py`
- Test: `tests/kasmina/test_gate_telemetry.py` (create)

**Step 1: Write failing tests (pass + fail)**

```python
# tests/kasmina/test_gate_telemetry.py
from unittest.mock import Mock

import torch

from esper.kasmina.slot import SeedSlot, SeedState
from esper.leyline import SeedStage, TelemetryEventType


def _collect_events():
    events = []
    def on_event(e):
        events.append(e)
    return events, on_event


def test_gate_event_emitted_on_pass():
    events, on_event = _collect_events()
    slot = SeedSlot(slot_id="mid", channels=8, device="cpu", on_telemetry=on_event)
    slot.seed = torch.nn.Identity()
    slot.state = SeedState(seed_id="s1", blueprint_id="conv", slot_id="mid")
    slot.state.stage = SeedStage.GERMINATED  # G1 pass condition for TRAINING

    slot.advance_stage(target_stage=SeedStage.TRAINING)

    assert any(e.event_type == TelemetryEventType.SEED_GATE_EVALUATED for e in events)


def test_gate_event_emitted_on_fail():
    events, on_event = _collect_events()
    slot = SeedSlot(slot_id="mid", channels=8, device="cpu", on_telemetry=on_event)
    slot.seed = torch.nn.Identity()
    slot.state = SeedState(seed_id="s1", blueprint_id="conv", slot_id="mid")
    slot.state.stage = SeedStage.TRAINING  # Attempt BLENDING (G2), default metrics should fail

    result = slot.advance_stage(target_stage=SeedStage.BLENDING)

    assert result.passed is False
    gate_events = [e for e in events if e.event_type == TelemetryEventType.SEED_GATE_EVALUATED]
    assert gate_events, "Expected a gate-evaluated event on failure"
    assert gate_events[-1].data["passed"] is False
    assert gate_events[-1].data["target_stage"] == "BLENDING"
```

**Step 2: Run tests to verify failure**

Run: `uv run pytest tests/kasmina/test_gate_telemetry.py -q`
Expected: FAIL because gate event isn’t emitted.

**Step 3: Implement emission in `SeedSlot`**

In `src/esper/kasmina/slot.py`, after every `gate_result = self.gates.check_gate(...)` call:
- Emit `TelemetryEventType.SEED_GATE_EVALUATED` **before** any transition.
- Include at least:
  - `gate`: `gate_result.gate.name`
  - `passed`: `gate_result.passed`
  - `target_stage`: `target_stage.name` (or `"GERMINATED"` for G0 in `germinate`)
  - `checks_passed`: `gate_result.checks_passed`
  - `checks_failed`: `gate_result.checks_failed`
  - `message`: `gate_result.message`

Example payload:

```python
self._emit_telemetry(
    TelemetryEventType.SEED_GATE_EVALUATED,
    data={
        "gate": gate_result.gate.name,
        "passed": gate_result.passed,
        "target_stage": target_stage.name,
        "checks_passed": list(gate_result.checks_passed),
        "checks_failed": list(gate_result.checks_failed),
        "message": gate_result.message,
    },
)
```

**Step 4: Re-run tests**

Run: `uv run pytest tests/kasmina/test_gate_telemetry.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/kasmina/slot.py tests/kasmina/test_gate_telemetry.py
git commit -m "feat(telemetry): emit SEED_GATE_EVALUATED from kasmina gates"
```

---

### Task 3: Include per-slot health fields in lifecycle telemetry payloads

**Intent:** When UIs show slot chips/inspectors, health fields must be present when known (and absent when unknown).

**Files:**
- Modify: `src/esper/kasmina/slot.py`
- Test: `tests/kasmina/test_slot_telemetry.py` (extend)

**Step 1: Write failing test**

```python
# tests/kasmina/test_slot_telemetry.py
from unittest.mock import Mock

from esper.kasmina.slot import SeedSlot, SeedState
from esper.leyline import TelemetryEventType


def test_lifecycle_events_include_health_fields_when_available():
    events = []
    slot = SeedSlot(slot_id="mid", channels=8, device="cpu", on_telemetry=events.append)
    slot.seed = Mock()
    slot.state = SeedState(seed_id="s1", blueprint_id="conv", slot_id="mid")
    # Pretend telemetry and metrics have been synced by the training loop
    slot.state.metrics.seed_gradient_norm_ratio = 0.42
    slot.state.metrics.isolation_violations = 3
    slot.state.telemetry.gradient_health = 0.9
    slot.state.telemetry.has_vanishing = True
    slot.state.telemetry.has_exploding = False

    slot._emit_telemetry(TelemetryEventType.SEED_STAGE_CHANGED, data={"from": "A", "to": "B"})
    payload = events[-1].data
    assert payload["seed_gradient_norm_ratio"] == 0.42
    assert payload["isolation_violations"] == 3
    assert payload["gradient_health"] == 0.9
    assert payload["has_vanishing"] is True
    assert payload["has_exploding"] is False
```

**Step 2: Run test (expect fail)**

Run: `uv run pytest tests/kasmina/test_slot_telemetry.py::test_lifecycle_events_include_health_fields_when_available -q`
Expected: FAIL (keys missing).

**Step 3: Implement**

In `SeedSlot._emit_telemetry`, extend the default payload population:
- From `self.state.metrics`: `seed_gradient_norm_ratio`, `isolation_violations`
- From `self.state.telemetry`: `gradient_health`, `has_vanishing`, `has_exploding`
- Use `payload.setdefault(...)` so explicit caller-provided values win.

**Step 4: Re-run test**

Run: `uv run pytest tests/kasmina/test_slot_telemetry.py::test_lifecycle_events_include_health_fields_when_available -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/kasmina/slot.py tests/kasmina/test_slot_telemetry.py
git commit -m "feat(telemetry): include slot health fields on lifecycle events"
```

---

### Task 4: Complete throughput telemetry with `fps`

**Files:**
- Modify: `src/esper/simic/vectorized.py`
- Test: `tests/simic/test_vectorized.py` (extend)

**Step 1: Write failing test**

```python
# tests/simic/test_vectorized.py
def test_throughput_metrics_include_fps():
    from esper.simic import vectorized
    from unittest.mock import Mock

    hub = Mock()
    vectorized._emit_throughput(
        hub=hub,
        env_id=0,
        batch_idx=1,
        episodes_completed=4,
        step_time_ms=20.0,
        dataloader_wait_ms=2.0,
    )
    data = hub.emit.call_args[0][0].data
    assert data["fps"] == 50.0
```

**Step 2: Run test (expect fail)**

Run: `uv run pytest tests/simic/test_vectorized.py::test_throughput_metrics_include_fps -q`
Expected: FAIL (missing `fps`).

**Step 3: Implement**

In `_emit_throughput`, compute:
- `fps = 1000.0 / step_time_ms` when `step_time_ms > 0`
- else `fps = None`

Add to emitted payload as `fps`.

**Step 4: Re-run test**

Run: `uv run pytest tests/simic/test_vectorized.py::test_throughput_metrics_include_fps -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/vectorized.py tests/simic/test_vectorized.py
git commit -m "feat(telemetry): add fps to throughput telemetry"
```

---

### Task 5 (Optional): DiagnosticTracker cadence + snapshot emission

**Goal:** When explicitly enabled, emit a rich diagnostics snapshot every K epochs for offline replay/UI.

**Files:**
- Modify: `src/esper/scripts/train.py` (CLI wiring)
- Modify: `src/esper/simic/vectorized.py`
- Modify: `src/esper/simic/training.py`
- Tests: `tests/scripts/test_train.py` (extend), plus a small unit test for a helper (new)

**Step 1: Add CLI flags (write failing test)**

```python
# tests/scripts/test_train.py
def test_diagnostics_flags_wired():
    import esper.scripts.train as train

    parser = train.build_parser()
    args = parser.parse_args(["heuristic", "--diagnostics-profile", "diagnostic", "--diagnostics-every", "5"])
    assert args.diagnostics_profile == "diagnostic"
    assert args.diagnostics_every == 5
```

**Step 2: Run test (expect fail)**

Run: `uv run pytest tests/scripts/test_train.py::test_diagnostics_flags_wired -q`
Expected: FAIL (unknown args).

**Step 3: Implement CLI wiring**

In `src/esper/scripts/train.py`, add:
- `--diagnostics-profile` (str, default `None`)
- `--diagnostics-every` (int, default `10`)

**Step 4: Emit diagnostics snapshot event (implementation sketch)**

In training loops, when diagnostics are enabled:
- Create `DiagnosticTracker(model, TelemetryConfig.from_profile(profile), device=device)`
- Every `diagnostics_every` epochs, call `tracker.end_epoch(...)`
- Emit `TelemetryEventType.ANALYTICS_SNAPSHOT` with:

```python
TelemetryEvent(
    event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
    severity="info",
    data={"kind": "diagnostic_snapshot", "snapshot": snapshot.to_dict()},
)
```

**Step 5: Tests**

Add a pure unit test for the helper that builds the event payload (no GPU required).

**Step 6: Commit**

```bash
git add src/esper/scripts/train.py src/esper/simic/vectorized.py src/esper/simic/training.py tests/scripts/test_train.py tests/...
git commit -m "feat(telemetry): optional DiagnosticTracker snapshot emission"
```

---

### Final verification (always)

Run:
- `uv run pytest -m "not slow" -q`
- `uv run ruff check src/ tests/`

Update docs (optional but recommended):
- `docs/specifications/telemetry-audit.md` (mark Phase 2 items as complete)
- `docs/plans/telemetry-remediation.md` (reduce the “UI truthfulness gaps” section to only remaining items)
