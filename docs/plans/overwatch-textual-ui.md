# Esper‑Overwatch Textual UI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ship the Textual “Esper‑Overwatch” UI (snapshot-first, outliers-first, truthful under missing telemetry) that replaces the Rich Karn TUI and supports offline replay.

**Architecture:** Snapshot-first pipeline: telemetry hub → aggregator (contracts + anomaly/bound scoring) → `TuiSnapshot` JSON-serializable schema → Textual renderer. Two cadences (fast flight board/pulse, slow vitals). All panels render from immutable snapshots; UI never computes analytics.

**Tech Stack:** Python 3.11, Textual, dataclasses, pytest, telemetry events from Simic/Kasmina/Tolaria/Nissa, JSONL for replay.

---

### Task 1: Create Overwatch package and snapshot schema

**Files:**
- Create: `src/esper/karn/overwatch/__init__.py`
- Create: `src/esper/karn/overwatch/snapshot.py`
- Tests: `tests/karn/overwatch/test_snapshot_schema.py`

**Step 1: Write failing test**
```python
# tests/karn/overwatch/test_snapshot_schema.py
from esper.karn.overwatch.snapshot import TuiSnapshot, EnvSummary, SlotChipState

def test_snapshot_serialises_to_json():
    snap = TuiSnapshot(
        schema_version=1,
        captured_at="2025-12-15T12:00:00Z",
        run={"id": "run1", "task": "cifar10", "algo": "ppo", "reward_mode": "shaped"},
        timing={"epoch_id": 3, "inner_epoch": 5, "batch": 10, "episodes_completed": 40, "episodes_total": 100, "elapsed_s": 12.3, "eta_s": 99.0},
        policy_pulse={"kl_divergence": {"value": 0.02, "status": "ok", "threshold": 0.1, "ts": "now"}},
        system={"devices": [], "cpu": {}, "ram": {}, "io": {}, "staleness": {}},
        flight_board=[
            EnvSummary(
                env_id=0,
                device_id=0,
                throughput={"fps": 100.0, "step_time_ms": 10.0, "dataloader_wait_ms": 1.0},
                metrics={"task_metric": 82.1, "task_metric_delta": 0.5, "reward": 0.3, "rent": -0.05},
                action={"op": "GERMINATE", "slot_id": "mid", "blueprint_id": "conv", "blend_id": "sigmoid", "masked": False, "success": True, "reason": None},
                slots={"mid": SlotChipState(slot_id="mid", stage="BLENDING", blueprint_id="conv", alpha=0.7, epochs_in_stage=3, epochs_total=7, gate={"last": "G2", "passed": True, "reason": ""})},
                anomaly={"score": 0.1, "reasons": ["ok"]},
                status="OK",
                staleness={"ts": "now"},
            )
        ],
        event_feed=[],
        ui_meta={"snapshot_age_ms": 5, "last_render_ms": 8, "avg_render_ms": 9},
    )
    import json
    json.dumps(snap.to_dict())  # should not raise
```

**Step 2: Run test (expect fail)**
- `pytest tests/karn/overwatch/test_snapshot_schema.py -q`

**Step 3: Implement**
- Add dataclasses for `SlotChipState`, `EnvSummary`, `TuiSnapshot` with `to_dict()` producing JSON-safe structures; set defaults for optional fields; include schema_version.

**Step 4: Re-run test**
- `pytest tests/karn/overwatch/test_snapshot_schema.py -q`

**Step 5: Commit**
```bash
git add src/esper/karn/overwatch/snapshot.py src/esper/karn/overwatch/__init__.py tests/karn/overwatch/test_snapshot_schema.py
git commit -m "feat(overwatch): add TuiSnapshot schema and JSON serialization"
```

---

### Task 2: Define aggregator contracts and telemetry intake

**Files:**
- Create: `src/esper/karn/overwatch/aggregator.py`
- Tests: `tests/karn/overwatch/test_aggregator_contracts.py`

**Step 1: Write failing test**
```python
def test_aggregator_accepts_events_and_builds_snapshot():
    from esper.karn.overwatch.aggregator import TelemetryAggregator
    from esper.leyline import TelemetryEvent, TelemetryEventType
    agg = TelemetryAggregator(schema_version=1)
    agg.handle_event(TelemetryEvent(event_type=TelemetryEventType.TRAINING_STARTED, data={"task": "cifar10", "algo": "ppo"}))
    agg.handle_event(TelemetryEvent(event_type=TelemetryEventType.PPO_UPDATE_COMPLETED, data={"kl_divergence": 0.02, "entropy": 1.2, "clip_fraction": 0.05, "explained_variance": 0.1, "lr": 0.0003, "grad_norm": 1.1, "update_time_ms": 12.0}))
    snap = agg.build_snapshot()
    assert snap.policy_pulse["kl_divergence"]["value"] == 0.02
    assert snap.schema_version == 1
```

**Step 2: Run test**
- `pytest tests/karn/overwatch/test_aggregator_contracts.py::test_aggregator_accepts_events_and_builds_snapshot -q`

**Step 3: Implement**
- Aggregator class: maintain minimal state (run context, timing, policy pulse) and build `TuiSnapshot`.
- Accept TelemetryEvents via `handle_event`, store last values; `build_snapshot` returns `TuiSnapshot`.
- Stub anomaly/bound/flight board with empty lists initially.

**Step 4: Re-run test**
- `pytest tests/karn/overwatch/test_aggregator_contracts.py -q`

**Step 5: Commit**
```bash
git add src/esper/karn/overwatch/aggregator.py tests/karn/overwatch/test_aggregator_contracts.py
git commit -m "feat(overwatch): add TelemetryAggregator scaffold and intake contracts"
```

---

### Task 3: Add anomaly scoring + bound detector stubs

**Files:**
- Modify: `src/esper/karn/overwatch/aggregator.py`
- Tests: `tests/karn/overwatch/test_anomaly_and_bounds.py`

**Step 1: Write failing test**
```python
def test_anomaly_reasons_and_bound_state():
    from esper.karn.overwatch.aggregator import score_anomaly, detect_bounds
    env = {"throughput": {"fps": 10}, "metrics": {"reward": -0.5}}
    score, reasons = score_anomaly(env)
    assert isinstance(score, float)
    assert reasons
    bounds = detect_bounds(dev={"util": 95, "mem_pct": 90}, io={"wait_ms": 15})
    assert "compute-bound" in bounds or "memory-bound" in bounds
```

**Step 2: Run test**
- `pytest tests/karn/overwatch/test_anomaly_and_bounds.py -q`

**Step 3: Implement**
- Add simple heuristic functions `score_anomaly(env_summary)` and `detect_bounds(dev, io)`; integrate into aggregator when building EnvSummary (placeholder calculations, no GPU sync).

**Step 4: Re-run test**
- `pytest tests/karn/overwatch/test_anomaly_and_bounds.py -q`

**Step 5: Commit**
```bash
git add src/esper/karn/overwatch/aggregator.py tests/karn/overwatch/test_anomaly_and_bounds.py
git commit -m "feat(overwatch): add anomaly scoring and bound detection stubs"
```

---

### Task 4: Snapshot writer/reader for replay

**Files:**
- Create: `src/esper/karn/overwatch/replay.py`
- Tests: `tests/karn/overwatch/test_replay.py`

**Step 1: Write failing test**
```python
def test_snapshot_round_trip(tmp_path):
    from esper.karn.overwatch.snapshot import TuiSnapshot
    from esper.karn.overwatch.replay import SnapshotWriter, SnapshotReader
    p = tmp_path / "snaps.jsonl"
    snap = TuiSnapshot(schema_version=1, captured_at="now", run={}, timing={}, policy_pulse={}, system={}, flight_board=[], event_feed=[], ui_meta={})
    writer = SnapshotWriter(p)
    writer.write(snap)
    reader = SnapshotReader(p)
    snaps = list(reader)
    assert len(snaps) == 1
    assert snaps[0].schema_version == 1
```

**Step 2: Run test**
- `pytest tests/karn/overwatch/test_replay.py::test_snapshot_round_trip -q`

**Step 3: Implement**
- SnapshotWriter appends JSONL; SnapshotReader yields `TuiSnapshot` from file.

**Step 4: Re-run test**
- `pytest tests/karn/overwatch/test_replay.py -q`

**Step 5: Commit**
```bash
git add src/esper/karn/overwatch/replay.py tests/karn/overwatch/test_replay.py
git commit -m "feat(overwatch): add snapshot writer/reader for replay"
```

---

### Task 5: Telemetry ingestion glue (hub listener)

**Files:**
- Create: `src/esper/karn/overwatch/listener.py`
- Tests: `tests/karn/overwatch/test_listener.py`

**Step 1: Write failing test**
```python
def test_listener_registers_and_forwards_events(mocker):
    from esper.karn.overwatch.listener import TelemetryListener
    agg = mocker.Mock()
    hub = mocker.Mock()
    listener = TelemetryListener(hub, agg)
    listener.start()
    assert hub.add_backend.called
    # simulate emit
    event = mocker.Mock()
    backend = hub.add_backend.call_args[0][0]
    backend.emit(event)
    agg.handle_event.assert_called_with(event)
```

**Step 2: Run test**
- `pytest tests/karn/overwatch/test_listener.py::test_listener_registers_and_forwards_events -q`

**Step 3: Implement**
- Lightweight backend that forwards TelemetryEvents to aggregator; register via `hub.add_backend`.

**Step 4: Re-run test**
- `pytest tests/karn/overwatch/test_listener.py -q`

**Step 5: Commit**
```bash
git add src/esper/karn/overwatch/listener.py tests/karn/overwatch/test_listener.py
git commit -m "feat(overwatch): add telemetry listener backend"
```

---

### Task 6: Textual app scaffold

**Files:**
- Create: `src/esper/karn/overwatch/app.py`
- Create: `src/esper/karn/overwatch/widgets/{header.py,flight_board.py,context_pane.py,feed.py}`
- Create: `src/esper/karn/overwatch/styles.css`
- Tests: `tests/karn/overwatch/test_app_scaffold.py`

**Step 1: Write failing test**
```python
def test_textual_app_launches_safely():
    from esper.karn.overwatch.app import OverwatchApp
    app = OverwatchApp(snapshot_provider=lambda: None)
    # Textual apps are hard to fully test; ensure instantiation works
    assert app is not None
```

**Step 2: Run test**
- `pytest tests/karn/overwatch/test_app_scaffold.py::test_textual_app_launches_safely -q`

**Step 3: Implement**
- Basic Textual App with placeholder widgets/layout; wire CSS tokens.
- Snapshot provider callable injected (for live vs replay).

**Step 4: Re-run test**
- `pytest tests/karn/overwatch/test_app_scaffold.py -q`

**Step 5: Commit**
```bash
git add src/esper/karn/overwatch/app.py src/esper/karn/overwatch/widgets/*.py src/esper/karn/overwatch/styles.css tests/karn/overwatch/test_app_scaffold.py
git commit -m "feat(overwatch): add Textual app scaffold and widgets"
```

---

### Task 7: Render TuiSnapshot (fast path: header + flight board)

**Files:**
- Modify: `src/esper/karn/overwatch/widgets/header.py`
- Modify: `src/esper/karn/overwatch/widgets/flight_board.py`
- Tests: `tests/karn/overwatch/test_render_flight_board.py`

**Step 1: Write failing test**
```python
def test_flight_board_renders_envs(snapshot):
    from esper.karn.overwatch.widgets.flight_board import render_flight_board
    snap = snapshot
    text = render_flight_board(snap)
    assert "env 0" in text.lower() or snap.flight_board[0].slots
```

**Step 2: Run test**
- `pytest tests/karn/overwatch/test_render_flight_board.py -q`

**Step 3: Implement**
- Render functions that take `TuiSnapshot` and return textual markup; handle dynamic slots; show unknown markers when missing.
- Header displays policy pulse + staleness.

**Step 4: Re-run test**
- `pytest tests/karn/overwatch/test_render_flight_board.py -q`

**Step 5: Commit**
```bash
git add src/esper/karn/overwatch/widgets/header.py src/esper/karn/overwatch/widgets/flight_board.py tests/karn/overwatch/test_render_flight_board.py
git commit -m "feat(overwatch): render header and flight board from snapshots"
```

---

### Task 8: Context Pane and Telemetry Feed rendering

**Files:**
- Modify: `src/esper/karn/overwatch/widgets/context_pane.py`
- Modify: `src/esper/karn/overwatch/widgets/feed.py`
- Tests: `tests/karn/overwatch/test_render_context_and_feed.py`

**Step 1: Write failing test**
```python
def test_context_pane_shows_vitals_and_selection(snapshot):
    from esper.karn.overwatch.widgets.context_pane import render_context
    text = render_context(snapshot, selected_env_id=0)
    assert "system" in text.lower()
    assert "env" in text.lower()
```

**Step 2: Run test**
- `pytest tests/karn/overwatch/test_render_context_and_feed.py -q`

**Step 3: Implement**
- Render system vitals, bound detector outputs, alarm rail.
- Env detail overlay: slot chips, recent actions, lifecycle events.
- Feed render: structured events with filters.

**Step 4: Re-run test**
- `pytest tests/karn/overwatch/test_render_context_and_feed.py -q`

**Step 5: Commit**
```bash
git add src/esper/karn/overwatch/widgets/context_pane.py src/esper/karn/overwatch/widgets/feed.py tests/karn/overwatch/test_render_context_and_feed.py
git commit -m "feat(overwatch): render context pane and telemetry feed"
```

---

### Task 9: Interaction model (navigation, filters, overlays)

**Files:**
- Modify: `src/esper/karn/overwatch/app.py`
- Tests: `tests/karn/overwatch/test_interactions.py`

**Step 1: Write failing test**
```python
def test_navigation_state_changes(mocker):
    from esper.karn.overwatch.app import OverwatchApp
    app = OverwatchApp(snapshot_provider=lambda: mocker.Mock(flight_board=[]))
    app.selected_env = 0
    app.move_selection(delta=1)
    assert app.selected_env == 1
```

**Step 2: Run test**
- `pytest tests/karn/overwatch/test_interactions.py -q`

**Step 3: Implement**
- Keyboard navigation (j/k or arrows), selection state, filter toggles, help overlay.
- Stable sort/hysteresis hooks on flight board ordering (logic in aggregator or render helper).

**Step 4: Re-run test**
- `pytest tests/karn/overwatch/test_interactions.py -q`

**Step 5: Commit**
```bash
git add src/esper/karn/overwatch/app.py tests/karn/overwatch/test_interactions.py
git commit -m "feat(overwatch): add navigation and filter interactions"
```

---

### Task 10: Wiring CLI and modes (live vs replay)

**Files:**
- Modify: `scripts/train.py` (replace Rich TUI flags with Overwatch flags)
- Create: `scripts/overwatch.py` (entry for replay/live viewer)
- Tests: `tests/scripts/test_overwatch_entry.py`

**Step 1: Write failing test**
```python
def test_overwatch_cli_parses_modes():
    import esper.scripts.overwatch as ow
    parser = ow.build_parser()
    args = parser.parse_args(["--replay", "snap.jsonl"])
    assert args.replay == "snap.jsonl"
```

**Step 2: Run test**
- `pytest tests/scripts/test_overwatch_entry.py::test_overwatch_cli_parses_modes -q`

**Step 3: Implement**
- New CLI: `esper.scripts.overwatch` supporting `--replay <file>` or live mode attaching to hub listener; options for top-N, compact mode.
- Update train CLI help to reference Overwatch (remove Rich TUI references).

**Step 4: Re-run test**
- `pytest tests/scripts/test_overwatch_entry.py -q`

**Step 5: Commit**
```bash
git add scripts/overwatch.py scripts/train.py tests/scripts/test_overwatch_entry.py
git commit -m "feat(overwatch): add CLI for live/replay Textual UI and update train flags"
```

---

### Task 11: Performance hardening and instrumentation

**Files:**
- Modify: `src/esper/karn/overwatch/app.py`
- Modify: `src/esper/karn/overwatch/aggregator.py`
- Tests: `tests/karn/overwatch/test_perf_guardrails.py`

**Step 1: Write failing test**
```python
def test_snapshot_throttling(mocker):
    from esper.karn.overwatch.aggregator import TelemetryAggregator
    agg = TelemetryAggregator(schema_version=1, fast_hz=5, slow_hz=1)
    agg._last_fast_emit = 0.0
    assert agg.should_emit_fast(now=0.05) is False  # throttled
```

**Step 2: Run test**
- `pytest tests/karn/overwatch/test_perf_guardrails.py -q`

**Step 3: Implement**
- Snapshot cadence throttling; decimate heavy metrics.
- UI instrumentation (render ms, snapshot age); no GPU sync points.

**Step 4: Re-run test**
- `pytest tests/karn/overwatch/test_perf_guardrails.py -q`

**Step 5: Commit**
```bash
git add src/esper/karn/overwatch/app.py src/esper/karn/overwatch/aggregator.py tests/karn/overwatch/test_perf_guardrails.py
git commit -m "feat(overwatch): add throttling and perf instrumentation"
```

---

### Task 12: Cutover (remove Rich TUI, docs update)

**Files:**
- Delete: `src/esper/karn/tui.py` (and related Rich-specific code/flags)
- Modify: `README.md`, `docs/plans/esper_overwatch.md` (add Textual usage), `docs/specifications/telemetry-audit.md` (note UI consumption), CLI help.
- Tests: adjust any references to Rich TUI (search in `tests/karn`).

**Step 1: Write failing test**
- Update/adjust existing tests that import old TUI to fail, then fix after removal (e.g., rename skipped tests).

**Step 2: Remove old TUI and update docs**
- Remove Rich imports/flags, add Overwatch CLI usage examples.

**Step 3: Run test suite**
- `pytest -m "not slow" -q`

**Step 4: Commit**
```bash
git add README.md docs/plans/esper_overwatch.md docs/specifications/telemetry-audit.md scripts/train.py
git rm src/esper/karn/tui.py
git commit -m "chore(overwatch): remove Rich TUI and document Textual Overwatch"
```

---

## Final Verification
- Run unit tests: `pytest -m "not slow" -q`
- Manual smoke: start training with live Overwatch: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --task cifar10 --telemetry-level normal --overwatch`
- Replay mode smoke: `PYTHONPATH=src uv run python -m esper.scripts.overwatch --replay telemetry_snaps.jsonl`

## Execution Handoff
Plan complete and saved to `docs/plans/overwatch-textual-ui.md`. Two execution options:
1. **Subagent-Driven (this session)** — use superpowers:subagent-driven-development to execute tasks sequentially with code review between tasks.
2. **Parallel Session** — new session using superpowers:executing-plans to run the plan in batches with checkpoints.
