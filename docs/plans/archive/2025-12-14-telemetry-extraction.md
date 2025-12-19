# Telemetry Extraction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace ad-hoc print statements with structured telemetry events, maximizing observability and enabling post-hoc analysis.

**Architecture:** Add new `TelemetryEventType` entries to `leyline/telemetry.py`, wire emitters at print sites, add console formatters to `nissa/output.py`. Existing hub infrastructure handles routing.

**Tech Stack:** Python dataclasses, Nissa telemetry hub, existing `TelemetryEvent` contract.

---

## Overview

**Prints to convert:** 23 print statements across 5 files
**New event types:** 6
**Files modified:** 5
**Tests required:** Unit tests for new event types and console formatting

---

## Task 1: Add Governor Event Types

**Files:**
- Modify: `src/esper/leyline/telemetry.py:34-72` (TelemetryEventType enum)

**Step 1: Add GOVERNOR event types to the enum**

In `src/esper/leyline/telemetry.py`, add after line 71 (after `NUMERICAL_INSTABILITY_DETECTED`):

```python
    # === Governor Events (Tolaria) ===
    GOVERNOR_PANIC = auto()           # Vital signs check failed
    GOVERNOR_ROLLBACK = auto()        # Emergency rollback executed
    GOVERNOR_SNAPSHOT = auto()        # LKG checkpoint saved
```

**Step 2: Run test to verify enum compiles**

Run: `python -c "from esper.leyline import TelemetryEventType; print(TelemetryEventType.GOVERNOR_ROLLBACK)"`
Expected: `TelemetryEventType.GOVERNOR_ROLLBACK`

**Step 3: Commit**

```bash
git add src/esper/leyline/telemetry.py
git commit -m "feat(telemetry): add Governor event types"
```

---

## Task 2: Add Batch Progress Event Types

**Files:**
- Modify: `src/esper/leyline/telemetry.py:34-72` (TelemetryEventType enum)

**Step 1: Add BATCH event types to the enum**

In `src/esper/leyline/telemetry.py`, add after the Governor events:

```python
    # === Training Progress Events ===
    BATCH_EPOCH_COMPLETED = auto()          # PPO batch finished
    EPISODE_COMPLETED = auto()        # Single episode finished (already exists as EPOCH_COMPLETED for epochs)
    TRAINING_STARTED = auto()         # Training run initialized
    CHECKPOINT_SAVED = auto()         # Model checkpoint saved
    CHECKPOINT_LOADED = auto()        # Model checkpoint restored
```

**Step 2: Run test to verify enum compiles**

Run: `python -c "from esper.leyline import TelemetryEventType; print(TelemetryEventType.BATCH_EPOCH_COMPLETED)"`
Expected: `TelemetryEventType.BATCH_EPOCH_COMPLETED`

**Step 3: Commit**

```bash
git add src/esper/leyline/telemetry.py
git commit -m "feat(telemetry): add batch progress event types"
```

---

## Task 3: Add Counterfactual Event Type

**Files:**
- Modify: `src/esper/leyline/telemetry.py:34-72` (TelemetryEventType enum)

**Step 1: Add COUNTERFACTUAL event type to the enum**

In `src/esper/leyline/telemetry.py`, add after the Training Progress events:

```python
    # === Counterfactual Attribution Events ===
    COUNTERFACTUAL_COMPUTED = auto()  # Per-slot counterfactual contribution measured
```

**Step 2: Run test to verify enum compiles**

Run: `python -c "from esper.leyline import TelemetryEventType; print(TelemetryEventType.COUNTERFACTUAL_COMPUTED)"`
Expected: `TelemetryEventType.COUNTERFACTUAL_COMPUTED`

**Step 3: Commit**

```bash
git add src/esper/leyline/telemetry.py
git commit -m "feat(telemetry): add counterfactual event type"
```

---

## Task 4: Add Console Formatters for Governor Events

**Files:**
- Modify: `src/esper/nissa/output.py:75-165` (ConsoleOutput._emit_summary)

**Step 1: Add Governor event formatting**

In `src/esper/nissa/output.py`, in the `_emit_summary` method, add a new elif block after the `REWARD_COMPUTED` block (around line 163):

```python
        elif event_type == "GOVERNOR_ROLLBACK":
            data = event.data or {}
            reason = data.get("reason", "unknown")
            loss = data.get("loss_at_panic", "?")
            threshold = data.get("loss_threshold", "?")
            panics = data.get("consecutive_panics", "?")
            if isinstance(loss, float):
                loss = f"{loss:.4f}"
            if isinstance(threshold, float):
                threshold = f"{threshold:.4f}"
            print(f"[{timestamp}] GOVERNOR | 🚨 ROLLBACK: {reason} (loss={loss}, threshold={threshold}, panics={panics})")
        elif event_type == "GOVERNOR_PANIC":
            data = event.data or {}
            loss = data.get("current_loss", "?")
            panics = data.get("consecutive_panics", 0)
            if isinstance(loss, float):
                loss = f"{loss:.4f}"
            print(f"[{timestamp}] GOVERNOR | ⚠️  PANIC #{panics}: loss={loss}")
```

**Step 2: Run test to verify formatting**

Run: `python -c "
from esper.nissa.output import ConsoleOutput
from esper.leyline import TelemetryEvent, TelemetryEventType
c = ConsoleOutput()
c.emit(TelemetryEvent(
    event_type=TelemetryEventType.GOVERNOR_ROLLBACK,
    data={'reason': 'Structural Collapse', 'loss_at_panic': 15.3, 'loss_threshold': 5.2, 'consecutive_panics': 2}
))
"`
Expected: `[HH:MM:SS] GOVERNOR | 🚨 ROLLBACK: Structural Collapse (loss=15.3000, threshold=5.2000, panics=2)`

**Step 3: Commit**

```bash
git add src/esper/nissa/output.py
git commit -m "feat(telemetry): add console formatters for Governor events"
```

---

## Task 5: Add Console Formatters for Batch Progress Events

**Files:**
- Modify: `src/esper/nissa/output.py:75-165` (ConsoleOutput._emit_summary)

**Step 1: Add Batch event formatting**

In `src/esper/nissa/output.py`, add after the Governor formatters:

```python
        elif event_type == "BATCH_EPOCH_COMPLETED":
            data = event.data or {}
            batch_idx = data.get("batch_idx", "?")
            episodes = data.get("episodes_completed", "?")
            total = data.get("total_episodes", "?")
            avg_acc = data.get("avg_accuracy", 0.0)
            rolling_acc = data.get("rolling_accuracy", 0.0)
            avg_reward = data.get("avg_reward", 0.0)
            print(f"[{timestamp}] BATCH {batch_idx} | Episodes {episodes}/{total}: acc={avg_acc:.1f}% (rolling={rolling_acc:.1f}%), reward={avg_reward:.1f}")
        elif event_type == "COUNTERFACTUAL_COMPUTED":
            data = event.data or {}
            env_idx = data.get("env_idx", "?")
            slot_id = data.get("slot_id", "?")
            real_acc = data.get("real_accuracy", 0.0)
            baseline_acc = data.get("baseline_accuracy", 0.0)
            contribution = data.get("contribution", 0.0)
            print(f"[{timestamp}] env{env_idx} | Counterfactual {slot_id}: {real_acc:.1f}% real, {baseline_acc:.1f}% baseline, Δ={contribution:+.1f}%")
        elif event_type == "CHECKPOINT_SAVED":
            data = event.data or {}
            path = data.get("path", "?")
            avg_acc = data.get("avg_accuracy", 0.0)
            print(f"[{timestamp}] CHECKPOINT | Saved to {path} (acc={avg_acc:.1f}%)")
        elif event_type == "CHECKPOINT_LOADED":
            data = event.data or {}
            path = data.get("path", "?")
            episode = data.get("start_episode", 0)
            print(f"[{timestamp}] CHECKPOINT | Loaded from {path} (resuming at episode {episode})")
```

**Step 2: Run test to verify formatting**

Run: `python -c "
from esper.nissa.output import ConsoleOutput
from esper.leyline import TelemetryEvent, TelemetryEventType
c = ConsoleOutput()
c.emit(TelemetryEvent(
    event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
    data={'batch_idx': 3, 'episodes_completed': 24, 'total_episodes': 100, 'avg_accuracy': 67.2, 'rolling_accuracy': 65.1, 'avg_reward': 2.3}
))
"`
Expected: `[HH:MM:SS] BATCH 3 | Episodes 24/100: acc=67.2% (rolling=65.1%), reward=2.3`

**Step 3: Commit**

```bash
git add src/esper/nissa/output.py
git commit -m "feat(telemetry): add console formatters for batch progress events"
```

---

## Task 6: Wire Governor Rollback to Telemetry

**Files:**
- Modify: `src/esper/tolaria/governor.py:162-226` (execute_rollback method)

**Step 1: Add import at top of file**

In `src/esper/tolaria/governor.py`, add after line 14 (`import torch.nn as nn`):

```python
from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.nissa import get_hub
```

**Step 2: Replace print with telemetry emission**

In `src/esper/tolaria/governor.py`, replace line 174:

```python
        print(f"[GOVERNOR] CRITICAL INSTABILITY DETECTED. INITIATING ROLLBACK.")
```

With:

```python
        # Emit telemetry event (replaces print)
        hub = get_hub()
        hub.emit(TelemetryEvent(
            event_type=TelemetryEventType.GOVERNOR_ROLLBACK,
            severity="critical",
            message="Critical instability detected - initiating rollback",
            data={
                "reason": "Structural Collapse",
                "loss_at_panic": self._panic_loss,
                "consecutive_panics": self.consecutive_panics,
            },
        ))
```

**Step 3: Run test to verify import works**

Run: `python -c "from esper.tolaria.governor import TolariaGovernor; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add src/esper/tolaria/governor.py
git commit -m "feat(telemetry): wire Governor rollback to telemetry system"
```

---

## Task 7: Wire Tamiyo Stabilization to Telemetry (Remove Redundant Print)

**Files:**
- Modify: `src/esper/tamiyo/tracker.py:116-136`

**Step 1: Remove redundant print statements**

In `src/esper/tamiyo/tracker.py`, the telemetry emission already exists (lines 124-136). Remove the print statements at lines 118-121:

Replace:

```python
                        env_str = f"ENV {self.env_id}" if self.env_id is not None else "Tamiyo"
                        if self.stabilization_epochs == 0:
                            # Stabilization disabled - just note when it happened
                            print(f"[{env_str}] Host stabilized at epoch {epoch} - germination now allowed")
                        else:
                            print(f"[{env_str}] Host stabilized at epoch {epoch} "
                                  f"({self._stable_count}/{self.stabilization_epochs} stable) - germination now allowed")

                        # Emit TAMIYO_INITIATED telemetry
```

With:

```python
                        # Emit TAMIYO_INITIATED telemetry (console output via Nissa backend)
```

**Step 2: Add console formatter for TAMIYO_INITIATED**

In `src/esper/nissa/output.py`, add after the checkpoint formatters:

```python
        elif event_type == "TAMIYO_INITIATED":
            data = event.data or {}
            env_id = data.get("env_id")
            epoch = data.get("epoch", "?")
            stable_count = data.get("stable_count", 0)
            stabilization_epochs = data.get("stabilization_epochs", 0)
            env_str = f"env{env_id}" if env_id is not None else "Tamiyo"
            if stabilization_epochs == 0:
                print(f"[{timestamp}] {env_str} | Host stabilized at epoch {epoch} - germination now allowed")
            else:
                print(f"[{timestamp}] {env_str} | Host stabilized at epoch {epoch} ({stable_count}/{stabilization_epochs} stable) - germination now allowed")
```

**Step 3: Run test to verify**

Run: `python -c "from esper.tamiyo.tracker import SignalTracker; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add src/esper/tamiyo/tracker.py src/esper/nissa/output.py
git commit -m "refactor(telemetry): route Tamiyo stabilization through Nissa console backend"
```

---

## Task 8: Wire Counterfactual Computation to Telemetry

**Files:**
- Modify: `src/esper/simic/vectorized.py:1030-1035`

**Step 1: Replace print with telemetry emission**

In `src/esper/simic/vectorized.py`, find the counterfactual print (around line 1033):

```python
                            cf_contribution = val_acc - baseline_acc
                            print(f"  [ENV {env_idx}] Slot {slot_id} counterfactual: "
                                  f"{val_acc:.1f}% real, {baseline_acc:.1f}% baseline, "
```

Replace with:

```python
                            cf_contribution = val_acc - baseline_acc
                            if hub:
                                hub.emit(TelemetryEvent(
                                    event_type=TelemetryEventType.COUNTERFACTUAL_COMPUTED,
                                    slot_id=slot_id,
                                    data={
                                        "env_idx": env_idx,
                                        "slot_id": slot_id,
                                        "real_accuracy": val_acc,
                                        "baseline_accuracy": baseline_acc,
                                        "contribution": cf_contribution,
                                    },
                                ))
```

**Step 2: Verify TelemetryEventType import exists**

Check that `TelemetryEventType` is already imported at top of file. If not, add:

```python
from esper.leyline import TelemetryEventType
```

**Step 3: Run syntax check**

Run: `python -m py_compile src/esper/simic/vectorized.py`
Expected: No output (success)

**Step 4: Commit**

```bash
git add src/esper/simic/vectorized.py
git commit -m "feat(telemetry): wire counterfactual computation to telemetry"
```

---

## Task 9: Wire Governor Rollback in Training Loop to Telemetry

**Files:**
- Modify: `src/esper/simic/vectorized.py:1186-1189`

**Step 1: Replace Governor rollback print in training loop**

In `src/esper/simic/vectorized.py`, find the Governor rollback print (around line 1188):

```python
                    batch_rollback_occurred = True  # Mark batch as having stale transitions
                    print(f"  [ENV {env_idx}] Governor rollback: {report.reason} "
                          f"(threshold={report.loss_threshold:.4f}, panics={report.consecutive_panics})")
```

Replace with:

```python
                    batch_rollback_occurred = True  # Mark batch as having stale transitions
                    if hub:
                        hub.emit(TelemetryEvent(
                            event_type=TelemetryEventType.GOVERNOR_ROLLBACK,
                            severity="warning",
                            data={
                                "env_idx": env_idx,
                                "reason": report.reason,
                                "loss_at_panic": report.loss_at_panic,
                                "loss_threshold": report.loss_threshold,
                                "consecutive_panics": report.consecutive_panics,
                            },
                        ))
```

**Step 2: Run syntax check**

Run: `python -m py_compile src/esper/simic/vectorized.py`
Expected: No output (success)

**Step 3: Commit**

```bash
git add src/esper/simic/vectorized.py
git commit -m "feat(telemetry): wire training loop Governor rollback to telemetry"
```

---

## Task 10: Wire Punishment Reward to Telemetry

**Files:**
- Modify: `src/esper/simic/vectorized.py:1257-1259`

**Step 1: Replace punishment print with telemetry**

In `src/esper/simic/vectorized.py`, find the punishment print (around line 1259):

```python
                    reward += punishment
                    print(f"  [ENV {env_idx}] Punishment reward: {punishment:.1f} (final reward: {reward:.1f})")
```

Replace with:

```python
                    reward += punishment
                    if hub:
                        hub.emit(TelemetryEvent(
                            event_type=TelemetryEventType.REWARD_COMPUTED,
                            severity="warning",
                            data={
                                "env_id": env_idx,
                                "action_name": "PUNISHMENT",
                                "total_reward": reward,
                                "punishment": punishment,
                                "reason": "governor_rollback",
                            },
                        ))
```

**Step 2: Run syntax check**

Run: `python -m py_compile src/esper/simic/vectorized.py`
Expected: No output (success)

**Step 3: Commit**

```bash
git add src/esper/simic/vectorized.py
git commit -m "feat(telemetry): wire punishment reward to telemetry"
```

---

## Task 11: Wire PPO Buffer Clear to Telemetry

**Files:**
- Modify: `src/esper/simic/vectorized.py:1404-1405`

**Step 1: Replace buffer clear print with telemetry**

In `src/esper/simic/vectorized.py`, find the buffer clear print (around line 1405):

```python
            agent.buffer.reset()
            print("[PPO] Buffer cleared due to Governor rollback - skipping update")
```

Replace with:

```python
            agent.buffer.reset()
            if hub:
                hub.emit(TelemetryEvent(
                    event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
                    severity="warning",
                    message="Buffer cleared due to Governor rollback - skipping update",
                    data={"reason": "governor_rollback", "skipped": True},
                ))
```

**Step 2: Run syntax check**

Run: `python -m py_compile src/esper/simic/vectorized.py`
Expected: No output (success)

**Step 3: Commit**

```bash
git add src/esper/simic/vectorized.py
git commit -m "feat(telemetry): wire PPO buffer clear to telemetry"
```

---

## Task 12: Wire Anomaly Detection Prints to Telemetry

**Files:**
- Modify: `src/esper/simic/vectorized.py:1431-1449`

**Step 1: Remove redundant anomaly prints**

The anomaly detection already emits telemetry events (lines 1451-1474). The print statements are redundant. Remove lines 1432-1449:

```python
            if anomaly_report.has_anomaly:
                print(f"\n⚠️  TRAINING ANOMALY DETECTED at episode {episodes_completed}:")
                for anomaly_type in anomaly_report.anomaly_types:
                    print(f"   - {anomaly_type}: {anomaly_report.details.get(anomaly_type, '')}")

                # Escalate to debug telemetry - collect diagnostic data
                print("   📊 Collecting debug diagnostics...")
                gradient_stats = collect_per_layer_gradients(agent.network)
                stability_report = check_numerical_stability(agent.network)

                # Log gradient health summary
                vanishing = sum(1 for gs in gradient_stats if gs.zero_fraction > 0.5)
                exploding = sum(1 for gs in gradient_stats if gs.large_fraction > 0.1)
                if vanishing > 0:
                    print(f"   ⚠️  {vanishing} layers with vanishing gradients (>50% zeros)")
                if exploding > 0:
                    print(f"   ⚠️  {exploding} layers with exploding gradients (>10% large values)")
                if stability_report.has_issues():
                    print(f"   🔥 NUMERICAL INSTABILITY detected in weights/gradients")
```

Replace with:

```python
            if anomaly_report.has_anomaly:
                # Collect diagnostic data for telemetry
                gradient_stats = collect_per_layer_gradients(agent.network)
                stability_report = check_numerical_stability(agent.network)
                vanishing = sum(1 for gs in gradient_stats if gs.zero_fraction > 0.5)
                exploding = sum(1 for gs in gradient_stats if gs.large_fraction > 0.1)
```

**Step 2: Add console formatters for anomaly events**

In `src/esper/nissa/output.py`, add after the TAMIYO_INITIATED formatter:

```python
        elif event_type in ("RATIO_EXPLOSION_DETECTED", "RATIO_COLLAPSE_DETECTED",
                           "VALUE_COLLAPSE_DETECTED", "NUMERICAL_INSTABILITY_DETECTED",
                           "GRADIENT_ANOMALY"):
            data = event.data or {}
            episode = data.get("episode", "?")
            detail = data.get("detail", "")
            # Clean event type name for display
            anomaly_name = event_type.replace("_DETECTED", "").replace("_", " ").title()
            print(f"[{timestamp}] ⚠️  ANOMALY | {anomaly_name} at episode {episode}: {detail}")
```

**Step 3: Run syntax check**

Run: `python -m py_compile src/esper/simic/vectorized.py`
Expected: No output (success)

**Step 4: Commit**

```bash
git add src/esper/simic/vectorized.py src/esper/nissa/output.py
git commit -m "refactor(telemetry): route anomaly detection through Nissa console backend"
```

---

## Task 13: Wire Batch Progress to Telemetry

**Files:**
- Modify: `src/esper/simic/vectorized.py:1488-1492`

**Step 1: Replace batch progress prints with telemetry**

In `src/esper/simic/vectorized.py`, find the batch progress prints (around line 1489):

```python
        episodes_completed += envs_this_batch
        print(f"Batch {batch_idx + 1}: Episodes {episodes_completed}/{n_episodes}")
        print(f"  Env accuracies: {[f'{a:.1f}%' for a in env_final_accs]}")
        print(f"  Avg acc: {avg_acc:.1f}% (rolling: {rolling_avg_acc:.1f}%)")
        print(f"  Avg reward: {avg_reward:.1f}")
```

Replace with:

```python
        episodes_completed += envs_this_batch
        if hub:
            hub.emit(TelemetryEvent(
                event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
                data={
                    "batch_idx": batch_idx + 1,
                    "episodes_completed": episodes_completed,
                    "total_episodes": n_episodes,
                    "env_accuracies": env_final_accs,
                    "avg_accuracy": avg_acc,
                    "rolling_accuracy": rolling_avg_acc,
                    "avg_reward": avg_reward,
                },
            ))
```

**Step 2: Update console formatter for BATCH_EPOCH_COMPLETED to show env accuracies**

In `src/esper/nissa/output.py`, update the BATCH_EPOCH_COMPLETED formatter to also show env accuracies:

```python
        elif event_type == "BATCH_EPOCH_COMPLETED":
            data = event.data or {}
            batch_idx = data.get("batch_idx", "?")
            episodes = data.get("episodes_completed", "?")
            total = data.get("total_episodes", "?")
            avg_acc = data.get("avg_accuracy", 0.0)
            rolling_acc = data.get("rolling_accuracy", 0.0)
            avg_reward = data.get("avg_reward", 0.0)
            env_accs = data.get("env_accuracies", [])
            env_acc_str = ", ".join(f"{a:.1f}%" for a in env_accs) if env_accs else "?"
            print(f"[{timestamp}] BATCH {batch_idx} | Episodes {episodes}/{total}")
            print(f"[{timestamp}]   Env accs: [{env_acc_str}]")
            print(f"[{timestamp}]   Avg: {avg_acc:.1f}% (rolling: {rolling_acc:.1f}%), reward: {avg_reward:.1f}")
```

**Step 3: Run syntax check**

Run: `python -m py_compile src/esper/simic/vectorized.py`
Expected: No output (success)

**Step 4: Commit**

```bash
git add src/esper/simic/vectorized.py src/esper/nissa/output.py
git commit -m "feat(telemetry): wire batch progress to telemetry"
```

---

## Task 14: Wire Action Distribution to Telemetry

**Files:**
- Modify: `src/esper/simic/vectorized.py:1500-1502`

**Step 1: Replace action distribution prints with telemetry**

In `src/esper/simic/vectorized.py`, find the action distribution prints (around line 1501):

```python
                successful_actions[a] += c
        print(f"  Actions: {total_actions}")
        print(f"  Successful: {successful_actions}")
```

Replace with:

```python
                successful_actions[a] += c
        # Action distribution is included in BATCH_EPOCH_COMPLETED event data
        if hub:
            # Update the previous BATCH_EPOCH_COMPLETED event with action data
            # Note: Actions are already tracked in analytics, this is redundant console output
            pass  # Removed - action counts visible in analytics.summary_table()
```

Actually, since action counts are already tracked in analytics and printed via `analytics.summary_table()`, we can simply remove these prints. Replace with nothing (delete the two print lines).

**Step 2: Run syntax check**

Run: `python -m py_compile src/esper/simic/vectorized.py`
Expected: No output (success)

**Step 3: Commit**

```bash
git add src/esper/simic/vectorized.py
git commit -m "refactor(telemetry): remove redundant action distribution prints (covered by analytics)"
```

---

## Task 15: Wire Checkpoint Save/Load to Telemetry

**Files:**
- Modify: `src/esper/simic/vectorized.py:1612-1628`

**Step 1: Replace checkpoint prints with telemetry**

In `src/esper/simic/vectorized.py`, find the checkpoint prints.

First, the "loaded best weights" print (around line 1613):

```python
        agent.network.load_state_dict(best_state)
        print(f"\nLoaded best weights (avg_acc={best_avg_acc:.1f}%)")
```

Replace with:

```python
        agent.network.load_state_dict(best_state)
        if hub:
            hub.emit(TelemetryEvent(
                event_type=TelemetryEventType.CHECKPOINT_LOADED,
                message="Loaded best weights",
                data={"source": "best_state", "avg_accuracy": best_avg_acc},
            ))
```

Second, the "model saved" print (around line 1628):

```python
        })
        print(f"Model saved to {save_path}")
```

Replace with:

```python
        })
        if hub:
            hub.emit(TelemetryEvent(
                event_type=TelemetryEventType.CHECKPOINT_SAVED,
                message=f"Model saved to {save_path}",
                data={"path": str(save_path), "avg_accuracy": best_avg_acc},
            ))
```

**Step 2: Run syntax check**

Run: `python -m py_compile src/esper/simic/vectorized.py`
Expected: No output (success)

**Step 3: Commit**

```bash
git add src/esper/simic/vectorized.py
git commit -m "feat(telemetry): wire checkpoint save/load to telemetry"
```

---

## Task 16: Wire Checkpoint Resume to Telemetry

**Files:**
- Modify: `src/esper/simic/vectorized.py:408-431`

**Step 1: Replace checkpoint resume prints with telemetry**

In `src/esper/simic/vectorized.py`, find the resume prints (around line 409):

```python
    if resume_path:
        print(f"Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
```

And later (around line 426):

```python
            print(f"  Restored observation normalizer state (momentum={obs_normalizer.momentum})")
```

And (around line 431):

```python
            print(f"  Resuming from episode {start_episode}")
```

Replace the block with:

```python
    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        # ... existing checkpoint loading code ...
        if hub:
            hub.emit(TelemetryEvent(
                event_type=TelemetryEventType.CHECKPOINT_LOADED,
                message=f"Resumed from checkpoint: {resume_path}",
                data={
                    "path": str(resume_path),
                    "start_episode": start_episode,
                    "obs_normalizer_momentum": obs_normalizer.momentum if 'obs_normalizer_state' in metadata else None,
                },
            ))
```

(Consolidate multiple prints into single telemetry event)

**Step 2: Run syntax check**

Run: `python -m py_compile src/esper/simic/vectorized.py`
Expected: No output (success)

**Step 3: Commit**

```bash
git add src/esper/simic/vectorized.py
git commit -m "feat(telemetry): wire checkpoint resume to telemetry"
```

---

## Task 17: Wire PPO Loss Metrics to Telemetry

**Files:**
- Modify: `src/esper/simic/vectorized.py:1505-1507`

**Step 1: Check if PPO metrics are already in telemetry**

The PPO update already emits `PPO_UPDATE_COMPLETED` events (lines 1527-1541). The print at line 1506 is redundant:

```python
            print(f"  Policy loss: {metrics['policy_loss']:.4f}, "
                  f"Value loss: {metrics['value_loss']:.4f}, "
```

**Step 2: Add console formatter for PPO_UPDATE_COMPLETED**

In `src/esper/nissa/output.py`, add:

```python
        elif event_type == "PPO_UPDATE_COMPLETED":
            data = event.data or {}
            if data.get("skipped"):
                reason = data.get("reason", "unknown")
                print(f"[{timestamp}] PPO | Update skipped ({reason})")
            else:
                policy_loss = data.get("policy_loss", 0.0)
                value_loss = data.get("value_loss", 0.0)
                entropy = data.get("entropy", 0.0)
                entropy_coef = data.get("entropy_coef", 0.0)
                print(f"[{timestamp}] PPO | policy={policy_loss:.4f}, value={value_loss:.4f}, entropy={entropy:.3f} (coef={entropy_coef:.4f})")
```

**Step 3: Remove redundant print**

In `src/esper/simic/vectorized.py`, remove the PPO loss print (lines 1505-1507).

**Step 4: Run syntax check**

Run: `python -m py_compile src/esper/simic/vectorized.py`
Expected: No output (success)

**Step 5: Commit**

```bash
git add src/esper/simic/vectorized.py src/esper/nissa/output.py
git commit -m "refactor(telemetry): route PPO metrics through Nissa console backend"
```

---

## Task 18: Update Leyline __init__.py Exports

**Files:**
- Modify: `src/esper/leyline/__init__.py`

**Step 1: Verify new event types are exported**

Check that `TelemetryEventType` is exported from `leyline/__init__.py`. The new enum values should be automatically available since they're part of the enum class.

Run: `python -c "from esper.leyline import TelemetryEventType; print(TelemetryEventType.GOVERNOR_ROLLBACK, TelemetryEventType.BATCH_EPOCH_COMPLETED)"`
Expected: `TelemetryEventType.GOVERNOR_ROLLBACK TelemetryEventType.BATCH_EPOCH_COMPLETED`

**Step 2: Commit if changes needed**

```bash
git add src/esper/leyline/__init__.py
git commit -m "chore: export new telemetry event types"
```

---

## Task 19: Add Integration Test

**Files:**
- Create: `tests/integration/test_telemetry_events.py`

**Step 1: Write integration test for new events**

```python
"""Integration tests for telemetry event emission."""

import pytest
from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.nissa.output import ConsoleOutput, NissaHub


class MockBackend:
    """Capture events for testing."""
    def __init__(self):
        self.events = []

    def emit(self, event: TelemetryEvent) -> None:
        self.events.append(event)

    def close(self) -> None:
        pass


def test_governor_events_exist():
    """Verify Governor event types are defined."""
    assert hasattr(TelemetryEventType, "GOVERNOR_ROLLBACK")
    assert hasattr(TelemetryEventType, "GOVERNOR_PANIC")


def test_batch_events_exist():
    """Verify batch progress event types are defined."""
    assert hasattr(TelemetryEventType, "BATCH_EPOCH_COMPLETED")
    assert hasattr(TelemetryEventType, "CHECKPOINT_SAVED")
    assert hasattr(TelemetryEventType, "CHECKPOINT_LOADED")


def test_counterfactual_event_exists():
    """Verify counterfactual event type is defined."""
    assert hasattr(TelemetryEventType, "COUNTERFACTUAL_COMPUTED")


def test_console_output_formats_governor_rollback(capsys):
    """Verify ConsoleOutput formats GOVERNOR_ROLLBACK correctly."""
    console = ConsoleOutput()
    event = TelemetryEvent(
        event_type=TelemetryEventType.GOVERNOR_ROLLBACK,
        data={
            "reason": "Structural Collapse",
            "loss_at_panic": 15.3,
            "loss_threshold": 5.2,
            "consecutive_panics": 2,
        },
    )
    console.emit(event)
    captured = capsys.readouterr()
    assert "GOVERNOR" in captured.out
    assert "ROLLBACK" in captured.out
    assert "Structural Collapse" in captured.out


def test_console_output_formats_batch_completed(capsys):
    """Verify ConsoleOutput formats BATCH_EPOCH_COMPLETED correctly."""
    console = ConsoleOutput()
    event = TelemetryEvent(
        event_type=TelemetryEventType.BATCH_EPOCH_COMPLETED,
        data={
            "batch_idx": 3,
            "episodes_completed": 24,
            "total_episodes": 100,
            "avg_accuracy": 67.2,
            "rolling_accuracy": 65.1,
            "avg_reward": 2.3,
        },
    )
    console.emit(event)
    captured = capsys.readouterr()
    assert "BATCH 3" in captured.out
    assert "24/100" in captured.out
    assert "67.2%" in captured.out


def test_hub_routes_events_to_backends():
    """Verify NissaHub routes events to all backends."""
    hub = NissaHub()
    mock1 = MockBackend()
    mock2 = MockBackend()
    hub.add_backend(mock1)
    hub.add_backend(mock2)

    event = TelemetryEvent(event_type=TelemetryEventType.GOVERNOR_ROLLBACK)
    hub.emit(event)

    assert len(mock1.events) == 1
    assert len(mock2.events) == 1
    assert mock1.events[0].event_type == TelemetryEventType.GOVERNOR_ROLLBACK
```

**Step 2: Run the test**

Run: `pytest tests/integration/test_telemetry_events.py -v`
Expected: All tests pass

**Step 3: Commit**

```bash
git add tests/integration/test_telemetry_events.py
git commit -m "test: add integration tests for new telemetry events"
```

---

## Task 20: Final Verification

**Step 1: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests pass

**Step 2: Run a short training to verify telemetry**

Run: `python -m esper.scripts.train cifar10 --max-epochs 5 --n-episodes 2`
Expected: See new telemetry format in console output (BATCH, GOVERNOR events if triggered)

**Step 3: Create final commit**

```bash
git add -A
git commit -m "docs: complete telemetry extraction implementation"
```

---

## Summary

| Category | Prints Converted | New Event Types |
|----------|------------------|-----------------|
| Governor | 2 | 3 (PANIC, ROLLBACK, SNAPSHOT) |
| Batch Progress | 6 | 3 (BATCH_EPOCH_COMPLETED, CHECKPOINT_SAVED, CHECKPOINT_LOADED) |
| Counterfactual | 1 | 1 (COUNTERFACTUAL_COMPUTED) |
| Anomaly | 6 (redundant) | 0 (already exist) |
| PPO | 2 (redundant) | 0 (already exist) |
| Tamiyo | 2 (redundant) | 0 (already exist) |
| **Total** | **19** | **7** |

**Remaining prints (intentionally kept as CLI output):**
- Training startup banners (lines 269-288)
- Data loading status (lines 310, 332)
- Analytics summary tables (lines 1585-1590)
- Telemetry file path confirmations (train.py)
