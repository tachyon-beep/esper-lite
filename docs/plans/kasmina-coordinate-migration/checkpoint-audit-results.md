# Checkpoint Audit Results

**Date:** 2025-12-16
**Auditor:** Claude Code

---

## Environment

| Component | Version |
|-----------|---------|
| PyTorch | 2.9.1+cu128 |
| CUDA | 12.8 |
| Python | 3.13.1 |
| Platform | linux |

---

## Summary

| Checkpoint Type | Stage | weights_only=True | Issues Found |
|-----------------|-------|-------------------|--------------|
| PPO Agent | N/A | PASS | 0 issues |
| MorphogeneticModel | GERMINATED | FAIL | 10 issues |
| MorphogeneticModel | BLENDING | FAIL | 15 issues |
| MorphogeneticModel | PROBATIONARY | FAIL | 17 issues |

---

## PPO Agent Checkpoint

**Generated:** `/tmp/esper_test_checkpoint.pt`

### weights_only=True Result

```
SUCCESS - checkpoint is already compatible!
```

### Issues Found

None. PPO Agent checkpoints are ALREADY COMPATIBLE with PyTorch 2.9 `weights_only=True`.

### Top-Level Structure

```
model_state_dict: dict[26 keys]
config: dict[1 keys]
```

### Analysis

PPO Agent checkpoints save minimal state in `extra_state`:
- No dataclasses (uses primitive types only)
- No enums (uses strings or ints)
- No datetime objects
- No deque or other complex collections
- No nn.Module serialization issues

This demonstrates that simple serialization design can achieve compatibility without additional conversion logic.

---

## MorphogeneticModel Checkpoint (GERMINATED)

**Generated:** `/tmp/esper_morphogenetic_checkpoint.pt`

### weights_only=True Result

```
FAILED: Weights only load failed. This file can still be loaded, to do so you have two options,
do those steps only if you trust the source of the checkpoint.
```

### Issues Found

```
* root['model_state_dict']['seed_slots.mid._extra_state']['seed_state']: dataclass esper.kasmina.slot.SeedState -> convert to dict via to_dict() method
* root['model_state_dict']['seed_slots.mid._extra_state']['seed_state'].stage: Enum esper.leyline.stages.SeedStage.GERMINATED -> convert to int (enum.value) for forward compatibility
* root['model_state_dict']['seed_slots.mid._extra_state']['seed_state'].previous_stage: Enum esper.leyline.stages.SeedStage.DORMANT -> convert to int (enum.value) for forward compatibility
* root['model_state_dict']['seed_slots.mid._extra_state']['seed_state'].stage_entered_at: datetime.datetime -> convert to ISO 8601 string
* root['model_state_dict']['seed_slots.mid._extra_state']['seed_state'].metrics: dataclass esper.kasmina.slot.SeedMetrics -> convert to dict via to_dict() method
* root['model_state_dict']['seed_slots.mid._extra_state']['seed_state'].stage_history: collections.deque -> convert to list
* root['model_state_dict']['seed_slots.mid._extra_state']['seed_state'].stage_history[0][0]: Enum esper.leyline.stages.SeedStage.GERMINATED -> convert to int (enum.value) for forward compatibility
* root['model_state_dict']['seed_slots.mid._extra_state']['seed_state'].stage_history[0][1]: datetime.datetime -> convert to ISO 8601 string
* root['model_state_dict']['seed_slots.mid._extra_state']['seed_state'].telemetry: dataclass esper.leyline.telemetry.SeedTelemetry -> convert to dict via to_dict() method
* root['model_state_dict']['seed_slots.mid._extra_state']['seed_state'].telemetry.captured_at: datetime.datetime -> convert to ISO 8601 string
```

### Top-Level Structure

```
model_state_dict: dict[26 keys]
config: dict[1 keys]
```

### Analysis

At GERMINATED stage:
- `alpha_schedule` is None (not yet started blending)
- All issues stem from `SeedState` dataclass and its nested structures
- Primary problematic types: dataclass, Enum, datetime, deque

---

## MorphogeneticModel Checkpoint (BLENDING) - CRITICAL

**Generated:** `/tmp/esper_blending_checkpoint.pt`

### weights_only=True Result

```
FAILED: Weights only load failed. This file can still be loaded, to do so you have two options,
do those steps only if you trust the source of the checkpoint.
```

### Issues Found

```
* root['model_state_dict']['seed_slots.mid._extra_state']['seed_state']: dataclass esper.kasmina.slot.SeedState -> convert to dict via to_dict() method
* root['model_state_dict']['seed_slots.mid._extra_state']['seed_state'].stage: Enum esper.leyline.stages.SeedStage.BLENDING -> convert to int (enum.value) for forward compatibility
* root['model_state_dict']['seed_slots.mid._extra_state']['seed_state'].previous_stage: Enum esper.leyline.stages.SeedStage.TRAINING -> convert to int (enum.value) for forward compatibility
* root['model_state_dict']['seed_slots.mid._extra_state']['seed_state'].stage_entered_at: datetime.datetime -> convert to ISO 8601 string
* root['model_state_dict']['seed_slots.mid._extra_state']['seed_state'].metrics: dataclass esper.kasmina.slot.SeedMetrics -> convert to dict via to_dict() method
* root['model_state_dict']['seed_slots.mid._extra_state']['seed_state'].stage_history: collections.deque -> convert to list
* root['model_state_dict']['seed_slots.mid._extra_state']['seed_state'].stage_history[0][0]: Enum esper.leyline.stages.SeedStage.GERMINATED -> convert to int (enum.value) for forward compatibility
* root['model_state_dict']['seed_slots.mid._extra_state']['seed_state'].stage_history[0][1]: datetime.datetime -> convert to ISO 8601 string
* root['model_state_dict']['seed_slots.mid._extra_state']['seed_state'].stage_history[1][0]: Enum esper.leyline.stages.SeedStage.TRAINING -> convert to int (enum.value) for forward compatibility
* root['model_state_dict']['seed_slots.mid._extra_state']['seed_state'].stage_history[1][1]: datetime.datetime -> convert to ISO 8601 string
* root['model_state_dict']['seed_slots.mid._extra_state']['seed_state'].stage_history[2][0]: Enum esper.leyline.stages.SeedStage.BLENDING -> convert to int (enum.value) for forward compatibility
* root['model_state_dict']['seed_slots.mid._extra_state']['seed_state'].stage_history[2][1]: datetime.datetime -> convert to ISO 8601 string
* root['model_state_dict']['seed_slots.mid._extra_state']['seed_state'].telemetry: dataclass esper.leyline.telemetry.SeedTelemetry -> convert to dict via to_dict() method
* root['model_state_dict']['seed_slots.mid._extra_state']['seed_state'].telemetry.captured_at: datetime.datetime -> convert to ISO 8601 string
* root['model_state_dict']['seed_slots.mid._extra_state']['alpha_schedule']: nn.Module esper.kasmina.blending.GatedBlend -> CRITICAL: store state_dict separately, serialize config only
```

### Top-Level Structure

```
model_state_dict: dict[30 keys]
config: dict[1 keys]
```

### Critical Finding: alpha_schedule

The `alpha_schedule` field stores an `nn.Module` (e.g., `GatedBlend`), which cannot be serialized with `weights_only=True`.

**Current behavior:**
```python
# SeedSlot.get_extra_state()
return {
    "seed_state": self.state,           # SeedState dataclass
    "alpha_schedule": self.alpha_schedule,  # nn.Module - FAILS
    "isolate_gradients": self.isolate_gradients,
}
```

**Required fix for M4:**
- Discard `alpha_schedule` after BLENDING completes
- Or serialize only config, not the nn.Module itself
- GatedBlend weights should be saved via normal state_dict() mechanism, not extra_state

**Observed during generation:**
```
alpha_schedule type: GatedBlend
```

This confirms that during BLENDING, `alpha_schedule` is a GatedBlend nn.Module instance.

---

## MorphogeneticModel Checkpoint (PROBATIONARY)

**Generated:** `/tmp/esper_probationary_checkpoint.pt`

### weights_only=True Result

```
FAILED: Weights only load failed. This file can still be loaded, to do so you have two options,
do those steps only if you trust the source of the checkpoint.
```

### Issues Found

```
* root['model_state_dict']['seed_slots.mid._extra_state']['seed_state']: dataclass esper.kasmina.slot.SeedState -> convert to dict via to_dict() method
* root['model_state_dict']['seed_slots.mid._extra_state']['seed_state'].stage: Enum esper.leyline.stages.SeedStage.PROBATIONARY -> convert to int (enum.value) for forward compatibility
* root['model_state_dict']['seed_slots.mid._extra_state']['seed_state'].previous_stage: Enum esper.leyline.stages.SeedStage.BLENDING -> convert to int (enum.value) for forward compatibility
* root['model_state_dict']['seed_slots.mid._extra_state']['seed_state'].stage_entered_at: datetime.datetime -> convert to ISO 8601 string
* root['model_state_dict']['seed_slots.mid._extra_state']['seed_state'].metrics: dataclass esper.kasmina.slot.SeedMetrics -> convert to dict via to_dict() method
* root['model_state_dict']['seed_slots.mid._extra_state']['seed_state'].stage_history: collections.deque -> convert to list
* root['model_state_dict']['seed_slots.mid._extra_state']['seed_state'].stage_history[0][0]: Enum esper.leyline.stages.SeedStage.GERMINATED -> convert to int (enum.value) for forward compatibility
* root['model_state_dict']['seed_slots.mid._extra_state']['seed_state'].stage_history[0][1]: datetime.datetime -> convert to ISO 8601 string
* root['model_state_dict']['seed_slots.mid._extra_state']['seed_state'].stage_history[1][0]: Enum esper.leyline.stages.SeedStage.TRAINING -> convert to int (enum.value) for forward compatibility
* root['model_state_dict']['seed_slots.mid._extra_state']['seed_state'].stage_history[1][1]: datetime.datetime -> convert to ISO 8601 string
* root['model_state_dict']['seed_slots.mid._extra_state']['seed_state'].stage_history[2][0]: Enum esper.leyline.stages.SeedStage.BLENDING -> convert to int (enum.value) for forward compatibility
* root['model_state_dict']['seed_slots.mid._extra_state']['seed_state'].stage_history[2][1]: datetime.datetime -> convert to ISO 8601 string
* root['model_state_dict']['seed_slots.mid._extra_state']['seed_state'].stage_history[3][0]: Enum esper.leyline.stages.SeedStage.PROBATIONARY -> convert to int (enum.value) for forward compatibility
* root['model_state_dict']['seed_slots.mid._extra_state']['seed_state'].stage_history[3][1]: datetime.datetime -> convert to ISO 8601 string
* root['model_state_dict']['seed_slots.mid._extra_state']['seed_state'].telemetry: dataclass esper.leyline.telemetry.SeedTelemetry -> convert to dict via to_dict() method
* root['model_state_dict']['seed_slots.mid._extra_state']['seed_state'].telemetry.captured_at: datetime.datetime -> convert to ISO 8601 string
* root['model_state_dict']['seed_slots.mid._extra_state']['alpha_schedule']: nn.Module esper.kasmina.blending.LinearBlend -> CRITICAL: store state_dict separately, serialize config only
```

### Top-Level Structure

```
model_state_dict: dict[26 keys]
config: dict[1 keys]
```

### Critical Finding: alpha_schedule PERSISTS

**The alpha_schedule nn.Module persists even after transitioning to PROBATIONARY.**

**Observed during generation:**
```
Stage: PROBATIONARY
alpha_schedule: LinearBlend()
```

This is a design flaw. After BLENDING completes and the seed transitions to PROBATIONARY, the alpha_schedule is no longer needed (alpha is permanently 1.0). The nn.Module should be discarded during the BLENDING -> PROBATIONARY transition.

**Impact:**
- Serialization issue persists throughout entire lifecycle after BLENDING starts
- Wastes memory keeping unused nn.Module alive
- Violates principle of minimal state

**Required fix:**
```python
# In SeedSlot.transition() or similar:
if target_stage == SeedStage.PROBATIONARY:
    self.alpha_schedule = None  # Discard - no longer needed
    self.state.alpha = 1.0      # Permanent full blend
```

---

## Type Conversion Requirements for M4

| Current Type | Location | Convert To | Priority |
|--------------|----------|------------|----------|
| `SeedState` dataclass | `extra_state['seed_state']` | `dict` via `to_dict()` | High |
| `SeedStage` Enum | `SeedState.stage` | `int` (enum.value) | High |
| `datetime` | `SeedState.stage_entered_at` | ISO 8601 string | High |
| `deque` | `SeedState.stage_history` | `list` | High |
| `GatedBlend` nn.Module | `extra_state['alpha_schedule']` | Config dict OR discard | Critical |
| `LinearBlend` nn.Module | `extra_state['alpha_schedule']` | Config dict OR discard | Critical |
| `SeedMetrics` dataclass | `SeedState.metrics` | `dict` via `to_dict()` | Medium |
| `SeedTelemetry` dataclass | `SeedState.telemetry` | `dict` via `to_dict()` | Medium |

---

## Recommended Changes for M4

### 1. SeedSlot.get_extra_state() - CRITICAL

```python
def get_extra_state(self) -> dict:
    """Persist SeedState for PyTorch 2.9+ weights_only=True compatibility."""
    state_dict = {
        "isolate_gradients": self.isolate_gradients,
    }

    if self.state is not None:
        state_dict["seed_state"] = self.state.to_dict()

    # Alpha schedule: DO NOT serialize the nn.Module
    # If restoration is needed, serialize config only
    if self.alpha_schedule is not None:
        state_dict["alpha_schedule_config"] = {
            "algorithm_id": getattr(self.alpha_schedule, "algorithm_id", None),
            "total_steps": getattr(self.alpha_schedule, "total_steps", None),
            # GatedBlend weights saved in state_dict(), not here
        }

    return state_dict
```

### 2. SeedState.to_dict() / from_dict()

```python
def to_dict(self) -> dict:
    """Convert to primitive dict for serialization."""
    return {
        "seed_id": self.seed_id,
        "blueprint_id": self.blueprint_id,
        "slot_id": self.slot_id,
        "stage": self.stage.value,  # Enum -> int
        "previous_stage": self.previous_stage.value if self.previous_stage else None,
        "stage_entered_at": self.stage_entered_at.isoformat(),  # datetime -> str
        "alpha": self.alpha,
        "stage_history": list(self.stage_history),  # deque -> list
        "metrics": self.metrics.to_dict() if self.metrics else None,
        "telemetry": self.telemetry.to_dict() if self.telemetry else None,
        "blend_algorithm_id": self.blend_algorithm_id,
    }

@classmethod
def from_dict(cls, data: dict) -> "SeedState":
    """Reconstruct from primitive dict."""
    return cls(
        seed_id=data["seed_id"],
        blueprint_id=data["blueprint_id"],
        slot_id=data["slot_id"],
        stage=SeedStage(data["stage"]),  # int -> Enum
        previous_stage=SeedStage(data["previous_stage"]) if data["previous_stage"] else None,
        stage_entered_at=datetime.fromisoformat(data["stage_entered_at"]),
        alpha=data["alpha"],
        stage_history=deque(data["stage_history"]),
        metrics=SeedMetrics.from_dict(data["metrics"]) if data["metrics"] else None,
        telemetry=SeedTelemetry.from_dict(data["telemetry"]) if data["telemetry"] else None,
        blend_algorithm_id=data["blend_algorithm_id"],
    )
```

### 3. Discard alpha_schedule after BLENDING

```python
# In SeedSlot, after BLENDING -> PROBATIONARY transition:
if target_stage == SeedStage.PROBATIONARY:
    self.alpha_schedule = None  # No longer needed
    self.state.alpha = 1.0  # Permanent full blend
```

### 4. SeedMetrics.to_dict() / from_dict()

```python
# Similar pattern for nested dataclasses
def to_dict(self) -> dict:
    return {
        "epochs_in_current_stage": self.epochs_in_current_stage,
        "blending_steps_completed": self.blending_steps_completed,
        # ... other fields
    }

@classmethod
def from_dict(cls, data: dict) -> "SeedMetrics":
    return cls(
        epochs_in_current_stage=data["epochs_in_current_stage"],
        blending_steps_completed=data["blending_steps_completed"],
        # ... other fields
    )
```

### 5. SeedTelemetry.to_dict() / from_dict()

```python
def to_dict(self) -> dict:
    return {
        "captured_at": self.captured_at.isoformat(),  # datetime -> str
        # ... other fields
    }

@classmethod
def from_dict(cls, data: dict) -> "SeedTelemetry":
    return cls(
        captured_at=datetime.fromisoformat(data["captured_at"]),
        # ... other fields
    )
```

---

## Verification

After M4 implementation, re-run audits to confirm:
```bash
PYTHONPATH=src python scripts/checkpoint_audit.py --generate-blending
```

Expected: `SUCCESS - checkpoint is already compatible!`

---

## Key Takeaways

1. **PPO Agent checkpoints are ALREADY COMPATIBLE** - demonstrates that simple design achieves compatibility
2. **MorphogeneticModel requires dataclass serialization** - all SeedState types need to_dict()/from_dict()
3. **CRITICAL: alpha_schedule persists unnecessarily** - should be discarded after BLENDING completes
4. **17 total issues in PROBATIONARY state** - more than BLENDING (15) due to longer stage_history
5. **All issues are solvable** - no fundamental blockers, just need systematic type conversion
