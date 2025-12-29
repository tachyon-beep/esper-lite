# Tamiyo UX Specialist Enhancements Implementation Plan (v5)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement PyTorch and DRL specialist recommendations for enhanced Tamiyo dashboard monitoring.

**Architecture:** Extend existing TamiyoState schema using nested dataclasses for new metrics, update telemetry emitters to collect them, modify existing widgets to display new metrics, and add gradient flow as a footer row in the Attention Heads panel (preserving bar visualizations).

**Tech Stack:** Python 3.11+, Textual (TUI framework), Rich (text rendering), PyTorch (metric collection), dataclasses (schema)

**Revision Notes (v5):** Addresses code reviewer feedback from v4:
- Task 3.3: `collect_cuda_memory_metrics()` now returns ALL fields even when no CUDA (0.0 defaults), not empty dict
- Task 3.4: Uses DIRECT ACCESS on `cuda_metrics` dict (no `.get()`) - collect function guarantees all fields present
- Task 4.2: Fixed commit message that incorrectly mentioned `.get()` usage
- **Wired `cuda_memory_peak_gb`** through entire pipeline (was in schema but missing from collection/emission/aggregation)

**Revision Notes (v4):** Addresses PyTorch expert and code reviewer feedback from v3:
- Uses `self.train_steps` instead of non-existent `self._update_count` (PyTorch expert)
- Removes `compile_healthy` runtime detection (not implementable via API) - just uses `compile_enabled` (PyTorch expert)
- **Always populates** gradient quality fields (even when throttled) - uses DIRECT ACCESS not `.get()` to comply with CLAUDE.md prohibition on defensive programming (Code reviewer)
- Fixes Task 3.4 test logic - directional clip always present, gradient_cv only on throttled batches

**Revision Notes (v3):** Addresses code reviewer feedback from v2:
- TrainingStartedPayload test includes all required fields
- Task 3.4 explicitly shows PPO agent wiring

**Revision Notes (v2):** Incorporates feedback from UX, PyTorch, DRL, and Code Review specialists:
- Fixed inverted SNR → uses Coefficient of Variation (CV)
- Removed `param_update_magnitude` (use existing `grad_norm`)
- Removed `minibatch_gradient_variance` (not applicable to recurrent PPO)
- Removed `graph_break_count` (not accessible via API - see v4 notes)
- Added complete telemetry collection pipeline (was missing in v1)
- Uses nested dataclasses to prevent TamiyoState bloat
- Keeps HeadsPanel bars, adds gradient flow as footer row
- Moves compile indicator to border title, compresses memory to percentage

---

## Phase 1: Schema Extensions with Nested Dataclasses

### Task 1.1: Create InfrastructureMetrics Nested Dataclass

**Files:**
- Modify: `src/esper/karn/sanctum/schema.py`
- Test: `tests/karn/sanctum/test_schema.py`

**Step 1: Write the failing test**

```python
# tests/karn/sanctum/test_schema.py
def test_infrastructure_metrics_dataclass():
    """InfrastructureMetrics should contain CUDA memory and compile status."""
    from esper.karn.sanctum.schema import InfrastructureMetrics

    metrics = InfrastructureMetrics()

    # Memory fields
    assert metrics.cuda_memory_allocated_gb == 0.0
    assert metrics.cuda_memory_reserved_gb == 0.0
    assert metrics.cuda_memory_peak_gb == 0.0
    assert metrics.cuda_memory_fragmentation == 0.0

    # Compile status (static session metadata - no runtime health detection)
    assert metrics.compile_enabled is False
    assert metrics.compile_backend == ""
    assert metrics.compile_mode == ""
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_schema.py::test_infrastructure_metrics_dataclass -v`
Expected: FAIL with "ImportError: cannot import name 'InfrastructureMetrics'"

**Step 3: Write minimal implementation**

Add before `TamiyoState` in `src/esper/karn/sanctum/schema.py`:

```python
@dataclass
class InfrastructureMetrics:
    """PyTorch infrastructure health metrics.

    Grouped separately to prevent TamiyoState bloat (per code review).
    Collected every N batches to amortize CPU-GPU sync overhead.
    """
    # CUDA Memory (PyTorch expert recommendation)
    cuda_memory_allocated_gb: float = 0.0   # torch.cuda.memory_allocated()
    cuda_memory_reserved_gb: float = 0.0    # torch.cuda.memory_reserved()
    cuda_memory_peak_gb: float = 0.0        # torch.cuda.max_memory_allocated()
    cuda_memory_fragmentation: float = 0.0  # 1 - (allocated/reserved), >0.3 = pressure

    # torch.compile Status (captured at training start - static session metadata)
    # Note: graph_break_count/compile_healthy removed - not accessible via PyTorch API
    # Compile issues will surface in throughput metrics (fps, step_time_ms)
    compile_enabled: bool = False
    compile_backend: str = ""    # "inductor", "eager", etc.
    compile_mode: str = ""       # "default", "reduce-overhead", "max-autotune"

    @property
    def memory_usage_percent(self) -> float:
        """Memory usage as percentage for compact display."""
        if self.cuda_memory_reserved_gb <= 0:
            return 0.0
        return (self.cuda_memory_allocated_gb / self.cuda_memory_reserved_gb) * 100
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_schema.py::test_infrastructure_metrics_dataclass -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/schema.py tests/karn/sanctum/test_schema.py
git commit -m "feat(schema): add InfrastructureMetrics nested dataclass

Add CUDA memory tracking (allocated, reserved, peak, fragmentation) and
torch.compile status (binary healthy indicator) per specialist review.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 1.2: Create GradientQualityMetrics Nested Dataclass

**Files:**
- Modify: `src/esper/karn/sanctum/schema.py`
- Test: `tests/karn/sanctum/test_schema.py`

**Step 1: Write the failing test**

```python
def test_gradient_quality_metrics_dataclass():
    """GradientQualityMetrics should contain gradient CV and directional clip."""
    from esper.karn.sanctum.schema import GradientQualityMetrics

    metrics = GradientQualityMetrics()

    # Gradient coefficient of variation (NOT SNR - per DRL review)
    # Low CV (<0.5) = high signal quality, High CV (>2.0) = noisy
    assert metrics.gradient_cv == 0.0

    # Directional clip fraction (per DRL expert)
    # clip+ = probability increases capped (r > 1+ε)
    # clip- = probability decreases capped (r < 1-ε)
    assert metrics.clip_fraction_positive == 0.0
    assert metrics.clip_fraction_negative == 0.0
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_schema.py::test_gradient_quality_metrics_dataclass -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Add before `TamiyoState` in `src/esper/karn/sanctum/schema.py`:

```python
@dataclass
class GradientQualityMetrics:
    """Gradient quality diagnostics for DRL training health.

    Grouped separately to prevent TamiyoState bloat (per code review).

    Note: Uses Coefficient of Variation (CV) not SNR per DRL expert review.
    The original plan had inverted SNR (var/mean² is noise-to-signal).
    CV = sqrt(var)/|mean| is standard and self-explanatory.
    """
    # Gradient Coefficient of Variation (per DRL expert - replaces inverted SNR)
    # Low CV (<0.5) = high signal quality, High CV (>2.0) = noisy gradients
    gradient_cv: float = 0.0

    # Directional Clip Fraction (per DRL expert recommendation)
    # These track WHERE clipping occurs, not WHETHER policy improved:
    # clip+ = r > 1+ε (probability increases were capped)
    # clip- = r < 1-ε (probability decreases were capped)
    # Asymmetry indicates directional policy drift; symmetric high values are normal
    clip_fraction_positive: float = 0.0
    clip_fraction_negative: float = 0.0

    # Note: param_update_magnitude REMOVED per PyTorch expert review
    # - Conflates gradient magnitude with learning rate
    # - Existing grad_norm, dead_layers, exploding_layers already provide this signal

    # Note: minibatch_gradient_variance REMOVED per PyTorch expert review
    # - Not applicable to recurrent PPO (single-batch processing due to LSTM coherence)
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_schema.py::test_gradient_quality_metrics_dataclass -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/schema.py tests/karn/sanctum/test_schema.py
git commit -m "feat(schema): add GradientQualityMetrics nested dataclass

Add gradient CV (coefficient of variation) and directional clip fraction.
Uses CV instead of SNR per DRL expert (original formula was inverted).
Removes param_update_magnitude and minibatch_variance per PyTorch expert.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 1.3: Add Nested Dataclasses to TamiyoState

**Files:**
- Modify: `src/esper/karn/sanctum/schema.py`
- Test: `tests/karn/sanctum/test_schema.py`

**Step 1: Write the failing test**

```python
def test_tamiyo_state_has_nested_metrics():
    """TamiyoState should have infrastructure and gradient_quality nested fields."""
    from esper.karn.sanctum.schema import (
        TamiyoState,
        InfrastructureMetrics,
        GradientQualityMetrics,
    )

    state = TamiyoState()

    # Nested dataclasses should be present with defaults
    assert isinstance(state.infrastructure, InfrastructureMetrics)
    assert isinstance(state.gradient_quality, GradientQualityMetrics)

    # Access nested fields
    assert state.infrastructure.cuda_memory_allocated_gb == 0.0
    assert state.gradient_quality.gradient_cv == 0.0
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_schema.py::test_tamiyo_state_has_nested_metrics -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Add to `TamiyoState` dataclass after the existing fields (around line 667):

```python
    # === Nested Metric Groups (per code review - prevents schema bloat) ===
    infrastructure: InfrastructureMetrics = field(default_factory=InfrastructureMetrics)
    gradient_quality: GradientQualityMetrics = field(default_factory=GradientQualityMetrics)
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_schema.py::test_tamiyo_state_has_nested_metrics -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/schema.py tests/karn/sanctum/test_schema.py
git commit -m "feat(schema): add nested metric groups to TamiyoState

Add infrastructure and gradient_quality nested dataclasses to TamiyoState.
This prevents schema bloat (was heading to 84+ flat fields) per code review.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Phase 2: Telemetry Payload Extensions

### Task 2.1: Extend PPOUpdatePayload with Directional Clip and Gradient CV

**Files:**
- Modify: `src/esper/leyline/telemetry.py` (PPOUpdatePayload dataclass)
- Test: `tests/leyline/test_telemetry.py`

**Step 1: Write the failing test**

```python
# tests/leyline/test_telemetry.py
def test_ppo_update_payload_has_gradient_quality_fields():
    """PPOUpdatePayload should have directional clip and gradient CV."""
    from esper.leyline import PPOUpdatePayload

    payload = PPOUpdatePayload(
        policy_loss=0.1,
        value_loss=0.2,
        entropy=1.0,
        grad_norm=0.5,
        kl_divergence=0.01,
        clip_fraction=0.15,
        clip_fraction_positive=0.10,
        clip_fraction_negative=0.05,
        gradient_cv=0.42,
    )

    assert payload.clip_fraction_positive == 0.10
    assert payload.clip_fraction_negative == 0.05
    assert payload.gradient_cv == 0.42
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/leyline/test_telemetry.py::test_ppo_update_payload_has_gradient_quality_fields -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Add to `PPOUpdatePayload` dataclass in `src/esper/leyline/telemetry.py`:

```python
    # === Gradient Quality Metrics (DRL expert recommendation) ===
    # Directional clip fraction - tracks WHERE clipping occurs
    clip_fraction_positive: float = 0.0  # r > 1+ε (probability increases capped)
    clip_fraction_negative: float = 0.0  # r < 1-ε (probability decreases capped)

    # Gradient Coefficient of Variation: sqrt(var)/|mean|
    # Low (<0.5) = high signal, High (>2.0) = noisy
    gradient_cv: float = 0.0
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/leyline/test_telemetry.py::test_ppo_update_payload_has_gradient_quality_fields -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/leyline/telemetry.py tests/leyline/test_telemetry.py
git commit -m "feat(telemetry): add gradient quality fields to PPOUpdatePayload

Add clip_fraction_positive, clip_fraction_negative, and gradient_cv.
Uses CV (coefficient of variation) per DRL expert - original SNR was inverted.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 2.2: Extend PPOUpdatePayload with Infrastructure Metrics

**Files:**
- Modify: `src/esper/leyline/telemetry.py` (PPOUpdatePayload dataclass)
- Test: `tests/leyline/test_telemetry.py`

**Step 1: Write the failing test**

```python
def test_ppo_update_payload_has_infrastructure_fields():
    """PPOUpdatePayload should have CUDA memory fields."""
    from esper.leyline import PPOUpdatePayload

    payload = PPOUpdatePayload(
        policy_loss=0.1,
        value_loss=0.2,
        entropy=1.0,
        grad_norm=0.5,
        kl_divergence=0.01,
        clip_fraction=0.15,
        cuda_memory_allocated_gb=4.2,
        cuda_memory_reserved_gb=8.0,
        cuda_memory_peak_gb=6.5,
        cuda_memory_fragmentation=0.475,
    )

    assert payload.cuda_memory_allocated_gb == 4.2
    assert payload.cuda_memory_reserved_gb == 8.0
    assert payload.cuda_memory_peak_gb == 6.5
    assert payload.cuda_memory_fragmentation == 0.475
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/leyline/test_telemetry.py::test_ppo_update_payload_has_infrastructure_fields -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Add to `PPOUpdatePayload` dataclass:

```python
    # === Infrastructure Metrics (PyTorch expert recommendation) ===
    # Collected every N batches to amortize CPU-GPU sync overhead
    cuda_memory_allocated_gb: float = 0.0
    cuda_memory_reserved_gb: float = 0.0
    cuda_memory_peak_gb: float = 0.0        # torch.cuda.max_memory_allocated()
    cuda_memory_fragmentation: float = 0.0  # 1 - (allocated/reserved), >0.3 = pressure
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/leyline/test_telemetry.py::test_ppo_update_payload_has_infrastructure_fields -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/leyline/telemetry.py tests/leyline/test_telemetry.py
git commit -m "feat(telemetry): add CUDA memory fields to PPOUpdatePayload

Add cuda_memory_allocated_gb, cuda_memory_reserved_gb, cuda_memory_peak_gb,
and cuda_memory_fragmentation for OOM risk monitoring per PyTorch expert.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Phase 3: PPO Agent Metric Collection

### Task 3.1: Compute Directional Clip Fraction in PPO Agent

**Files:**
- Modify: `src/esper/simic/agent/ppo.py`
- Test: `tests/simic/agent/test_ppo.py`

**Step 1: Write the failing test**

```python
# tests/simic/agent/test_ppo.py
import torch

def test_compute_directional_clip_fraction():
    """Directional clip should be computed from ratio tensor."""
    from esper.simic.agent.ppo import compute_directional_clip_fraction

    # ratio = pi_new / pi_old
    ratio = torch.tensor([0.5, 0.9, 1.0, 1.1, 1.5, 2.0])
    clip_ratio = 0.2  # Standard PPO clip

    clip_pos, clip_neg = compute_directional_clip_fraction(ratio, clip_ratio)

    # r > 1.2: indices 4, 5 → 2/6 = 0.333...
    # r < 0.8: indices 0 → 1/6 = 0.166...
    assert abs(clip_pos - 2/6) < 1e-5
    assert abs(clip_neg - 1/6) < 1e-5
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/simic/agent/test_ppo.py::test_compute_directional_clip_fraction -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Add to `src/esper/simic/agent/ppo.py`:

```python
def compute_directional_clip_fraction(
    ratio: torch.Tensor,
    clip_ratio: float,
) -> tuple[float, float]:
    """Compute directional clip fractions from importance sampling ratio.

    Args:
        ratio: Importance sampling ratio π_new(a|s) / π_old(a|s)
        clip_ratio: PPO clip epsilon (typically 0.2)

    Returns:
        (clip_fraction_positive, clip_fraction_negative)
        - positive: fraction where r > 1 + clip_ratio (probability increases capped)
        - negative: fraction where r < 1 - clip_ratio (probability decreases capped)
    """
    ratio_minus_one = ratio - 1.0
    clip_pos = (ratio_minus_one > clip_ratio).float().mean().item()
    clip_neg = (ratio_minus_one < -clip_ratio).float().mean().item()
    return clip_pos, clip_neg
```

Then integrate into the PPO update loop where `clip_fraction` is computed, adding results to the metrics dict:

```python
# In compute_ppo_loss or update method, after computing ratio:
clip_pos, clip_neg = compute_directional_clip_fraction(ratio, self.clip_ratio)
metrics["clip_fraction_positive"] = clip_pos
metrics["clip_fraction_negative"] = clip_neg
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/simic/agent/test_ppo.py::test_compute_directional_clip_fraction -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/agent/ppo.py tests/simic/agent/test_ppo.py
git commit -m "feat(ppo): compute directional clip fraction during update

Add compute_directional_clip_fraction() and integrate into PPO update.
Tracks clip+ (r > 1+ε) and clip- (r < 1-ε) separately per DRL expert.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 3.2: Compute Gradient Coefficient of Variation in PPO Agent

**Files:**
- Modify: `src/esper/simic/agent/ppo.py`
- Test: `tests/simic/agent/test_ppo.py`

**Step 1: Write the failing test**

```python
def test_compute_gradient_cv():
    """Gradient CV should be sqrt(var)/|mean|."""
    from esper.simic.agent.ppo import compute_gradient_cv
    import torch
    import torch.nn as nn

    # Simple model with known gradients
    model = nn.Linear(10, 2)
    loss = model(torch.randn(1, 10)).sum()
    loss.backward()

    cv = compute_gradient_cv(model)

    # CV should be non-negative
    assert cv >= 0.0
    # CV should be finite
    assert not (cv != cv)  # Not NaN
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/simic/agent/test_ppo.py::test_compute_gradient_cv -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Add to `src/esper/simic/agent/ppo.py`:

```python
def compute_gradient_cv(model: torch.nn.Module) -> float:
    """Compute Coefficient of Variation of gradients: sqrt(var)/|mean|.

    CV interpretation (per DRL expert):
    - CV < 0.5: High signal quality (gradients are consistent)
    - CV 0.5-2.0: Normal range
    - CV > 2.0: Noisy gradients (high variance relative to mean)

    Note: This is O(params) and requires a sync. Only compute every N updates.

    Returns:
        Coefficient of variation, or 0.0 if no gradients.
    """
    grads = [p.grad.flatten() for p in model.parameters() if p.grad is not None]
    if not grads:
        return 0.0

    all_grads = torch.cat(grads)
    mean = all_grads.mean()
    std = all_grads.std()

    # Avoid division by zero
    if abs(mean.item()) < 1e-10:
        return 0.0

    cv = (std / mean.abs()).item()
    return cv
```

Then integrate into PPO update (compute every N updates to amortize cost):

```python
# In PPO update, after backward pass:
if batch_idx % 10 == 0:  # Every 10 updates
    metrics["gradient_cv"] = compute_gradient_cv(self.policy.network)
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/simic/agent/test_ppo.py::test_compute_gradient_cv -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/agent/ppo.py tests/simic/agent/test_ppo.py
git commit -m "feat(ppo): compute gradient coefficient of variation

Add compute_gradient_cv() using sqrt(var)/|mean| formula per DRL expert.
Computed every 10 updates to amortize O(params) cost.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 3.3: Collect CUDA Memory Metrics in PPO Agent

**Files:**
- Modify: `src/esper/simic/agent/ppo.py`
- Test: `tests/simic/agent/test_ppo.py`

**Step 1: Write the failing test**

```python
def test_collect_cuda_memory_metrics():
    """CUDA memory collection should return dict with expected keys."""
    from esper.simic.agent.ppo import collect_cuda_memory_metrics

    metrics = collect_cuda_memory_metrics()

    # Should always return dict (empty if no CUDA)
    assert isinstance(metrics, dict)

    # If CUDA available, should have these keys
    if torch.cuda.is_available():
        assert "cuda_memory_allocated_gb" in metrics
        assert "cuda_memory_reserved_gb" in metrics
        assert "cuda_memory_peak_gb" in metrics
        assert "cuda_memory_fragmentation" in metrics
        assert metrics["cuda_memory_peak_gb"] >= 0.0
        assert metrics["cuda_memory_fragmentation"] >= 0.0
        assert metrics["cuda_memory_fragmentation"] <= 1.0
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/simic/agent/test_ppo.py::test_collect_cuda_memory_metrics -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Add to `src/esper/simic/agent/ppo.py`:

```python
def collect_cuda_memory_metrics() -> dict[str, float]:
    """Collect CUDA memory metrics for telemetry.

    Note: Each call involves CPU-GPU sync. Call every N batches, not every update.

    Returns:
        Dict with cuda_memory_allocated_gb, cuda_memory_reserved_gb,
        cuda_memory_peak_gb, cuda_memory_fragmentation. Returns 0.0 values
        if CUDA not available (always populates all fields per CLAUDE.md).
    """
    if not torch.cuda.is_available():
        # Always return all fields per CLAUDE.md - enables direct access in callers
        return {
            "cuda_memory_allocated_gb": 0.0,
            "cuda_memory_reserved_gb": 0.0,
            "cuda_memory_peak_gb": 0.0,
            "cuda_memory_fragmentation": 0.0,
        }

    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9

    # Fragmentation: 1 - (allocated/reserved)
    # High fragmentation (>0.3) indicates memory pressure even with low allocation
    fragmentation = 0.0
    if reserved > 0:
        fragmentation = 1.0 - (allocated / reserved)

    return {
        "cuda_memory_allocated_gb": allocated,
        "cuda_memory_reserved_gb": reserved,
        "cuda_memory_peak_gb": peak,
        "cuda_memory_fragmentation": fragmentation,
    }
```

Then integrate into PPO update (throttled):

```python
# In PPO update, every N batches:
if batch_idx % 10 == 0:
    metrics.update(collect_cuda_memory_metrics())
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/simic/agent/test_ppo.py::test_collect_cuda_memory_metrics -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/agent/ppo.py tests/simic/agent/test_ppo.py
git commit -m "feat(ppo): collect CUDA memory metrics with fragmentation

Add collect_cuda_memory_metrics() with throttling recommendation.
Includes fragmentation metric (1 - allocated/reserved) per PyTorch expert.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 3.4: Wire Metric Collection into PPO Update Loop

**Files:**
- Modify: `src/esper/simic/agent/ppo.py` (PPOAgent.update method)
- Test: `tests/simic/agent/test_ppo.py`

**Context:** Tasks 3.1-3.3 created the metric collection functions. This task wires them into the PPO update loop so metrics flow to the emitter.

**Step 1: Write the failing test**

```python
def test_ppo_update_returns_gradient_quality_metrics():
    """PPO update should return gradient quality metrics in metrics dict."""
    # This is an integration test - verify metrics dict has expected keys
    # after a full update cycle
    from esper.simic.agent.ppo import PPOAgent
    import torch

    # Create minimal PPO agent for testing
    # (Use existing test fixtures from test_ppo.py)
    agent = create_test_ppo_agent()  # Assume fixture exists

    # Run update and check metrics dict
    metrics = agent.update(batch)

    # Directional clip fraction should ALWAYS be present (computed inline, not throttled)
    assert "clip_fraction_positive" in metrics
    assert "clip_fraction_negative" in metrics

    # Gradient CV and CUDA memory are throttled but ALWAYS populated (with 0.0 on non-throttle batches)
    # This ensures emitter can use direct access per CLAUDE.md requirements
    assert "gradient_cv" in metrics

    # Verify values make sense
    assert 0.0 <= metrics["clip_fraction_positive"] <= 1.0
    assert 0.0 <= metrics["clip_fraction_negative"] <= 1.0
    assert metrics["gradient_cv"] >= 0.0
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/simic/agent/test_ppo.py::test_ppo_update_returns_gradient_quality_metrics -v`
Expected: FAIL (metrics dict missing gradient quality keys)

**Step 3: Write minimal implementation**

In `PPOAgent.update()` method in `src/esper/simic/agent/ppo.py`, wire the metric collection:

```python
def update(self, batch: TamiyoRolloutBuffer) -> dict[str, Any]:
    """Run PPO update on collected batch."""
    metrics: dict[str, Any] = {}

    # ... existing PPO loss computation ...

    # After computing ratio tensor (existing code):
    ratio = (new_log_probs - old_log_probs).exp()

    # === NEW: Compute directional clip fraction (Task 3.1) ===
    # Always computed (cheap operation on existing ratio tensor)
    clip_pos, clip_neg = compute_directional_clip_fraction(ratio, self.clip_ratio)
    metrics["clip_fraction_positive"] = clip_pos
    metrics["clip_fraction_negative"] = clip_neg

    # ... existing backward pass ...
    loss.backward()

    # === NEW: Throttled metric collection (every 10 batches) ===
    # Per CLAUDE.md: ALWAYS populate fields so emitter can use direct access
    # Throttling controls WHEN we compute, not WHETHER we include the field
    if self.train_steps % 10 == 0:
        # Gradient CV (Task 3.2) - after backward, before optimizer step
        metrics["gradient_cv"] = compute_gradient_cv(self.policy.network)
        # CUDA memory (Task 3.3) - uses DIRECT ACCESS per CLAUDE.md
        # collect_cuda_memory_metrics() guarantees all fields present (even when no CUDA)
        cuda_metrics = collect_cuda_memory_metrics()
        metrics["cuda_memory_allocated_gb"] = cuda_metrics["cuda_memory_allocated_gb"]
        metrics["cuda_memory_reserved_gb"] = cuda_metrics["cuda_memory_reserved_gb"]
        metrics["cuda_memory_peak_gb"] = cuda_metrics["cuda_memory_peak_gb"]
        metrics["cuda_memory_fragmentation"] = cuda_metrics["cuda_memory_fragmentation"]
    else:
        # Non-throttle batches: populate with 0.0 defaults
        # This ensures emitter can always use metrics["field"] not metrics.get()
        metrics["gradient_cv"] = 0.0
        metrics["cuda_memory_allocated_gb"] = 0.0
        metrics["cuda_memory_reserved_gb"] = 0.0
        metrics["cuda_memory_peak_gb"] = 0.0
        metrics["cuda_memory_fragmentation"] = 0.0

    # ... existing optimizer step ...
    self.optimizer.step()
    self.train_steps += 1  # Use existing counter, not new _update_count

    return metrics
```

**Key wiring points:**
1. Directional clip: Computed inline after ratio calculation (always available, always computed)
2. Gradient CV: Computed every 10 batches but ALWAYS populated (0.0 on non-throttle)
3. CUDA memory: Computed every 10 batches but ALWAYS populated (0.0 on non-throttle or no CUDA)

**Why always populate:** Per CLAUDE.md prohibition on defensive programming, the emitter must use direct access `metrics["field"]` not `metrics.get("field", 0.0)`. This requires fields to always be present.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/simic/agent/test_ppo.py::test_ppo_update_returns_gradient_quality_metrics -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/agent/ppo.py tests/simic/agent/test_ppo.py
git commit -m "feat(ppo): wire gradient quality metrics into update loop

Integrate compute_directional_clip_fraction, compute_gradient_cv, and
collect_cuda_memory_metrics into PPOAgent.update(). Throttles expensive
metrics (CV, CUDA memory) to every 10 batches.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Phase 4: Telemetry Emission

### Task 4.1: Emit Gradient Quality Metrics in Telemetry

**Files:**
- Modify: `src/esper/simic/telemetry/emitters.py`
- Test: `tests/simic/telemetry/test_emitters.py`

**Step 1: Write the failing test**

```python
# tests/simic/telemetry/test_emitters.py
def test_emit_ppo_update_includes_gradient_quality():
    """emit_ppo_update_event should include gradient quality metrics."""
    from unittest.mock import MagicMock
    from esper.simic.telemetry.emitters import emit_ppo_update_event

    hub = MagicMock()
    metrics = {
        "policy_loss": 0.1,
        "value_loss": 0.2,
        "entropy": 1.0,
        "approx_kl": 0.01,
        "clip_fraction": 0.15,
        "clip_fraction_positive": 0.10,
        "clip_fraction_negative": 0.05,
        "gradient_cv": 0.42,
    }

    emit_ppo_update_event(
        hub=hub,
        metrics=metrics,
        episodes_completed=1,
        batch_idx=0,
        epoch=0,
        optimizer=None,
        grad_norm=0.5,
        update_time_ms=10.0,
    )

    # Verify emitted payload has gradient quality fields
    call_args = hub.emit.call_args
    event = call_args[0][0]

    # Verify fields are present and have expected values
    assert event.data.clip_fraction_positive == 0.10
    assert event.data.clip_fraction_negative == 0.05
    assert event.data.gradient_cv == 0.42
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/simic/telemetry/test_emitters.py::test_emit_ppo_update_includes_gradient_quality -v`
Expected: FAIL

**Step 3: Write minimal implementation**

In `emit_ppo_update_event` in `src/esper/simic/telemetry/emitters.py`, add to the PPOUpdatePayload construction:

```python
            # Gradient quality metrics (DRL expert recommendation)
            # Uses DIRECT ACCESS per CLAUDE.md - PPO agent (Task 3.4) always populates these
            clip_fraction_positive=metrics["clip_fraction_positive"],
            clip_fraction_negative=metrics["clip_fraction_negative"],
            gradient_cv=metrics["gradient_cv"],
```

**Note:** Uses DIRECT ACCESS per CLAUDE.md prohibition on defensive programming. Task 3.4 ensures these fields are ALWAYS populated (0.0 on non-throttle batches), so direct access is safe and will fail-fast if the wiring is broken.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/simic/telemetry/test_emitters.py::test_emit_ppo_update_includes_gradient_quality -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/telemetry/emitters.py tests/simic/telemetry/test_emitters.py
git commit -m "feat(telemetry): emit gradient quality metrics in PPO updates

Add clip_fraction_positive, clip_fraction_negative, gradient_cv to
PPOUpdatePayload emission. Uses direct dict access per code review.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 4.2: Emit Infrastructure Metrics in Telemetry

**Files:**
- Modify: `src/esper/simic/telemetry/emitters.py`
- Test: `tests/simic/telemetry/test_emitters.py`

**Step 1: Write the failing test**

```python
def test_emit_ppo_update_includes_infrastructure():
    """emit_ppo_update_event should include CUDA memory metrics when present."""
    from unittest.mock import MagicMock
    from esper.simic.telemetry.emitters import emit_ppo_update_event

    hub = MagicMock()
    metrics = {
        "policy_loss": 0.1,
        "value_loss": 0.2,
        "entropy": 1.0,
        "approx_kl": 0.01,
        "clip_fraction": 0.15,
        "clip_fraction_positive": 0.0,
        "clip_fraction_negative": 0.0,
        "gradient_cv": 0.0,
        "cuda_memory_allocated_gb": 4.2,
        "cuda_memory_reserved_gb": 8.0,
        "cuda_memory_peak_gb": 6.5,
        "cuda_memory_fragmentation": 0.475,
    }

    emit_ppo_update_event(
        hub=hub,
        metrics=metrics,
        episodes_completed=1,
        batch_idx=0,
        epoch=0,
        optimizer=None,
        grad_norm=0.5,
        update_time_ms=10.0,
    )

    call_args = hub.emit.call_args
    event = call_args[0][0]
    assert event.data.cuda_memory_allocated_gb == 4.2
    assert event.data.cuda_memory_reserved_gb == 8.0
    assert event.data.cuda_memory_peak_gb == 6.5
    assert event.data.cuda_memory_fragmentation == 0.475
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/simic/telemetry/test_emitters.py::test_emit_ppo_update_includes_infrastructure -v`
Expected: FAIL

**Step 3: Write minimal implementation**

In `emit_ppo_update_event`, add to PPOUpdatePayload construction:

```python
            # Infrastructure metrics (PyTorch expert recommendation)
            # Uses DIRECT ACCESS per CLAUDE.md - PPO agent (Task 3.4) always populates these
            # (0.0 for CPU-only training or non-throttle batches)
            cuda_memory_allocated_gb=metrics["cuda_memory_allocated_gb"],
            cuda_memory_reserved_gb=metrics["cuda_memory_reserved_gb"],
            cuda_memory_peak_gb=metrics["cuda_memory_peak_gb"],
            cuda_memory_fragmentation=metrics["cuda_memory_fragmentation"],
```

**Note:** Uses DIRECT ACCESS per CLAUDE.md. Task 3.4 ensures these fields are ALWAYS populated (0.0 for CPU-only training or non-throttle batches), so direct access is safe.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/simic/telemetry/test_emitters.py::test_emit_ppo_update_includes_infrastructure -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/telemetry/emitters.py tests/simic/telemetry/test_emitters.py
git commit -m "feat(telemetry): emit CUDA memory metrics in PPO updates

Add cuda_memory_allocated_gb, cuda_memory_reserved_gb, cuda_memory_peak_gb,
and cuda_memory_fragmentation to PPOUpdatePayload emission. Uses direct
dict access per CLAUDE.md - Task 3.4 ensures fields always populated.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Phase 5: Aggregator Integration

### Task 5.1: Populate Nested Metrics in Aggregator

**Files:**
- Modify: `src/esper/karn/sanctum/aggregator.py`
- Test: `tests/karn/sanctum/test_aggregator.py`

**Step 1: Write the failing test**

```python
# tests/karn/sanctum/test_aggregator.py
def test_aggregator_populates_nested_metrics():
    """Aggregator should populate infrastructure and gradient_quality nested fields."""
    from esper.karn.sanctum.aggregator import SanctumAggregator
    from esper.leyline import TelemetryEvent, TelemetryEventType, PPOUpdatePayload

    aggregator = SanctumAggregator()

    event = TelemetryEvent(
        event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
        data=PPOUpdatePayload(
            policy_loss=0.1,
            value_loss=0.2,
            entropy=1.0,
            grad_norm=0.5,
            kl_divergence=0.01,
            clip_fraction=0.15,
            clip_fraction_positive=0.10,
            clip_fraction_negative=0.05,
            gradient_cv=0.42,
            cuda_memory_allocated_gb=4.2,
            cuda_memory_reserved_gb=8.0,
            cuda_memory_peak_gb=6.5,
            cuda_memory_fragmentation=0.475,
        ),
    )

    aggregator.process_event(event)
    snapshot = aggregator.get_snapshot()

    # Gradient quality (nested)
    assert snapshot.tamiyo.gradient_quality.clip_fraction_positive == 0.10
    assert snapshot.tamiyo.gradient_quality.clip_fraction_negative == 0.05
    assert snapshot.tamiyo.gradient_quality.gradient_cv == 0.42

    # Infrastructure (nested)
    assert snapshot.tamiyo.infrastructure.cuda_memory_allocated_gb == 4.2
    assert snapshot.tamiyo.infrastructure.cuda_memory_reserved_gb == 8.0
    assert snapshot.tamiyo.infrastructure.cuda_memory_peak_gb == 6.5
    assert snapshot.tamiyo.infrastructure.cuda_memory_fragmentation == 0.475
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_aggregator.py::test_aggregator_populates_nested_metrics -v`
Expected: FAIL

**Step 3: Write minimal implementation**

In the PPO_UPDATE_COMPLETED handler in `aggregator.py`, add:

```python
        # Gradient quality metrics (nested dataclass)
        self._tamiyo.gradient_quality.clip_fraction_positive = payload.clip_fraction_positive
        self._tamiyo.gradient_quality.clip_fraction_negative = payload.clip_fraction_negative
        self._tamiyo.gradient_quality.gradient_cv = payload.gradient_cv

        # Infrastructure metrics (nested dataclass)
        self._tamiyo.infrastructure.cuda_memory_allocated_gb = payload.cuda_memory_allocated_gb
        self._tamiyo.infrastructure.cuda_memory_reserved_gb = payload.cuda_memory_reserved_gb
        self._tamiyo.infrastructure.cuda_memory_peak_gb = payload.cuda_memory_peak_gb
        self._tamiyo.infrastructure.cuda_memory_fragmentation = payload.cuda_memory_fragmentation
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_aggregator.py::test_aggregator_populates_nested_metrics -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/aggregator.py tests/karn/sanctum/test_aggregator.py
git commit -m "feat(aggregator): populate nested infrastructure and gradient_quality

Extract gradient quality and CUDA memory metrics from PPOUpdatePayload
into nested TamiyoState dataclasses.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 5.2: Populate Compile Status from TrainingStarted

**Files:**
- Modify: `src/esper/karn/sanctum/aggregator.py`
- Test: `tests/karn/sanctum/test_aggregator.py`

**Step 1: Write the failing test**

```python
def test_aggregator_populates_compile_status():
    """Aggregator should populate compile status from TrainingStartedPayload."""
    from esper.karn.sanctum.aggregator import SanctumAggregator
    from esper.leyline import TelemetryEvent, TelemetryEventType, TrainingStartedPayload

    aggregator = SanctumAggregator()

    # TrainingStartedPayload has many required fields - must include all
    event = TelemetryEvent(
        event_type=TelemetryEventType.TRAINING_STARTED,
        data=TrainingStartedPayload(
            # Required fields (from leyline/telemetry.py TrainingStartedPayload)
            n_envs=4,
            max_epochs=25,
            task="mnist",
            host_params=1000000,
            slot_ids=("slot_0", "slot_1", "slot_2"),
            seed=42,
            n_episodes=100,
            lr=3e-4,
            clip_ratio=0.2,
            entropy_coef=0.01,
            param_budget=500000,
            policy_device="cuda:0",
            env_devices=("cuda:0", "cuda:0", "cuda:0", "cuda:0"),
            reward_mode="shaped",
            # The fields we're testing
            compile_enabled=True,
            compile_backend="inductor",
            compile_mode="reduce-overhead",
        ),
    )

    aggregator.process_event(event)
    snapshot = aggregator.get_snapshot()

    assert snapshot.tamiyo.infrastructure.compile_enabled is True
    assert snapshot.tamiyo.infrastructure.compile_backend == "inductor"
    assert snapshot.tamiyo.infrastructure.compile_mode == "reduce-overhead"
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_aggregator.py::test_aggregator_populates_compile_status -v`
Expected: FAIL

**Step 3: Write minimal implementation**

In the TRAINING_STARTED handler in `aggregator.py`, add:

```python
        # Compile status (static configuration from training start)
        self._tamiyo.infrastructure.compile_enabled = payload.compile_enabled
        self._tamiyo.infrastructure.compile_backend = payload.compile_backend or ""
        self._tamiyo.infrastructure.compile_mode = payload.compile_mode or ""
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_aggregator.py::test_aggregator_populates_compile_status -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/aggregator.py tests/karn/sanctum/test_aggregator.py
git commit -m "feat(aggregator): populate compile status from TrainingStarted

Extract compile_enabled, compile_backend, compile_mode from
TrainingStartedPayload into infrastructure nested dataclass.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Phase 6: Widget Updates

### Task 6.1: Update Status Banner with Compact Memory and Compile in Title

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/tamiyo_brain_v2/status_banner.py`
- Test: `tests/karn/sanctum/widgets/tamiyo_brain_v2/test_status_banner.py`

**Step 1: Write the failing test**

```python
# tests/karn/sanctum/widgets/tamiyo_brain_v2/test_status_banner.py
def test_status_banner_shows_memory_percentage():
    """Status banner should display memory as percentage, not absolute."""
    from esper.karn.sanctum.schema import SanctumSnapshot, TamiyoState, InfrastructureMetrics
    from esper.karn.sanctum.widgets.tamiyo_brain_v2.status_banner import StatusBanner

    snapshot = SanctumSnapshot()
    snapshot.tamiyo = TamiyoState(ppo_data_received=True)
    snapshot.tamiyo.infrastructure = InfrastructureMetrics(
        cuda_memory_allocated_gb=4.2,
        cuda_memory_reserved_gb=8.0,
    )
    snapshot.current_batch = 60

    banner = StatusBanner()
    banner._snapshot = snapshot
    content = banner._render_content()

    content_str = str(content)
    # Should show percentage (52%), not absolute values
    assert "52%" in content_str or "53%" in content_str  # Allow rounding
    # Should NOT show absolute GB values in banner
    assert "4.2/8.0" not in content_str


def test_status_banner_compile_in_title():
    """Compile indicator should be in border title, not banner content."""
    from esper.karn.sanctum.schema import SanctumSnapshot, TamiyoState, InfrastructureMetrics
    from esper.karn.sanctum.widgets.tamiyo_brain_v2.status_banner import StatusBanner

    snapshot = SanctumSnapshot()
    snapshot.tamiyo = TamiyoState(ppo_data_received=True)
    snapshot.tamiyo.infrastructure = InfrastructureMetrics(compile_enabled=True)
    snapshot.current_batch = 60

    banner = StatusBanner()
    banner._snapshot = snapshot
    banner._update_status_classes()

    # Compile indicator should be in border_title
    assert "compiled" in banner.border_title.lower() or "[c]" in banner.border_title.lower()
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/widgets/tamiyo_brain_v2/test_status_banner.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Update `StatusBanner` class:

```python
def _update_status_classes(self) -> None:
    """Update CSS classes and border title based on current status."""
    status, _, _ = self._get_overall_status()

    # Remove all status classes
    self.remove_class("status-ok", "status-warning", "status-critical", "status-warmup")
    self.add_class(f"status-{status}")

    # Update border title with compile status (per UX review)
    # Note: compile_healthy removed (not accessible via API) - just show compile_enabled
    # Compile issues will surface in throughput metrics (fps, step_time_ms)
    if self._snapshot and self._snapshot.tamiyo.infrastructure.compile_enabled:
        self.border_title = "TAMIYO [compiled]"
    else:
        self.border_title = "TAMIYO"

    # ... rest of existing code
```

In `_append_metrics`, add memory as percentage (at end of banner):

```python
        # Memory as percentage (per UX review - more scannable than absolute)
        mem_pct = self._snapshot.tamiyo.infrastructure.memory_usage_percent
        if mem_pct > 0:
            if mem_pct > 90:
                mem_style = "red bold"
            elif mem_pct > 75:
                mem_style = "yellow"
            else:
                mem_style = "dim"
            banner.append(f"  [Mem:{mem_pct:.0f}%]", style=mem_style)
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/widgets/tamiyo_brain_v2/test_status_banner.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/widgets/tamiyo_brain_v2/status_banner.py tests/karn/sanctum/widgets/tamiyo_brain_v2/test_status_banner.py
git commit -m "feat(status-banner): add memory percentage and compile in title

Display memory as [Mem:53%] per UX review (more scannable than absolute).
Move compile indicator to border title per UX review (session metadata).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 6.2: Add Gradient Flow Footer to HeadsPanel

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/tamiyo_brain_v2/heads_grid.py`
- Test: `tests/karn/sanctum/widgets/tamiyo_brain_v2/test_heads_grid.py`

**Step 1: Write the failing test**

```python
# tests/karn/sanctum/widgets/tamiyo_brain_v2/test_heads_grid.py
def test_heads_panel_shows_gradient_flow_footer():
    """HeadsPanel should show gradient flow metrics as footer row."""
    from esper.karn.sanctum.schema import (
        SanctumSnapshot, TamiyoState, GradientQualityMetrics
    )
    from esper.karn.sanctum.widgets.tamiyo_brain_v2.heads_grid import HeadsPanel

    snapshot = SanctumSnapshot()
    snapshot.tamiyo = TamiyoState()
    snapshot.tamiyo.gradient_quality = GradientQualityMetrics(
        gradient_cv=0.42,
        clip_fraction_positive=0.10,
        clip_fraction_negative=0.05,
    )
    # Set existing dead/exploding layers (already tracked)
    snapshot.tamiyo.dead_layers = 0
    snapshot.tamiyo.exploding_layers = 0

    panel = HeadsPanel()
    panel.update_snapshot(snapshot)
    content = panel.render()

    content_str = str(content)
    # Should show gradient CV
    assert "CV" in content_str or "0.42" in content_str
    # Should show directional clip
    assert "↑" in content_str or "↓" in content_str or "10%" in content_str
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/widgets/tamiyo_brain_v2/test_heads_grid.py::test_heads_panel_shows_gradient_flow_footer -v`
Expected: FAIL

**Step 3: Write minimal implementation**

In `HeadsPanel._render_heads()`, add footer row after the existing state row:

```python
        # Row 6: Gradient Flow Footer (per UX review - keeps bars, adds flow as footer)
        result.append("\n")
        result.append("─ Flow: ", style="dim")

        # Gradient CV with status
        cv = self._snapshot.tamiyo.gradient_quality.gradient_cv
        cv_status = "stable" if cv < 0.5 else ("warn" if cv < 2.0 else "BAD")
        cv_style = "green" if cv < 0.5 else ("yellow" if cv < 2.0 else "red")
        result.append(f"CV:{cv:.2f} ", style=cv_style)
        result.append(f"{cv_status}   ", style="dim")

        # Dead/Exploding layers (use existing TamiyoState fields)
        dead = self._snapshot.tamiyo.dead_layers
        exploding = self._snapshot.tamiyo.exploding_layers
        total = 12  # TOTAL_LAYERS constant
        layers_style = "green" if (dead == 0 and exploding == 0) else "red"
        result.append(f"Dead:{dead}/{total}   Exploding:{exploding}/{total}   ", style=layers_style)

        # Directional clip (asymmetry indicator)
        clip_pos = self._snapshot.tamiyo.gradient_quality.clip_fraction_positive
        clip_neg = self._snapshot.tamiyo.gradient_quality.clip_fraction_negative
        result.append(f"Clip:↑{clip_pos:.0%}/↓{clip_neg:.0%}", style="dim")

        return result
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/widgets/tamiyo_brain_v2/test_heads_grid.py::test_heads_panel_shows_gradient_flow_footer -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/widgets/tamiyo_brain_v2/heads_grid.py tests/karn/sanctum/widgets/tamiyo_brain_v2/test_heads_grid.py
git commit -m "feat(heads-grid): add gradient flow footer row

Add footer showing CV, dead/exploding layers, and directional clip.
Keeps bar visualizations per UX review (patterns > numbers).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 6.3: Add Directional Clip to PPO Health Panel

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/tamiyo_brain_v2/ppo_health.py`
- Test: `tests/karn/sanctum/widgets/tamiyo_brain_v2/test_ppo_health.py`

**Step 1: Write the failing test**

```python
# tests/karn/sanctum/widgets/tamiyo_brain_v2/test_ppo_health.py
def test_ppo_health_shows_directional_clip():
    """PPO Health panel should show clip↑/↓ breakdown."""
    from esper.karn.sanctum.schema import (
        SanctumSnapshot, TamiyoState, GradientQualityMetrics
    )
    from esper.karn.sanctum.widgets.tamiyo_brain_v2.ppo_health import PPOHealthPanel

    snapshot = SanctumSnapshot()
    snapshot.tamiyo = TamiyoState(clip_fraction=0.15)
    snapshot.tamiyo.gradient_quality = GradientQualityMetrics(
        clip_fraction_positive=0.10,
        clip_fraction_negative=0.05,
    )
    snapshot.current_batch = 60

    panel = PPOHealthPanel()
    panel._snapshot = snapshot
    content = panel.render()

    content_str = str(content)
    # Should show directional breakdown with arrows
    assert "↑" in content_str or "↓" in content_str
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/widgets/tamiyo_brain_v2/test_ppo_health.py::test_ppo_health_shows_directional_clip -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Modify the Clip Frac row in `_render_gauges` to show directional breakdown:

```python
        # Row 3: Clip Fraction (with directional breakdown per UX review)
        result.append(self._render_gauge_row(
            label="Clip Frac",
            value=tamiyo.clip_fraction,
            min_val=0.0,
            max_val=0.5,
            status=self._get_clip_status(tamiyo.clip_fraction),
            is_warmup=is_warmup,
        ))
        # Add directional breakdown with arrows (per UX review)
        clip_pos = tamiyo.gradient_quality.clip_fraction_positive
        clip_neg = tamiyo.gradient_quality.clip_fraction_negative
        if clip_pos > 0 or clip_neg > 0:
            result.append(f" (↑{clip_pos:.0%} ↓{clip_neg:.0%})", style="dim")
        result.append("\n")
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/widgets/tamiyo_brain_v2/test_ppo_health.py::test_ppo_health_shows_directional_clip -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/widgets/tamiyo_brain_v2/ppo_health.py tests/karn/sanctum/widgets/tamiyo_brain_v2/test_ppo_health.py
git commit -m "feat(ppo-health): show directional clip fraction with arrows

Display (↑10% ↓5%) after clip fraction gauge per UX review.
Arrows match semantic meaning: ↑=probability increases capped.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Summary

This plan (v5) addresses all specialist feedback from v1-v4 reviews:

**Schema (Phase 1):**
- Uses nested dataclasses (`InfrastructureMetrics`, `GradientQualityMetrics`) to prevent bloat
- Removes `param_update_magnitude` (PyTorch expert: use existing `grad_norm`)
- Removes `minibatch_gradient_variance` (PyTorch expert: not applicable to recurrent PPO)
- Uses `gradient_cv` instead of inverted SNR (DRL expert)
- Removes `compile_healthy` (PyTorch expert v4: not accessible via API) - just uses `compile_enabled`
- Adds `cuda_memory_peak_gb` and `cuda_memory_fragmentation` (PyTorch expert)

**Telemetry (Phases 2-5):**
- Complete pipeline: PPO compute → emit → aggregate → schema (was missing in v1)
- Uses **DIRECT ACCESS** per CLAUDE.md - metrics always populated (code reviewer v4/v5)
- Task 3.3: `collect_cuda_memory_metrics()` returns ALL fields even when no CUDA (code reviewer v5)
- Task 3.4: Uses direct `cuda_metrics["field"]` not `.get()` (code reviewer v5)
- Throttles expensive metrics (every 10 batches) but still populates fields
- Uses `self.train_steps` not `self._update_count` (PyTorch expert v4)
- TrainingStartedPayload tests include all required fields (code reviewer v3)

**Widgets (Phase 6):**
- Keeps HeadsPanel bars, adds gradient flow as footer row (UX specialist)
- Moves compile indicator to border title (UX specialist)
- Displays memory as percentage `[Mem:53%]` (UX specialist)
- Uses arrows `↑↓` for directional clip (UX specialist)

**Total:** 16 tasks across 6 phases with TDD methodology.
