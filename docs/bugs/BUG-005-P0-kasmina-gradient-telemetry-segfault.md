# BUG-005: channels_last + isolate_gradients segfault

- **Title:** Segfault during backward() with channels_last and isolate_gradients=True
- **Context:** CNNHost (channels_last default) with any seed stage using isolate_gradients=True
- **Impact:** Blocker — process crashes (exit 139) during backward(), aborting training
- **Environment:** HEAD @ workspace, Python 3.11+, CPU or CUDA
- **Status:** FIXED (2025-12-17)

## Root Cause Analysis

**Original hypothesis was WRONG.** The crash was not in `capture_gradient_telemetry()` or `torch._foreach_norm`.

**Actual root cause:** PyTorch segfaults during `backward()` when:
1. Tensors are in `channels_last` memory format (non-contiguous)
2. Combined with `detach()` in the autograd graph
3. During backward pass through the detached branch

The bug affects BOTH:
- TRAINING stage (STE path)
- BLENDING stage (lerp path)

**Why the bug report was misleading:** The segfault happens during `backward()`, not in `capture_gradient_telemetry()`. Since `capture_gradient_telemetry()` was called immediately after `backward()`, it appeared to be the crash location.

## Fix

In `SeedSlot.forward()` (`src/esper/kasmina/slot.py`), make host_features contiguous BEFORE the detach when isolate_gradients=True:

```python
if self.isolate_gradients and not host_features.is_contiguous():
    host_features = host_features.contiguous()

seed_input = host_features.detach() if self.isolate_gradients else host_features
```

This ensures the entire computation and autograd graph use contiguous tensors, avoiding the PyTorch bug.

## Reproduction (Updated)

```bash
PYTHONPATH=src python - <<'PY'
import torch
from esper.kasmina.host import CNNHost, MorphogeneticModel
from esper.leyline import SeedStage

x = torch.randn(4, 3, 32, 32)
labels = torch.randint(0, 10, (4,))
model = MorphogeneticModel(CNNHost(num_classes=10), device="cpu", slots=["r0c1"])
model.germinate_seed("norm", "seed-1", slot="r0c1")
slot = model.seed_slots["r0c1"]
slot.state.stage = SeedStage.TRAINING
slot.isolate_gradients = True

out = model(x)
loss = torch.nn.functional.cross_entropy(out, labels)
loss.backward()  # <-- This was the actual crash location
slot.capture_gradient_telemetry()
print("done - ratio:", slot.state.metrics.seed_gradient_norm_ratio)
PY
```

## Validation

Regression tests added in `tests/kasmina/test_bug005_channels_last_segfault.py`:
- `test_training_stage_with_channels_last_no_crash`
- `test_training_stage_with_channels_last_gradient_telemetry`
- `test_training_stage_explicit_channels_last`
- `test_training_stage_contiguous_format_still_works`
- `test_blending_stage_channels_last_with_fix`
- `test_training_without_isolate_gradients_no_issue`

All 6 tests pass.

## Links

- Fix: `src/esper/kasmina/slot.py` (SeedSlot.forward, lines 1178-1191)
- Note: `src/esper/kasmina/isolation.py` (ste_forward docstring updated)
- Tests: `tests/kasmina/test_bug005_channels_last_segfault.py`
