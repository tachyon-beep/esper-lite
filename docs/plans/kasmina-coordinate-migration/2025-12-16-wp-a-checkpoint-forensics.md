# Work Package A: Checkpoint Forensics

**Status:** Ready for implementation
**Priority:** High (blocks M4)
**Effort:** ~2-3 hours
**Dependencies:** None (can start immediately)

---

## Goal

Understand exactly what types are serialized in Esper checkpoints today, before redesigning for PyTorch 2.9 `weights_only=True` compatibility.

## Why This De-risks M4

- We don't know what types are actually in checkpoints until we inspect them
- PyTorch 2.9's `weights_only=True` rejection gives cryptic errors
- Knowing the exact gap makes the M4 fix mechanical instead of exploratory

## Background

PyTorch 2.9 defaults `torch.load(..., weights_only=True)`, which rejects:
- Arbitrary Python classes (dataclasses, custom objects)
- datetime objects
- deque collections
- Enum instances
- nn.Module subclasses in extra_state

Current checkpoint paths:
- `src/esper/simic/ppo.py:606` — `torch.save(save_dict, path)`
- `src/esper/simic/ppo.py:611` — `torch.load(path, weights_only=False)`
- `src/esper/simic/vectorized.py:997` — `torch.load(resume_path, weights_only=False)`
- `src/esper/kasmina/slot.py:1428-1441` — `get_extra_state()` / `set_extra_state()`

---

## Tasks

### A.1 Create checkpoint audit diagnostic tool

**File:** `scripts/checkpoint_audit.py`

```python
"""Audit checkpoint contents for PyTorch 2.9 compatibility."""

import sys
from pathlib import Path
from datetime import datetime
from collections import deque
from enum import Enum
from dataclasses import is_dataclass

import torch

SAFE_TYPES = (
    int, float, str, bool, type(None),
    list, tuple, dict,
    torch.Tensor,
)

def audit_value(value, path="root", issues=None):
    """Recursively audit a value for non-primitive types."""
    if issues is None:
        issues = []

    value_type = type(value)

    # Check for known problematic types
    if isinstance(value, datetime):
        issues.append(f"{path}: datetime object")
    elif isinstance(value, deque):
        issues.append(f"{path}: deque (use list instead)")
    elif isinstance(value, Enum):
        issues.append(f"{path}: Enum {value_type.__name__}.{value.name}")
    elif is_dataclass(value) and not isinstance(value, type):
        issues.append(f"{path}: dataclass {value_type.__name__}")
    elif isinstance(value, torch.nn.Module):
        issues.append(f"{path}: nn.Module {value_type.__name__}")
    elif isinstance(value, dict):
        for k, v in value.items():
            audit_value(v, f"{path}[{k!r}]", issues)
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            audit_value(v, f"{path}[{i}]", issues)
    elif not isinstance(value, SAFE_TYPES):
        issues.append(f"{path}: unknown type {value_type.__name__}")

    return issues

def main(checkpoint_path: str):
    print(f"Auditing: {checkpoint_path}")
    print("=" * 60)

    # First, try loading with weights_only=True to see what fails
    print("\n1. Testing weights_only=True...")
    try:
        torch.load(checkpoint_path, weights_only=True)
        print("   SUCCESS - checkpoint is already compatible!")
        return
    except Exception as e:
        print(f"   FAILED: {e}")

    # Load with weights_only=False and audit
    print("\n2. Loading with weights_only=False and auditing...")
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    issues = audit_value(checkpoint)

    if issues:
        print(f"\n3. Found {len(issues)} compatibility issues:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("\n3. No issues found (but weights_only=True still failed?)")

    print("\n4. Top-level keys:")
    if isinstance(checkpoint, dict):
        for key in checkpoint.keys():
            print(f"   - {key}: {type(checkpoint[key]).__name__}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/checkpoint_audit.py <checkpoint.pt>")
        sys.exit(1)
    main(sys.argv[1])
```

### A.2 Test with actual checkpoints

- [ ] Run audit tool on any existing `.pt` checkpoint files
- [ ] If no checkpoints exist, create one by running:
  ```bash
  PYTHONPATH=src uv run python -m esper.scripts.train ppo \
      --preset cifar10 --episodes 1 --save /tmp/test_checkpoint.pt
  ```
- [ ] Capture full output of audit tool

### A.3 Test round-trip with weights_only=True

- [ ] Attempt `torch.load(path)` (default weights_only=True in PyTorch 2.9)
- [ ] Document exact error message
- [ ] Identify which specific types cause rejection

### A.4 Document findings

**File:** `docs/plans/checkpoint-audit-results.md`

Document:
- List of all non-primitive types found
- Which checkpoint locations contain them
- Exact PyTorch error messages
- Mapping: current type → required primitive representation

---

## Acceptance Criteria

- [ ] Audit tool exists and runs
- [ ] At least one checkpoint audited
- [ ] `docs/plans/checkpoint-audit-results.md` contains:
  - Complete inventory of problematic types
  - Error messages from `weights_only=True` attempt
  - Clear scope for M4 implementation

## Outputs

This work package produces:
1. `scripts/checkpoint_audit.py` — reusable diagnostic tool
2. `docs/plans/checkpoint-audit-results.md` — findings document

These outputs directly inform M4 implementation scope.
