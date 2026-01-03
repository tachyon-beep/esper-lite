# Defensive Pattern Fixes Proposal

## Overview

This document proposes fixes for ~25 inappropriate defensive programming patterns identified by the new `lint_defensive_patterns.py` linter. These patterns violate CLAUDE.md's prohibition on bug-hiding code.

## Category 1: Dynamic TUI Field Access (9 entries)

**File:** `src/esper/karn/sanctum/widgets/tamiyo_brain/action_heads_panel.py`

**Problem:** Uses `getattr(tamiyo, field_name, default)` to access fields by dynamically constructed string names. If field names change, code silently uses the default instead of failing.

**Current Code (lines 317, 339, 353-354, 374, 389):**
```python
entropy: float = getattr(tamiyo, ent_field)
grad: float = getattr(tamiyo, grad_field)
grad_prev: float = getattr(tamiyo, f"{grad_field}_prev")
ratio: float = getattr(tamiyo, ratio_field, 1.0)  # WORST: silently returns 1.0!
```

**Proposed Fix:** Create a typed accessor method on `TamiyoSnapshot` that fails loudly:

```python
# In src/esper/karn/sanctum/schema.py - add to TamiyoSnapshot:

def get_head_metrics(self, head_key: str) -> tuple[float, float, float, float]:
    """Get (entropy, gradient, gradient_prev, ratio_max) for a head.

    Raises KeyError if head_key is invalid - fails loudly instead of silently.
    """
    VALID_HEADS = {"op", "slot", "blueprint", "style", "tempo", "alpha_target", "alpha_speed", "alpha_curve"}
    if head_key not in VALID_HEADS:
        raise KeyError(f"Invalid head key: {head_key}. Valid: {VALID_HEADS}")

    entropy = getattr(self, f"head_{head_key}_entropy")
    grad = getattr(self, f"head_{head_key}_grad_norm")
    grad_prev = getattr(self, f"head_{head_key}_grad_norm_prev")
    ratio = getattr(self, f"head_{head_key}_ratio_max")
    return entropy, grad, grad_prev, ratio
```

Then update `action_heads_panel.py` to use this accessor:
```python
entropy, grad, grad_prev, ratio = tamiyo.get_head_metrics(head_key)
```

**Benefit:** If a field is missing or renamed, we get a clear `AttributeError` with the field name, not silent 1.0 defaults.

---

## Category 2: Enum Duck-Typing with hasattr (6 entries)

**Files:**
- `src/esper/karn/collector.py` (2 entries)
- `src/esper/karn/sanctum/aggregator.py` (1 entry)
- `src/esper/karn/serialization.py` (2 entries)
- `src/esper/karn/store.py` (2 entries)

**Problem:** Uses `hasattr(event_type, "name")` to check if something is an Enum, when we *know* the type should be an Enum. This is duck-typing when we have type information.

**Current Code:**
```python
if hasattr(event.event_type, "name"):
    event_type_str = event.event_type.name
```

**Proposed Fix:** Use explicit `isinstance` check with `Enum`:

```python
from enum import Enum

if isinstance(event.event_type, Enum):
    event_type_str = event.event_type.name
else:
    event_type_str = str(event.event_type)
```

**Note:** This moves from STRICT tier (hasattr) to AUDIT tier (isinstance), which is allowed by default. The isinstance here is legitimate typed payload discrimination.

---

## Category 3: Optional Interface Duck-Typing (4 entries)

**Files:**
- `src/esper/tolaria/governor.py` (2 entries) - `hasattr(self.model, 'seed_slots')`
- `src/esper/nissa/output.py` (2 entries) - `hasattr(self, '_file')`

### 3a: Governor seed_slots check

**Problem:** Uses `hasattr(model, 'seed_slots')` to check if model supports seed operations.

**Proposed Fix:** Create a Protocol in leyline:

```python
# In src/esper/leyline/protocols.py (or existing protocols file):
from typing import Protocol, runtime_checkable

@runtime_checkable
class SeedAwareModel(Protocol):
    """Protocol for models that support seed slot operations."""
    seed_slots: dict  # or more specific type
```

Then in governor.py:
```python
from esper.leyline import SeedAwareModel

if isinstance(self.model, SeedAwareModel):
    # ... use seed_slots
```

### 3b: FileOutput destructor guards

**Current Code:**
```python
def close(self):
    if hasattr(self, '_file') and not self._file.closed:
        ...

def __del__(self):
    if hasattr(self, '_file') and not self._file.closed:
        ...
```

**Assessment:** These are LEGITIMATE. Destructors can be called on partially initialized objects (if `__init__` fails). This is a Python best practice, not a bug-hiding pattern.

**Proposed:** Keep these whitelisted. The hasattr guards against `__init__` failure, not missing fields.

---

## Category 4: Protocol Validation via hasattr (2 entries)

**File:** `src/esper/tamiyo/policy/registry.py`

**Problem:** Uses `hasattr(cls, method)` to validate that a class implements required methods.

**Current Code:**
```python
missing_methods = [m for m in required_methods if not hasattr(cls, m)]
missing_props = [p for p in required_properties if not hasattr(cls, p)]
```

**Proposed Fix:** Use `@runtime_checkable Protocol` and `isinstance`:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class TamiyoPolicyProtocol(Protocol):
    """Protocol that all Tamiyo policies must implement."""

    def get_action(self, obs: Tensor, ...) -> ...: ...
    def get_value(self, obs: Tensor, ...) -> ...: ...
    # ... other required methods

    @property
    def device(self) -> torch.device: ...

# In decorator:
if not isinstance(cls, TamiyoPolicyProtocol):
    raise TypeError(f"{cls.__name__} does not implement TamiyoPolicyProtocol")
```

---

## Category 5: Dynamic Action Lookup (1 entry)

**File:** `src/esper/tamiyo/heuristic.py`

**Problem:** Uses `getattr(Action, f"{GERMINATE_PREFIX}_{slot}")` to dynamically construct action names.

**Current Code:**
```python
germinate_action = getattr(Action, f"{GERMINATE_PREFIX}_{slot.name}")
```

**Proposed Fix:** Use a mapping dict instead:

```python
# Define explicit mapping
SLOT_TO_GERMINATE_ACTION: dict[SlotPosition, Action] = {
    SlotPosition.R0C0: Action.GERMINATE_R0C0,
    SlotPosition.R0C1: Action.GERMINATE_R0C1,
    # ... etc
}

# Use mapping (fails loudly with KeyError if slot not in mapping)
germinate_action = SLOT_TO_GERMINATE_ACTION[slot]
```

---

## Category 6: to_dict() Duck-Typing (3 entries)

**Files:**
- `src/esper/karn/serialization.py` (1 entry)
- `src/esper/nissa/output.py` (1 entry)
- `src/esper/nissa/output.py:_emit_summary` (1 entry)

**Problem:** Uses `hasattr(obj, "to_dict")` to check for serialization method.

**Proposed Fix:** Create a Serializable Protocol in leyline:

```python
# In src/esper/leyline/protocols.py:
from typing import Protocol, runtime_checkable

@runtime_checkable
class Serializable(Protocol):
    """Protocol for objects that can serialize to dict."""
    def to_dict(self) -> dict: ...
```

Then use isinstance:
```python
if isinstance(obj, Serializable):
    return obj.to_dict()
```

---

## Summary of Changes

| Category | Files | Pattern | Fix | Remove from Whitelist? |
|----------|-------|---------|-----|------------------------|
| Dynamic TUI fields | action_heads_panel.py | getattr with default | Typed accessor method | Yes (9 entries) |
| Enum duck-typing | collector, aggregator, serialization, store | hasattr(.name) | isinstance(x, Enum) | Yes, moves to AUDIT tier (6 entries) |
| seed_slots check | governor.py | hasattr(model, 'seed_slots') | Protocol + isinstance | Yes (2 entries) |
| Destructor guards | output.py | hasattr(self, '_file') | **KEEP** - legitimate | No (2 entries) |
| Protocol validation | registry.py | hasattr for method check | runtime_checkable Protocol | Yes (2 entries) |
| Dynamic action | heuristic.py | getattr(Action, f"...") | Explicit mapping dict | Yes (1 entry) |
| to_dict() check | serialization, output | hasattr(obj, "to_dict") | Serializable Protocol | Yes (3 entries) |

**Total entries to remove after fixes: 23**
**Entries to keep (legitimate): 2 (destructor guards)**

---

## Implementation Order

1. Create Protocols in `leyline/protocols.py` (SeedAwareModel, Serializable, TamiyoPolicyProtocol)
2. Add `get_head_metrics()` accessor to TamiyoSnapshot
3. Create `SLOT_TO_GERMINATE_ACTION` mapping in heuristic.py
4. Update all call sites to use new patterns
5. Remove fixed entries from `defensive_patterns.yaml`
6. Verify lint passes

