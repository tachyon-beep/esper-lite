# Finding Ticket: Action Introspection Helpers Could Move to Leyline

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B9-DRL-07` |
| **Severity** | `P3` |
| **Status** | `open` |
| **Batch** | 9 |
| **Agent** | `drl` |
| **Domain** | `tamiyo/decisions` |
| **Assignee** | |
| **Created** | 2025-12-27 |
| **Updated** | 2025-12-27 |

---

## Location

| Field | Value |
|-------|-------|
| **File(s)** | `/home/john/esper-lite/src/esper/tamiyo/decisions.py` |
| **Line(s)** | `12-22` |
| **Function/Class** | `_is_germinate_action()`, `_get_blueprint_from_action()` |

---

## Summary

**One-line summary:** Action introspection helpers are module-private but logically belong in leyline.actions.

**Category:**
- [ ] Correctness bug
- [ ] Race condition / concurrency
- [ ] Memory leak / resource issue
- [ ] Performance bottleneck
- [ ] Numerical stability
- [ ] torch.compile compatibility
- [ ] Dead code / unwired functionality
- [x] API design / contract violation
- [ ] Test coverage gap
- [ ] Documentation / naming
- [ ] Defensive programming violation
- [ ] Legacy code policy violation

---

## Detailed Description

### What's Wrong

```python
# Lines 12-22
def _is_germinate_action(action_name: str) -> bool:
    """Check if action name indicates a germination action."""
    return action_name.startswith("GERMINATE_")

def _get_blueprint_from_action(action_name: str) -> str | None:
    """Extract blueprint ID from a germination action name."""
    if _is_germinate_action(action_name):
        return action_name[len("GERMINATE_"):]
    return None
```

These functions parse action enum names to detect germination and extract blueprints. They're currently:
1. Private (underscore prefix) to the decisions module
2. Only used by `TamiyoDecision.blueprint_id` property

However, action introspection is a cross-cutting concern that other modules might need:
- Logging/telemetry formatting
- Action space analysis
- Reward computation (for action-specific rewards)

### Impact

- **Code locality**: Action utilities scattered across modules
- **Discoverability**: Other modules may reimplement similar logic
- **Consistency**: No single source of truth for action parsing

---

## Recommended Fix

Move to `leyline.actions` for centralized action introspection:

```python
# leyline/actions.py
def is_germinate_action(action: ActionType | str) -> bool:
    """Check if action indicates germination."""
    name = action.name if hasattr(action, 'name') else str(action)
    return name.startswith("GERMINATE_")

def get_blueprint_from_action(action: ActionType | str) -> str | None:
    """Extract blueprint ID from germination action."""
    name = action.name if hasattr(action, 'name') else str(action)
    if is_germinate_action(name):
        return name[len("GERMINATE_"):]
    return None
```

Then in decisions.py:

```python
from esper.leyline.actions import is_germinate_action, get_blueprint_from_action

@property
def blueprint_id(self) -> str | None:
    return get_blueprint_from_action(self.action)
```

---

## Verification

### How to Verify the Fix

- [ ] Move functions to leyline.actions
- [ ] Update decisions.py imports
- [ ] Add tests in leyline test suite

---

## Related Findings

- B9-DRL-08: confidence field unused (related TamiyoDecision field)

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch9-drl.md`
**Section:** "D1 - Action introspection helpers should move to leyline.actions"
