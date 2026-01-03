# Telemetry Audit: ThreadDeathModal

**File:** `/home/john/esper-lite/src/esper/karn/sanctum/widgets/thread_death_modal.py`
**Purpose:** Displays a prominent, unmissable modal notification when the training thread stops unexpectedly, requiring operator attention.

---

## Telemetry Fields Consumed

### Source: None (Static Modal)

This widget does **not** consume any telemetry fields from `SanctumSnapshot`. It is a static notification modal with hardcoded content.

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| (none) | - | - | Modal displays static text only |

### Trigger Mechanism

The modal is triggered by the `SanctumApp` when it detects the training thread has died:

| Source | Field | Type | Trigger Condition |
|--------|-------|------|-------------------|
| `SanctumApp` | `_training_thread.is_alive()` | `bool` | `False` |
| `SanctumApp` | `_thread_death_shown` | `bool` | `False` (prevents duplicate modals) |

The app checks `training_thread_alive` during each poll cycle and pushes the modal screen when:
1. `thread_alive is False` (training thread has died)
2. `_thread_death_shown is False` (modal not already shown)

### Related SanctumSnapshot Field

While not consumed directly by the widget, the app populates:

| Field | Type | Description |
|-------|------|-------------|
| `SanctumSnapshot.training_thread_alive` | `bool \| None` | Debug field tracking thread status |

---

## Thresholds and Color Coding

| Metric | Threshold | Style | Meaning |
|--------|-----------|-------|---------|
| (none) | - | - | No conditional styling |

### Static Styling

| Element | CSS Class | Background/Color | Meaning |
|---------|-----------|------------------|---------|
| Modal overlay | `ThreadDeathModal` | `$error-darken-3 90%` | Error state backdrop |
| Container | `#death-container` | `$error-darken-2`, border `$error` | Error container |
| Title | `.death-title` | `$error` (bold) | Critical warning |
| Message | `.death-message` | `white` | Instruction text |
| Hint | `.death-hint` | `$text-muted` | Dismissal instructions |

---

## Rendering Logic

The widget renders a fixed modal overlay with three static text sections:

1. **Title:** "TRAINING THREAD DIED" with warning emoji and bold red styling
2. **Message:** Instructions to check the terminal for stack trace
3. **Hint:** Dismissal key hints (ESC, Q, Enter, or click)

### Dismissal Bindings

| Key | Action |
|-----|--------|
| `Escape` | Dismiss modal |
| `Q` | Dismiss modal |
| `Enter` | Dismiss modal |
| Click anywhere | Dismiss modal |

### Compose Structure

```
Container (#death-container)
  Static (.death-title)     -> "TRAINING THREAD DIED"
  Static (.death-message)   -> Crash notification + stack trace hint
  Static (.death-hint)      -> Dismissal instructions
```

---

## Data Flow Summary

```
Training Thread Dies
       |
       v
SanctumApp._poll_snapshot()
       |
       +-- thread_alive = _training_thread.is_alive()  # returns False
       |
       +-- if thread_alive is False and not _thread_death_shown:
       |       _thread_death_shown = True
       |       push_screen(ThreadDeathModal())  # No data passed
       v
ThreadDeathModal displays static content
```

---

## Notes

- This is a **static notification modal** - it contains no dynamic telemetry data
- The widget serves as a critical failure indicator requiring manual acknowledgment
- The `training_thread_alive` field in `SanctumSnapshot` is populated for debugging purposes but is not consumed by this widget
- The modal can only appear once per session (`_thread_death_shown` flag prevents duplicates)
