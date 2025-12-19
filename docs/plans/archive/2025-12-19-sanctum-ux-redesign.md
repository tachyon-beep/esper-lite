# Sanctum UX Redesign

**Date:** 2025-12-19
**Status:** Design Complete
**Goal:** Make Sanctum work as both a teaching tool and operational telemetry

---

## CRITICAL: Do Not Touch

> **EnvOverview and Scoreboard are SACRED. Do not modify them.**
>
> EnvOverview is "fantastic" — it shows the seed story beautifully.
> Scoreboard "will be great when it works" — leave it alone.
>
> This redesign is about everything ELSE.

---

## Design Goals

Sanctum must serve two purposes:

1. **Teaching tool** — Explain Esper concepts to newcomers watching over your shoulder
2. **Operational telemetry** — Diagnose training issues in real-time

The current layout buries the most important teaching panel (TamiyoBrain) while giving space to low-value panels (SystemResources, TrainingHealth).

---

## Layout Restructure

### Current Layout (Problems)

```
┌──────────────── RUN HEADER ────────────────────┐
│ EnvOverview (65%)      │ Scoreboard (35%)      │
├────────────────────────────────────────────────┤
│            TamiyoBrain (full width)            │  ← cramped, underutilized
├────────────────────────────────────────────────┤
│ EventLog (65%)         │ SysRes + TrainHealth  │  ← TrainHealth redundant
└────────────────────────────────────────────────┘
```

### New Layout

```
┌────────── RUN HEADER ───────────────────────────────── OK ┐
├───────────────────────────────────────────────────────────┤
│                                                           │
│   EnvOverview (65%)            │    Scoreboard (35%)      │
│   (DO NOT TOUCH)               │    (DO NOT TOUCH)        │
│                        (TOP ~55%)                         │
│                                                           │
├────────────────────────────────┬──────────────────────────┤
│                                │                          │
│   EventLog                     │   TamiyoBrain            │
│   "system velocity"            │   "learning + decisions" │
│         (BOTTOM LEFT ~50%)     │   (BOTTOM RIGHT ~50%)    │
│                                │                          │
└────────────────────────────────┴──────────────────────────┘
```

### What Moves Where

| Old Panel | New Location |
|-----------|--------------|
| SystemResources | Tiny alarm indicator in header corner |
| TrainingHealth | Merged into TamiyoBrain (was redundant) |
| TamiyoBrain | Expands to right half of bottom section |

---

## Header Changes

### System Resource Indicator

**Exception-based monitoring** — Don't clutter with numbers when healthy.

```
# Everything fine:
┌─────────── RUN STATUS ─────────────────────────────────── OK ┐

# cuda:0 memory-bound:
┌─────────── RUN STATUS ──────────────────── [cuda:0] RAM 92% ┐

# Multiple issues:
┌─────────── RUN STATUS ───────── [cuda:0] RAM │ [cuda:2] PROC ┐
```

**Threshold:** 90% for both RAM and PROC. Either has capacity or it doesn't.

### Header Row 2 (already updated)

```
● Live  |  Thread ✓  |  312 events (6.6/s)  |  4 OK  |  T:8 B:2 F:12  |  cifar10
```

---

## TamiyoBrain Redesign

The current TamiyoBrain is a black hole. The new design answers:
- "What is Tamiyo doing?"
- "Is she learning?"
- "What did she just decide and why?"

### Layout

```
┌─────────────────── TAMIYO ────────────────────────────────┐
│                                                           │
│  ┌─ LEARNING VITALS ───────────────────────────────────┐  │
│  │                                                     │  │
│  │  Actions: [▓▓▓▓▓▓▓▓░░░░░░▒▒▒▒▒▒▒▒░░░░]             │  │
│  │           Germinate 35%  Wait 25%  Blend 40%        │  │
│  │                                                     │  │
│  │  Entropy:   [████████░░░░] 0.42  "Getting decisive" │  │
│  │  Value Loss:[██░░░░░░░░░░] 0.08  "Learning well"    │  │
│  │  Advantage: [██████████░░] +0.31 "Choices working"  │  │
│  │                                                     │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                           │
│  ┌─ LAST DECISION (2.3s ago) ──────────────────────────┐  │
│  │ SAW:  r0c0: Training 12% │ r0c1: Empty │ Host: 67%  │  │
│  │ CHOSE: Germinate r0c1 (73%)                         │  │
│  │ EXPECTED: +0.42  →  GOT: +0.38 ✓                    │  │
│  │ Also: Wait (15%), Blend r0c0 (12%)                  │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

### Learning Vitals (always visible)

| Metric | Display | Question it Answers |
|--------|---------|---------------------|
| Action Distribution | Horizontal stacked bar | "What is she doing?" |
| Entropy | Gauge with label | "Is she becoming decisive?" |
| Value Loss | Gauge with label | "Is she learning to predict?" |
| Advantage Mean | Gauge with label | "Are her choices working?" |

### Last Decision Snapshot (sampled, ~1/minute or on-demand)

Priority order:
1. **What she saw** — The observation (slot states, host accuracy)
2. **What she chose** — Action + confidence % (e.g., "Germinate r0c1 (73%)")
3. **What she got** — Actual reward received
4. **What she expected** — Value estimate (compare to actual)
5. **Alternatives** — Runner-up actions considered

**Diagnostic value:** "She saw THIS and still chose THAT? That's not right!"

### Future Enhancement: Decision Carousel

Three decision slots that auto-rotate (oldest ages out every 30s, newest appears at bottom). Users can **pin** interesting decisions to prevent aging out.

```
┌─ DECISIONS ────────────────────────────────────────────────┐
│ 📌 SAW: Host 45%, r0c0 stalled → CULL r0c0 (91%)          │  ← pinned
│    SAW: Host 67%, slots full → WAIT (64%)                  │  ← ages out
│    SAW: Host 71%, r0c1 empty → GERMINATE r0c1 (73%)        │  ← newest
└────────────────────────────────────────────────────────────┘
```

---

## EventLog Enhancements

EventLog keeps its prominent position (left half of bottom) and must **fill the space**.

### Design

```
┌─ EVENTS ───────────────────────────────────────────────┐
│ ─── Episode 5 ─────────────────────────────────────── │
│ 12:34:56 (2s)  🌱 GERMINATED seed_0a3f in r0c1        │
│ 12:34:51 (7s)  📊 Tamiyo action: WAIT (confidence 64%)│
│ 12:34:48 (10s) ✅ seed_0b2e FOSSILIZED +3.2% contrib  │
│ 12:34:45 (13s) ⚠️  seed_0c1d CULLED (negative contrib)│
│ ─── Episode 4 ─────────────────────────────────────── │
│ 12:33:12       🏆 Episode complete: 78.4% accuracy    │
│ ...                                                    │
└────────────────────────────────────────────────────────┘
```

### Features

1. **Full-width rows** — Use all available horizontal space
2. **Color-coded by type:**
   - Seed lifecycle (green)
   - Tamiyo actions (cyan)
   - Warnings (yellow)
   - Errors (red)
3. **Timestamp + relative time** — "12:34:56 (2s ago)"
4. **Episode grouping** — Visual separators between episodes

**Purpose:** Show "system velocity" — the stream of events demonstrates the system is alive and moving, critical for demos and teaching.

---

## Panels Deleted

| Panel | Reason | Where it Went |
|-------|--------|---------------|
| SystemResources | Rarely actionable, clutters UI | 90% threshold alarm in header |
| TrainingHealth | Redundant with TamiyoBrain vitals | Merged into TamiyoBrain |

---

## Implementation Notes

### Phase 1: Layout Restructure
1. Modify `app.py` layout (move panels, adjust sizing)
2. Delete SystemResources widget
3. Delete TrainingHealth widget
4. Add system alarm indicator to RunHeader

### Phase 2: TamiyoBrain Redesign
1. Clear existing TamiyoBrain implementation
2. Implement Learning Vitals section (gauges + action bar)
3. Implement Decision Snapshot section
4. Wire up telemetry data sources

### Phase 3: EventLog Enhancement
1. Redesign row layout for full width
2. Add color coding by event type
3. Add episode grouping with separators
4. Ensure it fills vertical space

### Future: Decision Carousel
- Three-slot rotating display
- Pin functionality
- 30-second auto-rotation

---

## Summary

| Panel | Purpose | Status |
|-------|---------|--------|
| EnvOverview | Seed story | **DO NOT TOUCH** |
| Scoreboard | Hall of fame | **DO NOT TOUCH** |
| Header | Run status + alarms | Minor update (add OK/alarm) |
| EventLog | System velocity | Enhance (fill space, color, grouping) |
| TamiyoBrain | Learning + decisions | **Complete redesign** |
| SystemResources | — | **Deleted** (→ header alarm) |
| TrainingHealth | — | **Deleted** (→ TamiyoBrain) |
