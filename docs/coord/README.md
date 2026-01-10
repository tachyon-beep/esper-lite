# Plan Coordination

This folder provides meta-planning infrastructure for the Esper project. As the project scales, coordinating across multiple concurrent plans, concepts, and workspaces requires explicit tooling.

## Purpose

Plan coordination addresses three challenges:

1. **Visibility** — Understanding what work is planned, in progress, or blocked
2. **Prioritization** — Making informed decisions about sequencing and resource allocation
3. **Dependency Management** — Tracking relationships between plans to avoid deadlocks or wasted effort

## Contents

| File | Purpose |
|------|---------|
| `PLAN_TRACKER.md` | Master tracking document with prioritization, dependencies, and commentary |

> **Note:** The metadata template lives at `docs/plans/PLAN_TEMPLATE.md` alongside the plans themselves.

## How to Use

### When Creating a New Plan

1. Add the metadata block from `docs/plans/PLAN_TEMPLATE.md` to your document
2. Fill in urgency, complexity, risk, and dependencies
3. Add an entry to `PLAN_TRACKER.md` in the appropriate tier

### When Updating Plan Status

1. Update the `status_notes` and `percent_complete` in your plan
2. Update the corresponding entry in `PLAN_TRACKER.md`
3. Check if the status change unblocks other plans

### When Prioritizing Work

1. Consult `PLAN_TRACKER.md` for the current priority matrix
2. Check the dependency graph before starting new work
3. Focus on Tier 1 (Critical Path) items first

## Relationship to Other Plan Folders

```
docs/plans/
├── concepts/      ← Early ideas, explorations (may not happen)
├── planning/      ← Active design workspaces (multi-file)
├── ready/         ← Approved, implementation-ready (single-file)
├── completed/     ← Historical record of executed work
├── abandoned/     ← Superseded or cancelled plans
└── coord/         ← THIS FOLDER: meta-layer for tracking all of the above
```

## Maintenance

The tracker should be updated:
- When a plan changes status (concept → planning → ready → in-progress → completed)
- When dependencies are identified or resolved
- At the start of each major work session (quick sanity check)
- When new plans or concepts are created

## Questions This System Answers

- "What should I work on next?" → Check Tier 1 in `PLAN_TRACKER.md`
- "Is this plan blocked on something?" → Check `depends_on` field
- "What's the risk of starting this?" → Check risk level and notes
- "How much effort is this?" → Check complexity (S/M/L/XL)
- "Why was this abandoned?" → Check abandoned section or `docs/plans/abandoned/`
