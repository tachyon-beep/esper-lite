# Plans

This folder contains all planning artifacts for the Esper project.

## Structure

| Folder | Purpose |
|--------|---------|
| `concepts/` | Early ideas and explorations (may never be implemented) |
| `planning/` | Active design workspaces for multi-phase efforts |
| `ready/` | Approved, scoped, implementation-ready plans |
| `completed/` | Historical record of executed work |
| `abandoned/` | Superseded, deferred, or cancelled plans |

## Key Files

| File | Purpose |
|------|---------|
| `PLAN_TEMPLATE.md` | Metadata schema for annotating plans (copy the YAML block to your plan) |

## Coordination

**All plans are tracked in `docs/coord/PLAN_TRACKER.md`.**

When creating, modifying, or obsoleting any plan:
1. Use the metadata block from `PLAN_TEMPLATE.md`
2. Update `docs/coord/PLAN_TRACKER.md` with the change

See `docs/coord/README.md` for full coordination workflow.

## Lifecycle

```
concepts/ ──► planning/ ──► ready/ ──► (execute) ──► completed/
                │              │
                └──────────────┴──────────────────► abandoned/
```

- **Concept → Planning**: When an idea is worth designing in detail
- **Planning → Ready**: When design is complete and scoped for implementation
- **Ready → Completed**: After successful execution
- **Any → Abandoned**: When superseded, deferred indefinitely, or cancelled
