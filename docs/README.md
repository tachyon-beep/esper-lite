# Docs Index

This folder is organized by intent. Put new docs in the most specific
section and keep the root lean.

Top-level files
- `docs/README.md`: this index and folder rules.

Core sections
- `docs/architecture/`: system-level architecture, diagrams, handovers.
- `docs/specifications/`: canonical module specs (start at `index.md`).
- `docs/plans/`: plans in `concepts/`, `ready/`, `completed/`, `abandoned/`.
- `docs/decisions/`: ADRs and major decision records.
- `docs/telemetry/`: telemetry panel and UI docs (start at `00-index.md`).
- `docs/manuals/`: setup guides and operational how-tos.
- `docs/optimization/`: performance analyses and optimization plans.

Research and review
- `docs/research/`: long-form research, with `archive/` for older items.
- `docs/findings/`: distilled findings and conclusions.
- `docs/analysis/`: audits and diagnostics, with `archive/` for older items.
- `docs/reviews/`: code and system reviews, with `archive/` for older items.
- `docs/results/`: experiment summaries and results.

Tracking and triage
- `docs/bugs/`: bug batches, tickets, and their status folders.

Misc
- `docs/misc/`: one-off artifacts that do not fit other sections.
  - `docs/misc/archive/`: historical snapshots and exports.

Conventions
- Use `YYYY-MM-DD-...` prefixes for time-bound docs.
- Prefer `index.md` or `00-index.md` as entry points for sections.
- Move superseded docs into the section's `archive/`.
- Avoid adding generated artifacts, telemetry dumps, or datasets to `docs/`.
