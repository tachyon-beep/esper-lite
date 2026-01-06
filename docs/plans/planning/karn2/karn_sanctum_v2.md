# Karn Sanctum V2 UX Notes (Tamiyo)

## Purpose
Capture UX improvements for Sanctum Tamiyo panels after the op-conditioned critic refactor, with an emphasis on truthful signals, fast scanability, and RL/PyTorch debugging clarity.

## Goals
- Make critic preference obviously tied to op-conditioned outputs (no ambiguity about validity or mask coverage).
- Improve first-pass scanning (F-pattern) with compact, high-signal summaries.
- Reduce duplicated information across panels without losing diagnostic value.
- Surface RL and PyTorch stability signals that are actionable in the moment.

## Principles
- Truthfulness over smoothness: missing data should be explicit, not inferred.
- Minimize cognitive load: show fewer, clearer rows that explain themselves.
- Keep panel boundaries stable: avoid moving key metrics between refreshes.
- Avoid color-only encoding: always include text or symbols for state.

## Implemented Updates (Jan 2026 pass)
- Decision cards swap every 5s, rotate across envs when possible, and label forced-choice heads as `[forced]` instead of `[collapsing]`.
- Torch Stability panel split out and populated with compile state, CUDA mem/reserved/peak, fragmentation, NaN/Inf grad counts, dataloader wait ratio, and PPO update time.
- Value Diagnostics panel now focuses on return distribution; compile status removed and panel placement swapped with Torch Stability.
- Dataloader wait ratio wired end-to-end (metrics → telemetry payload → snapshot → UI).
- `--dual-ab` now forwards TUI/telemetry wiring (ready/shutdown events, quiet_analytics, telemetry_config, profiler flags) so Sanctum behaves like standard PPO.
- PPO update skip path already populates `op_q_values`/`op_valid_mask` (no KeyError under finiteness-gate skips).

## Proposed Changes (by panel)

### Action Context (Critic Preference)
Source: `src/esper/karn/sanctum/widgets/tamiyo/action_distribution.py`
- Add a "valid ops N/NUM_OPS" line and mask coverage percentage so flatness can be interpreted correctly.
- Split masked ops into a short "Masked" sub-block (no bars/values) to keep ranked ops clean.
- Add a decisiveness marker: delta(top1-top2) or delta(best-median) near Spread.
- Highlight the most recent chosen op row (subtle marker) to connect critic ranking to action.
- Make separator width and bar width adapt to panel width to prevent ragged rules.

### Action Heads Panel
Source: `src/esper/karn/sanctum/widgets/tamiyo/action_heads_panel.py`
- Add an "Active Heads (last decision)" row and dim inactive columns to contextualize head health.
- Collapse NaN/Inf rows into a single conditional line, or move to the footer, to reclaim vertical space.
- Add a small legend for state glyphs (textual: healthy, confused, deterministic, exploding).
- Add a short ratio/clip target hint (e.g., "clip eps=0.2") near the ratio row.

### Critic Calibration Panel
Source: `src/esper/karn/sanctum/widgets/tamiyo/critic_calibration_panel.py`
- Move the calibration summary (ok/weak/bad) to the top line or panel title.
- Keep detailed rows but optimize the first line for fast scanning.

### Health Panel and PPO Update Panel
Sources:
- `src/esper/karn/sanctum/widgets/tamiyo/health_status_panel.py`
- `src/esper/karn/sanctum/widgets/tamiyo/ppo_losses_panel.py`
- Show entropy coefficient alongside entropy so exploration schedule is visible in-context.
- Add a "finiteness gate last failure" badge when present for quick NaN diagnosis.
- Keep warnings compact and consistent (one symbol per row).

### Value Diagnostics vs Returns Duplication
Sources:
- `src/esper/karn/sanctum/widgets/tamiyo/action_distribution.py`
- `src/esper/karn/sanctum/widgets/tamiyo/value_diagnostics_panel.py`
- Reduce duplication: keep percentiles/skew/trend in Value Diagnostics and a single-line recent return summary in Action Context.
- Ensure both panels show consistent terminology (p10/p50/p90, mean, sigma).

### PyTorch Stability and Performance Surfacing
Sources:
- `src/esper/karn/sanctum/widgets/tamiyo/ppo_losses_panel.py`
- `src/esper/karn/sanctum/widgets/tamiyo/value_diagnostics_panel.py`
- `src/esper/karn/sanctum/widgets/tamiyo/torch_stability_panel.py`
- Add AMP/GradScaler status (loss_scale and overflow flag) near PPO Update.
- (Done) CUDA memory fragmentation indicator (allocated/reserved + fragmentation ratio).
- (Done) dataloader wait vs step time ratio (CPU-bound vs GPU-bound hint).

## Data Contract and Wiring Notes
- Critic Preference requires: `op_q_values`, `op_valid_mask`, `q_variance`, `q_spread`.
- Active heads row requires access to last decision action (or derived head mask).
- Finiteness gate badge requires last failure sources from PPO update payload.
- AMP/GradScaler status requires telemetry fields for loss_scale and overflow detection.
- Dataloader wait vs step time requires existing throughput fields to be exposed in snapshot.
- Dual-ab TUI parity requires passing `telemetry_config`, `quiet_analytics`, and `ready_event` to vectorized training.

## Priority
- P0: Critic Preference validity clarity (valid ops line, masked block, chosen op highlight).
- P1: Action Heads context (active heads row, NaN/Inf compacting).
- P1: Calibration summary visibility.
- P2: Returns de-duplication and layout polish.
- P2: PyTorch stability/perf surfacing (AMP, fragmentation, dataloader wait).

## Implementation Map (suggestion -> files)
- Critic Preference clarity: `src/esper/karn/sanctum/widgets/tamiyo/action_distribution.py`
- Action Heads cleanup: `src/esper/karn/sanctum/widgets/tamiyo/action_heads_panel.py`
- Calibration summary priority: `src/esper/karn/sanctum/widgets/tamiyo/critic_calibration_panel.py`
- Entropy coef in Health: `src/esper/karn/sanctum/widgets/tamiyo/health_status_panel.py`
- Finiteness gate badge: `src/esper/karn/sanctum/widgets/tamiyo/ppo_losses_panel.py`
- Returns de-duplication: `src/esper/karn/sanctum/widgets/tamiyo/action_distribution.py`, `src/esper/karn/sanctum/widgets/tamiyo/value_diagnostics_panel.py`
- AMP/GradScaler status: `src/esper/karn/sanctum/widgets/tamiyo/ppo_losses_panel.py`
- CUDA fragmentation: `src/esper/karn/sanctum/widgets/tamiyo/torch_stability_panel.py`
- Dataloader wait vs step time: `src/esper/karn/sanctum/widgets/tamiyo/torch_stability_panel.py`

## Validation
- Update unit tests for Action Context to validate new lines and chosen-op marker.
- Add snapshot-based tests to verify new fields are rendered only when present.
- Manual TUI check: ensure no layout overflow in standard terminal widths.
- Sanity run: `--dual-ab --sanctum` should initialize Sanctum after DataLoader warmup and stop cleanly.

## Open Questions
- Should masked ops show reasons (e.g., lifecycle constraints) or just count and dim?
- Should delta(top1-top2) use raw Q values or normalized values?
- Do we want "chosen op" highlight in Critic Preference or in Decisions only?
- How to compute "active heads" for WAIT and mixed-action cases?
- Are loss_scale and overflow available in telemetry now, or do we need to wire them?
- Is dataloader wait exposed per batch or aggregated per run (and how to show it)?
