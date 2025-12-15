# Karn TUI → Textual Migration Plan (Superseded)

> **Status:** 🚫 **SUPERSEDED** on 2025-12-15  
> **Superseded by:** `docs/plans/esper_overwatch.md` (authoritative UI spec) and `docs/plans/overwatch-textual-ui.md` (execution plan)  
> **Reason:** Overwatch reframes the UI as ATC/outliers-first + snapshot-first, and makes telemetry parity/remediation in-scope.

## Goals
- Replace Rich-based Karn TUI with Textual while preserving current layout and telemetry coverage.
- Improve readability, compactness, and discoverability using Textual widgets and CSS theming.
- Remove Rich backend entirely once Textual is landed; first pass keeps the same layout/fields for parity, then polish.

## Scope & Non-Goals
- In scope: Karn TUI code, CLI surface, documentation, smoketests, and layout parity.
- Out of scope: changing telemetry schemas, trainer core logic, non-Karn consumers, or reward semantics.
- No backwards-compatibility modes, no dual backends, no shims (per `CLAUDE.md` No Legacy Code Policy).

## Current Layout (baseline to mirror first)
- Header: episode/batch/accuracy summary.
- Env Cards: per-env rows with reward/acc/rent, seed slots (early/mid/late) on one line, stats line with params/reward/acc sparklines, last action/status.
- Policy Stats: single-line strip (actions, health, losses, rewards).
- Bottom row: Event Log (wide) + Performance panel.
- Footer: keybind hints.

## Panel Deep Dive (what to show, by level)
- **Whole-of-System**
  - Header: episode, batches, best/current acc; optional run id/seed, reward hacking flag.
  - Aggregate: mean acc/reward across envs; count of envs by status; total seeds active/fossilized/culled.
  - Performance: GPU/CPU/mem utilization, throughput (epochs/sec, batches/hr), runtime.
  - Event Log: severity-colored stream with filters (env, action type), search, auto-scroll toggle.
- **Policy Level**
  - Action mix (%), recent action streak; mask hit rate.
  - Health: entropy, clip frac, KL, explained var; thresholds flagged inline.
  - Losses: policy/value/entropy, grad norm with status.
  - Rewards: current, rolling mean/std, best; tiny sparkline of recent rewards.
  - Optional detail drawer: advantage stats, ratio stats if space allows.
- **Per-Environment**
  - Summary: reward, acc, rent/penalty, seeds/params, status; last action + success flag.
  - History: reward and acc sparklines; best acc and epochs since improvement.
  - Params: host params (if known), fossilized params, active seed params; growth ratio if available.
  - Actions: per-env action mix and last N actions; per-env mask hits (if exposed).
  - Slots inline: Early/Mid/Late stage+blueprint, α, Δacc, grad ratio; stage color badges.
  - States: stalled/degraded/excellent; epochs since improvement; gate failures if exposed.
- **Per-Seed (per env)**
  - Stage, blueprint id, alpha, accuracy deltas (total, stage-local), grad ratio.
  - Counters: epochs in stage, epochs total; params (seed); contribution/counterfactual if present.
  - Lifecycle: last gate attempted/passed, reasons; fossilize/cull timestamps.
  - Actions affecting seed (if traceable) and probation/blending warnings.
  - Slot identity: include slot id and per-slot ordinal (e.g., `env2_mid_seed4` and slot index) in lifecycle events so stage changes read as `env2_seed_6 GERMINATED->TRAINING (mid#3)`.

## Migration Steps (technical)
1) **Dependencies**
   - Add Textual (0.60+ pinned) to deps (`pyproject`/`uv`); keep Rich only for non-TUI logging/console output.
2) **Scaffold**
   - Create `karn/textual_app.py` with `App` subclass; `compose()` wires Header, EnvCards, PolicyStats, EventLog, Performance, Footer using `DockLayout`/`Grid`.
   - Remove Rich TUI entrypoints; route TUI startup to Textual only (no backend selector, no fallback).
3) **State & Message Bus**
   - Reuse current TUI state; add a bridge that converts telemetry events to Textual `Message`s.
   - Central update loop batches messages per tick to avoid reflow storms.
4) **Widgets**
   - Header widget (Static/Label group).
   - EnvCards widget: `DataTable` or custom Grid; scrollable when many envs; inline slot badges.
   - PolicyStats widget: single-line `Static` with styled spans; optional expandable detail panel.
   - EventLog: `Log` with severity colors, filter/search controls.
   - Performance: compact grid with gauges/badges for GPU/CPU/mem/throughput.
   - Footer: keybind hints; optional help overlay.
5) **Styling**
   - `karn/textual.css` for spacing, borders, colors; stage color tokens; compact row heights.
   - Responsive tweaks for narrow terminals (stack policy/perf, compress labels).
6) **CLI Integration**
   - Update `esper/scripts/train.py` to launch Textual unconditionally (remove Rich flags/paths); rename/repurpose CLI help accordingly.
7) **Testing**
   - Add a telemetry replay harness (recorded or synthetic stream) to drive the Textual app without training.
   - Smoketest via `python -m py_compile` and a short PPO run with Textual TUI.
8) **Rollout**
   - Land Textual as the sole backend; delete Rich TUI code and docs; update help/README.

## UI Enhancements (Textual widgets to improve clarity)
- Env Cards: colored stage badges; tooltips for slot detail; compact chips for seeds/params; inline sparklines for reward/acc; per-env action mix mini-bar; status pill.
- Policy Stats: single-line strip with colored spans; optional collapsible drawer for losses/advantages.
- Event Log: filter/search, severity coloring, pause/auto-scroll toggle.
- Performance: gauges/badges for GPU/CPU/mem; thresholds highlighted; uptime and throughput inline.
- Interactions: focus navigation, help overlay, resize-friendly grid; consider modal for per-seed detail drilldown.

## Phases & Steps
- **Phase 1: Scaffolding & Plumbing**
  - Add Textual dependency.
  - Replace Rich entrypoints with Textual app skeleton; wire telemetry-to-Message bridge; layout placeholders for all panels.
- **Phase 2: Parity Rendering**
  - Implement Header, PolicyStats, Performance, Footer widgets with current fields.
  - Implement EnvCards with slots inline and stats row; EventLog basic rendering; add scroll and refresh throttling.
- **Phase 3: Enhancements**
  - Tooltips/badges for stages/slots; per-env action mix; status pills; inline sparklines via small canvases.
  - EventLog filters/search; performance gauges; help overlay; responsive tweaks and CSS polish.
- **Phase 4: Validation & Rollout**
  - Telemetry replay harness; short PPO smoketest on Textual backend.
  - Update docs/README/help; delete Rich TUI code and any flags; Textual becomes the sole backend.
