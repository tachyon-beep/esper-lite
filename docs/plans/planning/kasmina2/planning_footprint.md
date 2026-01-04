# Kasmina2 Planning Footprint (Cross-Domain Sprawl Map)

This document defines the *planning footprint*: what planning artifacts we maintain, how we split work into dev-sized slices, and which domains/files are in-scope per phase.

The non-negotiables from `CLAUDE.md` and `ROADMAP.md` apply:
- **Sensors match capabilities**: every new lever ships with telemetry + Obs support.
- **No legacy/backcompat**: contract changes update *all* call sites; no dual paths.
- **No bug-hiding defensive patterns**: contracts fail fast; no `.get()`, `getattr()`, `hasattr()`, silent excepts.
- **GPU-first inverted control**: avoid Python bottlenecks in `src/esper/simic/training/vectorized.py`.
- **DDP symmetry**: SeedSlot lifecycle/state changes must be rank-symmetric (`src/esper/kasmina/slot.py`).
- **Metaphors**: organism for domains; botanical only for seed lifecycle.

---

## 1) Planning artifact set (what lives in this folder)

**A. Roadmap (strategy)**
- Canonical roadmap for Tracks A+C: `docs/plans/planning/kasmina2/kasmina_tamiyo_submodule_intervention_roadmap.md`.

**B. Phase plans (execution)**
- One “dev-ready” plan per phase promoted into `docs/plans/ready/` once accepted.
- Inside this folder, we keep phase working notes as needed, but the *promotion target* is always a single ready plan file that can drive a PR.

**C. Contract diffs (Leyline-first)**
- For each phase that changes contracts, maintain a checklist of:
  - new/changed enums + dataclass fields (exact types)
  - all downstream consumers that must change (features, masks, telemetry store/UI, reward accounting, tests)

**D. Experiment packs**
- Run commands/config fields + expected outcomes + dashboards to watch (Karn/Sanctum/Overwatch).

---

## 2) File/directory footprint we should converge to

Recommended steady-state layout for this effort:

```
docs/plans/planning/kasmina2/
  README.md
  kasmina_tamiyo_submodule_intervention_roadmap.md
  planning_footprint.md

  phases/
    phase0_a0c0/
      scope.md
      leyline_contracts.md
      kasmina_mechanics.md
      tamiyo_obs_and_masks.md
      simic_execution_and_reward.md
      tolaria_runtime_tasks.md
      telemetry_and_dashboards.md
      experiments_and_gates.md
    phase1_a1c1/
    phase2_a2c2/
    phase3_subtargets/
    phase4_slot_transformer/
    phase5_container_seeds/
```

Notes:
- The `phases/` subfolders can start as *outlines*; do not gold-plate. The rule is: if a phase is ready to implement, it gets promoted to `docs/plans/ready/` as a single, implementation-ready plan.
- Keep “contracts” separate from “mechanics” so reviewers can validate Leyline changes independently.

---

## 3) Cross-domain impact matrix (what changes where)

This matrix is the baseline “sprawl map”. It should be updated whenever we add a new lever.

### Leyline (DNA/contracts) — **always first**
Primary files:
- `src/esper/leyline/injection_spec.py` (Track A surfaces: `InjectionSurface`, `InjectionSpec.order/surface`)
- `src/esper/leyline/slot_config.py` (deterministic slot ordering + slot metadata)
- `src/esper/leyline/factored_actions.py` (new `LifecycleOp`s, new `BlueprintAction`s, new heads if Phase 3)
- `src/esper/leyline/causal_masks.py` (credit assignment masks must remain single source of truth)
- `src/esper/leyline/reports.py` (`SeedStateReport` fields for microstructure + surfaces)
- `src/esper/leyline/telemetry.py` (new event types + typed payloads)
- `src/esper/leyline/__init__.py` (Obs V3 constants + embedding sizing; must stay consistent)

Downstream consumers to enumerate for every Leyline change:
- **Kasmina:** `src/esper/kasmina/host.py`, `src/esper/kasmina/slot.py`, `src/esper/kasmina/blueprints/*`
- **Tamiyo:** `src/esper/tamiyo/policy/features.py`, `src/esper/tamiyo/policy/action_masks.py`, `src/esper/tamiyo/networks/factored_lstm.py`
- **Simic:** `src/esper/simic/training/vectorized.py`, `src/esper/simic/rewards/rewards.py`, `src/esper/simic/agent/*` (rollout buffer shapes, PPO masks, Q-value telemetry)
- **Tolaria/runtime:** `src/esper/runtime/tasks.py`, `src/esper/tolaria/environment.py`, CLI surfaces in `src/esper/scripts/train.py` (task + slot presets)
- **Karn/Nissa:** event ingestion + rendering (Sanctum/Overwatch) keyed off `TelemetryEventType` + payload schema

### Kasmina (mechanics) — **host + slot + seeds**
Primary files:
- `src/esper/kasmina/host.py` (Track A routing + injection specs; deterministic ordering)
- `src/esper/kasmina/slot.py` (Track C internal ops execution + telemetry emission; DDP symmetry)
- `src/esper/kasmina/blueprints/cnn.py` and `src/esper/kasmina/blueprints/transformer.py` (new ladder seed families)

Non-negotiable invariants:
- No hooks; routing must be explicit and deterministic (`HostProtocol` contract).
- Internal state transitions must be action-driven and DDP-symmetric.
- torch.compile graph growth must be bounded (cap ladder levels; avoid Python list mutation in hot path).

### Tamiyo (decisions) — **Obs V3 + masks + policy network**
Primary files:
- `src/esper/tamiyo/policy/features.py` (Obs V3 fields for new levers; keep hot-path constraints)
- `src/esper/tamiyo/policy/action_masks.py` (mask only physically impossible actions; update for internal/subtarget ops)
- `src/esper/tamiyo/networks/factored_lstm.py` (input dim changes; new head only in Phase 3)

Hot-path constraints:
- Features are built CPU-side then moved once; avoid Python-per-slot GPU writes.
- Observation shape changes must update all buffer/assertion codepaths (no partial updates).

### Simic (training/reward) — **execution + ROI**
Primary files:
- `src/esper/simic/training/vectorized.py` (execute new ops; action validity; reward attribution wiring)
- `src/esper/simic/training/helpers.py` (rent/shock inputs if internal capacity impacts param accounting)
- `src/esper/simic/rewards/rewards.py` (intervention costs per op; rent/churn accounting)
- `src/esper/simic/agent/ppo.py` (Q(s,op) telemetry must include any new ops; causal mask integration)

Throughput constraint:
- Any new op must be executable without adding Python-side per-env overhead in the inner loop.

### Tolaria/runtime (execution surface) — **task presets + model creation**
Primary files:
- `src/esper/runtime/tasks.py` (add task presets for A0/A1 experiments, e.g., `tinystories_layerwise`)
- `src/esper/tolaria/environment.py` (model creation wiring is already strict; ensure new task specs are consistent)
- `src/esper/scripts/train.py` (CLI surface for selecting new tasks/presets if needed)

---

## 4) “Dev-sized slice” breakdown (PR footprint recommendation)

Because we can’t ship backcompat, the safest PR shape is **contract-first, end-to-end** for the minimal phase, then expand.

### PR0 (docs-only)
- Add/iterate plans and experiment packs (this folder + `docs/plans/ready/` promotion).

### PR1 (Phase 0, contracts + plumbing)
- Leyline: new ops + new report fields + new telemetry payload (Phase 0 only).
- Update *all* call sites to compile.
- Add minimal tests for invariants (slot ordering, obs dim consistency).

### PR2 (Phase 0, mechanics + policy + training)
- Kasmina: implement `conv_ladder` + internal ops in `SeedSlot`.
- Tamiyo: add `internal_level_norm` feature + internal op masks.
- Simic: execute internal ops in vectorized loop; add intervention costs; update Q(s,op) telemetry for new ops.
- Karn: show `SEED_INTERNAL_LEVEL_CHANGED` in event views (at least in table logs).

### PR3+ (Phase 1+)
- Phase 1: `InjectionSpec.order/surface` + transformer post-attn/post-MLP lattice + `lora_ladder`.
- Phase 2: CNN pre/post-pool surfaces + channel-group ladder.
- Phase 3: subtarget head + grouped addressing (requires policy and PPO shape work).
- Phase α: Slot Transformer policy pivot (representation change; larger refactor).

---

## 5) Phase 0 “planning checklist” (what must be written down before coding)

Before implementing Phase 0, we should have a short phase folder (or a promoted ready plan) that explicitly enumerates:

- **Leyline deltas:** new `LifecycleOp`s, new `BlueprintAction` (`CONV_LADDER`), new report fields, new telemetry event/payload, Obs V3 constant updates.
- **All consumers:** list every file that must change (Tamiyo features/masks/network; Simic vectorized + rewards + PPO telemetry; Kasmina slot/blueprints).
- **Telemetry parity:** where the event is emitted and where it is visible (Karn/Sanctum/Overwatch).
- **Gates:** “done means” checklists (ROI, invalid-action rate, entropy stability, governor rollback rate).

This keeps Phase 0 truly “implementation-ready” and prevents partial contract drift.
