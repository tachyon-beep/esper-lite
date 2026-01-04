# Kasmina + Tamiyo Submodule Intervention Roadmap (Track A + Track C)

**Author role:** Esper Research Architect  
**Status:** Research roadmap / planning deliverable (no code)  
**Baseline assumed:** Obs V3, Policy V2, op-conditioned critic, ~150-step horizon  
**Non-negotiables:** Sensors match capabilities; no legacy/backcompat shims; no bug-hiding defensive patterns; GPU-first inverted control flow; DDP symmetry invariants; metaphors (organism for domains, botanical only for seed lifecycle)

---

## 1) One-paragraph executive summary (what we’re building and why it fixes “conv_heavy dominance”)

We will evolve Esper from **module-level** morphogenesis (“pick one blueprint, at one coarse boundary”) to **submodule-level** morphogenesis where Tamiyo can act more often with **smaller, reversible** adjustments. Today, the cheapest way to get immediate learning signal is frequently “buy a big module” (e.g., `conv_heavy`) because (1) injection surfaces are coarse, and (2) blueprint capacity is step-function discrete. Track **A** increases *where* Tamiyo can act by exposing denser, semantically meaningful injection surfaces (transformer post-attention vs post-MLP; CNN pre-pool vs post-pool) with deterministic slot IDs and ordering. Track **C** reduces the *mass per decision* by introducing microstructured “ladder” seeds whose capacity is controlled by a small internal state and a tiny number of lifecycle ops (grow/shrink/enable/disable), so Tamiyo can dial signal up/down without repeatedly germinating/pruning heavy seeds. Every new lever ships with **Leyline-typed telemetry + Obs V3 support** (ROADMAP commandment #1), preserves GPU-first inverted control flow (`src/esper/simic/training/vectorized.py`), and maintains SeedSlot DDP symmetry invariants (`src/esper/kasmina/slot.py`).

---

## 2) A vs C design taxonomy (options table with pros/cons)

### Track A — Injection surfaces (denser, semantically meaningful places to attach slots)

| Option | Surfaces exposed | Stable IDs + ordering | Pros | Cons / when to avoid | Compile/DDP notes |
|---|---|---|---|---|---|
| **A0 (minimal): denser existing segments** | More *segment boundaries* only | Existing `slot_id="r0c{i}"`; preserve host order from `HostProtocol.injection_specs()` (`src/esper/leyline/host_protocol.py`) | Immediate density (CNN `n_blocks↑`, Transformer `num_segments=n_layer`); no new routing semantics | Still “boundary-only” meaning; policy sees bigger obs (flat concat) as slots increase | No new graphs; DDP unchanged |
| **A1: Transformer sublayer surfaces** | `POST_ATTN`, `POST_MLP` (per layer) | `slot_id="r{layer}c{surface}"` with deterministic `InjectionSpec.order` (see `src/esper/leyline/injection_spec.py`) | True submodule meaning; clearer credit assignment than block-only; aligns with Phase 3 transformer pivot | Doubles slot count; requires explicit routing changes in `src/esper/kasmina/host.py` | Avoid hooks; route with explicit sublayer loop; slot count pressure accelerates Phase α |
| **A2: CNN pre/post-pool surfaces** | `PRE_POOL`, `POST_POOL` (per block) | `slot_id="r{block}c{surface}"`, `order` block-major then surface | Adds resolution-aware choice (high-res vs cheap); avoids brittle intra-conv hooks; matches pooling semantics in `CNNHost.forward_to_segment()` | Requires CNNHost routing refactor; increases slot count; must keep channels_last invariants | No hooks; ensure conversion once; watch pool op memory format |
| **A3 (defer): intra-attn micro-surfaces** | Q/K/V/proj or per-head surfaces | Requires subtarget inventory IDs | Extremely fine control | Action-space explosion; brittle; torch.compile hostile | Defer until grouped subtarget control + Slot Transformer exists |
| **Reject** | Forward hooks / hidden callbacks | N/A | Prototype speed | Violates Train Anything protocol (#5) and compile expectations; hard to keep deterministic | Avoid |

### Track C — Microstructured seeds (lighter “mass per decision” inside one seed)

| Option | Internal state + ops | What becomes cheap | Pros | Cons / when to avoid | Compile/DDP notes |
|---|---|---|---|---|---|
| **C0 (minimal): level ladder** | `internal_level:int` + `GROW_INTERNAL/SHRINK_INTERNAL` | Incremental capacity vs discrete blueprints | Small action-space change (2 ops); creates “small bets”; directly attacks conv_heavy dominance | Must define strict invariants + telemetry; too many levels can multiply compile graphs | Cap `L≤4`; internal ops at epoch boundaries; keep DDP-symmetric transitions |
| **C1: LoRA rank-block ladder** | Level controls number of rank blocks | Cheap PEFT capacity, transformer-aligned | Smooth alternative to `lora_large`; clean ROI story | Needs block design to make params truly incremental | Rank blocks as ModuleList; append-only growth = deterministic |
| **C2: CNN block-count ladder (conv ladder)** | Level controls # residual conv micro-blocks | Cheap spatial capacity for CNN | Simple, shape-preserving; incremental params; no tricky grouping | Not as hardware-structured as channel groups | Small L; compile specialization acceptable |
| **C3: channel-group ladder** | Enable/disable channel groups | Structured capacity; smaller compute ramps | Better hardware alignment | Requires deterministic partition rule; risk of brittle slicing | Implement with fixed grouping + stable group IDs |
| **C4: head-group ladder** | Enable/disable attention head groups | True transformer submodule control | Strong interpretability | Needs stable head indexing + masks; risks credit assignment noise | Defer until Phase 3 inventory contracts land |
| **C5a: focused subtargets (headless)** | `focus_idx:int` + `FOCUS_NEXT/FOCUS_PREV` ops; reuse `GROW/SHRINK_INTERNAL` on focused subtarget | Targeted micro-growth without a new head | Adds only 2 ops; avoids PPO/policy schema refactor; bounded + DDP-friendly | Focus cycling adds temporal credit assignment; requires focus telemetry + Obs; K must stay small | focus_idx must be DDP-synchronized; compile impact minimal |
| **C5b: addressable subtargets (head)** | New head `subtarget_idx` + enable/disable ops | Direct selective control within a seed | Debuggable micro-decisions; faster than focus cycling; sets up Phase α scaling | Requires new action head + masks; increases learnability burden | Keep K small (≤8–16) and grouped; token policy likely required |

**Cross-cutting determinism (non-negotiable):**
- **Stable slot order:** add `InjectionSpec.order: int` and sort SlotConfig by `order` (not float `position`) in `src/esper/leyline/slot_config.py`.
- **Surface identity:** represent surfaces with a Leyline enum (e.g., `InjectionSurface`) instead of stringly-typed or module-name-based hooks.
- **Internal IDs:** microstructure must have deterministic level/group indexing; no float tie-breaking; no runtime-discovered hook names.

---

## 3) Roadmap table (phase, duration, dependencies, risk level, success metric)

| Phase | Duration | Dependencies | Risk | Success metric (ROI-style + decision quality) |
|---|---:|---|---|---|
| **0 (A0 + C0)** | 1–2w | Phase 2.5 gates (reward exam) recommended | Low–Med | `GROW/SHRINK_INTERNAL` used; conv_heavy selection drops; ROI ≥ baseline; no rollback/invalid-action spikes |
| **1 (A1 + C1)** | 2–3w | Phase 0 | Med | TinyStories ROI improves with post-attn/post-MLP lattice; LoRA ladder used vs `lora_large`; head entropies stable |
| **2 (A2 + C2)** | 2–3w | Phase 0 + Phase 1 injection contracts | Med | CIFAR ROI improves with pre/post-pool surfaces; ladder upgrades replace “step-function” blueprint jumps; fossilize/prune efficiency improves |
| **3 (A + C inventory)** | 3–4w | Phases 1–2 | High | Subtarget control used with low invalid-action rate; effective params reduced at same quality; stable entropy (new head if chosen) |
| **4 (Phase α: Slot Transformer policy)** | 3–5w | Phase 3 or objective “slot pressure” trigger | High | PPO stable at ≥32 slots; ≤10% throughput hit; improved decision quality vs flat concat |
| **5 (End-state)** | 4–8+w | Phase 4 | Very high | Replayable “grow then trim” run with stable governor outcomes + explainable telemetry |

---

## 4) Per-phase spec blocks (as described above)

### Phase 0 — **A0 + C0** minimal milestone (hard requirement)

**Objective:** Give Tamiyo a **cheap, reversible** way to adjust capacity without defaulting to heavyweight blueprints, using only existing injection surfaces (A0) and a single microstructured seed family (C0).

**Scope (what you will/won’t touch):**
- **Will touch (by domain boundary):**
  - **Leyline (contracts):** `src/esper/leyline/factored_actions.py`, `src/esper/leyline/causal_masks.py`, `src/esper/leyline/reports.py`, `src/esper/leyline/telemetry.py`, `src/esper/leyline/__init__.py` (Obs V3 constants).
  - **Kasmina (mechanics):** `src/esper/kasmina/slot.py` (internal ops execution + state), `src/esper/kasmina/blueprints/cnn.py` (one ladder seed family).
  - **Tamiyo (decisions):** `src/esper/tamiyo/policy/features.py` (Obs V3 field), `src/esper/tamiyo/policy/action_masks.py` (mask internal ops).
  - **Simic (training/reward):** `src/esper/simic/training/vectorized.py` (execute ops), `src/esper/simic/rewards/rewards.py` (intervention cost + ROI telemetry alignment).
  - **Nissa/Karn (telemetry/memory):** ingestion is automatic; UI surfaces may need small additions to display the new event/type.
- **Won’t touch:** sublayer host refactors (A1/A2), Slot Transformer policy (Phase α), addressable subtargets (Phase 3).

**Design sketch (contracts + dataflow):**
- Add a seed-local microstructure state: `internal_level` with a deterministic mapping `level → active submodules`.
- Tamiyo acts as: `GERMINATE(conv_ladder)` once, then `GROW_INTERNAL`/`SHRINK_INTERNAL` to change level without changing slot or seed identity.
- Kasmina executes internal ops as explicit lifecycle ops inside `SeedSlot` (`src/esper/kasmina/slot.py`), updating trainable parameter count and emitting telemetry.

**Required telemetry sensors + derived Obs V3 fields (exact names, locations, wiring):**
- **Leyline enums (new):**
  - `SeedInternalKind` (new file suggested: `src/esper/leyline/seed_internal.py`, or colocate in `src/esper/leyline/reports.py` if preferred)
    - `NONE = 0`
    - `CONV_LADDER = 1`
    - (reserved for later phases) `LORA_RANK_LADDER = 2`, `CHANNEL_GROUP_LADDER = 3`, `ATTN_HEAD_GROUP_LADDER = 4`
- **Leyline report fields (Kasmina → Tamiyo):** extend `src/esper/leyline/reports.py:SeedStateReport`:
  - `internal_kind: SeedInternalKind = SeedInternalKind.NONE`
  - `internal_level: int = 0` (invariant: `0 <= internal_level <= internal_max_level`)
  - `internal_max_level: int = 1` (invariant: `internal_max_level >= 1` even when `internal_kind=NONE`, to avoid division guards)
  - Optional (for dashboards / rent learnability): `internal_active_params: int = 0` (trainable params currently enabled by level)
  - **Emission point:** `src/esper/kasmina/slot.py:SeedState.to_report()` populates these fields from SeedSlot state.
  - **Consumption point:** `src/esper/tamiyo/policy/features.py` reads these fields directly (fail-fast if absent after contract change).
- **Obs V3 (policy input):** add one per-slot scalar field in `src/esper/tamiyo/policy/features.py`:
  - `internal_level_norm: float` in `[0.0, 1.0]`, defined as `internal_level / internal_max_level`.
  - **Normalization rule:** no clamping; contract guarantees range.
- **Leyline telemetry event (typed, must exist for learnability + debugging):** extend `src/esper/leyline/telemetry.py`:
  - `TelemetryEventType.SEED_INTERNAL_LEVEL_CHANGED`
  - `SeedInternalLevelChangedPayload` (dataclass, `slots=True, frozen=True`):
    - `slot_id: str`
    - `env_id: int` (Kasmina emits `-1` sentinel; replaced by `emit_with_env_context`)
    - `blueprint_id: str`
    - `from_level: int`
    - `to_level: int`
    - `max_level: int`
    - Optional: `active_params: int`
  - **Emission point:** `src/esper/kasmina/slot.py` on successful internal op execution (same style as `SEED_STAGE_CHANGED`).
  - **Storage/UI consumers:** Karn store views + Sanctum/Overwatch should display (at least) `from_level→to_level`, and correlate with ROI.
- **Obs V3 constants (must stay in sync):**
  - Adding new `LifecycleOp`s changes `NUM_OPS`, which changes the base “last_action_op one-hot” width in `src/esper/tamiyo/policy/features.py`.
  - Update `src/esper/leyline/__init__.py`:
    - `OBS_V3_BASE_FEATURE_SIZE` must become `17 + NUM_OPS` (not hard-coded 23), or be updated explicitly after op additions.
    - `OBS_V3_SLOT_FEATURE_SIZE` increments by `+1` for `internal_level_norm`.
    - `OBS_V3_NON_BLUEPRINT_DIM` recomputed accordingly.
  - Downstream consumers that must change: `src/esper/tamiyo/networks/factored_lstm.py` (feature net input dim), `src/esper/simic/agent/rollout_buffer.py` (state_dim assertions), and any observation normalization shapes.

**Action space changes (avoid combinatorial blow-up):**
- **New blueprint option (Leyline):** add to `src/esper/leyline/factored_actions.py:BlueprintAction`:
  - `CONV_LADDER` → `"conv_ladder"` (and include it in `CNN_BLUEPRINTS` so masking permits it on CNN tasks)
  - Downstream consumers that must change: `BLUEPRINT_IDS`/`BLUEPRINT_ID_TO_INDEX`, blueprint embedding table sizing (`BLUEPRINT_NULL_INDEX` in `src/esper/leyline/__init__.py`), and any UI blueprint-name tables.
- **New ops (Leyline):** add to `src/esper/leyline/factored_actions.py:LifecycleOp`:
  - `GROW_INTERNAL`
  - `SHRINK_INTERNAL`
- **Causal relevance (Leyline):** update `src/esper/leyline/causal_masks.py`:
  - For both internal ops: `op` and `slot` relevant; all other heads irrelevant.
- **No new head in Phase 0** (keeps factorization stable and learnable).
- **Intervention cost (Simic):** extend `src/esper/simic/rewards/rewards.py:INTERVENTION_COSTS` + `ContributionRewardConfig` fields for the new ops (cost should be small vs `GERMINATE`).

**Host changes (A0):**
- **No host contract changes.** Use existing surfaces via task presets:
  - CNN density: `src/esper/runtime/tasks.py` already provides `cifar_scale` (5 blocks → 5 injection points).
  - Transformer density A0: add a `tinystories_layerwise` task preset (or equivalent config field) setting `TransformerHost(num_segments=n_layer)` in `src/esper/runtime/tasks.py` / `src/esper/kasmina/host.py`. This preserves “boundary-only” semantics while increasing slot count.

**Seed changes (C0):**
- **Concrete seed family:** `conv_ladder` (CNN), implemented in `src/esper/kasmina/blueprints/cnn.py`.
  - Internal microstructure: `L` residual conv micro-blocks (`SeedConvBlock`-based), active blocks = `internal_level`.
  - Level invariants: `1 <= internal_level <= L` when `internal_kind=CONV_LADDER` (seed exists); `internal_level=0` only when no seed.
  - Param accounting: only active blocks have `requires_grad=True` so `SeedSlot.active_seed_params` and `SeedMetrics.seed_param_count` reflect level (ties directly into rent & churn economy without hacks).
  - Fossilization semantics: freeze `internal_level` at `SeedStage.FOSSILIZED` (internal ops masked out thereafter).

**Safety constraints (DDP symmetry, torch.compile, governor, throughput):**
- **DDP symmetry:** internal ops must be applied identically on all ranks. If actions are not already broadcast, mirror `_sync_gate_decision()` pattern for internal ops in `src/esper/kasmina/slot.py`.
- **torch.compile:** internal_level-dependent control flow can multiply graphs. Mitigation: cap `L≤4`, and allow internal ops only at epoch boundaries (where SeedSlot already tolerates specialization).
- **Governor interactions:** internal growth should not coincide with alpha ramps (`AlphaMode.UP/DOWN`). Mask internal ops unless `alpha_mode == HOLD` (in `src/esper/tamiyo/policy/action_masks.py`).

**Experiment plan:**
- **Task/preset:** `cifar_scale` (`src/esper/runtime/tasks.py`), slots `r0c0..r0c4`, `max_seeds=2`, `episode-length=150`, `n_envs=8`.
- **Cohorts:**
  - Control: current blueprint set (no ladder).
  - Treatment: ladder blueprint available + internal ops.
- **Primary success metrics (ROI-style):**
  - Accuracy ROI = `(final_acc - baseline_acc) / (added_trainable_params)` (use existing rent telemetry; verify with Karn).
  - Blueprint mix: drop in `CONV_HEAVY` selection frequency vs control.
  - Internal ops usage: non-trivial `SEED_INTERNAL_LEVEL_CHANGED` rate without oscillatory thrash.
- **Decision quality signals:**
  - Invalid-action rate (`last_action_success`), entropy trends per head, `GROW/SHRINK` selection entropy.
  - Governor events (`TelemetryEventType.GOVERNOR_ROLLBACK`) must not increase.

**Risks + mitigations:**
- **Thrash (grow/shrink oscillation):** introduce small per-op cost; add churn telemetry (`level_changes_per_100_steps`) in Karn.
- **Obs/shape drift:** update Obs V3 constants everywhere (Leyline is source of truth; no dual paths).

**Exit criteria (gates to unlock Phase 1):**
- `GROW/SHRINK_INTERNAL` used in ≥10% of non-WAIT decisions when ladder seed present, with invalid-action rate not materially worse than control.
- Conv-heavy dominance reduced (relative blueprint frequency down) without lowering ROI.
- No sustained entropy collapse or governor rollback increase.

---

### Phase 1 — **A1 + C1** Transformer sublayer surfaces + LoRA ladder

**Objective:** Make transformer interventions submodule-meaningful (post-attn vs post-MLP) and make transformer capacity adjustments incremental (LoRA ladder) so Tamiyo doesn’t have to jump straight to `LORA_LARGE`.

**Scope:**
- **Will touch:**
  - **Kasmina host routing:** `src/esper/kasmina/host.py:TransformerHost` (explicit sublayer routing; no hooks).
  - **Leyline injection contracts:** `src/esper/leyline/injection_spec.py`, `src/esper/leyline/slot_config.py`.
  - **Tamiyo obs/features:** `src/esper/tamiyo/policy/features.py` (surface/order features) + `src/esper/leyline/__init__.py` constants update.
  - **Kasmina seeds:** `src/esper/kasmina/blueprints/transformer.py` add `lora_ladder`.
- **Won’t touch:** per-head intra-attention surfaces; Slot Transformer policy (Phase α); addressable subtargets.

**Design sketch (A1 surfaces):**
- Expose two injection surfaces per transformer layer in `src/esper/kasmina/host.py`:
  - `POST_ATTN`: after `x = x + attn(ln1(x))`
  - `POST_MLP`: after `x = x + mlp(ln2(x))`
- Each surface becomes a distinct slot for Kasmina (`MorphogeneticModel._slot_order` uses `host.injection_specs()` ordering).

**Host changes (A1):**
- In `src/esper/kasmina/host.py:TransformerHost`:
  - Emit one `InjectionSpec` per `(layer_idx, surface)` with:
    - `slot_id = format_slot_id(layer_idx, surface.value)`
    - `order = (layer_idx * 2) + surface_rank` where `surface_rank(POST_ATTN)=0`, `surface_rank(POST_MLP)=1`
    - `position = (order + 1) / (2 * n_layer)` (visualization only)
  - Update `forward_to_segment()` / `forward_from_segment()` to route between these boundaries without hooks, using cached boundary metadata.

**Leyline contract changes required (and where):**
- **`src/esper/leyline/injection_spec.py`**
  - Add `InjectionSurface (IntEnum)`:
    - `SEGMENT_END = 0` (existing A0 semantics)
    - `POST_ATTN = 1`
    - `POST_MLP = 2`
    - `PRE_POOL = 3` (Phase 2)
    - `POST_POOL = 4` (Phase 2)
  - Extend `InjectionSpec`:
    - `surface: InjectionSurface`
    - `order: int` (strictly increasing in forward execution order; no float ties)
    - Keep `position: float` only for visualization; SlotConfig ordering must use `order`.
- **`src/esper/leyline/slot_config.py`**
  - Extend SlotConfig to carry static slot metadata derived from InjectionSpec:
    - `slot_orders: tuple[int, ...]` aligned with `slot_ids`
    - `slot_surfaces: tuple[int, ...]` (store `InjectionSurface.value` aligned with `slot_ids` for hot-path access)
  - Update `SlotConfig.from_specs()` to sort by `spec.order` (fail fast if duplicate/missing).

**Required telemetry sensors + derived Obs V3 fields:**
- **Obs V3 per-slot fields (Tamiyo):** add two scalars in `src/esper/tamiyo/policy/features.py`:
  - `slot_order_norm: float` (normalize `order / (max_order+1)`; range `[0,1]`)
  - `slot_surface_norm: float` (normalize `surface.value / max_surface_value`; small categorical proxy; optional to replace with small embedding later)
  - **Consumption points:** used by Tamiyo feature net; also used for action masking heuristics if needed later.
- **Contracts/constants:** update `src/esper/leyline/__init__.py:OBS_V3_SLOT_FEATURE_SIZE` (+2) and recompute derived dims; update any shape assertions in Simic/Tamiyo.
- **Emission points:** no new per-step emission; the host supplies metadata via InjectionSpec → SlotConfig at environment construction time (`src/esper/simic/training/vectorized.py` derives SlotConfig from injection specs).

**Action space changes:**
- No new head. Reuse Phase 0 internal ops for LoRA ladder capacity adjustment.
- Blueprint additions (Leyline): add `BlueprintAction.LORA_LADDER` (or reuse `LORA` with ladder semantics and deprecate `LORA_LARGE` later; choose one and update all call sites—no dual path).

**Seed changes (C1): LoRA ladder design**
- Add `lora_ladder` blueprint in `src/esper/kasmina/blueprints/transformer.py`.
  - Microstructure: `K` rank blocks (e.g., 4 blocks of rank=4 each → max rank=16) implemented as a ModuleList of small `down/up` pairs.
  - `internal_level` controls how many blocks are active/trainable; growth increases trainable params monotonically.
  - Fossilization freezes level.

**Safety constraints:**
- **torch.compile:** avoid Python hooks; explicit sublayer routing inside `TransformerHost.forward_to_segment()`; keep routing loops simple and layer-count constant.
- **DDP symmetry:** slot IDs and ordering are host-derived and deterministic; seed internal ops remain action-driven.
- **Throughput:** doubling slots increases obs size; measure policy forward throughput before unlocking Phase 2.

**Experiment plan:**
- **Task:** `tinystories` (`src/esper/runtime/tasks.py`).
- **Cohorts (curriculum):**
  1. A0 (3 segments) + existing LoRA vs LORA_LARGE.
  2. A0 (per-layer boundaries via `num_segments=n_layer`) + LoRA ladder (6 slots).
  3. A1 (post-attn only) + LoRA ladder (6 slots).
  4. A1 (post-attn + post-MLP) + LoRA ladder (12 slots).
- **Success criteria:**
  - Perplexity/accuracy ROI improves at similar effective rent.
  - Policy learns differentiated surface preference (post-attn vs post-MLP) in decision logs.
  - Uses internal ops instead of immediately choosing heavy adapters.

**Risks + mitigations:**
- **Slot pressure on flat obs:** introduce curriculum (post-attn only) and keep `max_seeds` small; Phase α trigger if entropy collapse emerges.
- **Surface semantics mismatch:** require deterministic unit tests for routing order; fail fast if `InjectionSpec.order` duplicates.

**Exit criteria:**
- Stable runs at 12 surfaces without sustained entropy collapse; lora ladder internal ops are used and correlate with ROI.

---

### Phase 2 — **A2 + C2** CNN pre/post-pool surfaces + channel-group ladder

**Objective:** Give CNN interventions submodule meaning aligned with spatial resolution changes (pre vs post pooling) and provide a structured microstructure option beyond “block-count ladder”.

**Scope:**
- **Will touch:** `src/esper/kasmina/host.py:CNNHost` routing + injection specs, `src/esper/kasmina/blueprints/cnn.py` (channel-group ladder), plus the Phase 1 injection contracts already landed.
- **Won’t touch:** intra-conv (conv/bn/relu) hook surfaces; addressable subtargets (Phase 3).

**Design sketch (A2 surfaces):**
- For each CNN block `i` in `CNNHost.forward_to_segment()` (`src/esper/kasmina/host.py`), expose:
  - `PRE_POOL`: boundary after block forward, before `MaxPool2d`
  - `POST_POOL`: boundary after pooling (only if `i < pool_layers`)
- Slot IDs: `slot_id = format_slot_id(block_idx, surface.value)` with deterministic `order = block_idx * 2 + surface_rank`.

**Required telemetry sensors + derived Obs V3 fields:**
- Reuse Phase 1 `InjectionSpec.surface` and `InjectionSpec.order` → SlotConfig metadata → Tamiyo per-slot `slot_surface_norm/slot_order_norm`.
- Reuse Phase 0 `internal_level_norm` for channel group ladder (`internal_level` = active groups).
- Optional (dashboard-only, not policy-critical): add `internal_group_size: int` and `internal_active_params: int` to `SeedStateReport` to make ROI attribution interpretable in Karn.

**Action space changes:**
- No new head; reuse `GROW_INTERNAL/SHRINK_INTERNAL`.

**Host changes (A):**
- Update `src/esper/kasmina/host.py:CNNHost.injection_specs()` to emit both surfaces with stable ordering:
  - `InjectionSpec.channels` unchanged (pool does not change channels)
  - `InjectionSpec.position` can remain derived from order for visualization
- Update `CNNHost.forward_to_segment()` routing to allow stopping at PRE_POOL boundaries without applying pool.
- Preserve channels_last contract: conversion at entry only; verify pooling doesn’t silently return contiguous_format (if it does, force `memory_format=torch.channels_last` once per boundary, not per op).

**Seed changes (C2): channel-group ladder**
- Add a channel-group ladder blueprint (e.g., `channel_ladder`) in `src/esper/kasmina/blueprints/cnn.py`.
  - Deterministic grouping rule (no data-dependent discovery):
    - `group_size` fixed (e.g., 16 channels/group) or derived from `TARGET_CHANNELS_PER_GROUP` (`cnn.py` already uses this heuristic).
    - `group_count = channels // group_size` (require divisibility; fail fast otherwise).
  - `internal_level` = number of active groups (0..group_count).
  - Active groups are trainable and contribute; inactive groups are identity.
  - Fossilization freezes the active group mask/level.
- Rent accounting: active trainable params scale with groups (requires_grad gating), so existing param-count rent remains meaningful.

**Safety constraints:**
- Determinism: grouping must be purely a function of `(channels, group_size)` and stable across runs.
- DDP symmetry: group mask changes action-driven; no rank-local decisions.
- Governor: PRE_POOL interventions are higher compute + higher risk; monitor rollbacks and gradient pathologies.

**Experiment plan:**
- **Tasks:** `cifar_baseline` and `cifar_scale` (`src/esper/runtime/tasks.py`).
- **Cohorts:**
  - A0 only (existing slots).
  - A2 surfaces enabled (pre/post pool).
  - C0 conv_ladder vs C2 channel_ladder (matched internal_max_level budgets).
- **Success criteria:**
  - Surface preference emerges (PRE_POOL used when high-res value exceeds rent).
  - Reduced prune rate due to overshoot (internal growth can stop early).
  - No regression in throughput >10% vs A0 at same env count.

**Risks + mitigations:**
- Slot-count pressure: cap enabled slots initially (e.g., only PRE_POOL on first 2 blocks) until Phase α.
- Channels_last regressions: add explicit throughput/perf gate before accepting.

**Exit criteria:**
- Stable training with A2 surfaces on `cifar_scale`; channel ladder internal ops reduce heavy blueprint usage with equal or better ROI.

---

### Phase 3 — **A + C inventory:** deterministic subtarget inventory + grouped addressing

**Objective:** Upgrade microstructure from “only grow/shrink the tail” to **addressable grouped subtargets** (enable/disable specific groups) with stable IDs, while keeping action space bounded and learnable.

**Scope:**
- **Will touch:**
  - **Leyline (new contracts):** subtarget enums + new action head spec.
  - **Tamiyo (new head):** add `subtarget_idx` head to policy network (`src/esper/tamiyo/networks/factored_lstm.py`) and masking (`src/esper/tamiyo/policy/action_masks.py`).
  - **Kasmina (seed reporting + ops):** seeds expose inventories; SeedSlot executes enable/disable.
  - **Simic:** update vectorized execution + PPO advantage masking for new head (causal masks + rollout buffer shapes).
- **Won’t touch:** nested morphogenetic planes (Phase 5); Phase α policy pivot unless needed for stability.

**Design sketch:**
- Each microstructured seed exposes an inventory:
  - `subtarget_kind` (what this index means)
  - `subtarget_count` (how many indices are valid)
  - `subtarget_active_count` (how many are enabled)
- Two viable encodings (choose one; both keep action growth bounded):
  - **(3A, head-based):** Tamiyo chooses `(slot, op, subtarget_idx)` where `op ∈ {ENABLE_SUBTARGET, DISABLE_SUBTARGET}`.
  - **(3B, focus-based / headless):** Tamiyo chooses `(slot, op)` where `op ∈ {FOCUS_NEXT_SUBTARGET, FOCUS_PREV_SUBTARGET, GROW_INTERNAL, SHRINK_INTERNAL}`, and `GROW/SHRINK_INTERNAL` applies to the **focused** subtarget.
- Subtarget indices are stable integers `0..subtarget_count-1` determined at germination.

**Option 3A — head-based addressing (direct `subtarget_idx` head)**

**Leyline contract changes required (exact types + fields):**
- **`src/esper/leyline/subtargets.py` (new file suggested):**
  - `SubtargetKind (IntEnum)`:
    - `NONE = 0`
    - `LORA_RANK_BLOCK = 1`
    - `CNN_CHANNEL_GROUP = 2`
    - `ATTN_HEAD_GROUP = 3`
- **`src/esper/leyline/reports.py:SeedStateReport` (add fields):**
  - `subtarget_kind: SubtargetKind = SubtargetKind.NONE`
  - `subtarget_count: int = 0` (0 means “no subtargets”)
  - `subtarget_active_count: int = 0`
  - Optional (dashboard-only): `subtarget_churn_epochs: int = 0` (epochs since last change)
- **`src/esper/leyline/factored_actions.py` (action space):**
  - Add new op(s) in `LifecycleOp`:
    - `ENABLE_SUBTARGET`
    - `DISABLE_SUBTARGET`
  - Add a new head spec (bounded K):
    - `ActionHeadSpec(name="subtarget_idx", enum=None, slot_dependent=False)` plus an explicit `NUM_SUBTARGETS: int = K` constant in Leyline, or
    - define a `SubtargetIndexAction(IntEnum)` sized `K` (single source of truth).
  - Update `ACTION_HEAD_SPECS` / `ACTION_HEAD_NAMES` and downstream uses (Tamiyo network, PPO buffer, telemetry).
- **`src/esper/leyline/causal_masks.py`:**
  - `subtarget_idx` relevant only for ENABLE/DISABLE ops; slot relevant for all non-WAIT ops.

**Required telemetry sensors + derived Obs V3 fields:**
- **Obs V3 per-slot scalars (Tamiyo):** in `src/esper/tamiyo/policy/features.py`:
  - `subtarget_active_frac = subtarget_active_count / subtarget_count` (define 0 when `subtarget_count==0`)
  - `subtarget_churn_norm` (e.g., `min(churn_epochs / 25, 1.0)`)
- **Telemetry event (Leyline):** extend `src/esper/leyline/telemetry.py`:
  - `TelemetryEventType.SEED_SUBTARGET_CHANGED`
  - Payload fields: `slot_id, env_id, blueprint_id, subtarget_kind, subtarget_idx, enabled`
- **Emission:** `src/esper/kasmina/slot.py` on subtarget change; **Consumption:** Tamiyo features + Karn dashboards.

**Host changes (A):**
- None (depends on Phases 1–2 ordering stability).

**Seed changes (C):**
- Extend `lora_ladder` and `channel_ladder` to map internal groups to stable subtarget indices.
- Add attention head-group seed only if needed and keep groups small.
- Fossilization freezes subtarget enablement mask.

**Safety constraints:**
- DDP symmetry: subtarget changes are purely action-driven; no rank-local enablement.
- torch.compile: represent enablement masks as device tensors; do not mutate Python lists in the hot path.
- Governor interactions: micro-churn can be reward-hacked; add telemetry + (optional) churn penalty knob.

**Experiment plan:**
- **Tasks:** `cifar_scale` + `tinystories`.
- **Cohorts:** ladder-only (Phase 0/1/2) vs addressable subtargets (Phase 3).
- **Success criteria:**
  - Subtarget ops used with low invalid-action rate.
  - Effective params reduced at equal quality (disable unneeded groups).
  - Head entropy for `subtarget_idx` does not collapse persistently.

**Risks + mitigations:**
- Head entropy collapse: curriculum unlock (enable subtarget ops only after N episodes) + per-head entropy floors (Policy V2 already supports differential entropy).
- K too large: enforce `K≤16` and group aggressively.

**Exit criteria:**
- Stable training with subtarget head enabled; measurable ROI improvement per added action capacity; no persistent governor anomalies.

**Option 3B — focus-based addressing (headless; focus ops + reuse grow/shrink)**

This is a bounded alternative that avoids adding a new action head in Phase 3.

**Action space changes (Leyline):**
- Add two new ops in `src/esper/leyline/factored_actions.py:LifecycleOp`:
  - `FOCUS_NEXT_SUBTARGET`
  - `FOCUS_PREV_SUBTARGET`
- No `subtarget_idx` head. `GROW_INTERNAL/SHRINK_INTERNAL` are reinterpreted as “grow/shrink the focused subtarget” for multi-subtarget seeds.

**Leyline report contract changes (Kasmina → Tamiyo):**
- Reuse `SubtargetKind/subtarget_count/subtarget_active_count` from Option 3A.
- Extend `src/esper/leyline/reports.py:SeedStateReport` with focused-subtarget visibility (few scalars; no per-subtarget vectors):
  - `subtarget_focus_idx: int = 0` (invariant: `0 <= focus_idx < subtarget_count` when `subtarget_count>0`, else `focus_idx=0`)
  - `subtarget_focus_level: int = 0` (level of the focused subtarget)
  - `subtarget_focus_churn_epochs: int = 0` (epochs since focus last changed; optional but helps thrash detection)
- Constraint (to keep contracts small): multi-subtarget seeds must use a **single** `internal_max_level` shared by all subtargets, so Tamiyo can normalize `subtarget_focus_level` without per-subtarget max metadata.

**Telemetry sensors (sensors match capabilities):**
- Extend `src/esper/leyline/telemetry.py`:
  - Add `TelemetryEventType.SEED_SUBTARGET_FOCUS_CHANGED`
  - Payload fields: `slot_id, env_id, blueprint_id, subtarget_kind, from_focus_idx, to_focus_idx`
- For grow/shrink events on focused subtargets, **do not lose the subtarget identity**:
  - Either extend `SEED_SUBTARGET_CHANGED` payload to include `from_level/to_level/max_level` (preferred), or extend `SEED_INTERNAL_LEVEL_CHANGED` payload with `subtarget_kind/subtarget_idx` when the op targets a focused subtarget.

**Obs V3 derived fields (Tamiyo):**
- Add per-slot scalars in `src/esper/tamiyo/policy/features.py`:
  - `subtarget_focus_idx_norm = subtarget_focus_idx / max(subtarget_count-1, 1)`
  - `subtarget_focus_level_norm = subtarget_focus_level / internal_max_level`
  - Keep `subtarget_active_frac` (can be defined as `active_count / count`, where `active_count` is “level>0” count for multi-level subtargets).

**Masking and validity rules (Tamiyo + Simic execution):**
- Focus ops valid only when a seed exposes `subtarget_count > 1`.
- Grow/shrink valid only when the focused subtarget level can change (no boundary NOOPs).

**Risks + mitigations (focus-based):**
- Focus cycling too slow / credit assignment messy:
  - keep `subtarget_count` small and semantically chunky (K≤4–8),
  - require focus-change telemetry + obs to detect thrash,
  - if usage remains low or oscillatory, pivot to Option 3A head-based addressing.

**Decision gate (choose 3A vs 3B):**
- Choose **3B (focus-based)** if Phase 3 needs to stay minimally invasive (no new head) and K is small.
- Choose **3A (head-based)** if focus cycling is too slow or if direct addressing materially improves ROI and stability (and Phase α is already planned for scaling pressure).

---

### Phase 4 — **Slot Transformer policy pivot** (Phase α alignment)

**Objective:** Keep Tamiyo learnable when slot count grows (Track A) and when microstructure richness increases (Track C) by replacing flat concatenation with a tokenized, masked Slot Transformer encoder (ROADMAP Phase α).

**Scope:**
- **Will touch:** Tamiyo policy architecture + Leyline token schema contracts; keep PPO factorization semantics intact if possible.
- **Won’t touch:** core Kasmina mechanics beyond exposing stable ordering/metadata.

**Design sketch:**
- Replace `obs_flat = concat(base, slot_0, slot_1, ...)` with token tensor:
  - `base_token`: global training context (loss/acc history, action feedback).
  - `slot_tokens`: one token per slot, containing per-slot features + blueprint embedding + (optional) surface/order embedding.
  - Optional `subtarget_tokens` only if Phase 3 shows aggregated scalars are insufficient.
- Encoder: small Transformer encoder over tokens with attention masks for variable slot counts.
- Heads:
  - `op` head from base token (global decision).
  - `slot` head as pointer distribution over slot tokens (logit from each slot token).
  - `subtarget_idx` head (if Phase 3) can be conditioned on chosen slot token.

**Leyline contract changes required:**
- Define a token schema contract in `src/esper/leyline/` (new module suggested `token_schema.py`):
  - `MAX_SLOTS` (or derive from SlotConfig) and `TOKEN_FEATURE_DIM`.
  - `SlotTokenFields` (documented list of fields and their normalization).
  - Mask tensor semantics: `token_mask: [batch, n_tokens]` boolean.
- Update Simic rollout buffer schemas to store token-shaped observations without Python-level per-slot loops in the hot path.

**Required telemetry sensors + Obs support:**
- Add observation stats telemetry to verify token shapes + sparsity (Simic already has ObservationStatsTelemetry).
- Karn: attention diagnostics only in “dense trace” mode to avoid overhead.

**Action space changes:**
- Keep head semantics unchanged; only representation changes.
- Causal masks remain the single source of truth (`src/esper/leyline/causal_masks.py`), now applied to token-conditioned heads.

**Host changes (A):**
- None beyond requiring `InjectionSpec.order` for token ordering stability (already introduced in Phase 1).

**Seed changes (C):**
- None required; microstructure becomes easier to exploit because policy can attend to relevant slots instead of consuming a huge flat vector.

**Safety constraints:**
- Preserve inverted control flow: token construction in `src/esper/tamiyo/policy/features.py` must remain CPU-filled + one H2D transfer (no Python writes into GPU tensors).
- DDP symmetry: variable masking must be deterministic given SlotConfig.
- Throughput gate: ≤10% slowdown at same `n_envs` and slot count; if exceeded, reduce encoder depth or token dim.

**Experiment plan:**
- Scale study on CNN and transformer tasks:
  - Slots: 3 vs 12 vs 20 vs 40 (enabled slots; `max_seeds` constant to isolate obs scaling effect).
  - Compare Policy V2 (flat concat) vs Slot Transformer.
- Success criteria:
  - PPO remains stable at ≥32 slots (no persistent entropy collapse).
  - Invalid-action rates do not spike with slot count.
  - Throughput within budget.

**Risks + mitigations:**
- Engineering scope: keep token schema minimal; postpone subtarget tokens until evidence.
- Overfitting to slot order: add slot/surface embeddings and rely on attention rather than positional hacks.

**Exit criteria:**
- Slot Transformer can train stably at high slot counts with acceptable throughput and improves decision quality metrics over flat concat.

---

### Phase 5 — **Full submodule morphogenesis** (end-state)

**Objective:** Reach an end-state where “submodule intervention” is not only about host surfaces, but also about seeds that expose their own internal microstructure under contract (ROADMAP Phase 6 direction), without introducing per-parameter free-form edits.

**Scope:**
- **Will:** define “container seed” families that expose internal injection surfaces and micro-lifecycles under Leyline contracts; two-tier fossilization (freeze microstructure then freeze macro integration).
- **Won’t:** arbitrary parameter edits, dynamic hooks, or backwards-compatible dual systems.

**Design sketch:**
- A container seed behaves like a small internal morphogenetic plane:
  - Exposes `injection_specs()` for its internal submodules (micro-lattice).
  - Supports a bounded set of internal ops (reuse Phase 3 subtarget addressing where possible).
  - Fossilization is staged: microstructure freezes before the seed is fossilized at the host surface.

**Telemetry + Obs support:**
- Always-on aggregated microstructure sensors (counts, churn, effective params).
- Dense per-subtarget traces only when governor/anomaly triggers fire (avoid overhead).

**Action space changes:**
- Hierarchical decision: (slot → op) plus optional microtarget selection (bounded K, masked).
- Likely requires Phase α token policy to remain learnable.

**Host changes (A):**
- Host lattices remain the macro “morphogenetic plane” (`src/esper/kasmina/host.py`); requirement is continued deterministic `InjectionSpec.order` stability and slot ordering.

**Seed changes (C):**
- Introduce at least one concrete “container seed” family (transformer-first recommended) that exposes internal `injection_specs()` under Leyline contracts and uses the same SeedSlot lifecycle semantics internally (no new metaphors; botanical only for lifecycle).

**Safety constraints:**
- Deterministic ordering at both macro and micro levels (explicit `order` fields; no float ties).
- DDP symmetry and torch.compile graph growth bounds (cap micro-lattice size).

**Experiment plan:**
- Storyboard run (single task, fixed seed budget):
  - Underfit host → grow microstructure → disable subtargets to reduce rent while holding quality.
  - Produce a replayable forensic trace: which micro-ops happened when, and what ROI they produced.

**Exit criteria:**
- Demonstrated cost reduction at equal quality with stable governor outcomes and explainable telemetry across one representative run.

---

## 5) “First experiment pack” (exact runs to do first, expected outcomes, what telemetry to watch)

> Note: Phase 0 adds new `LifecycleOp`s, which changes `NUM_OPS` and therefore the Obs V3 base feature width (`last_action_op` one-hot). Treat Phase 0 as a **new training surface**: do not compare against old checkpoints; compare against fresh-run controls.

**Run 1 — Control (A0 only, current baseline):**

```bash
PYTHONPATH=src uv run python -m esper.scripts.train ppo \
  --preset cifar_scale --task cifar_scale \
	  --slots r0c0 r0c1 r0c2 r0c3 r0c4 \
	  --max-seeds 2 \
	  --rounds 100 --envs 8 --episode-length 150 \
	  --dual-ab shaped-vs-simplified \
	  --sanctum
```

Expected outcomes:
- Blueprint usage skews toward “heavier” seeds when signal is needed.
- Slot usage concentrates in early/mid injection points; later slots under-used unless rent allows.

Telemetry to watch:
- Head entropies (slot/op/blueprint/style/alpha heads), decision success rates
- `seed_lifecycle` outcomes: prune/fossilize ratios, stage durations
- Rent proxies (`effective_seed_params`) and alpha shock (`alpha_delta_sq_sum`)
- Governor events (`GOVERNOR_ROLLBACK`) and anomaly flags

**Run 2 — A0 stress (slot/seed pressure):**

```bash
PYTHONPATH=src uv run python -m esper.scripts.train ppo \
  --preset cifar_scale --task cifar_scale \
	  --slots r0c0 r0c1 r0c2 r0c3 r0c4 \
	  --max-seeds 4 \
	  --rounds 100 --envs 8 --episode-length 200 \
	  --dual-ab shaped-vs-simplified \
	  --sanctum
```

Expected outcomes:
- Higher prune rate; entropy pressure and potential collapse in slot/op heads if observation scaling is insufficient.

Telemetry to watch:
- Entropy collapse diagnostics; invalid-action frequency (`last_action_success`)
- Governor rollback frequency and gradient pathology events

**Run 3 — Phase 0 validation (A0 + C0) once ladder + internal ops exist:**

Repeat Run 1, but validate the new signals exist:
- `TelemetryEventType.SEED_INTERNAL_LEVEL_CHANGED` events appear (Karn/Sanctum decision timeline)
- Obs V3 includes `internal_level_norm` and the updated `NUM_OPS`-sized last_action one-hot (observation stats telemetry)
- Internal ops appear in decision logs and correlate with ROI (more “small bets”, fewer blueprint jumps)

Expected outcomes:
- Reduced “conv_heavy dominance”
- More frequent small interventions (`GROW/SHRINK_INTERNAL`) rather than germinate/prune churn

Telemetry to watch (additions):
- `SEED_INTERNAL_LEVEL_CHANGED` rate and `(from_level,to_level)` distribution
- Op-conditioned Q-values include new internal ops (update required in `src/esper/simic/agent/ppo.py` metrics emission)

**Run 4 — Transformer A0 sanity (per-layer boundaries) once `tinystories_layerwise` exists:**

```bash
PYTHONPATH=src uv run python -m esper.scripts.train ppo \
  --preset tinystories --task tinystories_layerwise \
  --slots r0c0 r0c1 r0c2 r0c3 r0c4 r0c5 \
  --max-seeds 2 \
  --rounds 100 --envs 8 --episode-length 150 \
  --sanctum
```

Expected outcomes:
- Slot selection policy differentiates earlier vs later layers even with boundary-only semantics.

Telemetry to watch:
- Slot-head entropy and slot-choice distribution vs layer depth.

**Run 5 — Transformer A1 surfaces (Phase 1) once post-attn/post-MLP lattice exists:**

Repeat Run 4 with the A1 slot set enabled (post-attn only, then post-attn+post-MLP) and compare:
- Surface preference emergence (`POST_ATTN` vs `POST_MLP`)
- LoRA ladder growth usage vs immediately choosing heavy adapters
- Throughput impact from slot count increase (watch step time)

---

## 6) Open questions / unknowns list (ranked by risk)

1. **Credit assignment vs churn pricing:** Do internal ops create reward-hackable “micro-churn loops”, and what minimal penalty/telemetry makes this learnable without strangling exploration?
2. **Obs scaling trigger (Phase α):** What objective threshold (slot count, obs dim, entropy collapse rate, throughput) triggers Slot Transformer pivot with highest confidence?
3. **DDP determinism proof:** Where do we assert and test that internal_level/subtarget state is synchronized (action-stream broadcast vs explicit sync helper), per `src/esper/kasmina/slot.py` symmetry rules?
4. **Rent semantics for incubator compute:** Should internal capacity cost apply even at `alpha=0` (incubator still computes), or remain alpha-weighted as in `compute_rent_and_shock_inputs()` (`src/esper/simic/training/helpers.py`)?
5. **Transformer surfaces curriculum:** Is “post-attn only first” strictly better for stability than enabling both surfaces immediately?
6. **Telemetry granularity trade:** Can aggregated subtarget sensors preserve learnability, while dense per-subtarget traces remain anomaly-triggered, without violating commandment #1?
7. **Deterministic ordering contract:** Is `InjectionSpec.order` sufficient across all hosts, or do we need explicit `(row,col)` fields to prevent accidental drift in `SlotConfig.from_specs()`?
8. **Internal design choice for CNN:** Is “conv micro-block ladder” enough, or does ROI require channel-group ladders to be materially cheaper (compute) as well as lighter (params)?

---

## 7) Recommendation (your chosen path and why)

Ship **Phase 0 (A0 + C0)** first: add `conv_ladder` + `GROW/SHRINK_INTERNAL` + `internal_level_norm` + `SEED_INTERNAL_LEVEL_CHANGED`, using existing denser CNN injection points (`cifar_scale`). This is the minimum implementable change that directly reduces “conv_heavy dominance” while respecting strict contracts (Leyline owns enums/fields; no silent fallbacks) and preserving GPU-first training (`src/esper/simic/training/vectorized.py`). Then land **Phase 1 (A1 + C1)** as the transformer-aligned track: `InjectionSpec.order/surface` + post-attn/post-MLP lattice in `TransformerHost` + `lora_ladder`.

**Pivot triggers:**
- If internal ops show oscillatory thrash (high `SEED_INTERNAL_LEVEL_CHANGED` with no ROI gain), increase internal-op intervention cost and/or restrict internal ops to `alpha_mode==HOLD` stages until stable.
- If slot count ≥12 causes sustained entropy collapse or throughput regression, start **Phase α (Slot Transformer)** immediately (do not keep bolting scalars onto flat concat).
- If A1 post-MLP surfaces destabilize early, run curriculum: `POST_ATTN` only first, then unlock `POST_MLP` after reward exam stability.
