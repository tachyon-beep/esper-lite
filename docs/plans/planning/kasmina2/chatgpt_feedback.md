1. One-paragraph executive summary (what we’re building and why it fixes “conv_heavy dominance”)

We’re building a controlled “finer motor cortex” for Tamiyo: (A) many more *submodule-meaningful* injection surfaces (so Tamiyo can act frequently in small places instead of rarely at big block boundaries) and (C) *ladder seeds* whose capacity can be adjusted via a tiny set of internal levers (so Tamiyo can “turn the dial” on an existing seed—grow/shrink/enable/disable—rather than repeatedly paying for a whole conv_heavy germination just to get a learnable signal). The net effect is that adaptation pressure shifts from “buy one big organ” to “tune many small tissues”: better ROI (reward per params/FLOPs), lower churn, and far less conv_heavy dominance—because the system finally provides enough action surfaces and cheap internal degrees of freedom *with matching telemetry* to make incremental interventions learnable and stable under strict contracts + DDP symmetry.

---

1. A vs C design taxonomy (options table with pros/cons)

### Track A — injection surfaces (denser / finer injection points)

| Option                                                      | What it is                                                                                                                           | Pros                                                                                                 | Cons / gotchas                                                                | “Best use” phase                      |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ------------------------------------- |
| A0: “More segments” (already supported)                     | CNN: more blocks. Transformer: `num_segments = n_layer` (per-layer)                                                                  | Minimal contract churn; keeps compute predictable; aligns with existing `runtime/tasks.py` “A0” path | Still coarse inside a layer/block; slot count rises quickly                   | Phase 0                               |
| A1: Residual-stream submodule sites                         | Transformer per-layer *multi-site*: post-attn vs post-MLP (and optionally pre-attn). CNN intra-block: after conv1/conv2/residual-add | Submodule-meaningful boundaries without brittle hooks; stable tensor shapes; easy SurfaceUID scheme  | Requires injection_spec to encode `site`; doubles/triples surfaces            | Phase 1                               |
| A2: Projection-level sites (structured wrappers, not hooks) | Transformer: QKV/O, MLP up/down projection surfaces                                                                                  | Extremely targeted; pairs perfectly with LoRA ladders; strong signal with tiny params                | More invasive blueprint/host refactor; must preserve compile + DDP invariants | Phase 4                               |
| A3: Micro-sites (per-head/per-channel surfaces)             | Many micro-slots across heads/channels                                                                                               | Max controllability                                                                                  | Action space + telemetry overhead explodes; credit assignment pain            | Likely never (or only research spike) |

### Track C — lighter seeds with internal microstructure + controllable internal levers

| Option                                         | What it is                                                                                                                     | Pros                                                                                    | Cons / gotchas                                                                            | “Best use” phase        |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ----------------------- |
| C0: Single “prefix ladder” internal_level      | One scalar `internal_level ∈ [0..L]` activates a *prefix* of capacity (rank groups / channel groups)                           | Tiny action change (grow/shrink); fixed shapes → DDP + compile friendly; easy telemetry | Limited expressivity (can’t choose *which* sub-part gets capacity)                        | Phase 0                 |
| C1: Ladder families per host                   | CNN channel-group ladder + Transformer LoRA rank ladder                                                                        | Avoids conv_heavy; unifies semantics across hosts                                       | Requires consistent InternalLeverSpec + sensors                                           | Phase 1                 |
| C2: Multi-ladder with deterministic subtargets | A seed has a small set (≤6–8) subtargets, each with its own ladder; Tamiyo changes “focus” and grows/shrinks focused subtarget | Submodule control without huge combinatorics; stable SubtargetUIDs                      | Must expose focus telemetry + obs; more state → more ways to desync if sloppy             | Phase 2                 |
| C3: Continuous gating vectors                  | Policy outputs K gates                                                                                                         | Very flexible                                                                           | PPO stability risk; action/obs bandwidth; can become “silent bug-hiding” if gates ignored | Maybe later, cautiously |

**Recommended taxonomy path:** A0 → A1 → (C2) → Slot Transformer scaling → A2 (projection sites) → full morphogenesis. The key is *prefix ladders first* because they’re the smallest lever that produces learnable “cheap adjustments” while staying friendly to strict contracts, compile, and DDP symmetry.

---

1. Roadmap table (phase, duration, dependencies, risk level, success metric)

> “Duration” here is *relative engineering effort / integration complexity*, not a wall-clock promise.

| Phase                                                              | Duration | Dependencies                     | Risk level  | Success metric (ROI-style + decision quality)                                                                      |
| ------------------------------------------------------------------ | -------- | -------------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------ |
| 0 — **A0 + C0** minimal                                            | Small    | None (uses existing A0 surfaces) | Low         | ≥ same return with **↓ conv_heavy param share** and **↑ internal grow/shrink utilisation**; no DDP divergence      |
| 1 — **A1 surfaces** + C1 ladder families                           | Medium   | Phase 0                          | Medium      | Return ↑ per added FLOP/param; telemetry overhead within budget; action entropy stays healthy                      |
| 2 — **C2 deterministic subtargets** (multi-ladder focus)           | Medium   | Phase 1                          | Medium–High | Capacity allocated *meaningfully* (focused ladders correlate with benefit); churn ↓; governor veto rate low        |
| 3 — **Slot Transformer obs/encoder** (scaling)                     | Large    | Phase 1–2 slot count pressure    | Medium      | Matches/exceeds baseline at high slot counts; flat-concat retired cleanly; throughput remains GPU-first            |
| 4 — **A2 projection-level injection** + C3 per-projection ladders  | Large    | Phase 3 strongly recommended     | High        | Tiny seeds beat heavy seeds on ROI; per-projection surfaces show strong, stable signal; compile/DDP stable         |
| 5 — **Full submodule morphogenesis** (bud/merge/fossilise economy) | Large    | Phases 3–4                       | High        | “Rent & churn economy” stabilises: low churn, high reuse, high ROI; fossilise ratios sensible; no entropy collapse |

---

1. Per-phase spec blocks (as described above)

## Phase 0 — **Hard requirement milestone: A0 + C0 (minimal, concrete, no re-architecture)**

### Objective (what capability is added)

Enable **more frequent, less weighty interventions** by (A0) increasing surface count using already-supported host segmentation and (C0) introducing **one ladder seed** with a single scalar internal lever controllable via 1–2 new lifecycle ops.

### Scope (what you will/won’t touch)

* **Will touch**

  * Leyline: add minimal internal-lever contract surface (InternalLeverSpec + ops).
  * Kasmina: add internal_level plumbing to SeedSlot lifecycle + one new seed family.
  * Tamiyo: add a single per-slot obs scalar + support new ops in action head.
  * Telemetry (Nissa/Karn path): add internal lever vitals.
* **Won’t touch**

  * No new injection sites inside blocks/layers yet (use A0 only).
  * No Slot Transformer encoder yet (stay with current Obs V3 layout).
  * No projection-level wrappers.

### Design sketch (contracts, dataflow, how Tamiyo acts)

* **A0 surfaces**

  * TransformerHost: configure `num_segments = n_layer` (per-layer boundaries) *if that path already exists*.
  * CNNHost: use existing deeper hosts / more blocks path referenced in `src/esper/runtime/tasks.py` (“A0”).
* **C0 ladder seed**

  * Seed exposes `internal_level ∈ {0..Lmax}`; effect is monotonic with level.
  * Implementation strategy (DDP/compile friendly): allocate parameters for **max** capacity up front; apply a **prefix mask** based on internal_level so shapes never change.
* **Tamiyo behaviour**

  * Instead of germinating conv_heavy to get signal, Tamiyo can:

    1. germinate a light ladder seed at an existing surface
    2. incrementally `GROW_INTERNAL` to dial in capacity
    3. `SHRINK_INTERNAL` when rent exceeds benefit.

### Required telemetry sensors + derived observation fields (exact fields; where emitted; where consumed)

**Emitted (Kasmina; at application time of seed inside host forward)**

* `slot.surface_uid` (int32) — stable surface identity (even if only coarse initially)
* `slot.seed_state` (enum) — existing lifecycle state
* `slot.seed_param_count_active` (int32) — active params given internal_level mask
* `slot.seed_flops_active_est` (float32) — estimated active FLOPs
* `slot.internal_level` (int16)
* `slot.internal_level_norm` (float32) = internal_level / Lmax
* `slot.delta_l2_norm` (float32) — ‖y−x‖₂ (or normalised by ‖x‖₂)
* `slot.output_l2_norm` (float32)

**Consumed (Tamiyo; `src/esper/tamiyo/policy/features.py`)**

* Add **one new scalar** into per-slot features:

  * `obs.slot_internal_level_norm` (float32)
* Strongly recommended (still Phase 0-safe) add:

  * `obs.slot_delta_l2_norm` (float32)
    because “grow/shrink” is otherwise blind.

### Action space changes (new op(s) vs new head; justify to avoid combinatorial blow-up)

* Add **two new lifecycle ops** in `src/esper/leyline/factored_actions.py`:

  * `GROW_INTERNAL` (internal_level += 1, clamped)
  * `SHRINK_INTERNAL` (internal_level -= 1, clamped)
* No extra action heads. No indices. This is the smallest possible change that still creates a learnable internal lever.

### Host changes (A): injection_specs changes; slot ID scheme; how positions are ordered

* **No new site types** yet.
* Ensure `injection_specs` ordering is deterministic:

  * Transformer: ordered by layer index.
  * CNN: ordered by block index.
* Define **SurfaceUID** now even if coarse, so it can remain stable later:

  * `surface_uid = host_kind_offset + depth_idx`
  * (Do **not** use Python’s `hash()`; use a deterministic arithmetic encoding.)

### Seed changes (C): at least one concrete seed family with internal levers; fossilization semantics (merge vs keep)

* **Concrete seed family (pick one for C0):** `lora_rank_ladder`

  * Applies LoRA-style low-rank update to a chosen linear surface inside the seed (initially you can keep it “seed-internal” even if not projection-level yet).
  * `Lmax` corresponds to maximum rank groups (e.g., 4 levels × 2 rank each).
* **Fossilisation semantics**

  * Phase 0: *keep*, don’t merge. Fossilised means:

    * internal_level becomes immutable
    * parameters stop receiving updates (or are excluded from optimiser)
    * forward path remains identical (mask fixed)

### Safety constraints (DDP symmetry, compile behavior, governor interactions)

* **DDP symmetry**

  * internal_level updates must be applied identically across ranks (whatever mechanism SeedSlot already uses for lifecycle ops must include these new ops).
* **Compile stability**

  * No dynamic parameter creation; masking only.
* **Governor**

  * Governor must see `seed_param_count_active` / `seed_flops_active_est` change when internal_level changes.
  * If governor vetoes grow: emit explicit `slot.governor_veto_internal_grow = 1` (no silent NOOP).

### Experiment plan (tasks/presets, run lengths, cohorts, what success looks like)

* **Cohorts (minimum)**

  1. Baseline: current seed library, current A surfaces
  2. A0 only: deeper/more-segment hosts, current seeds
  3. C0 only: ladder seed enabled, baseline surfaces
  4. A0 + C0: both enabled
* **Run length**

  * Two-stage:

    * Smoke: short rollouts to validate contracts/telemetry (catch fast-fail)
    * Learning: fixed environment-step budget (same across cohorts)
* **Success looks like**

  * Same-or-better return with:

    * **↓ conv_heavy germination count**
    * **↑ usage of grow/shrink**
    * **↑ reward-per-active-param** (ROI)
    * No entropy collapse; veto rate not pathological

### Risks + mitigations (especially action-space explosion, credit assignment, telemetry overhead)

* **Risk:** policy ignores grow/shrink (no signal)

  * **Mitigation:** include `slot_delta_l2_norm` + param/FLOPs vitals so the critic can learn value of growth.
* **Risk:** telemetry overhead

  * **Mitigation:** compute norms in a vectorised way; avoid per-slot Python.
* **Risk:** governor-veto ambiguity

  * **Mitigation:** explicit veto telemetry + obs bit.

### Exit criteria (objective gates to unlock next phase)

* `GROW_INTERNAL` / `SHRINK_INTERNAL` used in ≥X% of episodes (non-zero, non-trivial).
* conv_heavy share of active params drops materially **without** return dropping.
* No DDP divergence warnings; compile path unchanged.

---

## Phase 1 — A1 submodule-meaningful surfaces + C1 ladder families (Transformer + CNN)

### Objective

Move from “more places” to **better places**: introduce submodule-meaningful injection sites (Transformer post-attn vs post-MLP; CNN intra-block splits) and give both host types a ladder seed family.

### Scope

* **Will touch**

  * Leyline `injection_spec.py`: add `InjectionSite` enum + include in spec.
  * Kasmina hosts/blueprints: generate injection_specs with per-layer/per-block *multi-site*.
  * Tamiyo obs: include site/depth descriptors (lightweight).
* **Won’t touch**

  * No Slot Transformer yet unless slot count crosses a pain threshold.
  * No projection-level wrappers yet.

### Design sketch

* **TransformerHost surfaces**

  * For each layer `ℓ`, create sites:

    * `POST_ATTN` (after attention residual add)
    * `POST_MLP` (after MLP residual add)
  * (Optionally `PRE_ATTN` later if needed, but start with two.)
* **CNNHost surfaces**

  * For each residual block `b`, create sites:

    * `POST_CONV1`
    * `POST_CONV2`
    * (Optional) `POST_RESID_ADD`
  * Do this via explicit blueprint structure (no brittle runtime hooks).
* **Stable ordering**

  * Primary: depth index (layer/block)
  * Secondary: site enum order

### Required telemetry + derived obs fields

**Emit**

* `slot.site_id` (uint8) — enum value
* `slot.depth_idx` (int16)
* `slot.depth_norm` (float32)
* Keep Phase 0 fields.

**Obs additions (Tamiyo features)**

* `obs.slot_depth_norm` (float32)
* `obs.slot_site_onehot` **or** `obs.slot_site_id_embed` (prefer embed to avoid widening too much)

### Action space changes

* None beyond Phase 0 (still grow/shrink internal).

### Host changes (A)

* `InjectionSpec` gains:

  * `site: InjectionSite`
  * `depth_idx: int`
  * `surface_uid: int32` computed deterministically from `(host_kind, depth_idx, site)`
* Slot IDs:

  * `slot_index` remains array index
  * `surface_uid` is the stable identity for telemetry/obs

### Seed changes (C)

* Add a CNN ladder family: `conv_channel_ladder`

  * internal_level controls active channel groups (prefix groups)
* Maintain LoRA ladder for Transformer (or add Transformer-specific ladder if Phase 0 was CNN-first)

### Fossilisation semantics

* Still “keep, don’t merge”.
* Fossilised slots continue to report vitals but disallow lifecycle changes (explicitly masked).

### Safety constraints

* Slot count increases: ensure vectorised storage is preallocated and consistent across DDP.
* Ensure action masking includes site-specific constraints if any (e.g., some seeds valid only at certain sites).

### Experiment plan

* Compare:

  * Baseline A0 surfaces vs A1 multi-site surfaces at equal compute budget.
  * Ladder seeds vs heavy seeds under same governor budget.
* Key outcomes:

  * Does Tamiyo start distributing light adaptations across sites instead of buying conv_heavy?

### Risks + mitigations

* **Risk:** slot explosion hurts policy scaling

  * **Mitigation:** set a threshold that triggers Phase 3 earlier (see below).
* **Risk:** CNN intra-block sites become brittle

  * **Mitigation:** only expose sites that are already explicit modules in the blueprint; no reflection/hooking.

### Exit criteria

* Clear evidence that Tamiyo uses distinct sites differently (site-conditioned behaviour in telemetry).
* ROI improves vs baseline for same governor budget.

---

## Phase 2 — C2 deterministic subtargets inside seeds (multi-ladder focus, minimal action growth)

### Objective

Enable **submodule control inside a seed**: Tamiyo can allocate internal capacity to a *chosen internal subtarget* (e.g., attention vs MLP, or QKV vs O) without exploding action space.

### Scope

* **Will touch**

  * Leyline: add Subtarget contracts (IDs, kinds) + minimal focus ops.
  * Seeds: implement multi-ladder microstructure.
  * Obs/telemetry: expose focus + focused ladder state.
* **Won’t touch**

  * No new host surfaces required (works on A1 sites).
  * Avoid continuous gate vectors.

### Design sketch

* Each ladder seed defines a **small fixed subtarget set** `K ≤ 6–8`.

  * Transformer LoRA multi-ladder example subtargets:

    * `ATTN_QKV` (grouped)
    * `ATTN_O`
    * `MLP_UP`
    * `MLP_DOWN`
  * CNN channel multi-ladder example:

    * `CONV1_GROUPS`
    * `CONV2_GROUPS`
* Seed maintains:

  * `focus_idx ∈ [0..K-1]`
  * `internal_level[k]` for each subtarget k
* Tamiyo acts with:

  * `FOCUS_NEXT` / `FOCUS_PREV` (cycle)
  * existing `GROW_INTERNAL` / `SHRINK_INTERNAL` apply to the focused subtarget

This keeps action space growth tiny (just two extra ops).

### Required telemetry + derived obs fields

**Emit**

* `slot.focus_idx` (uint8)
* `slot.focus_subtarget_uid` (int32)
* `slot.focus_internal_level_norm` (float32)
* `slot.internal_level_mean_norm` (float32)
* `slot.internal_level_max_norm` (float32)
* (Optional, but powerful) `slot.focus_delta_l2_norm` (float32)

**Obs additions**

* `obs.slot_focus_internal_level_norm`
* `obs.slot_internal_level_mean_norm`
* `obs.slot_internal_level_max_norm`
* `obs.slot_focus_subtarget_embed` (embedding for `focus_subtarget_uid` or `subtarget_kind`)

### Action space changes

* Add two lifecycle ops:

  * `FOCUS_NEXT_SUBTARGET`
  * `FOCUS_PREV_SUBTARGET`
* Reuse grow/shrink ops (no extra heads, no indices).

### Host changes (A)

* None required, but `InjectionSpec` should already carry stable IDs so subtarget telemetry can be associated cleanly per slot.

### Seed changes (C)

* Implement `lora_multi_ladder` and/or `conv_multi_ladder`:

  * Fixed max capacity per subtarget
  * Prefix mask per subtarget internal_level[k]

### Fossilisation semantics

* Phase 2 option:

  * fossilise locks focus + all internal levels
  * (No per-subtarget fossilisation yet; keep it simple.)

### Safety constraints

* Focus changes must be DDP-synchronised like any other lifecycle state update.
* Strict contract: only seeds declaring `K>1` support focus ops; mask ops otherwise (explicitly).

### Experiment plan

* Compare C0 vs C2 at equal budgets:

  * Does multi-ladder allocate capacity to the “right” subtargets?
* Validate with telemetry correlations:

  * Growth events should coincide with increased marginal return / value estimate.

### Risks + mitigations

* **Risk:** focus cycling too slow / credit assignment messy

  * **Mitigation:** keep K small and semantically chunky (4–6).
* **Risk:** obs insufficient to learn focus utility

  * **Mitigation:** focus-specific delta norm + focused level norm are mandatory sensors.

### Exit criteria

* Focus ops used non-trivially.
* Distribution of internal levels across subtargets is non-degenerate and stable.
* ROI improves vs single-ladder.

---

## Phase 3 — Slot Transformer pivot for Tamiyo (scaling and learned slot-slot interactions)

### Objective

Make policy scalable as slot count rises: move off flat concatenation obs to a **masked Slot Transformer** encoder that can handle many slots and learn cross-slot interactions.

### Scope

* **Will touch**

  * Tamiyo encoder (Policy V2 → “Slot Transformer track”).
  * Obs V3 representation format (from flat concat to per-slot tensor + mask).
  * Critic path (op-conditioned critic must consume the same slot embeddings).
* **Won’t touch**

  * Leyline contracts beyond what’s needed to represent slot tensors/masks cleanly.
  * Host/seed mechanics should remain stable during this phase.

### Design sketch

* Observations become:

  * `slot_features: [B, S, F]`
  * `slot_mask: [B, S]` (1 = valid)
* Encoder:

  * Per-slot MLP → embeddings
  * Add positional/site encoding derived from `surface_uid`, `depth_norm`, `site_id`
  * Slot Transformer layers with masking
* Outputs:

  * Per-slot action logits (shared head applied to each slot embedding)
  * Global value (pool slot embeddings + op-conditioning)

### Required telemetry + derived obs fields

* Ensure **slot_mask** and `surface_uid` are present and stable.
* Derived obs:

  * `obs.global_slot_count` (scalar) can help with normalisation.

### Action space changes

* None.

### Host changes (A)

* Ensure the slot list order is deterministic and matches `surface_uid` ordering so positional encodings are stable.

### Seed changes (C)

* None.

### Safety constraints

* Preserve GPU-first vectorisation: no Python loops over slots in the policy forward.
* DDP symmetry: mask and ordering must be identical across ranks.

### Experiment plan

* A/B:

  * Flat concat vs Slot Transformer at the same slot count (S).
  * Scaling test: S small → S large.
* Watch:

  * throughput regressions
  * entropy collapse
  * ability to keep decision quality as S increases

### Risks + mitigations

* **Risk:** regression from architecture swap

  * **Mitigation:** keep heads + critic conditioning identical; only swap encoder.
* **Risk:** hidden Python bottleneck in feature building

  * **Mitigation:** enforce tensorised feature packing in Kasmina/Tamiyo boundary.

### Exit criteria

* Slot Transformer matches or exceeds baseline performance at S where concat starts to wobble.
* Stable training, healthy entropy, no throughput cliffs.

**Rule of thumb for “mandatory to move off flat concat”:** once you routinely exceed ~64 slots *or* you add per-layer multi-site in a ≥24-layer Transformer (because you’ll hit 48–72 slots fast), Slot Transformer stops being “nice” and becomes “necessary”.

---

## Phase 4 — A2 projection-level injection surfaces + per-projection ladders (true submodule morphogenesis)

### Objective

Give Tamiyo truly submodule-level surfaces (QKV/O, MLP projections) so tiny ladder seeds can produce strong, targeted signal without buying conv_heavy.

### Scope

* **Will touch**

  * Transformer blueprint + host to expose structured projection-level injection points (no hooks).
  * Leyline injection_spec to encode `submodule_role`.
  * New seed family optimised for projection sites.
* **Won’t touch**

  * Don’t add per-head micro-slots (keep it bounded).
  * Avoid continuous gate vectors unless absolutely required.

### Design sketch

* Extend `InjectionSpec` with:

  * `site` (existing)
  * `submodule_role` (new enum), e.g.:

    * `ATTN_QKV_PROJ`
    * `ATTN_O_PROJ`
    * `MLP_UP_PROJ`
    * `MLP_DOWN_PROJ`
* Each projection surface hosts a **microseed**:

  * `lora_proj_ladder` with small `Lmax`, strong regularisation, explicit ROI sensors.
* This makes “internal levers” line up with “where the computation actually lives”.

### Required telemetry + obs fields

**Emit**

* `slot.submodule_role_id` (uint8)
* `slot.weight_delta_norm_est` (float32) (if available cheaply)
* `slot.proj_active_rank` (int16)
* `slot.proj_effect_norm` (float32)

**Obs**

* `obs.slot_submodule_role_embed`
* Continue internal-level features.

### Action space changes

* None required beyond earlier phases; grow/shrink remains the workhorse.

### Host changes (A)

* Injection specs now include projection roles; ordering becomes:

  * depth_idx → (attn roles) → (mlp roles)
* Stable SurfaceUID must incorporate role:

  * `surface_uid = pack(host_kind, depth_idx, site_id, role_id)`

### Seed changes (C)

* `lora_proj_ladder` (projection-specific)

  * Optionally different ladder schedules per role (e.g., smaller for QKV, larger for MLP), but keep in config, not hard-coded magic.

### Fossilisation semantics

* Still avoid online merging.
* Introduce *optional* “compile-time merge plan” concept (offline): a fossilised LoRA can be merged into base weights when exporting a frozen organism. (Keep training-time shapes stable.)

### Safety constraints

* Must preserve compile friendliness: projection wrappers are explicit modules, created at init, no runtime graph surgery.
* DDP: all ranks must have identical wrapper structure and slot list.

### Experiment plan

* Compare:

  * Residual-stream A1 sites vs projection A2 sites under same rent budget.
* Look for:

  * higher ROI from microseeds
  * less need to spawn many slots
  * reduced conv-heavy dependence

### Risks + mitigations

* **Risk:** invasive refactor breaks invariants

  * **Mitigation:** do it behind a clean Leyline contract bump and delete old path (no dual-path shims).
* **Risk:** telemetry too expensive

  * **Mitigation:** prefer activation-space effect norms over weight norms; estimate FLOPs/params analytically.

### Exit criteria

* Projection-level interventions show clear advantage (ROI and stability).
* No compile/DDP regressions.

---

## Phase 5 — Full submodule morphogenesis (rent & churn economy, bud/merge/fossilise discipline)

### Objective

Turn A + C into a coherent economy: many stable surfaces + microstructured seeds that can grow, shrink, bud into adjacent surfaces, and fossilise, while governor keeps the organism within rent limits.

### Scope

* **Will touch**

  * Add a small set of lifecycle ops enabling controlled expansion/consolidation.
  * Add stronger ROI telemetry and decision-quality signals.
* **Won’t touch**

  * No unbounded slot creation: all surfaces exist up front; “budding” activates an empty predeclared slot (DDP-safe).

### Design sketch

* New ops (minimal, deterministic, bounded):

  * `BUD_NEXT_SURFACE` / `BUD_PREV_SURFACE`
    (germinates a seed in the next/prev empty slot in the deterministic ordering)
  * `CONSOLIDATE_WITH_NEXT`
    (prune neighbour + grow internal here, or vice versa; deterministic)
* Governor mediates bud/consolidate based on ROI and rent.
* Tamiyo learns a stable rhythm:

  * explore with small buds
  * grow internal where signal is good
  * consolidate and fossilise where stable

### Telemetry + obs

**Emit**

* `slot.roi_estimate_ema` (float32): Δreturn proxy / active_param (or / flops)
* `slot.churn_ema` (float32)
* `slot.governor_event_counts` (vector or summarised scalars)
* `global.active_param_budget_util` (float32)
* `global.active_flops_budget_util` (float32)

**Obs**

* Provide budget utilisation scalars globally and/or per-slot “budget pressure”.

### Action space changes

* Add 2–3 deterministic adjacency ops (bounded, no indices).
* Keep everything else stable.

### Host changes (A)

* None (surfaces are already dense and stable).

### Seed changes (C)

* Seeds must support being budded with:

  * consistent initial internal_level (probably 0 or 1)
  * predictable initialisation (deterministic seed init if needed for DDP parity)

### Safety constraints

* Strong DDP symmetry requirements: budding/consolidation must be synchronised.
* Governor vetoes must be explicit and learnable.

### Experiment plan

* Evaluate rent & churn economy:

  * Does churn drop over training while ROI stays high?
  * Are fossilised slots stable and beneficial?

### Risks + mitigations

* **Risk:** credit assignment across bud/consolidate is hard

  * **Mitigation:** deterministic adjacency + strong ROI telemetry + slot transformer context.
* **Risk:** action entropy collapse (policy gets “stuck” fossilising everything)

  * **Mitigation:** add explicit entropy/degeneracy monitors as gates.

### Exit criteria

* Stable, interpretable economy:

  * low churn
  * high reuse
  * high ROI
  * minimal conv_heavy dependence
  * consistent governor behaviour

---

1. “First experiment pack” (exact runs to do first, expected outcomes, what telemetry to watch)

### Pack 0 — Phase 0 validation grid (must run before Phase 1)

**Common settings across runs**

* Same environment/task preset family (baseline)
* Same PPO/Simic hyperparams
* Same total environment-step budget per run
* ≥3 random seeds per cohort (to catch “one lucky organism” syndrome)

**Runs**

1. **Baseline**: current system (Obs V3, Policy V2), current hosts, current seed library

   * Expected: conv_heavy dominates when it wants signal.
2. **A0-only**: deeper CNN and/or Transformer per-layer segmentation, current seeds

   * Expected: some improvement from more surfaces, but still heavy-seed bias.
3. **C0-only**: ladder seed enabled + new ops, baseline surfaces

   * Expected: grow/shrink starts being used if telemetry is sufficient; performance similar.
4. **A0+C0**: A0 surfaces + ladder seed + grow/shrink

   * Expected: **largest drop** in conv_heavy share with equal or better return.

**Telemetry to watch (must-have dashboards)**

* **Rent & churn economy**

  * `active_param_count_total`, `active_flops_est_total`
  * `lifecycle_op_counts` including new grow/shrink
  * `conv_heavy_active_param_share`
* **Decision quality**

  * policy entropy per head
  * invalid-action rate (should be near zero with masks)
  * governor veto counts (especially internal grow veto)
* **Learnability**

  * correlation: grow events ↔ subsequent return improvement (lagged)
  * `slot_delta_l2_norm` distributions vs internal_level changes
* **Stability**

  * SeedSlot DDP divergence sentinel (whatever hash/guard exists)
  * fossilisation ratio (if used)

### Pack 1 — Phase 1 site validation (after Pack 0 passes)

* Compare A0 vs A1 sites at equal rent.
* Expected: Tamiyo differentiates post-attn vs post-MLP sites (different grow patterns, different ROI).

### Pack 2 — Phase 2 subtarget focus validation

* C0 ladder vs C2 multi-ladder focus under A1 sites.
* Expected: focus ops non-trivial; capacity allocates to semantically relevant subtargets; churn down.

---

1. Open questions / unknowns list (ranked by risk)

2. **DDP lifecycle synchronisation semantics** for new internal state (internal_level + focus_idx). If the existing SeedSlot symmetry mechanism assumes a closed set of ops, expanding it can introduce subtle divergence if not broadcast/synchronised correctly.

3. **Telemetry cost envelope**: per-slot norms at high slot counts can become non-trivial. Needs strict vectorisation and probably EMA summaries rather than raw per-step firehose.

4. **Policy scaling threshold**: exactly where flat concat breaks in your current implementation (memory/layout/throughput) and when Slot Transformer becomes mandatory in practice.

5. **Governor semantics for “internal growth”**: how vetoes are represented and learned. If vetoes are frequent without clear obs bits, policy can appear “buggy” despite correct code.

6. **Projection-level injection refactor risk**: exposing QKV/O and MLP projections without hooks requires careful blueprint restructuring to avoid brittle coupling and compile regressions.

7. **Credit assignment with many small interventions**: as you add surfaces, improvements become diffuse; strong ROI sensors and a better slot encoder become important earlier.

8. **Fossilisation strategy**: when to freeze vs keep plasticity, and whether merge-to-host is training-time or export-time only.

---

1. Recommendation (your chosen path and why)

Take the **A0 + C0** milestone immediately as the “minimum viable submodule control”: keep hosts as-is but increase segments (Transformer per-layer; CNN deeper blocks) and add a single **prefix ladder seed** with **only** `GROW_INTERNAL` / `SHRINK_INTERNAL` plus the mandatory `internal_level_norm` (and, realistically, `delta_l2_norm`) sensors. Then go **A1** (post-attn/post-MLP + CNN intra-block splits) while extending ladder families to both hosts. After that, add **C2 multi-ladder focus** (deterministic subtargets) *before* you attempt projection-level injection, because it teaches the system to do meaningful allocation without combinatorics. As slot count rises, pivot Tamiyo to the **Slot Transformer encoder** (Phase α) rather than trying to brute-force flat concatenation. Only then do **A2 projection-level injection**—which is where LoRA ladders become devastatingly effective and conv_heavy should finally stop being the default hammer.

**Confidence (WEP):** *Likely* correct at the architectural level (the A0/A1/C0/C2/S-Transformer/A2 sequencing is a well-trodden stability path for “more loci + smaller knobs” systems), but only *somewhat likely* that the exact field names and integration points match your repo without adjustment, because the referenced files weren’t available here and I’m inferring contracts from the anchors and constraints you provided.
