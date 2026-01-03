# Kasmina + Tamiyo Submodule Intervention Roadmap (Track A + Track C)

**Author role:** Esper Research Architect  
**Status:** Research roadmap / planning deliverable (no code)  
**Baseline assumed:** Obs V3, Policy V2, op-conditioned critic, ~150-step horizon  
**Non-negotiables:** Sensors match capabilities; no legacy/backcompat shims; no bug-hiding defensive patterns; GPU-first inverted control flow; DDP symmetry invariants; metaphors (organism for domains, botanical only for seed lifecycle)

---

## 1) One-paragraph executive summary (what we’re building and why it fixes “conv_heavy dominance”)

We will evolve Esper from **module-level** morphogenesis (“germinate a whole seed at a coarse boundary”) to **submodule-level** morphogenesis where Tamiyo can intervene *more frequently* with *smaller mass per decision*: **(A)** expose denser, semantically meaningful injection surfaces (e.g., transformer post-attention vs post-MLP; CNN pre-pool vs post-pool) and **(C)** introduce “ladder” seeds whose capacity is controlled by a small internal state with cheap ops (grow/shrink/enable/disable) so Tamiyo is not forced to “buy a conv_heavy” to get signal. Every new lever is paired with typed telemetry + Obs V3 fields so it is learnable under the rent & churn economy, while preserving GPU-first control flow, torch.compile viability, and strict DDP symmetry.

---

## 2) A vs C design taxonomy (options table with pros/cons)

| Track | Option | Stable IDs (required) | Pros | Cons / when to avoid |
|---|---|---|---|---|
| A (surfaces) | **A0:** Denser *existing* segments (already supported) | Slot IDs remain canonical `r0c{i}` (segment index order) | Minimal; immediate density via `CNNHost(n_blocks↑)` and `TransformerHost(num_segments=n_layer)`; no new semantics | Still “boundary-only” (no post-attn vs post-MLP); denser slots stress flat obs sooner |
| A (surfaces) | **A1:** Transformer per-layer *sublayer* surfaces (post-attn + post-MLP) | Slot IDs `r{layer}c{surface}` with `surface∈{POST_ATTN, POST_MLP}` | True submodule meaning; clean credit assignment; stable ordering is natural | Requires host refactor; slot count doubles; pushes earlier need for Slot Transformer policy |
| A (surfaces) | **A2:** CNN per-block pre-pool + post-pool surfaces | Slot IDs `r{block}c{surface}` with `surface∈{PRE_POOL, POST_POOL}` | Meaningful in CNNs without brittle hooks; aligns with resolution changes and compute trade-offs | Requires CNNHost routing refactor; increases slot count; must preserve channels_last invariants |
| A (surfaces) | **A3:** Transformer intra-attention “micro hooks” (QKV/proj) or per-head surfaces | Subtarget IDs per head/group | Very fine control | High brittleness + action-space explosion; torch.compile hostile; defer until grouped policies are stable |
| A (surfaces) | **Reject:** forward hooks / hidden callbacks | N/A | Fast to prototype | Brittle, non-contractual, hard to reason about; tends to break compilation and DDP symmetry expectations |
| C (microstructure) | **C0:** Scalar ladder state `internal_level` with `GROW_INTERNAL` / `SHRINK_INTERNAL` | Deterministic level mapping; no dynamic IDs | Minimal action-space expansion (1–2 ops); enables frequent “small bets”; reduces “heavy blueprint dominance” | Must define rent semantics; branching can inflate compile graphs if levels too many |
| C (microstructure) | **C1:** LoRA rank ladder (rank blocks) | Subtargets = rank-block indices (stable 0..K-1) | Lightweight; ideal for transformer pivot; avoids “buy lora_large” | Doesn’t directly address CNN conv_heavy; pair with channel/conv ladders |
| C (microstructure) | **C2:** CNN channel-group ladder | Subtargets = channel-group indices (stable) | Hardware-aligned structured capacity; cheap enable/disable | Needs deterministic grouping rules across arbitrary channel counts |
| C (microstructure) | **C3:** Attention head-group enable/disable ladder | Subtargets = head-group indices | Strong “submodule” feel with bounded K | Needs stable head/group addressing; add only after inventory contracts exist |
| C (microstructure) | **C4:** Full subtarget inventory + grouped addressing | Subtargets = stable IDs 0..K-1 per seed | Explicit “what changed” improves learnability + debugging | Requires new action head + slot-token policy scaling; higher integration risk |

**Key design rule:** Prefer **typed, deterministic inventories** + **small grouped controls** over per-parameter freedom. Submodule control must be addressable deterministically and observable directly.

---

## 3) Roadmap table (phase, duration, dependencies, risk level, success metric)

| Phase | Duration | Dependencies | Risk | Success metric (ROI-style + decision quality) |
|---|---:|---|---|---|
| **0 (A0 + C0)** | 1–2w | Ideally after Phase 2.5 reward exam stability | Low–Med | Internal ops used; conv_heavy selection drops; Accuracy ROI ≥ baseline; no governor rollback increase |
| **1 (A1 + C1)** | 2–3w | Phase 0 | Med | TinyStories ROI improves with post-attn/post-MLP lattice; rank ladder used vs jumping to heavy seeds; stable entropies |
| **2 (A2 + C2)** | 2–3w | Phase 0 | Med | CIFAR ROI improves with pre/post-pool surfaces; ladders replace heavy “step function” upgrades; fossilize/prune efficiency improves |
| **3 (A + C inventory)** | 3–4w | Phases 1–2 | High | Subtarget ops used; effective params reduced at same quality; no persistent entropy collapse in new head(s) |
| **4 (Slot Transformer policy)** | 3–5w | Phase 3 or “slot pressure” trigger | High | Stable PPO with ≥32 slots; throughput within 10% of baseline; masking + slot–slot interactions improve decisions |
| **5 (Full submodule morphogenesis)** | 4–8+w | Phase 4 | Very high | End-to-end “grow then trim” demo with replayable trace and stable governor outcomes |

---

## 4) Per-phase spec blocks (as described above)

### Phase 0 — **A0 + C0** minimal milestone (hard requirement)

- **Objective (capability added):** Tamiyo can make **more frequent, less weighty** interventions by using denser existing surfaces (A0) and adjusting seed capacity via a single internal lever (C0) instead of repeatedly germinating “heavy” modules.
- **Scope (will/won’t touch):**
  - **Will:** Leyline contracts (reports + ops + telemetry); Kasmina seed implementation for one ladder seed; Tamiyo Obs V3 feature extraction; action masking; reward/rent accounting updates if needed.
  - **Won’t:** Slot Transformer policy pivot; transformer host refactor; per-subtarget addressing.
- **Design sketch (contracts, dataflow, how Tamiyo acts):**
  - Add a seed-local integer state `internal_level ∈ [0..L]`.
  - Tamiyo workflow: `GERMINATE` a ladder seed once, then use `GROW_INTERNAL`/`SHRINK_INTERNAL` to tune capacity while staying in the same slot and lifecycle.
  - Kasmina enforces deterministic level→capacity mapping; internal changes are explicit lifecycle ops (no hidden growth).
- **Required telemetry sensors + derived observation fields (exact fields; where emitted; where consumed):**
  - **Leyline reports:** add to `src/esper/leyline/reports.py` `SeedStateReport`:
    - `internal_kind: int` (0 = none, >0 enumerated ladder types)
    - `internal_level: int`
    - `internal_max_level: int`
    - `internal_effective_params: int` (optional but recommended for rent learnability)
  - **Leyline telemetry:** add to `src/esper/leyline/telemetry.py`:
    - `TelemetryEventType.SEED_INTERNAL_LEVEL_CHANGED`
    - `SeedInternalLevelChangedPayload(slot_id, env_id, seed_id, blueprint_id, from_level, to_level, internal_effective_params)`
  - **Emission:** `src/esper/kasmina/slot.py` emits `SEED_INTERNAL_LEVEL_CHANGED` whenever an internal op succeeds.
  - **Consumption (policy):** `src/esper/tamiyo/policy/features.py` appends per-slot scalar `internal_level_norm = internal_level / max(internal_max_level, 1)` (Obs V3).
  - **Consumption (masking):** `src/esper/tamiyo/policy/action_masks.py` enables internal ops only when `state != None` and the seed is not FOSSILIZED and growth/shrink is feasible.
  - **Storage/UI:** Karn collects the new event; Sanctum/Overwatch can display internal level deltas and churn.
- **Action space changes (new op(s) vs new head; justify to avoid combinatorial blow-up):**
  - Add **two ops** in `src/esper/leyline/factored_actions.py`:
    - `LifecycleOp.GROW_INTERNAL`
    - `LifecycleOp.SHRINK_INTERNAL`
  - No new head yet (keeps combinatorics bounded). Update `src/esper/leyline/causal_masks.py` so only `op` and `slot` are causally relevant for these ops.
- **Host changes (A): injection_specs changes; slot ID scheme; how positions are ordered:**
  - **No host contract change required** in Phase 0.
  - A0 uses existing density knobs:
    - CNN: deeper host blocks (`cifar_scale` already exposes 5 slots).
    - Transformer: optionally set `TransformerHost(num_segments=n_layer)` for per-layer boundaries (still boundary-only semantics).
  - Ordering remains by `InjectionSpec.position` / host-provided order.
- **Seed changes (C): at least one concrete seed family with internal levers; fossilization semantics (merge vs keep):**
  - Add **one ladder seed family** (starting point recommendation):
    - **CNN:** `conv_ladder` where level controls number of `SeedConvBlock` stages (level 1 ≈ conv_light; level 2 ≈ conv_heavy; capped L≤4).
    - **Transformer alternative:** `lora_ladder` where level selects rank blocks (0/4/8/16/32).
  - **Fossilization semantics:** keep the seed as permanent (current architecture), freezing `internal_level` once in FOSSILIZED.
- **Safety constraints (DDP symmetry, compile behavior, governor interactions):**
  - DDP: internal_level transitions are driven by the same policy action stream; state change must be identical on all ranks.
  - torch.compile: cap ladder levels; prefer tensor masks over Python control flow when possible to prevent graph blow-up.
  - Governor: internal growth must be visible to rent/shock telemetry (effective params and alpha shock).
- **Experiment plan (tasks/presets, run lengths, cohorts, what success looks like):**
  - Task: `cifar_scale` with 5 slots enabled; 100 rounds, 8 envs, 150 episode length.
  - Cohorts: baseline blueprint set vs ladder-enabled action space.
  - Success: reduced conv_heavy selection rate, non-trivial internal op usage, Accuracy ROI not worse, no increase in rollback events.
- **Risks + mitigations:**
  - Thrash (grow/shrink oscillation): price internal ops lightly; track churn telemetry and penalize if needed.
  - Invalid-action spam: strict masking + last_action_success feedback already in Obs V3.
  - Telemetry overhead: emit internal-level event only on changes; keep per-step telemetry unchanged.
- **Exit criteria (objective gates):**
  - `GROW/SHRINK_INTERNAL` appears with meaningful frequency and correlates with improved ROI or reduced rent for similar accuracy.
  - No regression in entropy stability or governor outcomes.

---

### Phase 1 — **A1 + C1** Transformer sublayer surfaces + LoRA ladder

- **Objective:** TransformerHost gains submodule-meaningful injection points (post-attn vs post-MLP) and Tamiyo can adjust adapter capacity smoothly via a rank ladder (no forced “lora_large” jumps).
- **Scope:**
  - **Will:** TransformerHost routing; Leyline injection surface metadata; LoRA ladder seed; Obs V3 static per-slot surface metadata.
  - **Won’t:** per-head injection inside attention; full Slot Transformer policy.
- **Design sketch:**
  - Expose two surfaces per layer: **POST_ATTN** (after attention residual) and **POST_MLP** (after MLP residual).
  - Each surface is a distinct slot with stable ID and deterministic order.
  - Ladder seed is placed once per slot; capacity tuned via internal ops.
- **Telemetry + obs (exact fields; where):**
  - **Leyline injection contract:** extend `src/esper/leyline/injection_spec.py`:
    - `surface: int` (enum-backed, e.g., InjectionSurface)
    - `order: int` (stable global order index, not a float)
    - (optional) `host_layer: int` and `host_block: int` (typed, deterministic)
  - **SlotConfig:** extend `src/esper/leyline/slot_config.py` to carry `order_map` and `surface_map` for slots derived from specs.
  - **Obs V3:** add per-slot static fields in `src/esper/tamiyo/policy/features.py`:
    - `slot_order_norm`
    - `slot_surface_norm`
  - **Emission:** host produces richer `InjectionSpec`; no extra per-step emission required.
  - **Consumption:** Tamiyo uses these features to learn surface preferences.
- **Action space changes:**
  - No new head; reuse Phase 0 internal ops.
  - (Optional) keep blueprint set unchanged; lora_ladder is a new blueprint choice only.
- **Host changes (A):**
  - Slot ID scheme: `r{layer}c{surface}` (row = layer index; col = surface enum value).
  - Ordering: increasing `(layer, surface)`; `InjectionSpec.order` is authoritative; `position` can remain as a derived float for visualization.
- **Seed changes (C):**
  - Add `lora_ladder` with rank blocks; `internal_level` selects active blocks.
  - Fossilization freezes rank; pruning is unchanged.
- **Safety constraints:**
  - Keep explicit routing (no hooks); keep GPU-first (no per-token Python loops).
  - DDP symmetry across expanded slot set.
  - Compile graph count: keep levels small and discrete.
- **Experiment plan:**
  - Task: `tinystories`.
  - Compare: segment boundaries (3) vs per-layer boundaries (6) vs post-attn/post-MLP lattice (12).
  - Success: better perplexity/accuracy ROI at similar rent; non-uniform surface placement (learned preference); stable head entropies.
- **Risks + mitigations:**
  - Slot explosion: enable only a subset initially (e.g., POST_ATTN only) as a curriculum.
  - Credit assignment noise: rely on causal masks; avoid adding too many new heads at once.
- **Exit criteria:**
  - Policy differentiates post-attn vs post-MLP and uses rank growth rather than jumping to heavy blueprints.

---

### Phase 2 — **A2 + C2** CNN pre/post-pool surfaces + channel-group ladder

- **Objective:** CNNHost exposes resolution-aligned surfaces (pre-pool vs post-pool) and CNN seeds gain fine-grained capacity control via channel groups (no “conv_heavy step function”).
- **Scope:**
  - **Will:** CNNHost routing changes; injection surface metadata applied to CNN; channel-group ladder seed.
  - **Won’t:** intra-conv hooks; subtarget addressing head (still scalar internal ops only).
- **Design sketch:**
  - Each block yields up to two surfaces:
    - **PRE_POOL:** after conv block, before pooling
    - **POST_POOL:** after pooling (if pooling applied)
  - Tamiyo can act at higher resolution (PRE_POOL) or lower cost (POST_POOL).
- **Telemetry + obs:**
  - Reuse Phase 1 injection metadata (`surface`, `order`) for CNN specs.
  - Add ladder telemetry fields (reports):
    - `internal_group_size: int`
    - `internal_active_groups: int`
    - `internal_effective_params: int`
  - Obs V3: add `internal_level_norm` as in Phase 0 (level = active groups).
- **Action space changes:**
  - No new head; reuse internal ops.
- **Host changes (A):**
  - Slot IDs: `r{block}c{surface}`; ordering is block-major, then surface.
  - `InjectionSpec.order` is deterministic and used for `SlotConfig` ordering.
  - Preserve channels_last handling: conversion happens once at entry, and seeds receive correct format.
- **Seed changes (C):**
  - Add `channel_ladder` where channels are partitioned deterministically into groups (e.g., 16 channels/group); internal_level controls active groups.
  - Fossilization freezes group mask/level.
- **Safety constraints:**
  - Determinism: grouping rule must be identical across hosts/runs.
  - DDP symmetry: mask changes must be identical on all ranks.
  - Governor: internal growth impacts rent/shock; watch for instability at PRE_POOL surfaces.
- **Experiment plan:**
  - Task: `cifar_scale` (deep) and `cifar_baseline` (shallow) cohorts.
  - Compare: PRE_POOL vs POST_POOL utilization; conv_ladder vs channel_ladder.
  - Success: ROI improves; fewer prunes due to “overshoot”; stable governor outcomes.
- **Risks + mitigations:**
  - Increased slot count pressures flat obs; mitigate by limiting enabled slots or accelerating Phase 4.
- **Exit criteria:**
  - Learned surface preference correlates with ROI; internal ops replace heavy blueprint jumps.

---

### Phase 3 — **A + C inventory:** deterministic subtarget inventory + grouped addressing

- **Objective:** Promote “internal microstructure” from scalar ladders to **addressable groups** (heads/channels/rank blocks) with stable IDs, enabling true submodule control without action-space explosion.
- **Scope:**
  - **Will:** Leyline subtarget contracts; one new action head (small K); seed families expose inventories.
  - **Won’t:** full nested morphogenetic planes; Slot Transformer policy unless scaling requires it.
- **Design sketch:**
  - Each microstructured seed exposes up to K grouped subtargets with stable indices `0..K-1`.
  - Tamiyo selects a slot, an op (enable/disable), and a subtarget index.
  - Sensors include subtarget inventory summary + churn signals.
- **Telemetry + obs:**
  - **Leyline contracts (new):**
    - `SubtargetKind` enum (e.g., NONE, LORA_RANK_BLOCK, CNN_CHANNEL_GROUP, ATTN_HEAD_GROUP)
    - `SeedStateReport.subtarget_kind: int`
    - `SeedStateReport.subtarget_count: int`
    - `SeedStateReport.subtarget_active_count: int`
    - (optional) `SeedStateReport.subtarget_active_mask: tuple[int, ...]` (packed) or fixed-length float vector if K is fixed
  - **Leyline telemetry:** `TelemetryEventType.SEED_SUBTARGET_CHANGED` + payload with `subtarget_idx` and `enabled`.
  - **Obs V3:** per-slot scalars:
    - `subtarget_active_frac = active_count / max(count, 1)`
    - `subtarget_churn_norm` (e.g., time since last subtarget change)
  - Emission in `src/esper/kasmina/slot.py` (on success); consumed in `tamiyo/policy/features.py`; stored in Karn.
- **Action space changes:**
  - Add **one new head** `subtarget` with small bounded size K (fixed per run), and one or two ops:
    - `ENABLE_SUBTARGET`, `DISABLE_SUBTARGET` (or a binary head + single op)
  - Update causal masks so `subtarget` is only relevant for these ops.
- **Host changes (A):**
  - None required; this phase depends on stable slot ordering and surface metadata.
- **Seed changes (C):**
  - Extend `lora_ladder` and `channel_ladder` to report subtarget inventory explicitly.
  - Add attention head-group seed if needed (grouped to keep K small).
  - Fossilization freezes subtarget mask.
- **Safety constraints:**
  - DDP symmetry: subtarget mask updates are synchronized by the policy action stream.
  - torch.compile: represent masks as tensors cached on device; no per-step Python list mutations.
  - Governor: add churn diagnostics; excessive micro-churn should be visible and penalizable.
- **Experiment plan:**
  - Tasks: `cifar_scale` + `tinystories`.
  - Goal: reduce effective params at equal quality by disabling unneeded groups; show stable decision quality (entropy) and improved fossilize/prune ratio.
- **Risks + mitigations:**
  - New head entropy collapse: per-head entropy floor tuning + curriculum unlock after baseline stability.
  - K too big: keep K small (≤8–16) by grouping.
- **Exit criteria:**
  - Subtarget ops used; effective params decrease without quality loss; no persistent head collapse.

---

### Phase 4 — **Slot Transformer policy pivot** (Phase α alignment)

- **Objective:** Maintain learnability and throughput as slot count grows (Track A) and seeds expose richer internal state (Track C), moving from flat concatenation to masked slot tokens with learned slot–slot interactions.
- **Scope:**
  - **Will:** Tamiyo architecture change (Slot Transformer encoder); Leyline token schema; masking infrastructure.
  - **Won’t:** change PPO factorization semantics unless necessary (keep head meanings).
- **Design sketch:**
  - Replace flat `obs = concat(base, slot1, slot2, ...)` with tokenized representation:
    - one base token + N slot tokens (+ optional subtarget tokens)
  - Learned slot–slot interactions enable scalable coordination and better credit assignment in dense lattices.
- **Telemetry + obs:**
  - Leyline defines token schema fields and masks (variable N).
  - Karn tracks per-token attention diagnostics only in dense trace modes (avoid overhead).
- **Action space changes:**
  - Keep existing semantics; implement slot selection as pointer distribution over enabled slots; subtarget selection similarly.
- **Host changes (A):**
  - Larger injection lattices become practical (e.g., transformer 2 surfaces/layer across 12+ layers).
  - `InjectionSpec.order` becomes required for token ordering stability.
- **Seed changes (C):**
  - No new family required; existing ladders/inventories become easier to use at scale.
- **Safety constraints:**
  - Preserve inverted control flow: token construction must avoid per-slot GPU writes in Python loops.
  - DDP symmetry with variable masking is mandatory.
  - Throughput gate: ≤10% slowdown at equivalent env count.
- **Experiment plan:**
  - Scale study: 3 vs 12 vs 20 vs 40 slots; compare flat LSTM vs Slot Transformer.
  - Success: stable PPO learning, throughput acceptable, improved decision quality at higher slot counts.
- **Risks + mitigations:**
  - Engineering scope: keep token schema minimal; reuse existing SlotConfig ordering and masks.
- **Exit criteria:**
  - Stable runs with ≥32 slots and no sustained entropy collapse; throughput within budget.

---

### Phase 5 — **Full submodule morphogenesis** (end-state)

- **Objective:** Seeds become structured containers that can spawn internal substructure under contract (multi-scale morphogenesis), enabling “grow then trim” narratives with clear evidence.
- **Scope:**
  - **Will:** internal injection planes inside certain seed families; two-tier fossilization (micro then macro); richer telemetry tiering.
  - **Won’t:** per-parameter free-form edits (avoid scope explosion and kernel complexity).
- **Design sketch:**
  - A “container seed” exposes its own `injection_specs` (micro-lattice) and `SeedSlot`-like internal lifecycle under Kasmina mechanics, still governed by Leyline contracts.
- **Telemetry + obs:**
  - Inventory + churn summaries always; dense traces only on anomaly.
  - Ensure every micro-lever is directly observable (commandment #1).
- **Action space changes:**
  - Hierarchical decisions: macro slot → micro target selection (bounded by grouping + masking).
- **Host changes (A):**
  - Host lattice remains; novelty is internal planes inside seeds.
- **Seed changes (C):**
  - One concrete container seed family (transformer-first recommended) with internal levers and deterministic inventories.
  - Fossilization semantics: freeze microstructure before freezing macro integration.
- **Safety constraints:**
  - Deterministic compaction boundaries; DDP symmetry; keep compile graphs bounded.
- **Experiment plan:**
  - Storyboard: underfit host → grow microstructure → disable subtargets to reduce rent while holding quality; produce replayable forensic trace for one run.
- **Risks + mitigations:**
  - Scope risk: keep inventories small and grouped; avoid parameter-level operations.
- **Exit criteria:**
  - Demonstrated cost reduction at equal quality with stable governor outcomes and explainable telemetry.

---

## 5) “First experiment pack” (exact runs to do first, expected outcomes, what telemetry to watch)

> Note: Commands below use existing CLI surfaces (`esper.scripts.train ppo`) and are intended as the immediate evaluation harness for Phase 0 (A0) and then Phase 0+ (A0+C0) once implemented.

**Run 1 — Control (A0 only, current baseline):**

```bash
PYTHONPATH=src uv run python -m esper.scripts.train ppo \
  --preset cifar_scale --task cifar_scale \
  --slots r0c0 r0c1 r0c2 r0c3 r0c4 \
  --max-seeds 2 \
  --rounds 100 --envs 8 --episode-length 150 \
  --ab-test shaped-vs-simplified \
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
  --ab-test shaped-vs-simplified \
  --sanctum
```

Expected outcomes:
- Higher prune rate; entropy pressure and potential collapse in slot/op heads if observation scaling is insufficient.

Telemetry to watch:
- Entropy collapse diagnostics; invalid-action frequency (`last_action_success`)
- Governor rollback frequency and gradient pathology events

**Run 3 — Phase 0 validation (A0 + C0) once ladder + internal ops exist:**

Repeat Run 1, but validate the new signals exist:
- `SEED_INTERNAL_LEVEL_CHANGED` events appear
- Obs V3 includes `internal_level_norm` (verify via observation stats telemetry)
- Internal ops appear in decision logs and correlate with ROI

Expected outcomes:
- Reduced “conv_heavy dominance”
- More frequent small interventions (grow/shrink) rather than germinate/prune churn

**Run 4 — Transformer sanity (Phase 1 once A1+C1 exists):**

```bash
PYTHONPATH=src uv run python -m esper.scripts.train ppo \
  --preset tinystories --task tinystories \
  --max-seeds 2 \
  --rounds 100 --envs 8 --episode-length 150 \
  --sanctum
```

Then run with an A1 slot set (post-attn only, then post-attn+post-MLP) and compare:
- surface preference emergence (post-attn vs post-MLP)
- rank ladder usage vs heavy blueprint jumps

---

## 6) Open questions / unknowns list (ranked by risk)

1. **Credit assignment:** How much does frequent internal adjustment increase gradient noise, and when do we need explicit micro-churn pricing in reward beyond existing rent/shock terms?
2. **Scaling threshold:** At what slot count (and subtarget richness) does flat Obs V3 concatenation become unlearnable rather than merely slower—what objective trigger moves us to the Slot Transformer encoder?
3. **DDP determinism:** What is the authoritative synchronization mechanism for internal_level/subtarget state (purely action-driven vs explicit broadcast), and how do we prove no lifecycle divergence?
4. **Rent semantics:** Should internal capacity be priced even when alpha=0 (TRAINING with STE still computes), or remain alpha-weighted as in current effective param accounting?
5. **Transformer surface dominance:** Do we always need both post-attn and post-MLP surfaces, or is a curriculum (post-attn first) strictly better for stability and learnability?
6. **Telemetry granularity:** Can aggregated subtarget sensors preserve learnability, with dense per-subtarget traces only on anomalies, without violating “sensors match capabilities”?
7. **Bounded inventories:** What is the smallest K (group count) that yields real efficiency wins without inducing combinatorial explosion?

---

## 7) Recommendation (your chosen path and why)

Start with **Phase 0 exactly as A0 + C0**: use existing denser CNN injection points (e.g., `cifar_scale` blocks) and introduce one ladder seed (recommended: `conv_ladder`) controlled by **two new lifecycle ops** (`GROW_INTERNAL`, `SHRINK_INTERNAL`) plus one new per-slot Obs V3 scalar (`internal_level_norm`) and one typed telemetry event (`SEED_INTERNAL_LEVEL_CHANGED`). This is the smallest change that directly attacks “conv_heavy dominance” without re-architecting the policy or host. Next, prioritize **Phase 1 (A1 + C1)** to align with the Phase 3 transformer pivot (post-attn/post-MLP surfaces + LoRA rank ladder). Begin **Phase 4 (Slot Transformer encoder)** as soon as slot count approaches ~16+ or Phase 3 subtarget addressing is needed, to keep Tamiyo scalable while preserving GPU-first throughput and strict contracts.

