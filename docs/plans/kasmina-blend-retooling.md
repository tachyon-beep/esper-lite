# Kasmina Blend Retooling: `prune` + Partial-Alpha Holds

> **Status:** Locked (design decisions committed; ready to implement)
>
> **Date:** 2025-12-20
>
> **Goal:** Make “temporary support → gradual removal” a first-class, learnable behavior by:
> - replacing demolition-style `cull` with scheduled `prune`, and
> - allowing Tamiyo to blend to a target alpha and hold (including partial alpha).
>
> **Non-scope:** DDP enablement and DDP safety (tracked separately; this work should assume single-process / non-DDP).

---

## 1) Summary (What Changes)

### Change A — `cull` → `prune` (4 levels, scheduled)

Replace the single “remove immediately” operation with:
- `PRUNE_INSTANT`: set alpha to 0 immediately, then remove the seed module.
- `PRUNE_FAST | PRUNE_MEDIUM | PRUNE_SLOW`: schedule alpha from current → 0 over N controller ticks using a chosen curve.
- Once alpha hits 0 (or ≤ epsilon), we physically remove the seed module and stop paying rent.

**Proposed default schedule lengths (initial)**

| Speed | Steps (ticks) | Notes |
|---|---:|---|
| `PRUNE_INSTANT` | 0 | parity with old `cull` behavior |
| `PRUNE_FAST` | 3 | aligns with `TempoAction.FAST` |
| `PRUNE_MEDIUM` | 5 | aligns with `TempoAction.STANDARD` |
| `PRUNE_SLOW` | 8 | aligns with `TempoAction.SLOW` |

These can later become task-relative (e.g., % of `max_epochs`), but starting with fixed, symmetric schedules improves learnability and reduces wiring churn.

### Change B — Blend to target and hold

Give Tamiyo a stable knob:
- Ramp alpha to a target (discrete menu), then hold.
- Later, either promote (ramp upward) or prune (ramp downward).

**Why `BLEND_HOLD` exists (scaffolding example)**

The intent is not “half a module for aesthetics” — it’s controlled scaffolding across slots. Example policy we want PPO/Tamiyo to be able to execute:

1. Slot 2 germinates a stabilizer (e.g., a lightweight/partial norm seed).
2. Slot 2 blends to `alpha_target=0.5` and stays in `BLEND_HOLD`.
3. Slot 3 germinates a capacity seed (e.g., `conv_light`) and grows while the stabilizer is partially present.
4. Once Slot 3 is stable, Slot 2 is pruned (`PRUNE_SLOW` / scheduled DOWN), fading out the stabilizer without shock.

### Change C — Add a blend composition “algorithm knob” (valves)

Add an explicit operator choice for how a seed affects the host stream:
- `ADD` (default): stable convex mixing (`lerp`) composition.
- `MULTIPLY`: valve-style modulation (identity at alpha=0).
- `GATE`: enable learned per-sample gating (so effective alpha depends on input), while keeping amplitude scheduling separate.

This is primarily to enable “valve scaffolding” behaviors (e.g., temporary dampeners) and to make that choice legible/learnable as a discrete action.

### Change D — Remove HOLDING auto-cull (keep catastrophic safety only)

Remove “cleanup-by-architecture” auto-culls in the hold/decision region and make removal an explicit learned decision, while keeping a single catastrophic safety rail (Governor / NaN / divergence).

---

## 2) Current System (Pressure Points)

### Lifecycle contracts
- Stages and transitions: `src/esper/leyline/stages.py:13`
- Factored action space includes `LifecycleOp.CULL`: `src/esper/leyline/factored_actions.py:89`

### Kasmina state engine
- `SeedState.transition()` resets stage baselines: `src/esper/kasmina/slot.py:389`
- `SeedSlot.step_epoch()` handles mechanical stage progression and HOLDING auto-culls: `src/esper/kasmina/slot.py:1627`
- `SeedSlot.cull()` immediately clears the slot: `src/esper/kasmina/slot.py:1152`

### Blending semantics
- Blending is output gating (`torch.lerp`), not weight interpolation: `src/esper/kasmina/isolation.py:22`
- Blending “algorithm” is currently used for alpha generation (linear/sigmoid/gated): `src/esper/kasmina/blending.py:16`
- Composition is currently fixed to `lerp(host, seed, alpha)`; this plan generalizes the blend mode (ADD/MULTIPLY/GATE) while keeping the alpha controller unified.

---

## 3) Decisions (Locked)

| Decision | Options | Recommended | Reason |
|---|---|---|---|
| Terminal naming | legacy failure label vs `PRUNED` | `PRUNED` | avoids semantic mismatch once removal is gradual |
| Hold naming | legacy hold label vs `HOLDING` | `HOLDING` | matches new meaning and reduces confusion |
| Alpha control location | treat as new stages vs internal controller fields | internal controller fields | avoids exploding `SeedStage`, keeps code paths unified |
| Action encoding | add many ops vs add factored heads | factored heads | PPO learns discrete knobs better than a giant enum |
| Target alpha menu | `{0.5, 0.7, 1.0}` vs `{0.3, 0.5, 0.7, 1.0}` | start `{0.5, 0.7, 1.0}` | avoid starving learning at very low alpha initially |
| Blend operator knob | none vs `{ADD, MULTIPLY, GATE}` | add `{ADD, MULTIPLY, GATE}` | enables valve scaffolding + makes composition legible |
| Prune-out training | allow updates vs freeze seed params | **mandatory freeze** | prevents “fighting death” / death spirals during prune |
| Zombie-seed prevention | alpha-weighted rent only vs + slot rent | add slot occupation rent | prevents “alpha=0.01 squatting” for near-zero cost |

---

## 4) Proposed Design: Alpha Controller as the Primitive

### 4.1 Define an internal “alpha activity” controller

We treat alpha changes as an internal controller (not a top-level lifecycle stage):
- `alpha_target`: target amplitude in `[0,1]`
- `alpha_mode`: `{UP, HOLD, DOWN}`
- `alpha_curve`: `{LINEAR, COSINE, SIGMOID}` (symmetric for up/down)
- `alpha_steps_total`, `alpha_steps_done`
- `freeze_seed_weights`: bool (**mandatory `True` when `alpha_mode == DOWN` / `BLEND_OUT`**)

**Core invariants**
- UP is strictly monotone increasing; DOWN strictly monotone decreasing.
- HOLD keeps alpha constant (within tolerance).
- The controller retargets only from HOLD (prevents “alpha dithering” hacks).
- **Freeze invariant:** when `alpha_mode == DOWN`, seed parameters MUST NOT update:
  - set `requires_grad=False` on all seed parameters (and any learnable gating params; see §7.1),
  - do **not** use graph-breaking techniques (`.detach()`, `torch.no_grad()`) in the forward pass (see §10.2).
- **Snap-to-target:** if `alpha_steps_done >= alpha_steps_total`, set `alpha = alpha_target` exactly (then enter HOLD). This avoids float slop and makes gate checks unambiguous.
- “Target reached” is defined by distance-to-target, not absolute alpha:
  - `abs(alpha - alpha_target) <= eps`
  - `eps` should be tied to schedule granularity (e.g., `1/alpha_steps_total`), not a hard constant.

### 4.2 How this interacts with existing blending algorithms

We separate **amplitude** from **per-sample gating**:

`alpha_effective(x) = alpha_amplitude(t) * gate(x)`

- For non-gated modes: `gate(x) = 1`
- For gated blend: `gate(x) = GatedBlend(x)` from `src/esper/kasmina/blending.py:138`

Tamiyo controls `alpha_amplitude(t)` and schedule parameters. Per-sample gating is enabled when `alpha_algorithm_head == GATE` (see §7.1).

### 4.3 Torch compile friendliness (design constraint)

Today, alpha is often materialized via new `torch.tensor(alpha)` allocations, which can increase `torch.compile` graph specialization pressure.

Preferred design for the new alpha controller:
- store the amplitude as a persistent 0-dim tensor buffer on the slot (mutated in-place),
- avoid creating new alpha tensors on every update,
- keep the forward path using a stable tensor identity.

---

## 5) Lifecycle Semantics with Alpha Controller

### 5.1 High-level stages remain, but become cleaner

We keep the botanical lifecycle (with rename adjustments):
- `DORMANT → GERMINATED → TRAINING → BLENDING → HOLDING → FOSSILIZED`
- Removal path ends in `PRUNED → EMBARGOED → RESETTING → DORMANT` (cooldown is first-class and always occurs after removal).

Key semantics changes:
- `BLENDING` means “seed influences output and alpha is under controller management” (UP / HOLD-at-partial / DOWN).
- `HOLDING` is reserved for full-amplitude hold (alpha_target=1.0, alpha≈1.0) and is the decision point for fossilization. Partial holds remain inside `BLENDING` (`BLEND_HOLD`).

### 5.2 `BLENDING` as a hub: `BLEND_IN` / `BLEND_HOLD` / `BLEND_OUT`

Treat BLENDING as the “center” of the lifecycle terrain: you can drive toward any corner (alpha≈0 or alpha≈1) via the same controller, and you can loiter in the middle (partial holds) without changing the top-level stage.

```mermaid
stateDiagram-v2
    [*] --> DORMANT
    DORMANT --> GERMINATED: GERMINATE
    GERMINATED --> TRAINING: auto
    TRAINING --> BLENDING: G2

    state BLENDING {
        [*] --> BLEND_IN
        BLEND_IN --> BLEND_HOLD: alpha hits target
        BLEND_HOLD --> BLEND_IN: retarget up (HOLD-only)
        BLEND_HOLD --> BLEND_OUT: retarget down / PRUNE
    }

    BLENDING --> HOLDING: alpha_target==1 & alpha≈1 & G3
    HOLDING --> BLENDING: retarget < 1 / PRUNE_SCHEDULED
    HOLDING --> FOSSILIZED: FOSSILIZE + G5
    BLENDING --> PRUNED: alpha≈0 (prune complete)
    HOLDING --> PRUNED: PRUNE_INSTANT
    PRUNED --> EMBARGOED --> RESETTING --> DORMANT
```

Even if `SeedStage` stays minimal, we should expose a substage/mode for observability:
- `BLEND_IN` (UP)
- `BLEND_HOLD` (HOLD, possibly partial alpha)
- `BLEND_OUT` (DOWN; pruning)

This is a telemetry/debugging concept; it does not have to be part of `SeedStage`.

### 5.3 Leaving `BLENDING` (and how G3 changes)

**Rule:** we only leave `BLENDING` when we are at an extreme:
- `alpha≈1.0` (full-amplitude) → eligible to enter `HOLDING` (the “final decision” stage)
- `alpha≈0.0` → prune completion (physical removal), then `PRUNED`/cooldown

Implications:
- Partial targets (`alpha_target ∈ {0.5, 0.7}`) reach `BLEND_HOLD` but do **not** stage-advance.
- The “blend completion” concept currently baked into G3 becomes: “target reached” (for telemetry) plus “full-amplitude reached” (for stage progression).
- Scheduled prune from `HOLDING` requires a stage transition back to `BLENDING` (because `HOLDING` is reserved for alpha≈1.0); this implies updating `VALID_TRANSITIONS` in `src/esper/leyline/stages.py:57`.

Concrete gate change:
- Update G3 to use distance-to-target and to require full amplitude for stage advancement:
  - Replace `state.alpha >= DEFAULT_ALPHA_COMPLETE_THRESHOLD` in `src/esper/kasmina/slot.py:639` with `abs(alpha - alpha_target) <= eps`.
  - Require `alpha_target == 1.0` (and mode HOLD) for the stage advance that is currently `BLENDING → HOLDING`.

### 5.4 Where to remove HOLDING auto-culls

Currently, HOLDING auto-culls live in `SeedSlot.step_epoch()` (`src/esper/kasmina/slot.py:1759`).

Plan:
- remove negative-counterfactual auto-cull and timeout auto-cull from the slot engine;
- keep only catastrophic safety via Governor / divergence detection (Tolaria), not “smart cleanup”.
- add a Governor-only emergency override: Governor may force `PRUNE_INSTANT` from any mode/stage if NaN/divergence/catastrophic collapse is detected (policy constraints remain unchanged).

---

## 6) Commands and Their Meaning (Tamiyo’s Toolkit)

### 6.1 `GERMINATE` (unchanged conceptually)

On germination:
- create the seed module,
- enter TRAINING (incubator semantics) as today,
- set an initial alpha plan (typically `alpha_target ∈ {0.5, 0.7, 1.0}`, `alpha_mode=UP`).

### 6.2 `SET_ALPHA_TARGET` (new, replaces “promote” variants)

Only legal when `alpha_mode == HOLD` (hard rule to prevent dithering):
- if target > current: set mode UP + schedule,
- if target == current: remain HOLD,
- if target < current: set mode DOWN + schedule (a demotion to a *non-zero* target; removal uses `PRUNE_*`).

### 6.3 `PRUNE_*` (new, replaces `CULL`)

`PRUNE_*` is explicit policy intent (“remove this”), mapped internally to `alpha_target=0.0` plus a speed:
- `instant`: `alpha_target=0` and remove immediately
- others: schedule down with the chosen curve + speed

Legal when `alpha_mode == HOLD` (i.e., `BLEND_HOLD` at partial alpha or `HOLDING` at full alpha).

**Completion rule**
- when alpha reaches 0 (or ≤ epsilon): physically remove the seed module and emit a “pruned” terminal event.

### 6.4 `FOSSILIZE` (unchanged, but more constrained)

Keep the existing G5 “counterfactual required” rule (`src/esper/kasmina/slot.py:665`).
Add a constraint:
- fossilize only when alpha_target==1 and alpha is within tolerance of 1.0 (i.e., not fossilizing partial-alpha supports).

---

## 7) Action Space Encoding (Recommended)

### 7.1 Recommended: factored heads, small op set

Update `LifecycleOp` (the existing `op` head) and add discrete heads:
- `LifecycleOp`: `{WAIT, GERMINATE, SET_ALPHA_TARGET, PRUNE, FOSSILIZE}` (replace `CULL`)
- `alpha_target_head`: `{0.5, 0.7, 1.0}` (non-zero only; removal uses `PRUNE`)
- `alpha_speed_head`: `{instant, fast, medium, slow}`
- `alpha_curve_head`: `{linear, cosine, sigmoid}`
- `alpha_algorithm_head`: `{ADD, MULTIPLY, GATE}`

Notes:
- Terminology: `alpha_curve_head` is the *schedule shape*; `alpha_algorithm_head` is the *blend mode* (composition operator + optional per-sample gating).
- Restrict `alpha_target_head` changes to HOLD mode via action masks.
- Allow `alpha_curve_head` only on schedule creation (not mid-transition).
- Restrict `alpha_algorithm_head` changes to HOLD mode (not mid-transition).
- Default `alpha_algorithm_head = ADD`.
- `SET_ALPHA_TARGET` does not permit `target=0.0`; use `PRUNE` for removal intent.
- Freeze applies to *all learnable params that can fight decay* during `BLEND_OUT`, including:
  - seed module parameters, and
  - any learnable gate network parameters if `alpha_algorithm_head == GATE`.
- Recommended default speed→ticks mapping: `instant=0`, `fast=3`, `medium=5`, `slow=8`.

**Operator semantics (must be explicit for implementation)**

Let `a = alpha_effective(x)` (see §4.2), and let `h = host_features` and `s = seed_features`.

Note: most Kasmina seeds are *residual* modules (common shape: `seed(x) = x + Δ(x)`). For valve semantics we want a signal that is near-zero at birth when the seed is identity-like, so for `MULTIPLY` we treat the modulation signal as the residual around the seed’s *input reference*:
- `x = seed_input` (the tensor we actually feed into the seed; `host_features` when non-isolated, `host_features.detach()` when `isolate_gradients=True`)
- `m = (s - x)` (seed modulation / residual)

- `ADD` (default): convex mix / cross-fade (matches today’s `torch.lerp` path).
  - Proposed: `y = lerp(h, s, a) = h + a * (s - h)`
- `MULTIPLY` (valve): modulate host by an identity-at-zero multiplier (for “dampener/valve” scaffolds).
  - Locked: `y = h * (1 + a * tanh(m))`
  - Init: initialize the seed’s final affine layer weights/biases to `0` so `m≈0` at birth and the valve starts near identity.
  - Isolation note: `MULTIPLY` is allowed when `isolate_gradients=True`, but **must** define `m` around the detached `seed_input` reference (not raw `host_features`) to avoid host-gradient sign flips and seed-dependent `dy/dh` coupling.
- `GATE`: enable per-sample gating so `a` depends on input (use the learned gate network), while keeping the `ADD` composition.

### 7.2 Transitional option (if we want minimal wiring first)

Reuse existing heads:
- map current `TempoAction` to prune/blend speed,
- map existing `BlendAction` to curve/gate selection:
  - `linear|sigmoid` → `alpha_curve_head`, with `alpha_algorithm_head=ADD`
  - `gated` → `alpha_algorithm_head=GATE` (curve still applies to `alpha_amplitude`),
while we validate the semantics, then split into explicit heads.

---

## 8) Reward Shaping (Make the Tradeoff Learnable)

### 8.1 Alpha-weighted rent

Goal: keeping partial-alpha supports online should incur an ongoing cost proportional to how “present” they are.

Proposal:
- Replace or augment seed rent with **alpha-weighted effective params + a slot occupation floor**:
  - `effective_seed_params = BaseSlotRent + Σ(alpha_amplitude_slot * seed_params_slot)`
  - rent computed from `effective_seed_params / host_params`

**Why add `BaseSlotRent` (zombie-seed risk)**

Without a floor, Tamiyo can squat a seed at `alpha=0.01` for near-zero rent while still consuming:
- a scarce slot (topological opportunity cost), and
- some VRAM/compute overhead (module exists, hooks, bookkeeping).

`BaseSlotRent` is a small fixed cost for “slot occupied by a seed”, regardless of alpha. A reasonable starting point is to set it as an equivalent-params constant on the order of `~1% * host_params` (tunable).

### 8.2 Shock penalty (discourage abrupt changes + dithering)

Proposal:
- `shock = -k * Σ((alpha_t - alpha_{t-1})^2 * scale_slot)` (convex; makes fast changes hurt more than slow ramps)
- `scale_slot` candidates:
  - `1.0` (simplest),
  - `seed_params/host_params` (more physical; recommended).

This makes:
- instant prune expensive,
- gradual prune cheaper,
- and makes oscillation costly.

### 8.3 Keep indecision pressure only at the full-amplitude decision point

Existing HOLDING indecision penalties should apply only to the “final decision” region:
- i.e., only when alpha≈1 and we are in HOLDING, not during partial holds inside BLENDING.

---

## 9) Telemetry Changes (Must-Have for Debugging)

### 9.1 Naming cleanup

If we rename actions to `PRUNE`, we should avoid mixed terminology in operator-facing telemetry.

Proposal:
- rename terminal lifecycle event to `SEED_PRUNED`.
- add an explicit initiator field for removals: `prune_initiator ∈ {policy, governor, manual}` (and record the reason string / code).
- rename stage/event names consistently (`PRUNED`, `HOLDING`) so telemetry reads correctly.

### 9.2 Add alpha controller fields to telemetry

Expose (at least):
- `alpha`, `alpha_target`
- `alpha_mode` / `blend_substage`
- `alpha_curve`, `alpha_steps_done/total`
- `alpha_algorithm` (ADD/MULTIPLY/GATE)
- `freeze_seed_weights` (if enabled)

### 9.3 Observation parity (Simic/Nissa commandment)

Per ROADMAP “Sensors match capabilities”, the policy must be able to see the new knobs.

Plan:
- add `alpha_target`, `alpha_mode`/`blend_substage`, `alpha_algorithm`, schedule progress (`alpha_steps_done/total`), and:
  - `time_to_target` (remaining steps normalized to `[0,1]`),
  - `alpha_velocity` (Δalpha from last controller tick),
  to the observation schema (or confirm they already exist and are wired end-to-end),
- ensure action masks are derived from the same state fields (so the agent can learn the constraint, not just hit hidden walls).

---

## 10) Optimizer / Runtime Considerations

### 10.1 Optimizer state cleanup on physical removal

When a module is physically removed:
- seed optimizers must drop state for removed parameters (momentum, Adam moments, etc.).
- existing training loops already do cleanup when a slot no longer has an active seed; prune completion must trigger the same cleanup path.

### 10.2 Ghost gradients (mandatory requirement)

During `BLEND_OUT`, we freeze parameters to prevent “fighting death”, but the host must still be able to adapt to the fading contribution.

Requirement:
- freeze via parameters (`requires_grad=False`) only,
- do NOT detach seed outputs (no `seed_features = seed_features.detach()`),
- do NOT run the seed forward under `torch.no_grad()`,
- keep the seed module in the forward pass so gradients can flow *through* it to the host (especially important for `MULTIPLY`/valve behavior).

### 10.3 Checkpoint/resume: persist alpha controller state

If we add `alpha_target/mode/curve/steps` but fail to serialize them, resume semantics become undefined (schedules reset mid-flight).

Plan:
- add alpha controller fields to `SeedState.to_dict()/from_dict()` (see `src/esper/kasmina/slot.py:424`),
- add a minimal roundtrip test that a mid-transition seed resumes with the same alpha controller state (including prune-out).

### 10.4 DDP warning (explicit)

Dynamic parameter sets and stateful alpha overrides are generally incompatible with DDP unless carefully centralized and synchronized.
This plan assumes single-process for the initial implementation. If/when we add DDP, prune must be revisited as part of a dedicated DDP design.

---

## 11) Test Plan (What “Done” Means)

### 11.1 Property tests (schedule correctness)
- Alpha is monotone during UP/DOWN transitions.
- Alpha never overshoots target.
- Alpha reaches target within `eps` derived from schedule resolution.
- HOLD maintains alpha within tolerance.

### 11.2 Unit tests (slot behavior)
- `PRUNE_INSTANT` produces the same final slot emptiness as old `cull`.
- Scheduled prune removes the module exactly when alpha reaches 0.
- Fossilize remains counterfactual-gated and cannot occur at partial alpha.
- `BLEND_OUT` freeze invariant: seed params receive no grads/updates, while host still receives gradients (ghost-gradient requirement).

### 11.3 Integration tests (vectorized env)
- No auto-culls in the hold region (unless catastrophic safety triggers).
- Reward telemetry includes alpha-weighted rent and shock terms.
- BaseSlotRent prevents “zombie seeds” (non-zero rent even at tiny alpha while occupied).
- Action masks prevent target changes mid-transition.
- Action masks prevent algorithm changes mid-transition.

---

## 12) Implementation Sequence (No Code Yet)

1) **Confirm constants** (BaseSlotRent magnitude, multiply bounding, default target menu, EMBARGOED cooldown).
2) **Naming refactor**: `cull`→`prune` across APIs + telemetry naming consistency (`PRUNED`, `HOLDING`) + update docs/mermaid diagrams that mention `PRUNED`/`HOLDING`.
3) **Alpha controller**: implement alpha_mode/target/curve/steps and integrate into `SeedSlot.step_epoch()` tick logic.
4) **Partial holds**: enable blend-to-target and hold in BLENDING; constrain HOLDING to alpha≈1 decision point.
5) **Prune schedules**: implement DOWN trajectories and physical removal at alpha==0.
6) **Reward shaping**: add alpha-weighted rent and shock penalty; restrict indecision penalty to full-amplitude hold.
7) **Telemetry + tests**: ensure new modes are observable; add schedule property tests and instant-prune parity tests.

### 12.1 Phases, Tasks, and Risk Reduction

**Ratings scale**
- **Complexity (1–5):** engineering effort + cross-cutting surface area.
- **Risk (1–5):** likelihood of regressions, training instability, or hard-to-debug failures.

**Phase ratings (as-written)**

| Phase | Complexity | Risk |
|---|---:|---:|
| 0 — Preflight | 2/5 | 3/5 |
| 1 — Contracts/Naming | 3/5 | 4/5 |
| 2 — Alpha Controller Core | 4/5 | 4/5 |
| 3 — Blend Modes + Ghost Grads | 5/5 | 5/5 |
| 4 — Prune Pipeline + Safety | 4/5 | 4/5 |
| 5 — Simic Wiring (Actions/Obs/Rewards) | 5/5 | 5/5 |
| 6 — End-to-End Validation | 3/5 | 3/5 |

#### Phase 0 — Preflight (constants + invariants)

**Ratings:** Complexity 2/5, Risk 3/5

**Tasks**
- Choose constants: `BaseSlotRent`, `EMBARGOED` dwell ticks (default `5`), schedule tick lengths (`0/3/5/8`), and shock coefficient `k`.
- Decide initial `alpha_target_head` menu (`{0.5, 0.7, 1.0}`) and confirm it’s non-zero only (`PRUNE` is the only path to `alpha_target=0`).

**Pre-activities (to reduce Risk → 2/5)**
- Measure baseline reward component magnitudes from a short heuristic run (or existing telemetry): accuracy deltas, rent term, and expected shock range.
- Pick `k` by simulating the schedule deltas (`instant/fast/medium/slow`) and verifying: `shock_instant > shock_fast > shock_medium > shock_slow` under the chosen curve(s).
- Add a math unit test that asserts “slow ramps are cheaper than fast” for the convex shock formula (independent of model code).
- Keep a tiny forward/backward toy test validating `MULTIPLY` identity at birth (`s≈0` from init; `alpha=0` is exact identity).

**Phase 0 calibration (initial numbers)**

- Host param counts (TaskSpec defaults, CPU instantiation):
  - `cifar10`: `6,418` host params
  - `cifar10_deep`: `99,922` host params
  - `cifar10_blind`: `12,074` host params
- CNN seed param scales quickly with channels (examples):
  - `conv_light(C=32)=9,280` (≈`1.45×` `cifar10` host); `conv_heavy(C=32)=18,560` (≈`2.89×`)
  - worst-case ratios in current CNN presets are on the order of `~3×` host (so shock scaling by `seed_params/host_params` can exceed `1.0`)
- Convex shock schedule sums for a 1→0 ramp (snap-to-target endpoints), `Σ(Δalpha^2)`:

  | Curve | `n=1` | `n=3` | `n=5` | `n=8` |
  |---|---:|---:|---:|---:|
  | linear | 1.000 | 0.333 | 0.200 | 0.125 |
  | cosine | 1.000 | 0.375 | 0.239 | 0.152 |
  | sigmoid | 1.000 | 0.608 | 0.374 | 0.241 |

- Suggested starting point (so worst-case instant shock stays “noticeable but not dominant”):
  - `shock_scale = seed_params/host_params`
  - `k = 0.1` (⇒ worst-case instant prune shock ≈ `-0.3` when ratio ≈ `3`)
- Reward scale sanity (current `ContributionRewardConfig` defaults):
  - rent penalty (uncapped) is `rent_weight * log(1 + growth_ratio)` with `rent_weight=0.5`
    - `growth_ratio=0.1` ⇒ rent ≈ `-0.024`
    - `growth_ratio=1.0` ⇒ rent ≈ `-0.347`
    - `growth_ratio=3.0` ⇒ rent ≈ `-0.693`
  - with `k=0.1` and `shock_scale=seed_params/host_params`:
    - instant (`n=1`) shock is ≈ `-0.1 * ratio` (so `ratio≈3` ⇒ ≈ `-0.3`)
    - slow (`n=8`, linear) shock is ≈ `-0.0125 * ratio` (so `ratio≈3` ⇒ ≈ `-0.0375`)

#### Phase 1 — Contracts, naming, and transitions (Leyline + docs)

**Ratings:** Complexity 3/5, Risk 4/5

**Tasks**
- Update lifecycle contracts: `src/esper/leyline/stages.py` (align stage/event naming to `HOLDING`/`PRUNED`, ensure `EMBARGOED` dwell remains concrete).
- Update transitions to support scheduled prune from `HOLDING → BLENDING` (since `HOLDING` is “alpha≈1 only”).
- Update docs that codify lifecycle: `README.md` mermaid diagram + any plan references.

**Pre-activities (to reduce Complexity/Risk → 2/5)**
- Produce an explicit “rename inventory” checklist (`rg` targets + file list) for:
  - `HOLDING`, `PRUNED`, `CULL`, `cull(`, telemetry event names, and any UI labels.
- Identify all schema/serialization boundaries that embed stage/op integers (action tensors, saved reports, telemetry) so nothing silently deserializes to the wrong enum.
- Add a fast “import all” smoke test (or a minimal `pytest -q` target) that imports policy + env modules to catch missing enum members early.
- Keep the change strictly mechanical: update all call sites in one pass (no shims) and rely on existing import-time enum sync assertions (`OP_NAMES`, etc.) to prevent drift.

**Phase 1 risk-reduction artifacts (implemented now, no runtime changes)**
- Inventory script: `scripts/lifecycle_phase1_inventory.sh`
- Import smoke test: `tests/leyline/test_import_smoke.py`

#### Phase 2 — Alpha controller core (Kasmina state + checkpoint)

**Ratings:** Complexity 4/5, Risk 4/5

**Tasks**
- Add alpha-controller fields to `SeedState` and persist them via `to_dict()/from_dict()` in `src/esper/kasmina/slot.py`.
- Implement controller tick update in `SeedSlot.step_epoch()`:
  - monotone UP/DOWN, HOLD-only retarget, snap-to-target on completion.
- Surface telemetry fields (`alpha_target`, `alpha_mode`, `alpha_steps_done/total`, etc.).

**Pre-activities (to reduce Complexity/Risk → 2/5)**
- Prototype the alpha controller as a pure, isolated object (`step()` / `retarget()`), with full unit + property tests, before touching `SeedSlot`.
- Write down the exact mapping between “controller ticks” and existing counters (`blending_steps_done/total`) so we don’t double-advance or mix semantics.
- Decide the definitive state fields for gating (`alpha_target`, `alpha_mode`, `blend_substage`) and make G3 depend only on these (not implicit stage timing).
- Add a dedicated checkpoint/resume roundtrip test for a mid-transition seed (UP, HOLD, DOWN) before integrating into the full env.

**Phase 2 risk-reduction artifacts (implemented now, no runtime changes)**
- Alpha controller module: `src/esper/kasmina/alpha_controller.py`
- Alpha mode/curve contracts: `src/esper/leyline/alpha.py`
- Property + checkpoint math tests: `tests/kasmina/properties/test_alpha_controller_properties.py`

#### Phase 3 — Blend modes (ADD/MULTIPLY/GATE) + ghost gradients

**Ratings:** Complexity 5/5, Risk 5/5

**Tasks**
- Implement `alpha_algorithm_head` wiring (HOLD-only changes).
- Implement composition operators:
  - `ADD`: `lerp(h, s, a)`
  - `MULTIPLY`: `h * (1 + a * tanh(s - seed_input))` + zero-init of final seed layer
  - `GATE`: per-sample `gate(x)` multiplied into amplitude `alpha_amplitude(t)`
- Enforce ghost-gradient requirement during `BLEND_OUT` (freeze params only; no detach/no `no_grad`).

**Pre-activities (to reduce Complexity/Risk → 2/5)**
- Implement the blend operators in isolation (no Kasmina) and test:
  - identity-at-zero,
  - monotone behavior under alpha ramps,
  - gradients for host vs seed under each operator.
- Run `torch.compile` on the isolated blend ops (and a minimal `SeedSlot` forward) to ensure no graph breaks or pathological specialization.
- Explicitly audit interaction with `isolate_gradients` + channels_last safety constraints (BUG-005): ensure no new detach patterns are introduced in `BLENDING+`.
- Decide whether `GATE` reuses existing `GatedBlend` as the per-sample gate (and if so, how its parameters are frozen during `BLEND_OUT`), before any wiring changes.
- Identify every seed blueprint that needs “final layer zero-init” support, and confirm it’s implementable without breaking initialization conventions.

**Phase 3 risk-reduction artifacts (implemented now, no runtime changes)**
- Blend operator contracts: `src/esper/kasmina/blend_ops.py`
- Operator math tests: `tests/kasmina/test_blend_ops_contracts.py`
- Gradient + ghost-gradient tests: `tests/kasmina/test_blend_ops_gradients.py`
- Final-layer zero-init helper: `src/esper/kasmina/blueprints/initialization.py`
- Identity-init blueprint tests: `tests/kasmina/test_blueprint_zero_init_final_layer.py`
- torch.compile smoke tests: `tests/kasmina/test_compile_tensor_ops.py` + `tests/kasmina/test_pytorch_expert_compile.py`
- isolate_gradients + channels_last regression: `tests/kasmina/test_isolation_channels_last_contracts.py`

#### Phase 4 — Prune pipeline + cooldown + safety override

**Ratings:** Complexity 4/5, Risk 4/5

**Tasks**
- Replace `CULL` with explicit `PRUNE` op (see `src/esper/leyline/factored_actions.py`) and map `PRUNE_*` → internal `alpha_target=0`.
- Implement prune completion: physical removal at `alpha==0`, then `EMBARGOED` dwell ticks, then `RESETTING → DORMANT`.
- Remove HOLDING auto-culls; keep catastrophic safety only.
- Add Governor-only emergency override: force `PRUNE_INSTANT` from any mode/stage on NaN/divergence/collapse.

**Pre-activities (to reduce Complexity/Risk → 2/5)**
- Inventory all `cull` call sites and classify them: policy op vs safety mechanism vs manual/debug. This prevents accidental behavior loss.
- Write an explicit stage transition table (current vs proposed) and add a test that enumerates all legal transitions (prevents lifecycle dead-ends).
- Identify the exact “physical removal” pathway today (optimizer cleanup, seed module clearing, telemetry) and ensure prune completion reuses it (no new cleanup bugs).
- Add an embargo/thrash regression test that attempts rapid prune/germinate loops and asserts the dwell prevents it.

#### Phase 5 — Simic wiring (policy heads/masks, obs, rewards)

**Ratings:** Complexity 5/5, Risk 5/5

**Tasks**
- Update policy action space to include `SET_ALPHA_TARGET` + `PRUNE` operations and the new heads (`alpha_algorithm_head`, etc.).
- Add observation fields: `alpha_target/mode/steps`, `time_to_target`, `alpha_velocity`, `alpha_algorithm`.
- Implement reward updates: `BaseSlotRent` floor + convex shock + full-amplitude-only indecision pressure.

**Pre-activities (to reduce Complexity/Risk → 2/5)**
- Write a “shape contract” table: exact head sizes and enum orders, then update the policy network output dims accordingly.
- Add unit tests for action masking across stages/modes (including HOLD-only constraints and Governor-only overrides).
- Add observation schema tests that assert fields exist, have correct dtype/range, and are stable across env resets/checkpoints.
- Calibrate reward coefficients offline using telemetry from short runs (heuristic + a few random actions) to keep rent/shock on comparable scale and avoid immediate reward hacking.
- Add a deterministic “scripted policy” runner used only for smoke tests (no PPO training) to validate end-to-end wiring before RL introduces noise.

#### Phase 6 — End-to-end validation (CIFAR sanity + regressions)

**Ratings:** Complexity 3/5, Risk 3/5

**Tasks**
- Run a small heuristic episode and a short PPO episode (CIFAR) to validate:
  - no NaNs/divergence, prune/fossilize still reachable, and telemetry fields are populated.
- Compare against baseline runs for: prune rate, thrash rate, and reward component magnitudes.

**Pre-activities (to reduce Complexity/Risk → 2/5)**
- Define an explicit acceptance checklist (pass/fail) before running experiments:
  - freeze invariant holds,
  - shock differentiates speeds,
  - BaseSlotRent prevents squatting,
  - `prune_initiator` always populated,
  - no stage dead-ends in telemetry.
- Decide the minimal run matrix (configs + seeds + episode count) so validation is time-bounded and comparable.
- Add a small telemetry summary script/query (even a pytest helper) that asserts the key invariants from the logged run artifacts.

---

## 13) Wiring Gotchas (PPO/Rewards)

- `shock` should be convex in Δalpha (e.g., `Δalpha^2`) so “fast hurts more than slow,” not just “movement is bad”.
- Freeze during `BLEND_OUT` must be param-only (`requires_grad=False`), not graph-breaking (`detach`/`no_grad`), to preserve ghost gradients.
- Governor may force `PRUNE_INSTANT` from any mode/stage for catastrophic safety; the policy cannot (constraints remain HOLD-only).
- Schedules are defined in controller ticks (`alpha_steps_total`), not the word “epoch”; clamp `alpha=target` exactly on completion.
- Add `time_to_target` and `alpha_velocity` to observations to improve learnability of “hold then promote/prune” patterns.
- Always record `prune_initiator` in telemetry so disappearances are debuggable.
- Contract reminder: `HOLDING` is full-amplitude only; partial holds live inside `BLENDING` (`BLEND_HOLD`).

---

## 14) Final Decisions (Resolved)

1) **Embargoed/resetting semantics**
   - Decision: `EMBARGOED` is a **first-class `SeedStage`** and must be observable to masks and telemetry.
   - Contract: after a prune completes, the slot does **not** clear its state immediately. It enters `PRUNED`, then `EMBARGOED` for a fixed dwell (default `5` controller ticks), then `RESETTING`, then finally `DORMANT` (available).
   - Operational detail: the **seed module is physically removed at prune completion** (so compute/rent stops), but the slot remains occupied/unavailable until `DORMANT` to prevent prune/germinate thrashing.

2) **Explicit `PRUNE_*` ops vs `alpha_target=0.0`**
   - Decision: keep `PRUNE` as an explicit lifecycle operation in the policy action space.
   - Internal mapping: `PRUNE_*` maps to `alpha_target=0.0` in the alpha controller (with speed/curve determining the schedule).
   - Constraint: `SET_ALPHA_TARGET` does not allow `target=0.0`; removal intent must be expressed via `PRUNE`.

3) **Explicit `PROMOTE` vs `SET_ALPHA_TARGET(1.0)`**
   - Decision: no `PROMOTE` op; promotion is `SET_ALPHA_TARGET(1.0)`.

4) **`MULTIPLY` bounding**
   - Decision: use `y = h * (1 + a * tanh(s))`.
   - Critical init note: initialize the seed’s final layer weights/biases to `0` so `s≈0` at birth and the valve starts near identity.
