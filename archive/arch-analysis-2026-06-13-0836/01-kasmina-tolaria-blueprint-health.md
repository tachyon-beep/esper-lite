# Kasmina, Tolaria, and Blueprint Health Deep Dive

Date: 2026-06-13

Scope: Kasmina seed/slot/host lifecycle, Tolaria governor and rollback, blueprint registry and growth identity, Tamiyo observation surfaces that describe blueprints/seeds, and the Simic execution/reward path that applies morphogenetic actions.

This is a read-only investigation report. It records bugs, architectural issues, and structural risks found in the current working tree. It does not apply fixes.

## Executive Summary

Esper's core morphogenetic body is real: Kasmina has a coherent host/slot lifecycle, shape-preserving seed probes, alpha blending, gradient isolation, and counterfactual contribution concepts. Tolaria can snapshot, detect catastrophic loss, roll back model state, prune live experimental seeds, and inject punishment into PPO. The blueprint catalogue gives the controller a bounded action space for morphology rather than arbitrary module construction.

The main health issue is not that these parts are absent. It is that their authority boundaries are misaligned:

1. Simic can still mutate the organism directly. Germination, pruning, alpha retargeting, and stage advancement are applied inline in `execute_actions()` rather than passing through a Tolaria-owned pre-flight adjudication layer.
2. Tolaria is currently a post-failure watchdog, not a full growth governor. It observes validation loss after the fact, snapshots periodically, and rolls back after damage.
3. Tamiyo observations and telemetry still contain bug-hiding defaults that can present missing or non-causal evidence as healthy morphogenetic state.
4. Blueprint growth is not replay-identifiable. Module initialization and probes use ambient RNG, and growth events are named descriptively rather than tied to stable action, observation, and RNG identity.
5. The proof/evaluation surface is improving, but the current reward and telemetry evidence still cannot certify blueprint health without confounder caveats.

Overall health: promising but not yet safe to treat as a proof-grade morphogenetic control plane. The top actions are to create a proposal/governor/mutation boundary, fix telemetry truthfulness, and make growth events replayable.

## Models and Components Investigated

| Component | Health | Key evidence | Key actions |
| --- | --- | --- | --- |
| `MorphogeneticModel` and `CNNHost` | Mostly healthy. Host boundaries and slot routing are centralized. | `src/esper/kasmina/host.py` routes active slots through host boundaries and exposes `germinate_seed()`/`prune_seed()` as mutation APIs. | Keep the host simple, but move policy-driven mutation authority behind a governor command surface. |
| `SeedSlot` and `SeedState` | Internally disciplined, but too much complexity sits in one object. | `SeedSlot.germinate`, `advance_stage`, `forward`, `step_epoch` are high-coupling hotspots. | Split adjudication/proposal from mutation side effects; add tests that exercise public lifecycle paths rather than direct state writes. |
| Blueprint registry and blueprints | Useful bounded catalogue, but identity is not hardened. | Duplicate registration overwrites keys at `registry.py:70`; listing sorts only by param estimate at `registry.py:89`; unknown IDs become `-1` at `slot.py:550`. | Reject duplicate registrations, add stable tie breakers/action IDs, and fail loudly on unknown blueprint IDs. |
| Tolaria governor | Rollback mechanics are substantial and tested; governor authority is incomplete. | `check_vital_signs()` is loss-only at `governor.py:186`; snapshots can run before panic checks at `vectorized_trainer.py:1167`. | Add pre-flight veto, robust health windows, action watch windows, and deterministic event logs. |
| Simic action execution | Reward/accounting is broad, but lifecycle mutation is inline and duplicated with unused handlers. | Direct calls at `action_execution.py:781`, `918`, `933`, `947`; handler registry is only used by tests. | Replace inline mutation block with one authoritative action executor or wire the handler registry into production. |
| Tamiyo observations | Shape-invariant structure is good; semantic hygiene is not complete. | Missing telemetry becomes healthy at `features.py:317` and `features.py:750`; missing counterfactual contribution falls back to host drift at `features.py:270` and `features.py:709`. | Make missing evidence explicit and fail closed where policy or gates depend on causal truth. |
| Simic reward and proof surface | Stronger than the older system, but not proof-grade yet. | `SIMPLIFIED` omits structural rent at `contribution.py:1101`; dual A/B is documented as sequential at `dual_ab.py:12`. | Keep proof packets blocked when confounded; add lockstep baselines and structural cost to any reward mode used as health evidence. |

## Findings

### P1. Snapshot Can Capture Bad State Before Panic Detection

Current training order snapshots first, then checks Tolaria vital signs:

- `src/esper/simic/training/vectorized_trainer.py:1167` calls `env_state.governor.snapshot()`
- `src/esper/simic/training/vectorized_trainer.py:1172` then calls `check_vital_signs(val_loss)`

If divergence lands on a snapshot epoch, the last-known-good snapshot can be overwritten by the divergent state immediately before panic detection. That weakens rollback's core promise.

Action: check vital signs first, then snapshot only after the state is known clean. Fossilization snapshots should also pass through a "safe to bless" check rather than being blindly promoted.

### P1. Rollback Does Not Halt Current Action Execution

`execute_actions()` rolls back at the top of the per-env loop, but then continues into reward computation and lifecycle mutation:

- Rollback path: `src/esper/simic/training/action_execution.py:447`
- Mutation block starts later: `src/esper/simic/training/action_execution.py:765`

That means a sampled action based on pre-rollback observations can be applied immediately after emergency recovery.

Action: after rollback, skip normal lifecycle action execution for that env and write a typed terminal rollback transition. The next policy decision should be based on post-rollback observations.

### P1. No Independent Pre-Flight Governor Veto

Policy-side masks and Kasmina local checks are not the same as governor authority. Simic directly invokes lifecycle mutation:

- Germinate: `src/esper/simic/training/action_execution.py:781`
- Schedule prune: `src/esper/simic/training/action_execution.py:918`
- Set alpha target: `src/esper/simic/training/action_execution.py:933`
- Advance stage: `src/esper/simic/training/action_execution.py:947`

Tolaria checks validation loss after the fact through `check_vital_signs()` at `src/esper/tolaria/governor.py:186`. This gives Tolaria rollback power, not pre-mutation authority.

Action: introduce a Tolaria-owned pre-flight API: proposal in, verdict out. Inputs should include operation, slot, blueprint, alpha schedule, current loss envelope, gradient/weight health, budget/rent state, cooldowns, and event identity. Simic should submit proposals and apply only approved mutation commands.

### P1. Missing Sensor Data Is Represented As Healthy State

Tamiyo converts missing slot telemetry into healthy-looking observations:

- Scalar path: `src/esper/tamiyo/policy/features.py:317` to `features.py:322`
- Batched path: `src/esper/tamiyo/policy/features.py:750` to `features.py:755`
- Previous gradient health also defaults to healthy through `.get(..., 1.0)` at `features.py:326` and `features.py:758`

Kasmina permissive G2 has a related issue: `gradient_health_ok` defaults true when telemetry is absent at `src/esper/kasmina/slot.py:823`.

This violates the "sensors match capabilities" rule. Missing telemetry should not look safe to the controller or to lifecycle gates.

Action: for active slots, require telemetry once a seed enters training. If missing telemetry is expected during a narrow bootstrap window, encode it explicitly with a missing/freshness feature and do not convert it to healthy.

### P1. Non-Causal Host Drift Can Enter Policy Observations

Kasmina correctly distinguishes causal contribution:

- `src/esper/kasmina/slot.py:155` says `counterfactual_contribution` is the true causal attribution.
- `src/esper/kasmina/slot.py:247` warns that some improvement metrics include host drift and are for telemetry/logging only.

Tamiyo still falls back from missing counterfactual contribution to `improvement_since_stage_start`:

- `src/esper/tamiyo/policy/features.py:270` to `features.py:273`
- `src/esper/tamiyo/policy/features.py:709` to `features.py:712`

The freshness feature helps, but the value channel remains contaminated. The policy can learn from host drift as if it were seed contribution.

Action: remove the fallback. Use neutral contribution plus explicit freshness/missing flags when counterfactual data is unavailable.

### P1. Blueprint Shape Probe Can Let Invalid Return Types Through

Kasmina validates blueprint output shape only when the output is a tensor:

- `src/esper/kasmina/slot.py:1382` calls the seed on a probe tensor.
- `src/esper/kasmina/slot.py:1383` checks shape only under `isinstance(seed_out, torch.Tensor)`.

If a malformed blueprint returns a tuple/list/object, germination can pass the smoke test and fail later in forward/blending. That is a bug-hiding pattern: the contract is "blueprint returns a tensor with the same shape," so non-tensor output should fail immediately.

Action: raise when `seed_out` is not a tensor, then validate shape. Add a malformed-blueprint regression test.

### P2. Growth Events Are Not Replay-Identifiable Or RNG-Isolated

Germination creates modules through `BlueprintRegistry.create()` using ambient torch RNG:

- `src/esper/kasmina/slot.py:1361`

Shape probes also use `torch.randn()` without an explicit generator:

- `src/esper/kasmina/slot.py:1151`
- `src/esper/kasmina/slot.py:1159`

Simic seed IDs are descriptive strings:

- `src/esper/simic/training/action_execution.py:783`

These are useful for humans, but they are not stable morphogenesis event identities. A deterministic replay system needs event IDs, action IDs, observation hashes, RNG state/seed, topology, slot, blueprint ID, and governor verdicts.

Action: add deterministic growth-event records and pass a scoped generator or captured RNG state through blueprint creation and probes.

### P2. Tolaria Is Loss-Only And Mean/Std Based

`TolariaGovernor.check_vital_signs()` handles NaN/Inf, random-guess lobotomy, and mean/std loss spikes:

- NaN/Inf: `src/esper/tolaria/governor.py:208`
- Lobotomy: `src/esper/tolaria/governor.py:216`
- Mean/std anomaly: `src/esper/tolaria/governor.py:238`

It does not incorporate gradient norm, weight norm, budget/rent state, action legality, alpha rate, per-event watch windows, or robust median/MAD statistics.

Action: treat current Tolaria as a watchdog layer and add a fuller `MorphogenesisGovernor` surface around it. That surface should own pre-flight, watch, commit, rollback, cooldown, and audit events.

### P2. Rollback Signal Exists But Is Coarsely Attributed

PPO does receive a rollback penalty:

- `src/esper/simic/training/ppo_coordinator.py:117` to `ppo_coordinator.py:124`
- `src/esper/simic/agent/rollout_buffer.py:709`

But it overwrites the last transition without a typed causal event/watch-window record. Advantages are normalized globally in `rollout_buffer.py:543`, so rare rollback events can be washed into ordinary advantage statistics.

Action: mark rollback transition type, severity, triggering proposal/action ID, and watch-window evidence. Consider stratified rollback metrics or advantage handling so catastrophic morphology signals stay visible.

### P2. Telemetry Event IDs Exist, But Morphology Events Are Not A Causal Log

The generic telemetry envelope has a UUID event ID at `src/esper/leyline/telemetry.py:121`, and Karn exposes raw event views. That is observability, not deterministic morphogenesis replay.

For growth control, the missing unit is a monotonic causal event log: proposal, governor verdict, mutation commit, post-event watch summary, and rollback/fossilization links.

Action: add morphology event IDs that survive across Simic, Kasmina, Tolaria, Nissa, and Karn. Each event should carry operation, slot, blueprint/action indices, RNG identity, observation hash, governor verdict, and reward mode.

### P2. Lifecycle Handler Refactor Is Not Wired Into Production

`src/esper/simic/training/handlers/` defines strategy handlers and a registry, but production still calls `execute_actions()` directly from `vectorized_trainer.py:1457` and then applies lifecycle side effects inline. Current handler references are tests and package exports, not the hot path.

This creates two sources of operation semantics and normalizes a legacy/incomplete split.

Action: either wire the handler registry into `execute_actions()` as the authoritative mutation executor or delete the unused abstraction. Do not keep both paths.

### P2. Simplified Reward Does Not Pay Structural Rent

`compute_simplified_reward()` contains PBRS, action cost, and terminal accuracy/fossilization bonuses:

- `src/esper/simic/rewards/contribution.py:1101`
- `src/esper/simic/rewards/contribution.py:1116` to `contribution.py:1125`

It does not include parameter rent or structural cost. That may be fine for a diagnostic reward, but it is not a blueprint-economy health signal.

Action: either add structural cost to any reward mode used for blueprint health evidence, or label `SIMPLIFIED` as a non-economy diagnostic that cannot be used to prove complexity pays rent.

### P2. Reward And Decision Telemetry Can Mislead

`RewardComponentsTelemetry.action_success` defaults true at `src/esper/simic/rewards/reward_telemetry.py:90`, but the actual action outcome is not known until later in `src/esper/simic/training/action_execution.py:955`. The stale component is emitted through `src/esper/simic/telemetry/emitters.py:337`.

Per-decision entropy is also explicitly unavailable during sampling:

- `src/esper/simic/training/vectorized_trainer.py:1443` to `vectorized_trainer.py:1445`

Action: set action success only after action execution, or remove it from reward components and keep it exclusively on action outcome. Compute real per-head entropy if telemetry displays it.

### P3. Proof Baselines Are Still Not Strong Enough For Final Blueprint Health Claims

The roadmap states Phase 2.5 gates are pending before Phase 3. The current proof-confounder plan also records that evidence must fail closed on confounders, and the latest proof rehearsal is blocked by value-collapse and gradient-anomaly confounders.

Dual A/B is also currently sequential rather than lockstep:

- `src/esper/simic/training/dual_ab.py:12`
- `src/esper/simic/training/dual_ab.py:182`

Action: implement proof cohorts as first-class experimental controls: off-switch, static initial, static final, fixed schedule, and lockstep reward A/B. Use proof packets as health evidence only when the confounder ledger is clean.

## Architecture Direction

The healthier control shape is:

1. Tamiyo samples an action proposal against policy masks.
2. Tolaria adjudicates the proposal before mutation.
3. Kasmina applies an approved mutation command.
4. Nissa records a causal event: proposal, verdict, mutation, RNG identity, observation hash, reward mode, and post-event watch window.
5. Tolaria commits or rolls back after the watch window.
6. Simic receives a typed reward/penalty signal tied to the same event ID.

This keeps the metaphors aligned with the code: Tamiyo proposes, Tolaria protects the organism, Kasmina changes the body, Simic supplies selection pressure, and Nissa/Karn make the evidence inspectable.

## Priority Action Plan

1. Fix rollback ordering and halt semantics.
   - Check Tolaria before snapshotting.
   - Skip lifecycle mutation after rollback in the same env step.
2. Fix observation truthfulness.
   - Missing telemetry must not become healthy.
   - Missing counterfactual contribution must not fall back to host drift.
3. Add a Tolaria pre-flight mutation API.
   - Proposal in, verdict out, mutation command only on approval.
4. Harden blueprint contracts.
   - Reject non-tensor blueprint outputs.
   - Reject duplicate registry keys.
   - Make unknown blueprint IDs fail loudly.
5. Add deterministic morphogenesis event identity.
   - Scoped RNG, event IDs, action IDs, observation hashes, governor verdicts, and replay fields.
6. Collapse the duplicate Simic action executor.
   - Use the handler registry in production or remove it.
7. Make proof evidence fit the claim.
   - Structural rent in proof reward modes.
   - Lockstep A/B and static/off-switch/fixed-schedule controls.

## Suggested Tracker Breakdown

1. P1 bug: Tolaria snapshot must not overwrite last-known-good state before panic detection.
2. P1 bug: rollback env must skip normal lifecycle action execution for the current step.
3. P1 bug: Tamiyo active-slot missing telemetry must not emit healthy observation features.
4. P1 bug: Tamiyo missing counterfactual contribution must not use host-drift fallback.
5. P1 bug: Kasmina germination must reject non-tensor blueprint outputs.
6. P2 architecture: introduce Tolaria pre-flight governor proposal/verdict API.
7. P2 architecture: add deterministic morphogenesis event identity and RNG isolation.
8. P2 cleanup: wire or remove Simic lifecycle handlers.
9. P2 proof: make `SIMPLIFIED` structural-cost aware or mark it non-economy diagnostic.
10. P3 proof: implement lockstep baseline cohorts and confounder-clean proof packet gating.

## Verification Performed

- Read project instructions from `CLAUDE.md`, `README.md`, and `ROADMAP.md`.
- Used Loomweave project status and entity/coupling queries for Kasmina and Tolaria orientation.
- Ran line-level source inspections over Kasmina, Tolaria, Tamiyo, Simic reward/training, telemetry, and proof-plan files.
- Used three read-only subagents for independent Kasmina/blueprint, Tolaria/governor, and telemetry/reward/evaluation passes.

No source fixes or tests were run as part of this report.
