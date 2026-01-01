# Detailed Specification: Esper Immune System Phase 4

Status: Draft Specification (Concept Locked)
Date: 2025-12-31

This document expands the original Phase 4 immune system design (Narset, Emrakul, Phages, Lysis) and incorporates the later control-layer work: Narset as endocrine allocator with leases and multi-colour warnings, Emrakul as conservative hygiene/triage agent, and Tamiyo’s interaction model (already implemented) through shared safety signals.

This spec aims to be implementable without runtime graph surgery, per-op hooks, or compile-hostile dynamic control flow.

---

## 0. Glossary and Naming

* **Host**: the base model topology (CNN, Transformer, etc) excluding Seeds.
* **Seed**: grafted capacity (Tamiyo/Kasmina system).
* **Phage**: a gate + sensors attached to a host segment to test redundancy and enable removal.
* **Gate**: compile-friendly scalar alpha controlling contribution of a host segment.
* **Functional Lysis**: runtime “disable” via gradient detachment/freeze (no physical deletion).
* **Physical Lysis**: offline rewrite removing host block and leaving bypass.
* **Narset**: endocrine allocator controlling who may act and under what constraints.
* **Lease**: time-bounded control permission for a child agent to act.
* **Cyan**: time pressure (lease nearing end).
* **Yellow**: risk pressure (stability degrading).
* **Red**: emergency mode (hard abort).
* **Scope**: resources protected by a lease (seed slots and/or gate IDs).
* **Turbulence**: instability regime where hygiene is disabled.

---

## 1. Purpose and Boundaries

### Goal

Autonomously audit and remove obsolete host structure after successful seed takeovers while preserving training stability and compile performance.

### Non-goals

* No new seed lifecycle logic: Kasmina/Tamiyo remains sole seed operator.
* No runtime graph surgery: no module deletion, no FX rewrites, no dynamic layer removal mid-training.
* No per-op hooks: gating is explicit and segment-level to preserve `torch.compile`.
* No “make it faster by magic”: runtime forward FLOP reduction is not guaranteed until Physical Lysis; Functional Lysis primarily reduces backward/optimizer traffic.

### Success definition

* Host detritus decreases over time without destabilising training.
* Training does not regress significantly under stable conditions.
* Hygiene does not activate in turbulence.
* Physical lysis yields measurable throughput and memory improvements on subsequent runs.

---

## 2. System Overview: Metabolic Equilibrium

The immune system is an equilibrium between growth and hygiene, mediated by a parent allocator.

* **Tamiyo (growth)** adds capacity and integrates knowledge.
* **Emrakul (hygiene)** tests host redundancy and removes dead structure.
* **Narset (endocrine)** budgets time, authority, and safety constraints.

Key property: the parent does not micromanage actions; it sets permissions, warnings, and emergency rules.

---

## 3. Subsystems

### 3.1 Narset: Endocrine Control Layer

#### Role

Narset is the strategic allocator of “agency time” and the safety arbiter. Initially heuristic, later learnable. Narset does not know task details, only whether the organism is stable enough for certain kinds of work.

#### Inputs

Global health telemetry (from Nissa or global system monitors):

* loss trend and volatility (short and medium windows)
* gradient norm trend and volatility
* frequency/magnitude of loss spikes
* training throughput proxies (optional)
* recent rollback/restore counts (from both children)
* lease outcomes (aggregate slips)

#### Outputs

A small set of control signals provided to both Tamiyo and Emrakul each epoch:

1. **Lease allocation**

* `lease_owner ∈ {NONE, TAMIYO, EMRAKUL}`
* `lease_time_remaining` (epochs)
* `scope_lock` (seed slots, gate IDs, or both)
* optional: `operation_budget` (max non-WAIT ops)

1. **Multi-colour warnings**

* **Cyan**: time warning, independent of risk
* **Yellow**: risk warning, independent of time
* **Red**: emergency mode

Representations exposed to policies:

* `time_warning` (cyan) boolean
* `risk_warning` (yellow) boolean
* `hard_abort` (red) boolean
* `lease_time_remaining_norm` scalar in [0,1]
* optional: `stability_score` scalar in [0,1]

#### Core policy (initial heuristic)

Narset operates as a thermostat with hysteresis and cooldown:

* Define stability score `S ∈ [0,1]` from volatility/spike signals.
* Enable hygiene only if `S` remains high for a sustained window.
* Disable hygiene quickly if `S` drops below a lower threshold.

Example regime:

* **Turbulent**: `S < 0.6`

  * hygiene disabled, Emrakul lease never granted
  * yellow may be raised for risk
  * red if hard limits breached
* **Calm**: `0.6 ≤ S < 0.85`

  * Tamiyo leases normal length
  * Emrakul leases short and rare
* **Stagnant**: `S ≥ 0.85` for N epochs

  * hygiene leases allowed and longer
  * encourage Emrakul audits

#### Lease concept

A lease is a time-limited “control lease”, not a plan. A policy can act freely within lease constraints.

Lease fields:

* `owner`: Tamiyo or Emrakul
* `duration_epochs`
* `scope_lock`:

  * seed slot IDs (Tamiyo scope)
  * gate IDs (Emrakul scope)
  * optional combined scope for coordinated sequences
* `abort_thresholds`: stability limits that trigger red
* `cooldown`: minimum time after lease before the same scope is re-leased

#### Cyan, Yellow, Red semantics

* **Cyan**: “wrap up, lease ending soon”

  * triggered by time remaining below a threshold
  * does not imply instability
* **Yellow**: “risk rising, de-risk now”

  * triggered by stability degradation
  * independent of lease timing
* **Red**: “emergency mode”

  * triggered by hard stability breach or yellow failure
  * sticky for minimum duration to prevent flicker

#### Parent intervention model

Narset does not override actions by default. Instead it changes the safety regime and the child must react. Only if the child fails to respond to red for N steps does Narset perform a last-resort hard intervention (optional).

---

### 3.2 Emrakul: Hygiene and Triage Policy

#### Role

Emrakul operates on host segments instrumented with Phages. She never touches seeds directly, but she must understand seed adjacency and scope conflicts.

Emrakul performs:

* hygiene: remove redundant host structure
* triage: identify struggling blocks for Tamiyo to assist (optional mode)

#### Action space (conceptual)

Per candidate gate/segment:

* **WITHER**: decrease host contribution alpha toward 0
* **RESTORE**: increase alpha toward 1 (abort audit)
* **NECROSE**: lock alpha at 0 and mark segment for Functional Lysis; eligible for Physical Lysis offline
* **SLEEP**: do nothing / defer

Action granularity is segment-level, not per-layer.

#### Conservatism principle

Emrakul is rewarded for safe traffic reduction and penalised heavily for churn or regressions. “Do nothing” must be a valid choice.

#### Observation

Emrakul’s obs is a combination of:

* global stability and Narset regime signals
* a candidate set of phages (shortlist preferred)
* adjacency signals to seed activity (conflict prevention)
* audit context: which segments are under test, time since last alpha change

Candidate feature per phage gate should include:

* current gate alpha
* gate state (DORMANT, AUDITING, ESSENTIAL, NECROTIC)
* gradient flow measures (norm, health, trend)
* contribution proxy (counterfactual or controlled delta if available)
* time since last intervention
* whether a seed is currently active in this scope
* whether this segment recently got “superseded” by a fossilised seed
* whether the host block is shape-critical (downsample, projection) metadata

#### Reward and evaluation signal

Two-tier measurement is recommended:

1. Fast learning signal (low weight)

* counterfactual-ish estimate or local churn proxy

1. Verified signal (high weight)

* controlled micro-intervention eval: same batches, alpha toggled (1.0 vs 0.5 vs 0.0), measure delta

Reward should emphasise:

* no regression under controlled test
* reduced backward/optimizer traffic during functional lysis
* large reward only when physical lysis completes and subsequent run confirms no regression

#### Audit pacing

Rate limit audits:

* only one active audit per model or per region at a time (default)
* limit necrosis events per window
* per-gate refractory: a restored gate cannot be re-audited for X epochs

---

### 3.3 Phage: Tactical Interface and Gate Mechanism

#### Core requirement

Phage gating must be explicit in forward, compile-friendly, stable shape, and architecture-specific.

#### Task-level phage wrappers

Phage wrappers are defined at the task level, not in generic core, to ensure correct semantics:

* CNN tasks use `CNNPhageGate` attached to residual branches or residual outputs.
* Transformer tasks use `TransformerPhageGate` attached to attention and MLP residual contributions.

Leyline defines the protocol and spec; tasks bind it to concrete module locations.

#### Gate placement: preferred patterns

CNN residual block:

* standard: `y = shortcut(x) + F(x)`
* gated: `y = shortcut(x) + alpha * F(x)`

Transformer block:

* `y = x + alpha_attn * Attn(x)`
* `z = y + alpha_mlp * MLP(y)`

This avoids inventing bypass layers most of the time, because bypass is already the residual identity/shortcut.

#### Functional lysis (runtime)

When a gate is necrotic:

* host branch stops participating in backward:

  * either compute host output detached, or run host branch under no-grad in a way that does not introduce compile-hostile control flow
* host parameters are frozen and removed from optimizer param groups if practical

Forward FLOPs for the host branch may still be paid until Physical Lysis. The immediate savings come from backward and optimizer work.

#### Physical lysis (offline)

During offline save/load:

* rewrite the architecture definition to remove the host block from the forward path
* leave only bypass/shortcut
* reinitialise optimizer state (recommended) due to parameter topology change
* generate a mapping report so telemetry can trace “gate_id was excised”

---

### 3.4 Lysis: Terminal State

#### Trigger

Gate enters lysis when:

* `alpha == 0` (or below epsilon) AND
* Narset stability regime permits (calm/stagnant) AND
* gate is not in conflict with any active seed scope

#### States

Each gate maintains a small state machine:

* DORMANT: normal operation, observed
* AUDITING: alpha being adjusted under Emrakul
* ESSENTIAL: audit failed; protected from re-audit for refractory period
* NECROTIC: alpha locked to 0; functional lysis applied; queued for physical lysis
* EXCISED: physical lysis complete (offline); gate spec persists for historical tracking

---

## 4. Lifecycle: Infection and Digestion

### Step 1: Infection (Marker)

When Tamiyo fossilises a seed in a region, the host segment at that region may become “suspect”.

Actions:

* attach or activate a phage gate for the relevant host segments
* initialise gate state to DORMANT
* record adjacency: which seed takeover triggered suspicion (for telemetry only)

Rules:

* never infect segments currently hosting active (non-fossilised) seeds
* infection is tagging and sensing only, not withering

### Step 2: Audit Initiation (Narset’s call)

Narset grants Emrakul a lease only under stability.
Emrakul selects one candidate gate and begins an audit:

* set state to AUDITING
* decay alpha gradually or stepwise (eg 1.0 → 0.5)

### Step 3: Stress Test (Controlled)

Preferred: controlled micro-interventions:

* run K evaluation batches with alpha=1.0 vs alpha=0.5 (or 0)
* compute churn score, confidence, and stability impact

Outcomes:

* Redundancy: churn small and stable

  * continue wither steps toward 0
* Dependency: churn large or unstable

  * restore alpha toward 1
  * mark ESSENTIAL and apply refractory period

### Step 4: Necrosis (Runtime disable)

If alpha reaches 0 and stability remains calm:

* mark NECROTIC
* apply functional lysis (freeze/detach)
* queue gate for offline physical lysis

### Step 5: Physical Lysis (Offline)

On next save/load or explicit maintenance run:

* rewrite topology removing host block
* regenerate model config and state dict
* reinitialise optimizer (safe default)
* mark gate EXCISED

---

## 5. Host Gating Surface

### Leyline ownership

Leyline defines:

* `HostGateSpec`
* `HostHygieneProtocol`
* constants: number of stages/ops, etc
* stable gate id conventions

### Protocol

Minimal interface:

* `gate_specs() -> list[HostGateSpec]`
* `set_gate_alpha(gate_id: str, alpha: float) -> None`
* `get_gate_alpha(gate_id: str) -> float`

Additional recommended methods:

* `get_gate_state(gate_id) -> GateState`
* `set_gate_state(gate_id, state)`
* `gate_metadata(gate_id)` for reporting and debugging
* `list_conflicts(seed_slots_active, gate_ids)` helper

### HostGateSpec fields

Each spec should include:

* `gate_id` stable string
* `kind` (cnn_residual, transformer_attn, transformer_mlp, etc)
* `shape_class` metadata (identity vs projection shortcut)
* `param_count` (approx)
* `compute_cost_estimate` (optional)
* `supports_physical_lysis` boolean
* `seed_conflict_scope` (which seed regions overlap)

---

## 6. Narset Lease and Alert Enforcement

### Lease scheduling rules

* only one child holds a lease over a given scope at a time
* child leases are time-blocks measured in epochs
* lease ends by expiry unless terminated early by yellow/red escalation

### Cyan (time warning)

* raised when remaining time below threshold
* does not restrict actions as harshly as yellow
* encourages wrap-up actions via mild penalties and action masks

### Yellow (risk warning)

* raised when stability falls but not yet red
* action masks narrow to reversible actions
* encourages alpha dial-down and avoidance of irreversible operations

### Red (hard abort)

* raised when hard limits breached or yellow not resolved
* emergency mode: policy is expected to execute a stabilising reflex

Tamiyo-specific red regime:

* large escalating penalty for any op not in emergency set
* reward dampening to prevent “gaming” in red

If purge exists and is the emergency reflex:

* in red: only PURGE and limited alpha-de-risk ops remain valid via mask
* penalty escalates for non-purge if non-purge remains unmasked

---

## 7. Coordination: Tamiyo and Emrakul Interaction

### Conflict prevention

Hard rules:

* Emrakul may not audit or necrose any gate that overlaps an active seed lifecycle (non-fossilised) region.
* Narset scope locks prevent simultaneous meddling in the same region.

### Cooperative signals

Optional but recommended:

* Emrakul can emit triage flags: “this region is unstable / needs capacity”
* Tamiyo can incorporate those as hints, not commands

### Non-stationarity handling

* Narset leases reduce interleaving of actions by different agents
* multi-colour warnings provide shared safety language
* per-gate refractory prevents repeated tug-of-war

---

## 8. Safety Guardrails

Mandatory:

1. Turbulence lock: no audits when stability low.
2. Conflict prevention: never necrose a segment hosting an active seed.
3. Rate limiting: cap necrosis events per window.
4. Single active audit per model by default.
5. Hysteresis and cooldowns on Narset warnings and leases.
6. Red regime action masking / emergency reflex.

Recommended:

* per-gate refractory after restore/essential marking
* “reason code” for red triggers (optional) to avoid wrong reflex (deadline vs risk vs external)

---

## 9. Telemetry and Metrics

### Core metrics

* `detritus_ratio`: fraction of host params in gates marked suspect/necrotic/excised
* `hygiene_throughput_gain`: measured after physical lysis
* `audit_success_rate`: fraction of audits that reach necrosis without regression
* `rollback_rate`: restores per audit window
* `stability_violations`: red events per epoch

### Audit telemetry

Per gate:

* alpha trajectory over time
* churn score under controlled tests
* gradient health trend
* seed adjacency status
* state transitions (DORMANT → AUDITING → …)

### Lease outcome slips

Per lease:

* tokens/epochs used
* outcome metric delta
* stability delta
* confidence score
* rollback count
* whether yellow/red occurred

---

## 10. Validation Plan

### Unit tests

* gate alpha updates are compile-safe and deterministic
* inactive and necrotic states behave correctly (no-grad, frozen params)
* conflict checks prevent illegal audits
* lease and alert logic produces correct masks and transitions

### Integration tests

* run with Emrakul OFF vs ON with identical seeds schedule; compare regression
* ensure no necrosis during turbulence
* verify that restored gates enter refractory and cannot be re-audited prematurely

### A/B testing

Compare:

* detritus_ratio vs throughput after physical lysis
* learning curves and stability metrics
* number of red events and purge events

### Controlled redundancy benchmark

Construct a known-redundant host region and ensure Emrakul can remove it safely under calm conditions.

---

## 11. Implementation Notes and Constraints

### Compile and performance

* gating must remain explicit arithmetic, no hooks
* avoid per-step `.to(device)` allocations for any lookup tables
* avoid per-forward tensor allocations in hot paths (register buffers where needed)

### Optimizer state and physical lysis

* physical lysis invalidates optimizer state; reinit is safest
* if later you want migration, it must be explicit and tested

### Task-level gating binders

Each task must provide a binder that:

* defines gate points and gate IDs
* implements the host hygiene protocol
* provides metadata for selection and safe bypass semantics
* ensures no shape changes from gating

---

## 12. Open Decisions

1. Do we require controlled eval micro-interventions for necrosis eligibility, or allow heuristics early?
2. How many concurrent audits are allowed in mature phase (default 1)?
3. Whether Narset learns as contextual bandit vs auction vs remains heuristic for long.
4. Red reason codes: do we distinguish deadline red vs stability red vs external disturbance?
5. Whether Emrakul has a triage mode that feeds hints to Tamiyo.

---

## Summary

This expanded immune system design introduces a clean control hierarchy:

* Narset allocates time and safety regimes through leases and multi-colour alerts (cyan deadline, yellow risk, red emergency).
* Emrakul performs conservative hygiene on task-defined phage gates, primarily using controlled tests and strict guardrails.
* Phage gates provide compile-friendly, shape-stable segment-level soft pruning with offline physical deletion.
* Tamiyo is shaped to stabilise under yellow and execute emergency reflex under red, without parent overrides by default.

This produces a biologically-inspired, engineering-realistic system that can remove dead host structure without turning training into a chaotic multi-agent knife fight.
