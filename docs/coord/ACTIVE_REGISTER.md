# Active Register

**Last Updated:** 2026-06-13
**Purpose:** Human-readable register of active work that is too important to live only in Filigree.
**Source of Truth:** Filigree is still the operational tracker. This document is the readable register for coordination and planning.

---

## Morphogenesis Governor Integrity Fault Drain

| Field | Value |
| --- | --- |
| Parent work package | `esper-lite-04d45ad04c` |
| Title | Morphogenesis Governor Integrity fault drain |
| Priority | P1 |
| Status | `defined` |
| Type | `work_package` |
| Parent release | `esper-lite-88a71d93f9` (`Future`) |
| Source report | `docs/arch-analysis-2026-06-13-0836/01-kasmina-tolaria-blueprint-health.md` |
| Slotting note | `docs/analysis/2026-06-13-arch-health-task-slotting.md` |
| Plan | `docs/plans/planning/2026-06-13-morphogenesis-governor-integrity.md` |

### Why This Register Exists

The architecture health report found that the main confounders are not missing core mechanics. They are ordering, authority-boundary, observation-truthfulness, and proof-evidence problems around existing mechanics. These tickets are the active fault drain before reward-efficiency or blueprint-health results should be treated as proof evidence.

The practical sequencing is:

1. Drain the P1 rollback, observation, blueprint-contract, and pre-flight authority faults.
2. Add the minimum event identity and telemetry repairs needed to make failures traceable.
3. Only then run PPO stability/oracle-sandbox and proof baseline work.

---

## P1: Immediate Faults

| ID | Type | Status | Title | Component | Evidence / Expected Outcome |
| --- | --- | --- | --- | --- | --- |
| `esper-lite-45a066bc37` | bug | `confirmed` | Snapshot can overwrite last-known-good state before panic detection | `simic/training/vectorized_trainer.py` | Snapshot currently precedes `check_vital_signs`; vital signs should be checked before blessing state. |
| `esper-lite-7cb8273acf` | bug | `confirmed` | Rollback env continues into stale lifecycle action execution | `simic/training/action_execution.py` | Rollback should produce a terminal rollback outcome for that env step and skip stale sampled lifecycle mutation. |
| `esper-lite-623f17c88f` | feature | `proposed` | Lifecycle mutations bypass independent Tolaria pre-flight veto | `tolaria/simic/kasmina` | Simic should submit lifecycle proposals to Tolaria and apply only approved mutation commands. |
| `esper-lite-a31b7d3b3e` | bug | `confirmed` | Active-slot missing telemetry is represented as healthy policy state | `tamiyo/policy/features.py` | Missing telemetry for active training slots must not become healthy gradient features. |
| `esper-lite-d2c74e7031` | bug | `confirmed` | Missing counterfactual contribution falls back to non-causal host drift | `tamiyo/policy/features.py` | Missing causal contribution should use neutral value plus explicit missing/freshness signal, not host drift. |
| `esper-lite-08e18c5ed8` | bug | `confirmed` | Blueprint shape probe accepts non-tensor outputs | `kasmina/slot.py` | Germination smoke test should reject non-tensor blueprint outputs before shape validation. |
| `esper-lite-a0c50bbe15` | bug | `confirmed` | Blueprint registry identity is not hardened against duplicate or unknown IDs | `kasmina/blueprints` | Duplicate blueprint registrations and unknown blueprint IDs should fail loudly rather than degrade into ambiguous identity. |

### P1 Acceptance Shape

- A panic epoch cannot overwrite the last-known-good snapshot.
- A rollback step cannot apply a stale lifecycle mutation afterward.
- Tamiyo cannot see missing active-slot telemetry or host drift as healthy causal evidence.
- Kasmina rejects malformed blueprint outputs at germination.
- Blueprint identity is deterministic enough that later proof packets can name what actually happened.

---

## P2: Foundation and Telemetry Faults

| ID | Type | Status | Title | Component | Evidence / Expected Outcome |
| --- | --- | --- | --- | --- | --- |
| `esper-lite-8f008c76ba` | feature | `proposed` | Growth events are not replay-identifiable or RNG-isolated | `kasmina/simic/leyline` | Growth records should carry event/action IDs, observation hashes, RNG identity, topology, slot, blueprint, and governor verdict. |
| `esper-lite-f6d8e49701` | feature | `proposed` | Tolaria governor is loss-only and mean/std based | `tolaria/governor.py` | Add a fuller morphogenesis governor surface around the watchdog: pre-flight, watch, commit, rollback, cooldown, and audit. |
| `esper-lite-3032cb39f8` | bug | `confirmed` | Rollback penalty signal is coarsely attributed and may wash out | `simic/agent/rollout_buffer.py` | Rollback transitions need typed severity, proposal/action ID, and watch-window evidence so catastrophic signals stay visible. |
| `esper-lite-45a64bfb36` | feature | `proposed` | Telemetry event UUIDs are not a causal morphology log | `leyline/telemetry.py` | Generic UUIDs are not enough; morphology needs a causal log from proposal through verdict, mutation, watch, rollback, and fossilization. |
| `esper-lite-7431b05440` | bug | `confirmed` | Reward and decision telemetry can report misleading action success or entropy | `simic/rewards/reward_telemetry.py` | Reward telemetry should not claim action success before execution; entropy should be real or explicitly unavailable. |
| `esper-lite-c170aa1198` | task | `open` | Lifecycle handler refactor is not wired into production | `simic/training/handlers` | Production should have one authoritative lifecycle execution path; wire handlers in or delete the unused abstraction. |
| `esper-lite-1d0c51a2ff` | bug | `confirmed` | SIMPLIFIED reward lacks structural rent but can be read as economy evidence | `simic/rewards/contribution.py` | Add structural cost to proof-grade reward modes or label `SIMPLIFIED` as diagnostic-only for economy claims. |

### P2 Acceptance Shape

- Morphology events can be traced across proposal, verdict, mutation, reward, and rollback/fossilization.
- Tolaria has an explicit safety-authority surface instead of only post-failure loss checks.
- Rollback penalties and proof telemetry are specific enough to explain failure causes.
- Simic has one production lifecycle executor.

---

## P3: Proof Controls

| ID | Type | Status | Title | Component | Evidence / Expected Outcome |
| --- | --- | --- | --- | --- | --- |
| `esper-lite-8066f63b04` | task | `open` | Proof baselines are insufficient for final blueprint-health claims | `simic/training/dual_ab.py` | Add off-switch, static initial/final, fixed-schedule, and lockstep reward A/B controls before final blueprint-health claims. |

### P3 Acceptance Shape

- Proof packets are used as health evidence only when the confounder ledger is clean.
- Final blueprint-health claims compare against meaningful control cohorts, not only sequential A/B runs.

---

## Simic/Tamiyo Training Loop Health Review

| Field | Value |
| --- | --- |
| Source report | `docs/arch-analysis-2026-06-13-0836/02-simic-tamiyo-training-loop-health.md` |
| Filigree label | `from-architecture-review` |
| Report label | `report:2026-06-13` |
| Area label | `simic-tamiyo-training-loop` |
| Tracker status | Five new tickets created; four existing tickets reused/annotated to avoid duplicates |

### New Tickets

| ID | Priority | Type | Status | Title | Parent / Link | Evidence / Expected Outcome |
| --- | --- | --- | --- | --- | --- | --- |
| `esper-lite-2163eafb91` | P1 | task | `open` | Add rollout decision entropy for all factored action heads | Child of `esper-lite-7431b05440` | `GetActionResult` should carry real per-head rollout entropy from the same masked/floored logits used for sampling; decision telemetry should stop recording zero placeholders. |
| `esper-lite-51ac22cbd5` | P1 | task | `open` | Promote PPO finiteness failures to a run-level governor signal | Standalone architecture-review ticket | Repeated `ppo_update_performed=False` or finiteness-failure streaks should halt/mark the run invalid rather than remaining local PPO degradation. |
| `esper-lite-8931d07670` | P2 | task | `open` | Expose value-head gradient telemetry in PPO update events | Standalone architecture-review ticket | PPO update telemetry should expose value-head gradient norm/state, or explicitly replace it with a documented equivalent signal. |
| `esper-lite-8ecf7d6023` | P2 | task | `open` | Replace generic PPO metric aggregation with typed reducers | Standalone architecture-review ticket | Proof-critical PPO aggregation should use explicit reducers per metric family, not generic dict/list/value merging. |
| `esper-lite-00cd1b3f66` | P3 | task | `open` | Add Obs V3 shape doc-test guard against dimension drift | Standalone architecture-review ticket | Public Obs V3 shape claims should be guarded against Leyline/runtime constants so README/spec/ROADMAP drift is caught. |

### Reused / Annotated Existing Tickets

| Existing ID | Status | Existing Register Location | Reused For |
| --- | --- | --- | --- |
| `esper-lite-a31b7d3b3e` | `confirmed` | P1 immediate faults | Active-slot missing telemetry remains the canonical ticket for missing sensor data becoming healthy policy state. |
| `esper-lite-c170aa1198` | `open` | P2 foundation and telemetry faults | Lifecycle handler production wiring remains the canonical ticket for duplicate lifecycle execution semantics. |
| `esper-lite-1d0c51a2ff` | `confirmed` | P2 foundation and telemetry faults | `SIMPLIFIED` reward structural rent remains the canonical ticket for proof/economy reward semantics. |
| `esper-lite-7431b05440` | `confirmed` | P2 foundation and telemetry faults | Broad reward/decision telemetry remains the parent ticket; rollout decision entropy is tracked as child `esper-lite-2163eafb91`. |

### Training Loop Acceptance Shape

- Rollout decisions expose real per-head entropy at decision time.
- PPO finiteness failures can invalidate or halt proof runs at the run level.
- Value-head health is visible in update telemetry, not hidden behind action-head-only metrics.
- PPO metric aggregation has explicit reducer semantics for proof-critical fields.
- Obs V3 public dimensions stay aligned with runtime constants.

---

## Operating Notes

- The P1 bugs are confirmed and actionable.
- The Simic/Tamiyo training-loop review tickets are labeled `from-architecture-review` in Filigree.
- The feature tickets are proposed and should be reviewed/promoted before implementation.
- The parent work package remains `defined`; start the package only when an agent is ready to own the whole slice.
- This register should be refreshed whenever any listed Filigree ticket changes status, priority, parentage, or scope.
