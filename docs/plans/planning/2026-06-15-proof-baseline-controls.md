# Proof Baseline Control Cohorts Plan

> **For Claude:** Do not mark blueprint-health or reward-efficiency evidence proof-grade unless control cohorts are real. Unsupported controls must block; they must never degrade into WAIT-only impostors.

**Goal:** Implement and enforce the full blueprint-health baseline set: off-switch, static-initial, static-final, fixed-schedule, and lockstep reward A/B controls.

**Architecture:** Keep control identity in Leyline, runtime enforcement in Simic/Tolaria, evidence in Nissa/Karn, and final acceptance in `scripts/proof_packet.py`. Controls must constrain the actual runner before action sampling or action execution; packet-only labels are not enough.

**Tech Stack:** Python 3.11, PyTorch, pytest, Karn views, Nissa telemetry, `scripts/proof_packet.py`.

```yaml
# Plan Metadata
id: proof-baseline-controls
title: Proof Baseline Control Cohorts
type: planning
created: 2026-06-15
updated: 2026-06-15
owner: Codex

urgency: medium
value: Prevents Esper from comparing morphogenetic runs against fake or incomplete controls, so failures can be attributed to math, mechanics, algorithm, or theory.

complexity: M
risk: medium
risk_notes: The risk is silent control collapse: static-final or fixed-schedule could look present in metadata while running as an initial-topology WAIT-only cohort.

depends_on:
  - correctness-proof-strategy
soft_depends:
  - morphogenesis-governor-integrity
blocks:
  - ppo-stability-oracle-sandbox
  - reward-efficiency

status_notes: Current code declares all required modes, emits mode/pair/lifecycle policy/seed/schedule provenance, exposes it through Karn `runs`, and makes reward-efficiency packets block incomplete baseline provenance, missing outcome-bearing baseline evidence, missing/mismatched fixed-schedule provenance, missing/mismatched fixed-schedule realized morphology traces, missing/malformed/mismatched static-final source/replay topology manifests, static-final post-replay lifecycle mutations, and malformed lockstep reward A/B pairs. Runtime support executes declared fixed-schedule and static-final source schedules by forcing typed actions through masks before policy sampling; all proof-controlled forced actions are excluded from actor loss. Static-final topology replay has typed manifest capture/materialization helpers, trainer-side source and replay evidence emission, governor snapshot refresh, action-control freezing after validation, and `train_blueprint_health_proof_baselines()` source-preflight orchestration. The live full-baseline rehearsal on 2026-06-15 produced isolated run directories and a reward-efficiency-default packet with complete precision, baseline, fixed-schedule, and static-final replay evidence; the packet correctly fails closed as `BLOCKED_MECHANICS` on tiny-run value-collapse/numerical-instability confounders. Remaining proof gap: multi-seed statistical discipline before final reward-efficiency claims.
percent_complete: 94

reviewed_by:
  - reviewer: python-engineering
    date: 2026-06-15
    verdict: needs-revision
    notes: Requires formal review before promotion to ready; current artifact documents existing code and missing runner support.
  - reviewer: quality-engineering
    date: 2026-06-15
    verdict: needs-revision
    notes: Requires acceptance tests that prove each control affects behavior, not only metadata.
```

## Current Implemented Surface

- `src/esper/leyline/proof_baselines.py` defines the required baseline modes.
- `src/esper/simic/training/proof_baselines.py` builds a complete control plan and refuses unsupported cohorts.
- `apply_proof_baseline_action_controls()` supports:
  - `force_wait_only`
  - `freeze_initial_topology`
  - `paired_lockstep_reward_comparison`
  - `apply_declared_lifecycle_schedule`
  - `evolve_source_final_topology`
- `apply_proof_baseline_action_controls()` freezes `freeze_replayed_final_topology` only after trainer-side static-final replay validation; without that evidence it refuses loudly instead of creating an initial-topology WAIT-only impostor.
- The declared fixed schedule `fixed-schedule-germinate-r0c0-v1` forces epoch 1 to `GERMINATE` the first canonical slot with the NORM blueprint, then forces declared `WAIT` steps. Invalid scheduled actions fail closed instead of degrading to WAIT.
- Scheduled proof-control steps, including the static-final source schedule, are marked as forced for PPO actor loss, so scripted control decisions do not train Tamiyo as if they were policy agency.
- `FIXED_SCHEDULE_GERMINATE_R0C0_HASH` is pinned to a literal expected hash and fails loudly if action enum/order drift changes the declared schedule without a version bump.
- `TRAINING_STARTED` carries `seed`, `proof_baseline_lifecycle_policy`, and fixed-schedule id/hash/version/action-count; Karn `runs` and W&B expose the proof baseline schedule provenance.
- `scripts/proof_packet.py` now defaults programmatic and CLI callers to `--proof-profile reward-efficiency`, rejects explicit precision/baseline opt-outs under that profile, and requires the full baseline set, precision provenance, non-empty baseline pair IDs, non-empty lifecycle policies, valid outcome evidence for every baseline cohort, fixed-schedule provenance for `fixed_schedule`, committed fixed-schedule morphology traces matching the declared schedule, no fixed-schedule provenance on other baselines, and a lockstep reward A/B pair with two outcome-bearing runs under the same pair/seed and distinct reward modes.
- `TOPOLOGY_MANIFEST_RECORDED` is a proof-critical event family with source-final and static-final-replay roles; Karn exposes it through `topology_manifests`, and the lossy analytics store marks it non-proof-grade when observed-but-dropped.
- `scripts/proof_packet.py --proof-profile reward-efficiency` blocks any labeled `static_final` cohort until a joined source-final/replay manifest pair proves matching topology hash, positive topology delta, positive fossilized seed count, acceptable replay weight policy, manifest match, and no lifecycle commit/fossilization rows in the static-final run, so a valid outcome under `freeze_replayed_final_topology` cannot be accepted as proof-grade by label alone. Ungrouped source manifests are accepted when the source event identity joins cleanly.
- `src/esper/simic/training/static_final_replay.py` captures source-final topology manifests, materializes a fossilized topology into a fresh model, preserves alpha algorithm semantics, and captures static-final replay manifests with source linkage and manifest-match evidence.
- `VectorizedPPOTrainer` can materialize validated static-final topology before action sampling, refresh the governor snapshot, emit `TOPOLOGY_MANIFEST_RECORDED`, and then force lifecycle actions to WAIT for the replayed topology.
- `VectorizedPPOTrainer` now emits source-final `TOPOLOGY_MANIFEST_RECORDED` evidence for the internal `static_final_source` preflight and returns the source event/run/group/episode identity through training history for replay handoff.
- `train_blueprint_health_proof_baselines()` now executes an internal one-env/one-episode dynamic source preflight before `static_final`, requires exactly one returned source-final manifest, and passes the exact `TopologyManifestPayload` plus source identity into the static-final replay cohort.

## Missing Executable Machinery

1. **Multi-seed statistical proof:** the packet validates pair shape and cohort evidence, but the final reward-efficiency verdict still needs the planned multi-seed evaluation discipline before becoming architecture evidence.

## Implementation Tasks

1. ✅ Emit or persist source-final `TOPOLOGY_MANIFEST_RECORDED` evidence from the dynamic source cohort at the final evolved topology.
2. ✅ Thread source-final manifest identity through `train_blueprint_health_proof_baselines()` into the static-final cohort call.
3. ✅ Add an end-to-end static-final baseline rehearsal that proves replayed topology differs from static-initial and remains frozen during the run.
4. ✅ Add tests:
   - static-final replay differs from static-initial and is frozen,
   - unsupported controls still fail loudly until implemented,
   - packet blocks incomplete lockstep pairs,
   - reward-efficiency profile blocks when any required control provenance is absent,
   - reward-efficiency profile is the programmatic default and rejects explicit proof-gate opt-outs,
   - ungrouped but event-joined static-final source manifests do not false-close.
5. Add multi-seed aggregation and confidence interval reporting before final reward-efficiency claims.

## Definition of Done

- `train_blueprint_health_proof_baselines()` can execute all required cohorts without raising unsupported-control errors.
- Every control mode has a behavior test proving it constrains the runner, including realized trace validation for fixed-schedule and topology replay validation for static-final.
- The proof packet blocks missing, outcome-empty, under-counted, or unpaired controls.
- The proof packet blocks missing pair ID or lifecycle policy on every labeled baseline run.
- The proof packet blocks missing, mismatched, or misplaced fixed-schedule provenance, plus missing or mismatched fixed-schedule committed morphology traces.
- The proof packet blocks labeled static-final cohorts until joined source-final/replay topology manifests, replay weight policy, positive topology delta, positive fossilized seed count, and freeze-by-no-mutation evidence exist.
- Lockstep reward A/B controls must share a pair ID and seed, carry distinct reward modes, and have valid outcomes on both sides.
- The final reward-efficiency command and programmatic default use `--proof-profile reward-efficiency` and cannot accidentally omit control/precision gates.
