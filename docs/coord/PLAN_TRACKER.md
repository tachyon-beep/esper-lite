# Esper Plan Tracker

**Last Updated:** 2026-06-21 (Post-P0-1 Hardening Sprint closeout: PR #111, "Release 0.2.0: merge 0.1.1 -> main", merged on 2026-06-19 at `d57ecf65`; PR #114, "Land 0.3.0 post-merge closeout line", merged on 2026-06-21 at `b1d558fe` after the `0.3.0` head advanced to `3ad2e897`. Filigree children `esper-lite-26e96f0578`, `esper-lite-a20b180e26`, `esper-lite-d289d208ac`, `esper-lite-569292a32b`, and `esper-lite-224fdba503` are closed. The umbrella plan and defect report moved to `docs/plans/completed/`.)
**Purpose:** Rack-and-stack all plans and concepts for prioritization and dependency tracking.

---

## Executive Summary

### Green-State Recovery (2026-06-12)

The recovery stabilization program has reached steady state. PR #52 (`env-refactor`) was made green and
merged into `main` as the new baseline at merge commit `cdff9c43`; post-merge
main CI passed. The completed recovery plan is
`docs/plans/completed/2026-06-12-green-state-recovery.md`.

Current operating rule: drain high-risk correctness bugs before feature work.
Recovery PR #72 is merged. Follow-up PRs #78 and #79 merged telemetry and
training-control correctness fixes with passing CI. PR #80 merged the first
P2 contract batch and closed three tracker bugs. PR #81 merged the second P2
action/reward contract batch and closed two tracker bugs. PR #82 merged the P2
counterfactual telemetry batch and closed two tracker bugs. PR #83 merged the
P2 GPU-sync batch and closed two tracker bugs. PR #84 merged the P2 Dual-AB
config contract batch and closed two tracker bugs. PR #85 merged the P2
config-contract batch and closed two tracker bugs. PR #86 merged the P2
telemetry-contract batch and closed three tracker bugs. The import-hygiene
batch landed in PR #87 and closed three tracker bugs. The final bug-drain
batch landed in PR #88 and closed the last two recovery bugs. The 2026-06-19
queue reconciliation confirmed the stale recurrent PPO P1 is closed, the
observation queue is empty, and current ready work is explicit Filigree sprint
work rather than stale recovery bugs.

### Post-Hiatus Audit (2026-02-21)

Full codebase audit performed after returning from a month off. Many plans marked "0%" were
actually completed during the Jan 9-17 implementation sprint but the tracker was never updated.

### ✅ Resolved: Op/Value Mismatch Bug

**CRITICAL-op-value-mismatch** (in `docs/bugs/investigations/`): Q(s,op) value head samples
op twice independently — once in `forward()` for value computation, once in `get_action()` for
the stored action. These ops frequently diverge, corrupting advantage estimates. This formerly
blocked Phase 7.

Resolution verified 2026-06-13: `get_action()` now reuses the sampled op in stochastic rollout mode,
recomputes Q(s,argmax op) in deterministic bootstrap mode, and focused regression tests cover both paths.

### Current Focus Areas
1. **Green State Recovery** - ✅ Completed; baseline green and recovery bug drain closed
2. **Dependency/Branch Drain** - ✅ Completed; patch dependency PRs consolidated, stale branches drained, checkout returned to `main`
3. **Karn Telemetry Quality Arc** - Drafted; next upgrade package focused on Sanctum, Overwatch, MCP analytics, and telemetry contracts
4. **Proof Confounder Drain** - ✅ Implemented; ledger, learnability, freshness, reward-accounting closure, proof packet, and blocked rehearsal packet verified
5. **Correctness Proof Strategy** - Drafted 2026-06-15; proof packet now emits typed machine verdicts (`BLOCKED_*`, `CONTINUE`, `REVISE_ALGORITHM`, `STOP_THEORY`) and CLI/API defaults to the reward-efficiency proof profile for control/precision gates, outcome-bearing baseline evidence, fixed-schedule provenance/hash-pin, fixed-schedule realized-trace validation, joined static-final source/replay topology-manifest validation, static-final freeze validation, and lockstep reward A/B pair validation; static-final topology replay now has a runtime primitive, trainer source/replay evidence emission, runner handoff, and live baseline rehearsal; current packet advances past math/control and blocks on mechanics (`BLOCKED_MECHANICS`)
6. **Morphogenesis Governor Integrity** - Drafted from the Kasmina/Tolaria/Blueprint health report; next P1 correctness and evidence-hygiene package
7. **PPO Stability / Oracle Sandbox** - Planning artifact created 2026-06-15; needed after governor-integrity to isolate value-collapse and gradient-anomaly proof blockers
8. **P1 Stability Batch 1** - ✅ Completed and merged; six high-risk PPO/telemetry correctness bugs closed
9. **P0 Filigree Bug Drain** - ✅ Initial six P0s fixed and closed
10. **Op/Value Mismatch** - ✅ Resolved; focused regression tests cover rollout and bootstrap consistency
11. **Reward Efficiency Experiment** - Deferred; reward-efficiency-default proof rehearsal has complete control evidence but is blocked by value-collapse and numerical-instability mechanics confounders
12. **Phase3-TinyStories** - 85% IMPLEMENTED, needs validation runs
13. **Drip Reward Implementation** - ~70% done, needs integration completion
14. **Telemetry Domain Separation** - ~30% done
15. **Blueprint Compiler** - 0% (correctly deferred until entropy confirmed stable)
16. **Correctness Defect Burndown Strategy** - Drafted 2026-06-19; Package A tracker/ready-folder/defect-report reconciliation executed on 2026-06-19 with Filigree IDs recorded for the next sprint work

### P-EV-RECAL Execution Note (2026-06-19)

Filigree task `esper-lite-26e96f0578` removes the EV calibration ambiguity for
`esper-lite-a20b180e26`. The chosen public query path is `ppo_updates`, not
`raw_events`; `ppo_updates` must expose `return_std`, `value_loss`,
`bellman_error`, and `v_return_correlation` for the Step 0 calibration
preflight. The exact fail-loud SQL is recorded in
`docs/plans/completed/2026-06-18-ev-telemetry-robustness-plan.md` under
`Step 0 executable preflight (P-EV-RECAL, 2026-06-19)`. Focused evidence command:
`uv run pytest tests/karn/mcp/test_views.py::test_ppo_updates_exposes_ev_robustness_columns tests/karn/mcp/test_views.py::test_ppo_updates_exposes_ev_calibration_preflight_fields tests/karn/mcp/test_views.py::test_ev_calibration_preflight_raises_when_required_evidence_missing tests/karn/mcp/test_views.py::test_run_confounders_view_empty_on_clean_run -q`
-> 4 passed. Live preflight evidence on `telemetry_2026-06-16_160350`:
`preflight_status=ok`, `updates=10`, `missing_required_rows=0`, value-loss
range `0.10646478831768036..0.8902218341827393`, bellman-error range
`0.25563251972198486..0.45969972014427185`.

### EV Telemetry Robustness Closeout (2026-06-20)

Filigree task `esper-lite-a20b180e26` is implemented and ready to close after
the final closeout commit. The completed plan moved to
`docs/plans/completed/2026-06-18-ev-telemetry-robustness-plan.md`.

Semantics now recorded in source, tests, and plan:

- `explained_variance` is honest diagnostic telemetry under the op-marginal
  `V(s)` critic; it is not a proof-blocking trigger.
- `VALUE_COLLAPSE_DETECTED` fires from robust signals only:
  `value_loss > 5.0 OR bellman_error > 5.0`; equality at `5.0` does not fire.
- Emitted `VALUE_COLLAPSE_DETECTED` rows stay proof-blocking through Karn
  `run_confounders` and `scripts/proof_packet.py`, even with
  `ev_low_return_variance=True`.
- Artifact suppression belongs upstream in detector/gate emission, not in
  `run_confounders`.

Closeout verification:

- `uv run pytest tests/simic/test_ppo_value_metrics.py tests/simic/test_telemetry_fields.py tests/simic/test_vectorized.py -q` -> 102 passed.
- `uv run pytest tests/simic/telemetry tests/simic/training/test_ppo_coordinator.py -q` -> 130 passed.
- `uv run pytest tests/karn/mcp/test_views.py tests/karn/sanctum/test_reward_health.py tests/karn/sanctum/test_aggregator.py tests/nissa/test_wandb_backend.py tests/telemetry/test_batch_stats_ev_field.py tests/scripts/test_proof_packet.py -q` -> 140 passed.
- `uv run pytest tests/simic/test_ppo_update_golden.py tests/simic/test_ppo.py tests/simic/test_ppo_normalization.py -q` -> 27 passed.
- `npm --prefix src/esper/karn/overwatch/web test -- --run -t "HealthGauges|ExperimentVerdictPanel"` -> 29 passed, 271 skipped.
- `uv run ruff check src/ tests/` -> passed.
- `uv run python scripts/lint_leyline_types.py` -> stale whitelist entries 0.
- `uv run python scripts/lint_defensive_patterns.py` -> violations 0.
- `uv run python scripts/lint_gpu_sync.py` -> violations 0.
- `MYPYPATH=src uv run mypy -p esper` -> success, 214 source files.
- `npm --prefix src/esper/karn/overwatch/web run build` -> passed.
- `wardline scan . --fail-on ERROR` -> exit 0, 0 active findings.
- `git diff --check` -> passed.

### Critical Path (Updated)
```
correctness-proof-strategy ──► morphogenesis-governor-integrity ──► ppo-stability-oracle-sandbox ──► reward-efficiency verdict ──► counterfactual-oracle ──► emrakul-phase1
                 │                                     │                         │
                 │                                     │                         └──► blueprint-compiler ──► kasmina2-phase0
                 └──► proof-baseline-controls ◄────────┘
                                                       └──► phase3-tinystories validation
```

### Health Summary
| Status | Count | Notes |
|--------|-------|-------|
| 🔴 Critical | 0 | New governor-integrity issues are high-priority proof blockers, not active Tier 0 mainline breakage |
| Completed | 19 | simic2 (3) + entropy fixes (2) + holding-warning + simic-audit + dual-state lifecycle (2) + drip-reward design + 4 telemetry + op/value mismatch + training-perf-master (2026-06-14) + green-state-recovery + p1-stability-batch-1 + post-p01-hardening-sprint |
| Ready | 10 | Implementation-ready plans after moving completed recovery/stability plans and post-P0-1 closeout plans out of `ready/` |
| In Progress | 2 | phase3-tinystories (85%); weft-phase-a-ci-migration (Phase A shadow-CI) |
| Planning | 12 | Active design workspaces, including correctness defect burndown, correctness proof strategy, governor-integrity, PPO oracle sandbox, and proof baseline controls |
| Concept | 4 | counterfactual-oracle, emrakul-sketch, scaled-counterfactuals, gil-throughput-profiler |
| Abandoned | 3 | shaped-delta-clip, emrakul-submodule-editing, scry-design |
| **Total Active** | **33** |

---

## Priority Matrix

### Tier 0: 🔴 CRITICAL (Fix Immediately)

| ID | Title | Type | Urgency | Complexity | Risk | Status |
|----|-------|------|---------|------------|------|--------|
| green-state-recovery-2026-06-12 | Green State Recovery Program | completed-batch | 🔴 critical | M | high | Completed: PRs #52, #72, #78-#88 merged; recovery bugs closed |
| p1-stability-batch-1 | PPO/Telemetry Stability Batch 1 | completed-batch | 🔴 critical | M | high | Completed and merged; six bugs closed, broad gates passed |
| filigree-p0-drain | Critical Filigree P0 Bug Drain | completed-batch | 🔴 critical | L | high | Initial six P0s fixed, verified, and closed |
| op-value-mismatch | Q(s,op) Double-Sampling Bug | investigation | 🔴 critical | M | high | Completed: stochastic rollout uses one sampled op; deterministic bootstrap recomputes Q(s,argmax op); regression tests pass |

### Tier 1: High Priority (This Week)

| ID | Title | Type | Urgency | Complexity | Risk | Status |
|----|-------|------|---------|------------|------|--------|
| correctness-defect-burndown | Correctness Defect Burndown Strategy | planning | high | XL | high | Drafted 2026-06-19; Package A executed 2026-06-19: stale P1 closed, observation queue empty, June 18 defect report reconciled, and sprint work bound to Filigree IDs |
| morphogenesis-governor-integrity | Morphogenesis Governor Integrity | planning | high | L | high | Drafted 2026-06-13 from the Kasmina/Tolaria/Blueprint health report; owns rollback ordering, observation truthfulness, blueprint contracts, minimal Tolaria pre-flight, and causal morphology event identity |
| correctness-proof-strategy | Correctness Proof Strategy | planning | high | L | high | Drafted 2026-06-15; owns the evidence ladder and typed proof-packet verdict taxonomy for instrumentation, precision, mechanics, math, algorithm revision, and theory stop decisions; packet now blocks outcome-empty baseline controls, missing/mismatched fixed-schedule provenance, misplaced schedule metadata, missing/mismatched fixed-schedule realized traces, missing/malformed/mismatched static-final source/replay manifests, static-final lifecycle mutations, and malformed lockstep pairs; runner-side static-final source handoff and live full-baseline rehearsal implemented; current blocker is mechanics, not proof math |
| ppo-stability-oracle-sandbox | PPO Stability / Oracle Sandbox | planning | high | M | high | Artifact created 2026-06-15; current smoke test proves scripted lifecycle mechanics only, missing proof-grade oracle telemetry and packet profile |
| reward-efficiency | Phase 1 Final Exam (A/B Testing) | ready | high | S | low | ⚠️ Infra 100% done, experiment deferred until governor-integrity, PPO oracle sandbox, and a mechanics-clean proof rehearsal packet |
| karn-telemetry-quality-arc | Karn Telemetry Quality Strategic Arc | planning | high | L | medium | Drafted 2026-06-13; establishes Karn as the next quality-upgrade package |
| karn-telemetry-sprint-1 | Karn Telemetry Quality Sprint 1 | planning | high | M | medium | Drafted 2026-06-13; dependency drain, Sanctum CI determinism, branch hygiene, Overwatch contract inventory |
| proof-confounder-drain | Proof Confounder Drain | completed | high | L | high | Implemented on `confounder-drain`; proof packet correctly blocks the rehearsal on value-collapse and gradient-anomaly confounders |
| drip-reward-impl | Post-Fossilization Drip Reward (impl) | ready | high | M | medium | ~70% done - dataclass + config complete, integration pending |
| telemetry-domain-sep | Telemetry Domain Separation | ready | high | L | medium | ~30% done (3/9 DRL fields), no event renaming |
| counterfactual-aux | Counterfactual Auxiliary Supervision | ready | high | M | medium | 0% - None of 4 phases started |
| blueprint-compiler | Blueprint Compiler (Phase 3 only) | ready | high | XL | medium | 0% - Correctly deferred until entropy stable |
| training-perf-master | Training Pipeline Performance (Simic+Tolaria) | completed | high | L | medium | EXECUTED 2026-06-14 (→ completed/). Phase 0 (allocator/TF32/fragprobe), Phase 1 all 6 incl. CRITICAL-1 BLOCKER (FP32 masked-logit seam + BF16 symmetry, V0 joint_ratio<1e-3 GPU-validated) + sync folds, Phase 2 (FRAGMETRIC telemetry + stream pool + fenced del; CUDA_LAUNCH_BLOCKING clean, bit-identical val_acc), Phase 3 (DYN + pinned SNAP; GATE compile-works-without-sanctum validated). Deliberate calls: P2-RESET NO-GO (retries=0/ooms=0, frag cured), P3-HOST off (gated on RESET), P3-CLONE deferred (esper-lite-472b6477d2). Also deferred: op-sampler (esper-lite-05b4113bc1), carry-clamp (esper-lite-9827eb6bfe). Pending: real-run A/B wall-clock + TUI compile narrowing |
| post-p01-hardening-sprint | Post-P0-1 Hardening & Integration Sprint | completed | high | M | medium | Filigree epic `esper-lite-5e6ff9f907` closed after all children closed: P-EV-RECAL `esper-lite-26e96f0578`, EV robustness `esper-lite-a20b180e26`, dependency triage `esper-lite-d289d208ac`, original 0.1.1 main merge reconciliation `esper-lite-569292a32b`, and 0.3.0 closeout landing `esper-lite-224fdba503`. PR #111 landed 0.1.1 to main at `d57ecf65`; PR #114 landed 0.3.0 to main at `b1d558fe`. Plan moved to `docs/plans/completed/2026-06-18-post-p01-hardening-sprint.md` |
| ev-telemetry-robustness | EV-Telemetry Robustness (low-return-variance artifact) | completed | high | M | medium | Filigree task `esper-lite-a20b180e26`; implemented after P-EV-RECAL `esper-lite-26e96f0578` exposed executable preflight fields. Robust-only gate: EV diagnostic-only; value collapse fires from `value_loss > 5.0 OR bellman_error > 5.0`; plan moved to `completed/` |
| main-merge-integration | 0.1.1 → main Merge & Integration | completed | high | L | high | Live reconciliation found this already completed by GitHub PR #111 (`d57ecf65`, merged 2026-06-19) and present in `origin/main` at `f8089677`; historical plan moved to `docs/plans/completed/2026-06-18-main-merge-integration-plan.md`; post-merge `0.3.0` branch landing is separate task `esper-lite-224fdba503` |
| weft-phase-a-ci-migration | Weft Phase A CI Migration | in-progress | high | M | medium | Phase A merged to `main` via PR #110 (f8089677): non-blocking `weft-shadow` CI job + `weft_parity.py`/`ci_weft_parity.py` parity report. Post-merge hardening landed: readiness now gates on homegrown linter exit codes + Loomweave index freshness (`runs.analyzed_at_commit` vs HEAD); defensive/leyline checks carry `comparison: "deferred"` (no Wardline/Loomweave equivalent yet). Homegrown gates stay blocking; no gate retires until a real comparison shows zero homegrown-only burn-in evidence |

### Tier 2: Medium Priority (Next 2 Weeks)

| ID | Title | Type | Urgency | Complexity | Risk | Status |
|----|-------|------|---------|------------|------|--------|
| phase3-tinystories | Transformer Domain Pivot | in-progress | medium | L | medium | ✅ 85% complete, needs validation runs |
| proof-baseline-controls | Proof Baseline Control Cohorts | planning | medium | M | medium | Artifact created 2026-06-15; mode/pair/lifecycle/seed/schedule provenance is emitted and packet-gated with valid outcome evidence, fixed-schedule provenance/hash-pin, fixed-schedule realized-trace validation, joined static-final source/replay topology-manifest validation, static-final freeze validation, and lockstep reward A/B pair validation; fixed-schedule and static-final source schedules execute through action-mask forcing with actor-loss suppression, static-final replay has source capture/materialization/trainer evidence/runner orchestration, and live rehearsal plus isolated run directories are verified; remaining gap is multi-seed statistical discipline |
| kasmina2-phase0 | Submodule Intervention Foundation | planning | high | L | medium | Design complete, simic2 blocker removed |
| defensive-patterns | Defensive Pattern Fixes | ready | medium | M | low | Removes 23 inappropriate defensive patterns |
| sanctum-help | Sanctum Help System | ready | medium | L | low | Contextual help modals for TUI |
| heuristic-tamiyo | Heuristic Tamiyo Tempo Parity | ready | medium | S | low | 5-head support for fair A/B comparison |

### Tier 3: Strategic (Plan Ahead)

| ID | Title | Type | Urgency | Complexity | Risk | Status |
|----|-------|------|---------|------------|------|--------|
| counterfactual-oracle | Learned Contribution Probe | concept | medium | XL | high | Blocked on reward-efficiency |
| emrakul-immune | Emrakul Immune System Architecture | planning | critical | XL | high | Master architecture doc, Phase 1 infra active |
| kasmina-multichannel | Multichannel Slot Grid (2×N) | planning | medium | M | low | Expand injection surfaces |
| esika-superstructure | Esika Host Superstructure | planning | medium | L | medium | Multi-cell coordination (future scaling) |

### Tier 4: Backlog (Someday)

| ID | Title | Type | Urgency | Complexity | Risk | Status |
|----|-------|------|---------|------------|------|--------|
| blueprint-antipatterns | Blueprint Anti-Patterns Appendix | ready | low | L | medium | 10 bad blueprints for curriculum (Phase 4+) |
| blueprint-future | Blueprint Future Appendix | ready | low | L | medium | 7 advanced CNN blueprints (Phase 3) |
| narset1 | Meta-Coordination Layer | planning | low | L | medium | Speculative, part of Emrakul design |
| karn2 | Karn Sanctum v2 | planning | low | M | low | Nice-to-have TUI improvements |
| tamiyo4 | Slot Transformer Architecture | ready | low | L | medium | Updated 2026-01: Q(s,op) value, contrib predictor, ResidualLSTM |
| emrakul-sketch | Immune System Sketch | concept | medium | XL | high | Concept version (see emrakul-immune for planning) |
| scaled-counterfactuals | Shapley Validation | concept | low | S | low | Diagnostic approach |
| gil-throughput-profiler | Tiered GIL/Throughput Profiler | concept | high | L | medium | Drafted 2026-06-16. Decision tool between free-threading (3.13t/3.14t), 3.x upgrade, and Rust offload. **Phase A1 (Tier 0) IMPLEMENTED & MERGED** (0ee51801 core + 1af4e22b wiring): leyline types + profiler + run-loop wiring + nissa PHASE_PROFILE_COMPLETED + Karn phase_occupancy view; V1/V3/V5 green, gpu_sync 118/0, mypy clean. reviewed_by: pytorch-expert + determinism-reviewer + drl-expert all signed off (A1 wiring). Remaining: A2 Tier-1 CUDA events, B Tier-2 GIL attribution, A3 characterization run, C/D migration decision. `docs/plans/concepts/2026-06-16-gil-throughput-profiler.md` |

### Completed (in docs/plans/completed/)

| ID | Title | Type | Status | Location |
|----|-------|------|--------|----------|
| green-state-recovery-2026-06-12 | Green State Recovery Program | ✅ completed | PRs #52, #72, #78-#88 merged; recovery bugs closed | `docs/plans/completed/2026-06-12-green-state-recovery.md` |
| p1-stability-batch-1 | PPO/Telemetry Stability Batch 1 | ✅ completed | Six high-risk PPO/telemetry correctness bugs closed | `docs/plans/completed/2026-06-12-p1-stability-batch-1.md` |
| post-p01-hardening-sprint | Post-P0-1 Hardening & Integration Sprint | ✅ completed | P-EV-RECAL, EV robustness, dependency triage, 0.1.1 main integration, and 0.3.0 closeout landing completed through PR #111 and PR #114 | `docs/plans/completed/2026-06-18-post-p01-hardening-sprint.md` |
| op-entropy-collapse | Op Head Entropy Collapse Fix | ✅ completed | Two-pronged fix: probability floors + entropy floors. Jan 9-11 sprint. | `docs/plans/completed/` |
| entropy-collapse | Per-Head Entropy Collapse Fix | ✅ completed | All 7 tasks, tests passing | `docs/plans/completed/` |
| holding-warning | SET_ALPHA_TARGET Turntabling Fix | ✅ completed | Committed 2026-01-08, DRL signed | `docs/plans/completed/` |
| simic-audit | Simic Audit Remediation | ✅ completed | Governor tests, action handlers, PPOCoordinator extraction. Jan 10-11. | `docs/plans/completed/` |
| dual-state-lifecycle | Historical Dual-State Lifecycle | ✅ completed | SeedLifecycleEvent, LifecyclePanel, Peak/End toggle. Jan 10. | `docs/plans/completed/` |
| drip-reward-design | Post-Fossilization Drip Reward (design) | ✅ completed | DRL expert reviewed 2026-01-12, approved with modifications | `docs/plans/completed/` |
| simic2-phase1 | Vectorized Module Split | ✅ completed | — | `docs/plans/completed/simic2/` |
| simic2-phase2 | Typed Contracts & API | ✅ completed | — | `docs/plans/completed/simic2/` |
| simic2-phase3 | Simic Module Split | ✅ completed | — | `docs/plans/completed/simic2/` |
| diagnostic-panel-metrics | Diagnostic Panel Metrics Wiring | ✅ completed | 92% (11/12 tasks) | `docs/plans/completed/` |
| tele-340-lstm-health | TELE-340 LSTM Health Wiring | ✅ completed | 100% (27 tests passing) | `docs/plans/completed/` |
| tele-610-episode-stats | TELE-610 Episode Stats Wiring | ✅ completed | 95% (19/20 tasks) | `docs/plans/completed/` |
| value-function-metrics | Value Function Metrics Wiring | ✅ completed | 100% (97 tests passing) | `docs/plans/completed/` |
| ev-telemetry-robustness | EV-Telemetry Robustness | ✅ completed | Robust-only value-collapse gate, proof-blocking regressions, Karn/Sanctum/Overwatch/W&B consumers verified | `docs/plans/completed/2026-06-18-ev-telemetry-robustness-plan.md` |

### Abandoned

| ID | Title | Reason |
|----|-------|--------|
| emrakul-submodule-editing | BLENDING/HOLDING Mutations | Superseded by Track A+C Microstructured Ladders |
| shaped-delta-clip | SHAPED Mode Delta Clipping | Superseded by op-entropy-collapse (root cause is entropy, not reward inflation) |
| scry-design | Scry Interface | Superseded; no scry directory exists |

---

## Detailed Plan Cards

### simic2-phase1: Vectorized Module Split

```yaml
id: simic2-phase1
title: Vectorized Module Split
type: completed
created: 2025-12-20
updated: 2026-01-10

urgency: N/A (done)
value: Unblocked ALL Simic modifications. 4.4k LOC → 1.2k LOC + 4 extracted modules.

complexity: L
risk: N/A (completed successfully)

depends_on: []
blocks: []  # All unblocked

status_notes: |
  SPOT CHECK 2026-01-10: 100% COMPLETE

  DELIVERED:
  ✅ VectorizedPPOTrainer class (vectorized_trainer.py, 1,856 LOC)
  ✅ vectorized.py reduced to 1,192 LOC (from ~4.4k)
  ✅ All nested functions converted to module-level
  ✅ Four extracted modules:
     - env_factory.py (env creation, slot wiring)
     - batch_ops.py (train/val batch processing)
     - counterfactual_eval.py (fused validation)
     - action_execution.py (decode/validate/execute)

  READY TO MOVE TO: docs/plans/completed/
percent_complete: 100
```

**Commentary:**
> ✅ **COMPLETE.** The refactor succeeded. vectorized.py went from a 4.4k LOC monolith
> with nested closures to a clean 1.2k LOC orchestrator with 4 focused modules.
> All extraction targets achieved. No behavioral regressions detected.

---

### op-entropy-collapse: Op Head Entropy Collapse Fix

```yaml
id: op-entropy-collapse
title: Op Head Entropy Collapse Fix
type: completed
created: 2026-01-09
updated: 2026-02-21
location: docs/plans/completed/2026-01-11-op-head-entropy-collapse-fix.md

urgency: N/A (done)
value: |
  Fixed multi-stage entropy collapse that froze the policy.

complexity: M
risk: N/A (completed successfully)

depends_on: []
blocks: []

status_notes: |
  ✅ COMPLETED (Jan 9-11 implementation sprint)

  TWO-PRONGED FIX:
  1. HARD FLOOR: PROBABILITY_FLOOR_PER_HEAD in MaskedCategorical (op=0.05)
     - Guarantees minimum probability mass on all valid actions
     - 741 lines across 13 files (commit 288643a7)
  2. SOFT FLOOR: Updated ENTROPY_FLOOR_PER_HEAD (op: 0.15→0.25)
     - Quadratic penalty, per-head coefficients, late-training decay
     - Blueprint/tempo penalty coef: 0.1→0.3

  ALSO DELIVERED:
  - Availability masks (compute_availability_masks in leyline/causal_masks.py)
  - Per-head entropy collapse detection with hysteresis in anomaly_detector.py
  - Governor rollback telemetry
  - 379-line probability floor test suite

  Plan moved to completed/ on 2026-01-11.
percent_complete: 100
```

**Commentary:**
> ✅ **COMPLETE.** Implemented during the Jan 9-11 sprint. The two-pronged fix
> (probability floors + entropy floors) addresses both the hard guarantee
> (gradients always flow) and the soft pressure (penalty pushes entropy up).
> The PLAN_TRACKER was stale — this was marked "0%" but was fully committed.

---

### reward-efficiency: Phase 1 Final Exam

```yaml
id: reward-efficiency
title: Phase 1 Final Exam - Reward A/B Testing
type: ready
created: 2025-12-19
updated: 2026-01-10

urgency: high
value: |
  Determine the optimal reward signal for Phase 3 (Transformers).
  Currently 7-component SHAPED reward may be "unlearnable landscape".

complexity: S  # REVISED: Infrastructure is 100% complete
risk: low
risk_notes: |
  - All reward modes implemented (SHAPED, SIMPLIFIED, SPARSE, ESCROW)
  - dual_ab.py training infrastructure complete
  - --dual-ab CLI flag wired
  - Test configs exist in configs/ablations/
  - Risk is only wasted compute if wrong hypothesis

depends_on:
  - morphogenesis-governor-integrity
  - ppo-stability-oracle-sandbox
  - proof-baseline-controls  # Required before using results as final blueprint-health evidence
blocks:
  - counterfactual-oracle (explicitly gated on this)

status_notes: |
  SPOT CHECK 2026-01-10: Infrastructure is 100% complete!
  - RewardMode.SIMPLIFIED implemented (contribution.py:747-836)
  - RewardMode.SPARSE implemented (contribution.py:670-702)
  - dual_ab.py exists with train_dual_policy_ab()
  - CLI: --dual-ab shaped-vs-simplified ready
  - Configs: configs/ablations/{shaped,simplified,sparse}_baseline.json

  NEVER EXECUTED. Do not run the long exam until morphogenesis-governor-integrity and PPO stability / oracle sandbox clear the rollback/truthfulness, value-collapse, and gradient-anomaly blockers. Add proof-baseline-controls before using results as final blueprint-health evidence.
  Current CLI uses --rounds for PPO update rounds:
  PYTHONPATH=src uv run python -m esper.scripts.train ppo --task cifar_impaired --dual-ab shaped-vs-simplified --rounds 100 --envs 8 --episode-length 150
percent_complete: 100 (infra) / 0 (experiment)
```

**Commentary:**
> **MAJOR FINDING:** All the experiment code exists, but weaker-than-expected
> prior signal means this should now run behind morphogenesis-governor-integrity
> and the PPO stability / oracle-sandbox package rather than as a raw
> "press the button" experiment.
>
> The dual-policy A/B system trains separate PPO agents per reward mode with
> isolated environments, policies, and optimizers. Results would directly
> unblock counterfactual-oracle.
>
> **Action:** Keep the infrastructure idle until the rehearsal packet is valid.
> Complexity remains S once the upstream blockers are cleared.

---

### kasmina2-phase0: Submodule Intervention Foundation

```yaml
id: kasmina2-phase0
title: Kasmina2 Phase 0 - Submodule Intervention Foundation
type: planning
created: 2025-12-26
updated: 2026-01-05

urgency: high
value: |
  Enable finer-grained growth control. Currently Tamiyo must "buy a conv_heavy"
  even when she only needs 2k params of capacity. This is the "lumpiness problem".

complexity: L
risk: medium
risk_notes: |
  - Cross-domain changes (Leyline + Kasmina + Tamiyo + Simic + Tolaria)
  - Track A (surfaces) vs Track C (microstructure) need clear sequencing
  - torch.compile compatibility must be verified

depends_on:
  - simic2-phase1 (for clean Simic integration)
soft_depends:
  - reward-efficiency (clearer signal helps)
blocks:
  - kasmina2-phase1
  - emrakul v1 submodule surgery

status_notes: |
  Design is mature (see phase0-implementation/ tracks).
  Six parallel tracks identified:
  1. Leyline contracts
  2. Kasmina mechanics
  3. Tamiyo policy
  4. Simic training
  5. Telemetry
  6. Testing

  Awaiting simic2-phase1 completion before execution.
percent_complete: 10
```

**Commentary:**
> This is the next big capability unlock after the Simic refactor. The planning is
> thorough (6 tracks, clear ownership). The main risk is cross-domain coordination.
> Track C "microstructured ladders" was chosen over the abandoned submodule-editing
> approach after specialist review - good decision-making discipline shown.

---

### counterfactual-oracle: Learned Contribution Probe

```yaml
id: counterfactual-oracle
title: Counterfactual Oracle - Learned Contribution Inference
type: concept
created: 2026-01-09
updated: 2026-01-09

urgency: medium
value: |
  Enable scaling to 50-100+ seeds without compute explosion.
  "Oracle = expensive truth, Probe = cheap belief"

complexity: XL
risk: high
risk_notes: |
  - Probe could become reward-hack surface (Goodhart risk)
  - Auxiliary loss could destabilize PPO
  - Selection bias from probe-driven audits
  - Requires careful uncertainty calibration

depends_on:
  - reward-efficiency (explicitly stated in doc)
soft_depends:
  - simic2-phase2 (typed contracts help)
  - kasmina2-phase0 (more seeds to probe)
blocks:
  - emrakul-phase1 (needs cheap contribution estimates)
  - phase3-tinystories-scale (50+ seeds)

status_notes: |
  Comprehensive proposal exists. Well-reviewed by specialists.
  Phase-gated: "blocked on Phase 2.5 Reward Efficiency Protocol completion"

  DO NOT START until reward-efficiency is resolved.
percent_complete: 0
```

**Commentary:**
> This is the most sophisticated concept doc in the set. 700+ lines, expert-reviewed,
> clear phasing. The explicit phase gate is good discipline - it prevents premature
> optimization before we know the reward signal works.
>
> The risk analysis is thorough (6 enumerated risks with mitigations). This should
> be treated as a "major research effort" not a "feature."

---

### emrakul-sketch: Immune System Phase 4

```yaml
id: emrakul-sketch
title: Immune System Phase 4 Specification
type: concept
created: 2025-12-31
updated: 2025-12-31

urgency: medium
value: |
  Autonomously remove obsolete host structure after seed takeovers.
  Completes the "growth + decay" ecology.

complexity: XL
risk: high
risk_notes: |
  - Phage gating must be torch.compile friendly
  - Narset coordination layer is speculative
  - Physical lysis requires offline topology rewrite
  - Could destabilize training if decay is too aggressive

depends_on:
  - kasmina2-phase0 (Tamiyo must be stable)
  - simic2-phase2 (typed contracts)
soft_depends:
  - counterfactual-oracle (cheap contribution estimates help Emrakul)
blocks:
  - emrakul v1 implementation

status_notes: |
  "Concept Locked" - design is mature but not implementation-ready.
  Waiting for growth side (Tamiyo/Kasmina) to stabilize.

  Key innovation: Narset as "endocrine allocator" with leases and
  multi-color warnings (cyan/yellow/red).
percent_complete: 0
```

**Commentary:**
> This is the "other half" of Esper's ecology - the decay side. The design is
> comprehensive (600 lines) and addresses torch.compile concerns head-on.
>
> The framing of Narset as an "endocrine" system (not micromanaging, just setting
> hormonal signals) is clever architecture. But this is explicitly Phase 4 -
> we need Phases 1-3 working first.

---

### phase3-tinystories: Transformer Domain Pivot

```yaml
id: phase3-tinystories
title: Phase 3 - Transformer Domain Pivot (TinyStories)
type: in-progress
created: 2025-12-19
updated: 2026-01-10

urgency: medium
value: |
  Prove morphogenetic principles work on Transformers, not just CNNs.
  Critical for credibility - must not overfit to "convolutional dynamics."

complexity: L  # REVISED: Most work is done
risk: medium  # REVISED: Implementation exists, just needs validation
risk_notes: |
  - Remaining risk is validation, not implementation
  - Need to run baseline experiments to measure learning curves
  - May need ResidualSeed (full layer insertion) if current seeds insufficient

depends_on: []  # REVISED: Can run independently
soft_depends:
  - reward-efficiency (cleaner signal helps but not blocking)
blocks:
  - phase4+ (broader model families)

status_notes: |
  SPOT CHECK 2026-01-10: 80-90% COMPLETE! Was incorrectly tracked as "not started".

  IMPLEMENTED:
  ✅ TransformerHost (host.py:451-657) - GPT-2 style, 6 layers, full HostProtocol
  ✅ 6 transformer blueprints (blueprints/transformer.py):
     - norm, lora, lora_large, attention, mlp_small, mlp, flex_attention, noop
  ✅ TinyStoriesDataset (data.py:1009-1118) - HuggingFace integration
  ✅ Task specification (tasks.py:209-252) - "tinystories" task wired
  ✅ Zero-init output projections (gradient shock prevention)
  ✅ torch.compile compatible (flex_attention uses cache)
  ✅ Test coverage exists (tests/tolaria/test_tinystories.py)

  NOT IMPLEMENTED:
  ❌ ResidualSeed (full layer insertion) - may not be needed
  ❌ Baseline experiment runs - need learning curve data
  ❌ SlotTransformer for Tamiyo policy (separate plan: tamiyo4)
percent_complete: 85
```

**Commentary:**
> **MAJOR FINDING:** This was the biggest tracking error. The transformer pivot
> is largely implemented and ready for training experiments.
>
> The TransformerHost exists with full GPT-2 architecture. Six transformer-specific
> blueprints are registered including LoRA, attention heads, MLPs, and FlexAttention.
> The TinyStories dataset loader is complete with HuggingFace integration.
>
> **Action:** Run baseline training on TinyStories to validate the implementation.
> The "blocked on reward-efficiency" dependency was overstated - this can run now.

---

### simic2-phase2: Typed Contracts & API

```yaml
id: simic2-phase2
title: Simic Phase 2 - Typed Contracts & API
type: completed
created: 2025-12-22
updated: 2026-01-10

urgency: N/A (done)
value: Clean interfaces between Simic components, easier testing.

complexity: M
risk: N/A (completed successfully)

depends_on: []
blocks: []

status_notes: |
  SPOT CHECK 2026-01-10: 100% COMPLETE

  DELIVERED:
  ✅ vectorized_types.py (131 LOC) with 6 dataclasses:
     - ActionSpec (14 fields)
     - ActionMaskFlags (boolean flags)
     - ActionOutcome (9 fields)
     - RewardSummaryAccumulator
     - EpisodeRecord
     - BatchSummary (with to_dict() serialization)
  ✅ rewards/types.py with typed containers:
     - ContributionRewardInputs (20+ fields)
     - LossRewardInputs (8 fields)
     - SeedInfo NamedTuple (11 fields)
  ✅ All using @dataclass(slots=True) for memory efficiency

  READY TO MOVE TO: docs/plans/completed/
percent_complete: 100
```

---

### simic2-phase3: Simic Module Split

```yaml
id: simic2-phase3
title: Simic Phase 3 - Module Split
type: completed
created: 2025-12-22
updated: 2026-01-10

urgency: N/A (done)
value: Final structural cleanup of Simic.

complexity: M
risk: N/A (completed successfully)

depends_on: []
blocks: []

status_notes: |
  SPOT CHECK 2026-01-10: 100% COMPLETE

  REWARDS MODULE SPLIT:
  ✅ contribution.py (1,090 LOC) - contribution-primary reward
  ✅ loss_primary.py (73 LOC) - loss-primary reward
  ✅ shaping.py - PBRS utilities
  ✅ types.py (135 LOC) - typed containers
  ✅ reward_telemetry.py (11,820 LOC) - telemetry
  ✅ rewards.py (7,419 LOC) - dispatcher

  AGENT MODULE SPLIT:
  ✅ ppo_agent.py (1,354 LOC) - PPOAgent class
  ✅ ppo_update.py (366 LOC) - update math
  ✅ ppo_metrics.py (211 LOC) - metrics builder
  ✅ types.py (198 LOC) - TypedDicts

  READY TO MOVE TO: docs/plans/completed/
percent_complete: 100
```

---

### scaled-counterfactuals: Shapley Validation

```yaml
id: scaled-counterfactuals
title: Scaled Counterfactual Validation
type: concept
created: 2025-12-15
updated: 2025-12-15

urgency: low
value: |
  Validate that seed contributions show interaction effects (emergence).
  "The definitive proof that your Morphogenetic Engine is working."

complexity: S
risk: low
risk_notes: Diagnostic only, no production impact.

depends_on: []
blocks: []

status_notes: |
  This is a diagnostic/validation approach, not a feature.
  Suggests Monte Carlo Shapley for scaling beyond 5+ seeds.
  Useful reference when debugging contribution measurement.
percent_complete: N/A
```

---

### narset1: Meta-Coordination Layer

```yaml
id: narset1
title: Narset Meta-Coordination
type: concept
created: 2025-12-30
updated: 2025-12-30

urgency: low
value: |
  Slow-timescale coordinator for zone budgets.
  "Does not observe architecture, only telemetry."

complexity: L
risk: medium
risk_notes: |
  - Adds another policy to coordinate
  - Speculative - may not be needed

depends_on:
  - emrakul-sketch (Narset is part of immune system design)
blocks: []

status_notes: |
  Speculative extension. Mentioned in emrakul_outline.md.
  Not needed until Emrakul exists.
percent_complete: 0
```

---

### karn2: Karn Sanctum v2

```yaml
id: karn2
title: Karn Sanctum v2
type: planning
created: 2025-12-25
updated: 2025-12-25

urgency: low
value: Improved TUI for training monitoring.

complexity: M
risk: low
risk_notes: User-facing only, no training impact.

depends_on: []
blocks: []

status_notes: Nice-to-have. Current TUI is functional.
percent_complete: 0
```

---

### tamiyo4: Slot Transformer Architecture

```yaml
id: tamiyo4
title: Slot Transformer Architecture
type: ready
created: 2025-12-26
updated: 2026-01-12

urgency: low
value: |
  Replace flat observation concatenation with Slot Transformer Encoder.
  Enables O(1) parameters per slot, variable slot counts, learned slot-slot interactions.

complexity: L
risk: medium
risk_notes: |
  - Architectural change to policy network
  - Needs careful A/B validation
  - Must preserve Q(s,op) value conditioning

depends_on:
  - simic2-phase2 (cleaner training code) ✅ COMPLETE
blocks: []

status_notes: |
  READY FOR EXECUTION. Plan updated 2026-01-12 with critical features:
  - Task 2.5: Op-conditioned value head Q(s,op) not V(s)
  - Task 2.6: Contribution predictor auxiliary head
  - Task 2.7: ResidualLSTM integration (not vanilla nn.LSTM)
  - Task 2.8: BlueprintEmbedding for Obs V3
  Moved to docs/plans/ready/
percent_complete: 0
```

---

### blueprint-compiler: Blueprint Compiler & Curriculum Seeds

```yaml
id: blueprint-compiler
title: Blueprint Compiler & Curriculum Seeds
type: ready
created: 2026-01-09
updated: 2026-01-09
location: docs/plans/ready/2026-01-09-blueprint-compiler-and-curriculum-seeds.md

urgency: high
value: |
  Compiles BlueprintRegistry into manifests with global indices.
  Adds LayerScale helper & 4 curriculum blueprints.

complexity: XL
risk: medium
risk_notes: |
  - Phase 4 (new blueprints) must wait until entropy >0.10
  - Phased rollout: Phase 3 (LayerScale) NOW, Phase 1-2 any time, Phase 4 DEFER

depends_on: []
blocks:
  - Phase 4 curriculum learning

status_notes: |
  PHASED ROLLOUT:
  - Phase 3 (LayerScale + dead-branch fixes): DO NOW
  - Phase 1-2 (compiler infrastructure): Any time
  - Phase 4 (curriculum blueprints): DEFER until entropy stable >0.10

  Has two appendices:
  - blueprint-antipatterns: 10 bad blueprints for curriculum
  - blueprint-future: 7 advanced CNN blueprints
percent_complete: 0
```

---

### telemetry-domain-sep: Telemetry Domain Separation

```yaml
id: telemetry-domain-sep
title: Telemetry Domain Separation
type: ready
created: 2026-01-02
updated: 2026-01-02
location: docs/plans/ready/2026-01-02-telemetry-domain-separation.md

urgency: high
value: |
  Renames event types with domain prefixes (PPO_UPDATE_COMPLETED→TAMIYO_POLICY_UPDATE).
  Adds DRL specialist fields (approx_kl_max, trust_region_violations, return stats).

complexity: L
risk: medium
risk_notes: |
  - Breaks telemetry schema (migration required)
  - Best done soon before more runs accumulate

depends_on: []
blocks: []

status_notes: |
  5 phases:
  1. Rename events
  2. Rename payloads
  3. Add specialist fields
  4. Update docs
  5. Cleanup
percent_complete: 0
```

---

### holding-warning: Fix SET_ALPHA_TARGET Turntabling

```yaml
id: holding-warning
title: Fix SET_ALPHA_TARGET Turntabling Exploit
type: ready
created: 2026-01-08
updated: 2026-01-08
location: docs/plans/ready/2026-01-08-fix-set-alpha-target-turntabling.md

urgency: high
value: |
  Extends holding_warning penalty to ALL non-terminal actions in HOLDING.
  Closes exploit where Tamiyo spammed SET_ALPHA_TARGET to avoid penalty.

complexity: S
risk: low

depends_on: []
blocks: []

status_notes: |
  Simple fix: Terminal actions (FOSSILIZE, PRUNE) remain exempt.
  All other actions in HOLDING stage get penalty.
percent_complete: 0
```

---

### shaped-delta-clip: SHAPED Mode Delta Clipping

```yaml
id: shaped-delta-clip
title: SHAPED Mode Delta Clipping
type: ready
created: 2026-01-10
updated: 2026-01-10
location: docs/plans/ready/shaped-mode-delta-clipping.md

urgency: high
value: |
  Fixes reward inflation in SHAPED mode where long-lived seeds get
  unbounded rewards due to cumulative seed_contribution.
  Adds shaped_delta_clip parameter (default 2.0) to mirror ESCROW's escrow_delta_clip.

complexity: S
risk: low
risk_notes: |
  - Mirrors proven ESCROW approach
  - Rollback via shaped_delta_clip=0.0

depends_on: []
blocks: []

status_notes: |
  Telemetry analysis (2026-01-10) found episodes with 22-24% accuracy
  getting 700+ episode rewards due to cumulative inflation.

  DRL expert review confirmed delta-clipping is the correct fix.

  6 phases:
  1. Add shaped_delta_clip config param
  2. Add telemetry fields
  3. Implement delta clipping logic
  4. Add function parameter
  5. Wire through vectorized trainer
  6. Add tests
percent_complete: 0
```

---

### drip-reward: Post-Fossilization Drip Reward

```yaml
id: drip-reward
title: Post-Fossilization Drip Reward
type: split (design=completed, impl=in-progress)
created: 2026-01-12
updated: 2026-02-21

design_doc: docs/plans/completed/2026-01-12-post-fossilization-drip-reward.md
impl_plan: docs/plans/ready/2026-01-12-post-fossilization-drip-reward-impl.md

urgency: high
value: |
  Prevents seeds from gaming by fossilizing at peak metrics then degrading.
  Enforces long-term accountability for fossilized seeds via drip reward.

complexity: M
risk: medium

depends_on: []
blocks: []

reviewed_by:
  - reviewer: drl-expert
    date: 2026-01-12
    verdict: approved_with_modifications

status_notes: |
  DESIGN: ✅ COMPLETE (moved to completed/)
  IMPLEMENTATION: ~70% done (Jan 12 sprint)

  IMPLEMENTED:
  ✅ Task 1: BASIC_PLUS reward mode enum added
  ✅ Task 2: FossilizedSeedDripState dataclass (contribution.py)
  ✅ Task 3: Drip split in compute_basic_reward (70/30)
  ✅ Task 4-5: Golden tests and property tests for drip anti-gaming
  ✅ Task 6: BASIC_PLUS wired into reward dispatcher
  ✅ Task 7: Complete drip telemetry fields added

  REMAINING:
  ❌ Task 8: Full integration in VectorizedPPOTrainer (partial)
  ❌ End-to-end validation run

percent_complete: 70
```

**Commentary:**
> DRL Expert reviewed both the core design and the counterfactual vs Shapley question.
> Key insight: per-epoch counterfactual and terminal Shapley answer different causal questions.
> Counterfactual ("is this seed helping right now?") is correct for Markovian dense rewards.
> Shapley ("what was fair attribution of total gain?") is correct for terminal evaluation only.
> Implementation plan has 8 TDD tasks with complete code snippets and test commands.

---

### counterfactual-aux: Counterfactual Auxiliary Supervision

```yaml
id: counterfactual-aux
title: Counterfactual Auxiliary Supervision
type: ready
created: 2026-01-10
updated: 2026-01-10
location: docs/plans/ready/2026-01-10-counterfactual-auxiliary-supervision.md

urgency: high
value: |
  Adds ContributionPredictor head to predict per-slot seed contributions
  from counterfactual ablation. Improves sample efficiency.

complexity: M
risk: medium
risk_notes: |
  - MSE auxiliary loss (coef=0.05, warmup 1000 steps, stop-grad to LSTM)
  - Could destabilize PPO if coefficient too high

depends_on: []
blocks: []

status_notes: |
  4 phases:
  1. Add ContributionPredictor head
  2. Compute targets from counterfactual ablation
  3. Integrate MSE auxiliary loss
  4. Add telemetry
percent_complete: 0
```

---

### emrakul-immune: Emrakul Immune System Architecture

```yaml
id: emrakul-immune
title: Esper Morphogenetic AI - Full System Architecture
type: planning
created: 2025-12-30
updated: 2026-01-10
location: docs/plans/planning/emrakul1/

urgency: critical (design), low (implementation)
value: |
  Complete morphogenetic ecology: Tamiyo (growth) + Emrakul (decay)
  under economic pressure (Simic rent/churn).

complexity: XL
risk: high
risk_notes: |
  - Novel distributed architecture
  - Two-timescale learning complexity
  - Phase 1 uses expensive Shapley audits; Phase 2 deploys trained policy

depends_on:
  - simic2 (complete)
  - kasmina2-phase0 (for submodule work)
soft_depends:
  - counterfactual-oracle (helps with cheap contribution estimates)
blocks:
  - emrakul v1 implementation

status_notes: |
  MASTER ARCHITECTURE DOCUMENT covering 7 domains:
  1. Tolaria (substrate): Training engine, replay, safety
  2. Simic (substrate): Economy, credit attribution
  3. Kasmina (organism): Morphogenetic host
  4. Tamiyo (organism): Growth policy (8 heads)
  5. Emrakul (organism): Decay policy (probe-and-lysis)

  STORYBOARDED MILESTONES (emrakul-and-phage.md):
  - Stage 0: Deterministic replay + telemetry integrity
  - Stage 1: Tamiyo grows modules safely
  - Stage 2: Emrakul prunes with ScarSlot
  - Stage 3: Trauma surgery loop
  - Stage 4: Submodule work
percent_complete: 5
```

---

### kasmina-multichannel: Multichannel Slot Grid

```yaml
id: kasmina-multichannel
title: Multichannel Slot Grid Architecture (2×N)
type: planning
created: 2025-12-20
updated: 2025-12-20
location: docs/plans/planning/kasmina1.5/multichannel_drifting.md

urgency: medium
value: |
  Expand CNN host from single injection boundary to multi-surface topology.
  Enables more slots without changing traversal logic.

complexity: M
risk: low
risk_notes: |
  - Option 1 (2×N): ~evening of work + tests
  - Option 2 (3×5): More complex, ~sprint

depends_on:
  - Stable InjectionSpec interface
blocks: []

status_notes: |
  Two options documented:
  - 2×N grid (recommended): Pre/post-pool surfaces per block
  - 3×5 multi-lane (complex): True multi-lane with merge semantics

  Use boundary timeline abstraction for deterministic routing.
percent_complete: 0
```

---

### esika-superstructure: Esika Host Superstructure

```yaml
id: esika-superstructure
title: Esika Host Superstructure Container
type: planning
created: 2025-12-28
updated: 2025-12-28
location: docs/plans/planning/esika1/concept.md

urgency: medium (future scaling)
value: |
  Coordinates multiple Kasmina "cells", enforces safe boundaries,
  deconfliction rules, and hosts Narset budget allocator at scale.

complexity: L
risk: medium
risk_notes: |
  - Infrastructure, not intelligence
  - Avoids "god object" but introduces new system layer

depends_on:
  - Kasmina single-cell maturity (Stage 2-3)
  - Narset allocator design
blocks: []

status_notes: |
  POST-STAGE-3 work. Esika is infrastructure (not policy):
  - Topology and identity (region graph)
  - Deconfliction rules (physics, not strategy)
  - Safe-boundary scheduling
  - Host Narset (routes budget outputs)

  Does NOT choose slots/blueprints (Tamiyo/Emrakul do that).
percent_complete: 0
```

---

## Dependency Graph

```
                    CRITICAL PATH (implement in order)
                    ═══════════════════════════════════

┌──────────────────┐
│ 🔴 op-value      │    ← FIX FIRST (corrupts advantage estimates)
│    mismatch      │
└────────┬─────────┘
         │
         ├──────────────────────────────────────┐
         │                                      │
         ▼                                      ▼
┌──────────────────┐                   ┌──────────────────┐
│ reward-          │                   │ blueprint-       │
│ efficiency       │                   │ compiler         │
│ (run experiment) │                   │ (Phase 3 only)   │
└────────┬─────────┘                   └────────┬─────────┘
         │                                      │
         │    ┌─────────────────────────────────┘
         │    │
         ▼    ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ kasmina2         │───►│ counterfactual   │───►│ emrakul          │
│ phase0           │    │ oracle           │    │ phase1           │
└──────────────────┘    └──────────────────┘    └──────────────────┘

                    PARALLEL TRACKS (can proceed independently)
                    ═════════════════════════════════════════════

┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ drip-reward-impl │    │ telemetry        │    │ counterfactual   │
│ (70% done)       │    │ domain-sep (30%) │    │ -aux             │
└──────────────────┘    └──────────────────┘    └──────────────────┘

┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ phase3-          │    │ sanctum-help     │    │ defensive-       │
│ tinystories      │    │ (UX)             │    │ patterns         │
│ (85% done)       │    │                  │    │ (code quality)   │
└──────────────────┘    └──────────────────┘    └──────────────────┘

                    COMPLETED (Jan 9-17 sprint)
                    ════════════════════════════

┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ ✅ entropy       │    │ ✅ holding       │    │ ✅ simic-audit   │
│    collapse      │    │    warning       │    │    remediation   │
└──────────────────┘    └──────────────────┘    └──────────────────┘
┌──────────────────┐    ┌──────────────────┐
│ ✅ dual-state    │    │ ✅ drip-reward   │
│    lifecycle     │    │    design        │
└──────────────────┘    └──────────────────┘

                    FUTURE (after Stage 3 stable)
                    ════════════════════════════════

┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ kasmina          │    │ esika            │    │ narset1          │
│ multichannel     │    │ superstructure   │    │                  │
└──────────────────┘    └──────────────────┘    └──────────────────┘
```

---

## Risk Register

| Plan | Risk Level | Primary Risk | Mitigation |
|------|------------|--------------|------------|
| op-value-mismatch | RESOLVED | Formerly corrupted advantage estimates in all training | Fixed in factored_lstm.py; regression tests cover stochastic and deterministic paths |
| counterfactual-oracle | HIGH | Goodhart/reward hacking | Probe as observation only, never as reward |
| emrakul-immune | HIGH | Novel architecture, two-timescale learning | Phased rollout, Shapley labels in Phase 1 only |
| phase3-tinystories | HIGH | NaN spikes on graft | Zero-init projections, LayerNorm pre-injection |
| blueprint-compiler | MEDIUM | New blueprints could destabilize | Phase 4 deferred until entropy >0.10 |
| counterfactual-aux | MEDIUM | Auxiliary loss could destabilize PPO | Low coefficient (0.05), warmup, stop-grad |
| telemetry-domain-sep | MEDIUM | Schema migration | Do early before more runs accumulate |
| kasmina2-phase0 | MEDIUM | Cross-domain coordination | Six parallel tracks with clear ownership |
| esika-superstructure | MEDIUM | New coordination layer | Infrastructure only, no intelligence |
| defensive-patterns | LOW | Code quality only | No behavior change |
| sanctum-help | LOW | User-facing only | N/A |
| karn2 | LOW | User-facing only | N/A |

---

## Recommendations

### Immediate Actions (This Week)

1. **Review and execute Morphogenesis Governor Integrity** - The architecture health report found P1 confounders that must precede proof experiments: snapshot-before-panic ordering, rollback continuing into stale lifecycle mutation, missing telemetry presented as healthy, host-drift fallback in contribution observations, non-tensor blueprint smoke-test holes, and absent pre-flight governor authority. Draft plan: `docs/plans/planning/2026-06-13-morphogenesis-governor-integrity.md`.

2. **Plan PPO stability / oracle-sandbox package after the P1 governor fixes** - The proof-confounder drain now blocks invalid runs honestly. The next package should fix or isolate the value-collapse and gradient-anomaly blockers, prove lifecycle mechanics under an oracle or hardcoded heuristic, and add a cheap mathematical micro-sandbox for Tamiyo/Simic:
   ```bash
   PYTHONPATH=src uv run python -m esper.scripts.train heuristic --task cifar_impaired --episodes 1
   ```

3. **Do not run the long reward-efficiency exam as proof yet** - The short rehearsal is correctly `BLOCKED`; clear the governor/truthfulness, value-collapse, and gradient-anomaly confounders first, then repeat the rehearsal before the expensive pass:
   ```bash
   PYTHONPATH=src uv run python -m esper.scripts.train ppo --task cifar_impaired --dual-ab shaped-vs-simplified --rounds 2 --envs 2 --episode-length 25
   ```

4. **Treat proof baseline controls as rehearsed, not final statistical evidence** - Mode/pair/lifecycle/seed/schedule provenance, outcome-bearing baseline evidence, fixed-schedule provenance/hash-pin, fixed-schedule realized-trace validation, static-final source/replay topology-manifest validation, static-final freeze validation, and lockstep reward A/B pair shape are packet-gated and live-rehearsed with isolated run directories. The remaining work is mechanics cleanup plus multi-seed statistical aggregation before a final reward-efficiency verdict is used as architecture evidence.

5. **Run reward-efficiency experiment only after the rehearsal packet is valid** - Infrastructure is 100% complete and the op/value blocker is resolved, but the proof packet must be able to mark confounded runs invalid:
   ```bash
   PYTHONPATH=src uv run python -m esper.scripts.train ppo --task cifar_impaired --dual-ab shaped-vs-simplified --rounds 100 --envs 8 --episode-length 150
   ```

6. **Run TinyStories baseline after the CIFAR proof verdict** - Implementation is 85% complete, but transformer validation should not outrun the confounder drain.

### Short-Term (Next 2 Weeks)

7. **Complete drip-reward integration** - ~70% done, needs pipeline wiring and telemetry
8. **Implement telemetry-domain-sep** - Currently ~30% done. Break schema now.
9. **Implement counterfactual-aux** - 0% done. Adds ContributionPredictor head.
10. **Triage generated bugs** - 14 Codex analysis files in `docs/bugs/generated/` need review
11. **Analyze reward A/B results** - Declare winner or next confounder (SHAPED vs SIMPLIFIED).

### Medium-Term (Next Month)

12. **Begin kasmina2-phase0 implementation** - Design complete, simic2 blocker removed.
13. **Begin counterfactual-oracle Phase 1** - Unblocked once reward-efficiency has data.
14. **Blueprint compiler Phase 4** - New curriculum blueprints (ONLY if entropy stable >0.10).

### Parking Lot (Not Now)

- **emrakul-immune** - Master architecture doc, but implementation is Stage 4+
- **kasmina-multichannel** - Slot grid expansion (after kasmina2)
- **esika-superstructure** - Multi-cell coordination (post-Stage 3)
- **narset1** - Speculative, part of Emrakul design
- **karn2** - Nice-to-have TUI improvements
- **tamiyo4** - READY: SlotTransformer for policy scaling (updated 2026-01-12)
- **blueprint-antipatterns** - Bad blueprints for curriculum (Phase 4+)
- **blueprint-future** - Advanced CNN blueprints (Phase 3)

---

## Change Log

| Date | Change |
|------|--------|
| 2026-06-16 | **GIL PROFILER TIER-0 (PHASE A1) IMPLEMENTED & MERGED.** Wired the Tier-0 phase profiler into the vectorized PPO runtime (commits 0ee51801 core + 1af4e22b wiring on 0.1.1): per-phase wall + Python-CPU attribution at the 5 transaction-phase seams, drain -> nissa PHASE_PROFILE_COMPLETED -> Karn phase_occupancy view. Built test-first + adversarially reviewed via multi-agent workflow (determinism + pytorch approve; drl + test-coverage flagged 3 blocking items — ppo_update batch-misattribution + final-batch drop, untested exception-safety, narrow V1 digest — all fixed). Gates: 1476 passed/1 skipped, V1/V3/V5 integration green, gpu_sync unchanged 118/0, defensive + mypy clean. drl-expert sign-off (previously pending) now complete. Next: A2 Tier-1 CUDA-event occupancy, B Tier-2 GIL HOLD/WAIT, A3 characterization run. |
| 2026-06-16 | **GIL/THROUGHPUT PROFILER DRAFTED (concept).** Added `docs/plans/concepts/2026-06-16-gil-throughput-profiler.md` to address the user's report that the vectorized PPO runtime hits a single-thread GIL ceiling before saturating GPU/VRAM. Tiered instrument (Tier 0 always-on per-phase wall+Python-CPU; Tier 1 CUDA-event occupancy/NVTX/torch.profiler; Tier 2 GIL HOLD/WAIT). Designed via multi-agent workflow across pytorch/determinism/training-opt/system-architect lenses; load-bearing file:line citations spot-verified. Reframed as the decision instrument between three remediations: (A) free-threaded CPython 3.13t/3.14t (interpreter upgrade authorized 2026-06-15), (B) plain 3.13/3.14 upgrade, (C) Rust/PyO3 offload of scalar reward/governor/telemetry. Corrected a workflow hallucination (runtime is 3.11.11, not 3.12.3 → `sys.monitoring` unavailable until upgrade). reviewed_by: pytorch-expert + determinism-reviewer (design-input-folded), drl-expert pending. |
| 2026-06-13 | **ARCHITECTURE HEALTH REPORT TRIAGED.** Reviewed `docs/arch-analysis-2026-06-13-0836/01-kasmina-tolaria-blueprint-health.md`, added `docs/analysis/2026-06-13-arch-health-task-slotting.md`, and drafted `docs/plans/planning/2026-06-13-morphogenesis-governor-integrity.md`. The critical path now puts governor-integrity before PPO oracle sandbox and reward-efficiency proof runs. |
| 2026-06-13 | **PROOF CONFOUNDER DRAIN IMPLEMENTED.** Moved plan to `docs/plans/completed/2026-06-13-proof-confounder-drain.md`. Implemented run-level confounder ledger, action-head learnability telemetry, fail-closed counterfactual freshness, reward-accounting closure, and generated proof packets. The rehearsal packet is `BLOCKED` by value-collapse and gradient-anomaly confounders, so the long reward-efficiency exam remains deferred. |
| 2026-06-13 | **PROOF CONFOUNDER DRAIN DRAFTED.** Added `docs/plans/planning/2026-06-13-proof-confounder-drain.md` as the next major signal-recovery package. Framing: prior results strongly suggest Esper's underlying theory is sound, but weaker-than-expected effect size points to confounders. The package gates proof runs on anomaly/confounder ledger, action-head learnability, counterfactual freshness, reward-accounting closure, and a generated proof packet before the full reward-efficiency verdict. |
| 2026-06-13 | **OP/VALUE MISMATCH VERIFIED RESOLVED.** `get_action()` uses one selected op for action, log-prob, and Q(s,op) value in stochastic rollout mode, and recomputes Q(s,argmax op) for deterministic bootstrap. Added direct regression probes and updated active critical count to 0. |
| 2026-02-21 | **POST-HIATUS FULL AUDIT.** Returned after 1-month break. Comprehensive codebase audit using explore agents: |
| | **Plans moved to completed/ (5 files):** |
| | - op-entropy-collapse: FULLY IMPLEMENTED Jan 9-11 (probability floors + entropy floors) |
| | - simic-audit-remediation: Effectively completed Jan 10-11 (Governor tests, action handlers, PPOCoordinator) |
| | - historical-dual-state-lifecycle (2 files): Complete (SeedLifecycleEvent, LifecyclePanel) |
| | - post-fossilization-drip-reward design: DRL expert reviewed and approved |
| | **Plans moved to abandoned/ (1 file):** |
| | - emrakul-submodule-editing: Moved from concepts/ (self-documents supersession) |
| | **Plans moved to ready/ (1 file):** |
| | - phase3-tinystories-strategy: Moved from concepts/ (85% implemented, needs validation runs) |
| | **Critical bug surfaced, later resolved 2026-06-13:** |
| | - CRITICAL-op-value-mismatch in investigations/: Q(s,op) double-sampling corrupted advantages |
| | **Bugs folder audited:** 307+ files across fixed/triaged/wontfix/not-a-bug/generated. |
| | - 14 generated Codex analysis files need triage |
| | - 0 skip/xfail markers in tests (good hygiene) |
| | **Key takeaway:** The Jan 9-17 sprint completed far more than the tracker reflected. |
| | Entropy collapse, simic audit, dual-state lifecycle, drip reward design all done. |
| | Updated Health Summary: Completed 8→14, Concept 5→3, Abandoned 2→3. |
| 2026-01-12 | **POST-FOSSILIZATION DRIP REWARD.** Added new plan to close gaming exploit: |
| | - Seeds can fossilize at peak metrics then degrade with no accountability |
| | - Solution: 70/30 split - 30% immediate, 70% drip over remaining epochs |
| | - Drip based on continued contribution (negative contribution = penalty) |
| | - Epoch normalization prevents early-fossilization gaming |
| | - DRL expert reviewed and approved design decisions |
| | - Location: `docs/plans/ready/2026-01-12-post-fossilization-drip-reward.md` |
| 2026-01-11 | **OP-ENTROPY-COLLAPSE DIAGNOSIS.** DRL expert telemetry analysis of ESCROW run revealed: |
| | - Previous entropy-collapse fix was insufficient (op floor 0.15 too close to collapse point 0.14) |
| | - Policy freeze after batch 40: WAIT 99.9%, GERMINATE 0.07%, fossilizations 0 |
| | - Blueprint head collapsed to 0.000 entropy, picking conv_small (0% fossil rate) 72% of the time |
| | - Root cause is OP HEAD collapse, not sparse heads |
| | **ACTIONS:** |
| | - Created op-entropy-collapse plan (replaces entropy-collapse) |
| | - Moved shaped-delta-clip to abandoned (root cause is entropy, not reward inflation) |
| | - New fix: op probability floor 0.05, op entropy floor 0.25, blueprint/tempo penalty coef 0.3 |
| 2026-01-10 | **TRANSITORY PLANS VERIFIED.** Checked 4 telemetry wiring plans from docs/plans/ root: |
| | ✅ diagnostic-panel-metrics: 92% (11/12 tasks) |
| | ✅ tele-340-lstm-health: 100% (27 tests passing) |
| | ✅ tele-610-episode-stats: 95% (19/20 tasks) |
| | ✅ value-function-metrics: 100% (97 tests passing) |
| | All 4 moved to completed/. Health Summary: Completed 5→9. |
| 2026-01-10 | **CODEBASE VERIFICATION.** Checked all ready/ plans against actual code: |
| | ✅ entropy-collapse: 100% COMPLETE (all 7 tasks, tests passing) |
| | ✅ holding-warning: 100% COMPLETE (committed 2026-01-08, DRL signed) |
| | ⚠️ defensive-patterns: COMPLIANT via whitelisting (not refactored) |
| | ❌ blueprint-compiler: 0% (correctly deferred) |
| | ❌ telemetry-domain-sep: ~15% (3/9 DRL fields, no renaming) |
| | ❌ counterfactual-aux: 0% (none of 4 phases) |
| | ❌ sanctum-help: ~10% (only global help) |
| | ❌ heuristic-tamiyo: 0% (TamiyoDecision missing tempo) |
| | Updated Health Summary: Completed 3→5, Ready 11→9 |
| 2026-01-10 | **COMPREHENSIVE INVENTORY.** Discovered 14 untracked plans: |
| | **ready/ (11 plans added):** |
| | - 🔴 entropy-collapse (CRITICAL) - per-head entropy collapse fix |
| | - blueprint-compiler + 2 appendices - compiler & curriculum seeds |
| | - telemetry-domain-sep - event type renaming |
| | - holding-warning - turntabling exploit fix |
| | - counterfactual-aux - auxiliary supervision |
| | - defensive-patterns - code quality cleanup |
| | - sanctum-help - TUI help system |
| | - heuristic-tamiyo - tempo parity for A/B testing |
| | - simic2-vectorized (DUPLICATE - already completed) |
| | **planning/ (3 workspaces added):** |
| | - emrakul-immune (emrakul1/) - master architecture doc |
| | - kasmina-multichannel (kasmina1.5/) - slot grid expansion |
| | - esika-superstructure (esika1/) - multi-cell coordination |
| | Total active plans: 10 → 24 |
| 2026-01-10 | **Moved simic2 to completed/.** All 3 phases verified and moved to `docs/plans/completed/simic2/`. |
| 2026-01-10 | **Second spot check (simic2 deep dive).** All 3 phases complete: |
| | - simic2-phase1: 75% → 100% (VectorizedPPOTrainer + 4 modules extracted) |
| | - simic2-phase2: Started → 100% (vectorized_types.py + rewards/types.py complete) |
| | - simic2-phase3: 0% → 100% (rewards/ and agent/ fully decomposed) |
| | - Moved phase1-final-exam.md from concepts/ to ready/ |
| | - simic2 no longer blocks kasmina2-phase0 |
| 2026-01-10 | **First spot check via codebase exploration.** Major corrections: |
| | - phase3-tinystories: 0% → 85% (TransformerHost, blueprints, dataset all exist) |
| | - reward-efficiency: Infra 0% → 100% (just needs experiment execution) |
| | - Updated dependency graph and recommendations accordingly |
| 2026-01-10 | Initial tracker created. Catalogued 10 active plans. |

---

## File Index

Quick reference for all tracked plans:

### ready/ (Implementation-Ready)
| File | ID |
|------|-----|
| `2026-01-09-blueprint-compiler-and-curriculum-seeds.md` | blueprint-compiler |
| `2026-01-09-blueprint-compiler-appendix-antipatterns.md` | blueprint-antipatterns |
| `2026-01-09-blueprint-compiler-appendix-future-blueprints.md` | blueprint-future |
| `2026-01-02-telemetry-domain-separation.md` | telemetry-domain-sep |
| `2026-01-10-counterfactual-auxiliary-supervision.md` | counterfactual-aux |
| `2026-01-12-post-fossilization-drip-reward-impl.md` | drip-reward-impl |
| `defensive-pattern-fixes.md` | defensive-patterns |
| `2025-12-29-sanctum-help-system.md` | sanctum-help |
| `2025-12-26-slot-transformer-architecture.md` | tamiyo4 |
| `h-tamiyo-updates.md` | heuristic-tamiyo |
| `phase1-final-exam.md` | reward-efficiency |
| `phase3-tinystories-strategy.md` | phase3-tinystories |
| `2026-06-17-recurrent-ppo-multiepoch-plan.md` | recurrent-ppo-multiepoch |

### planning/ (Active Design)
| Folder | ID |
|--------|-----|
| `kasmina2/` | kasmina2-phase0 |
| `emrakul1/` | emrakul-immune |
| `kasmina1.5/` | kasmina-multichannel |
| `esika1/` | esika-superstructure |
| `karn2/` | karn2 |
| `narset1/` | narset1 |

### concepts/ (Early Ideas)
| File | ID |
|------|-----|
| `emrakul-sketch.md` | emrakul-sketch |
| `counterfactual_oracle.md` | counterfactual-oracle |
| `scaled_counterfactuals.md` | scaled-counterfactuals |

### abandoned/ (Superseded)
| File | ID |
|------|-----|
| `shaped-mode-delta-clipping.md` | shaped-delta-clip (superseded by op-entropy-collapse) |
| `emrakul-submodule-editing-blending-holding.md` | emrakul-submodule-editing (superseded by Track A+C) |

### completed/ (Historical)
| File/Folder | ID |
|-------------|-----|
| `simic2/` | simic2-phase1, simic2-phase2, simic2-phase3 |
| `2026-01-09-fix-per-head-entropy-collapse.md` | entropy-collapse |
| `2026-01-11-op-head-entropy-collapse-fix.md` | op-entropy-collapse |
| `2026-01-08-fix-set-alpha-target-turntabling.md` | holding-warning |
| `2026-01-11-simic-audit-remediation.md` | simic-audit |
| `2026-01-10-historical-dual-state-lifecycle-design.md` | dual-state-lifecycle (design) |
| `2026-01-10-historical-dual-state-lifecycle.md` | dual-state-lifecycle (impl) |
| `2026-01-12-post-fossilization-drip-reward.md` | drip-reward-design |
| `2026-01-03-diagnostic-panel-metrics-wiring.md` | diagnostic-panel-metrics |
| `2026-01-03-tele-340-lstm-health-wiring.md` | tele-340-lstm-health |
| `2026-01-04-tele-610-episode-stats-wiring.md` | tele-610-episode-stats |
| `2026-01-04-value-function-metrics-wiring.md` | value-function-metrics |

### bugs/ (Issue Tracking)
| Category | Count | Notes |
|----------|-------|-------|
| `investigations/` | 4 | Includes resolved CRITICAL-op-value-mismatch historical diagnosis |
| `fixed/` | 90 | Resolved with code changes |
| `triaged/` | 108 | Analyzed, awaiting implementation |
| `wontfix/` | 49 | Intentionally deferred |
| `not-a-bug/` | 56 | False positives |
| `generated/` | 14 | Codex analysis, needs triage |
