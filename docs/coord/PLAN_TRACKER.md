# Esper Plan Tracker

**Last Updated:** 2026-01-10 (spot-checked via codebase exploration)
**Purpose:** Rack-and-stack all plans and concepts for prioritization and dependency tracking.

---

## Executive Summary

### Current Focus Areas
1. **Simic2 Refactor** - ✅ COMPLETE. Moved to `docs/plans/completed/simic2/`
2. **Reward Efficiency Experiment** - Infrastructure complete, experiment never run (NEEDS EXECUTION)
3. **Phase3-TinyStories** - 80-90% IMPLEMENTED (was incorrectly tracked as "not started")

### Critical Path
```
[simic2 ✅] ──► kasmina2-phase0 ──► counterfactual-oracle ──► emrakul-phase1
                      │
    reward-efficiency ┘
```

### Health Summary
| Status | Count |
|--------|-------|
| Completed | 3 (simic2 phases 1-3) |
| In Progress | 1 |
| Planning | 4 |
| Concept | 5 |
| Abandoned | 1 |
| **Total Active** | **10** |

---

## Priority Matrix

### Tier 1: Critical Path (Do Now)

| ID | Title | Type | Urgency | Complexity | Risk | Status |
|----|-------|------|---------|------------|------|--------|
| reward-efficiency | Phase 1 Final Exam (A/B Testing) | ready | high | S | low | ⚠️ Infra 100% done, experiment never run |
| kasmina2-phase0 | Submodule Intervention Foundation | planning | high | L | medium | Design complete, ready to implement |

### Tier 2: Next Up (Queue)

| ID | Title | Type | Urgency | Complexity | Risk | Status |
|----|-------|------|---------|------------|------|--------|
| phase3-tinystories | Transformer Domain Pivot | in-progress | medium | L | medium | ✅ 80-90% complete (was wrongly "not started") |
| counterfactual-oracle | Learned Contribution Probe | ready | medium | XL | high | Plan in ready/, blocked on reward-efficiency |

### Completed (Moved to docs/plans/completed/)

| ID | Title | Type | Location |
|----|-------|------|----------|
| simic2-phase1 | Vectorized Module Split | ✅ completed | `docs/plans/completed/simic2/` |
| simic2-phase2 | Typed Contracts & API | ✅ completed | `docs/plans/completed/simic2/` |
| simic2-phase3 | Simic Module Split | ✅ completed | `docs/plans/completed/simic2/` |

### Tier 3: Strategic (Plan Ahead)

| ID | Title | Type | Urgency | Complexity | Risk | Status |
|----|-------|------|---------|------------|------|--------|
| kasmina2-phase0 | Submodule Intervention Foundation | planning | high | L | medium | Design complete, no code yet |
| counterfactual-oracle | Learned Contribution Probe | ready | medium | XL | high | Plan promoted to ready/ today, blocked on reward-efficiency |
| emrakul-sketch | Immune System Phase 4 | concept | medium | XL | high | Design locked, needs Tamiyo stable |

### Tier 4: Backlog (Someday)

| ID | Title | Type | Urgency | Complexity | Risk | Status |
|----|-------|------|---------|------------|------|--------|
| narset1 | Meta-Coordination Layer | concept | low | L | medium | Speculative |
| karn2 | Karn Sanctum v2 | planning | low | M | low | Nice-to-have TUI improvements |
| tamiyo4 | Slot Transformer Architecture | planning | low | L | medium | Research direction |

### Abandoned

| ID | Title | Reason |
|----|-------|--------|
| emrakul-submodule-editing | BLENDING/HOLDING Mutations | Superseded by Track A+C Microstructured Ladders |

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

depends_on: []  # No blockers - can run now
blocks:
  - counterfactual-oracle (explicitly gated on this)

status_notes: |
  SPOT CHECK 2026-01-10: Infrastructure is 100% complete!
  - RewardMode.SIMPLIFIED implemented (contribution.py:747-836)
  - RewardMode.SPARSE implemented (contribution.py:670-702)
  - dual_ab.py exists with train_dual_policy_ab()
  - CLI: --dual-ab shaped-vs-simplified ready
  - Configs: configs/ablations/{shaped,simplified,sparse}_baseline.json

  NEVER EXECUTED. Just run it:
  uv run python -m esper.scripts.train ppo --dual-ab shaped-vs-simplified --episodes 100
percent_complete: 100 (infra) / 0 (experiment)
```

**Commentary:**
> **MAJOR FINDING:** All the code exists. The experiment was simply never run.
> This is a "just press the button" situation, not a development task.
>
> The dual-policy A/B system trains separate PPO agents per reward mode with
> isolated environments, policies, and optimizers. Results would directly
> unblock counterfactual-oracle.
>
> **Action:** Run the experiment. Complexity is S (just execute), not M (build infrastructure).

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
type: planning
created: 2025-12-26
updated: 2025-12-26

urgency: low
value: |
  Replace LSTM policy backbone with Transformer.
  Better for variable-length slot sequences.

complexity: L
risk: medium
risk_notes: |
  - Architectural change to policy network
  - Needs careful A/B validation

depends_on:
  - simic2-phase2 (cleaner training code)
blocks: []

status_notes: |
  Research direction. Not blocking anything critical.
  Could help with scaling to many slots.
percent_complete: 0
```

---

## Dependency Graph

```
                                    ┌─────────────────┐
                                    │  phase3-tiny    │
                                    │   stories       │
                                    └────────▲────────┘
                                             │
┌──────────────┐    ┌──────────────┐    ┌────┴───────────┐    ┌─────────────────┐
│  simic2      │───►│  simic2      │───►│   reward       │───►│  counterfactual │
│  phase1      │    │  phase2      │    │   efficiency   │    │  oracle         │
│  [CRITICAL]  │    │              │    │   [NEEDS ATTN] │    │                 │
└──────┬───────┘    └──────┬───────┘    └────────────────┘    └────────┬────────┘
       │                   │                                           │
       │                   │                                           │
       ▼                   ▼                                           ▼
┌──────────────┐    ┌──────────────┐                           ┌───────────────┐
│  kasmina2    │───►│  kasmina2    │                           │  emrakul      │
│  phase0      │    │  phase1+     │                           │  phase1       │
│              │    │              │                           │               │
└──────────────┘    └──────────────┘                           └───────────────┘
```

---

## Risk Register

| Plan | Risk Level | Primary Risk | Mitigation |
|------|------------|--------------|------------|
| counterfactual-oracle | HIGH | Goodhart/reward hacking | Probe as observation only, never as reward |
| emrakul-sketch | HIGH | Training destabilization | Turbulence locks, Narset safety regimes |
| phase3-tinystories | HIGH | NaN spikes on graft | Zero-init projections, LayerNorm pre-injection |
| kasmina2-phase0 | MEDIUM | Cross-domain coordination | Six parallel tracks with clear ownership |
| simic2-phase1 | MEDIUM | Behavioral regression | Baseline capture, diff testing |
| reward-efficiency | MEDIUM | Wasted compute | Clear success criteria, stop-loss |
| simic2-phase2 | LOW | Pure refactor | Typed contracts, tests |
| karn2 | LOW | User-facing only | N/A |

---

## Recommendations

### Immediate Actions (This Week)

1. **Run reward-efficiency experiment** - Infrastructure is 100% complete. Just execute:
   ```bash
   uv run python -m esper.scripts.train ppo --dual-ab shaped-vs-simplified --episodes 100
   ```
2. **Run TinyStories baseline** - Implementation is 85% complete. Validate it works:
   ```bash
   uv run python -m esper.scripts.train ppo --task tinystories --episodes 50
   ```

### Short-Term (Next 2 Weeks)

3. **Analyze reward A/B results** - Declare winner (SHAPED vs SIMPLIFIED).
4. **Validate TinyStories learning curves** - Confirm transformer morphogenesis works.
5. **Begin kasmina2-phase0 implementation** - Design complete, simic2 blocker removed.

### Medium-Term (Next Month)

6. **Begin counterfactual-oracle Phase 1** - Unblocked once reward-efficiency has data.
7. **Document TinyStories results** - Update phase3 plan with findings.
8. **Consider promoting kasmina2 design to ready/** - If phase0 implementation proceeds.

### Parking Lot (Not Now)

- emrakul-sketch - Wait for Tamiyo stability (Phase 4+)
- narset1 - Speculative, defer
- karn2 - Nice-to-have, defer
- tamiyo4 - Research direction (SlotTransformer for policy scaling)

---

## Change Log

| Date | Change |
|------|--------|
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
