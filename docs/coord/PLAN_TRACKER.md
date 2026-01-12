# Esper Plan Tracker

**Last Updated:** 2026-01-12 (added post-fossilization drip reward plan)
**Purpose:** Rack-and-stack all plans and concepts for prioritization and dependency tracking.

---

## Executive Summary

### 🔴 CRITICAL: Op Head Entropy Collapse

**Previous entropy-collapse fix was INSUFFICIENT.** DRL expert telemetry analysis (2026-01-11) revealed:
- Op head entropy floor (0.15) was too close to collapse point (0.14)
- Policy freezes after batch 40: WAIT 99.9%, GERMINATE 0.07%
- All 48 fossilizations occurred in Q1; Q2-Q4 had ZERO
- Blueprint head collapsed to 0.000 entropy, picking worst blueprint 72% of the time

### Current Focus Areas
1. **Op Entropy Collapse Fix** - 🔴 CRITICAL! Must implement probability floors + higher entropy floors
2. **Holding Warning Fix** - ✅ COMPLETE! Committed 2026-01-08, DRL expert signed
3. **Simic2 Refactor** - ✅ COMPLETE. Moved to `docs/plans/completed/simic2/`
4. **Reward Efficiency Experiment** - Infrastructure complete, experiment never run (BLOCKED by entropy fix)
5. **Phase3-TinyStories** - 80-90% IMPLEMENTED (BLOCKED by entropy fix)
6. **Blueprint Compiler** - 0% (correctly deferred until entropy confirmed stable)

### Critical Path
```
[op-entropy-collapse 🔴] ──► reward-efficiency ──► counterfactual-oracle ──► emrakul-phase1
         │
         └──► blueprint-compiler ──► kasmina2-phase0
         │
         └──► phase3-tinystories (validation runs)
```

### Health Summary
| Status | Count | Notes |
|--------|-------|-------|
| 🔴 Critical | 1 | op-entropy-collapse (blocks all training) |
| Completed | 8 | simic2 (3) + holding-warning + 4 transitory telemetry |
| Ready | 10 | Implementation-ready plans (drip-reward added 2026-01-12) |
| In Progress | 1 | phase3-tinystories |
| Planning | 6 | Active design workspaces |
| Concept | 5 | Early ideas |
| Abandoned | 2 | Superseded (shaped-delta-clip, emrakul-submodule-editing) |
| **Total Active** | **23** |

---

## Priority Matrix

### Tier 0: 🔴 CRITICAL (Fix Immediately)

| ID | Title | Type | Urgency | Complexity | Risk | Status |
|----|-------|------|---------|------------|------|--------|
| op-entropy-collapse | Op Head Entropy Collapse Fix | ready | 🔴 critical | M | high | 0% - DRL expert diagnosed (2026-01-11) |

### Tier 1: High Priority (This Week)

| ID | Title | Type | Urgency | Complexity | Risk | Status |
|----|-------|------|---------|------------|------|--------|
| reward-efficiency | Phase 1 Final Exam (A/B Testing) | ready | high | S | low | ⚠️ Infra 100% done, experiment never run |
| telemetry-domain-sep | Telemetry Domain Separation | ready | high | L | medium | ~15% done (3/9 DRL fields), no renaming |
| counterfactual-aux | Counterfactual Auxiliary Supervision | ready | high | M | medium | 0% - None of 4 phases started |
| blueprint-compiler | Blueprint Compiler (Phase 3 only) | ready | high | XL | medium | 0% - Correctly deferred until entropy stable |

### Tier 2: Medium Priority (Next 2 Weeks)

| ID | Title | Type | Urgency | Complexity | Risk | Status |
|----|-------|------|---------|------------|------|--------|
| drip-reward | Post-Fossilization Drip Reward | ready | high | M | medium | 0% - DRL expert reviewed 2026-01-12 |
| phase3-tinystories | Transformer Domain Pivot | in-progress | medium | L | medium | ✅ 80-90% complete, needs validation runs |
| kasmina2-phase0 | Submodule Intervention Foundation | planning | high | L | medium | Design complete, ready to implement |
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

### Completed (in docs/plans/completed/)

| ID | Title | Type | Status | Location |
|----|-------|------|--------|----------|
| entropy-collapse | Per-Head Entropy Collapse Fix | ✅ completed | All 7 tasks, tests passing | `docs/plans/completed/` |
| holding-warning | SET_ALPHA_TARGET Turntabling Fix | ✅ completed | Committed 2026-01-08, DRL signed | `docs/plans/completed/` |
| simic2-phase1 | Vectorized Module Split | ✅ completed | — | `docs/plans/completed/simic2/` |
| simic2-phase2 | Typed Contracts & API | ✅ completed | — | `docs/plans/completed/simic2/` |
| simic2-phase3 | Simic Module Split | ✅ completed | — | `docs/plans/completed/simic2/` |
| diagnostic-panel-metrics | Diagnostic Panel Metrics Wiring | ✅ completed | 92% (11/12 tasks) | `docs/plans/completed/` |
| tele-340-lstm-health | TELE-340 LSTM Health Wiring | ✅ completed | 100% (27 tests passing) | `docs/plans/completed/` |
| tele-610-episode-stats | TELE-610 Episode Stats Wiring | ✅ completed | 95% (19/20 tasks) | `docs/plans/completed/` |
| value-function-metrics | Value Function Metrics Wiring | ✅ completed | 100% (97 tests passing) | `docs/plans/completed/` |

### Abandoned

| ID | Title | Reason |
|----|-------|--------|
| emrakul-submodule-editing | BLENDING/HOLDING Mutations | Superseded by Track A+C Microstructured Ladders |
| shaped-delta-clip | SHAPED Mode Delta Clipping | Superseded by op-entropy-collapse (root cause is entropy, not reward inflation) |

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
type: ready
created: 2026-01-09
updated: 2026-01-11
location: ~/.claude/plans/reactive-churning-mist.md

urgency: critical
value: |
  Fixes multi-stage entropy collapse that freezes the policy.
  Op head collapses to 98% WAIT, starving sparse heads (blueprint, tempo)
  of gradient signal. Policy stops learning after batch 40.

complexity: M
risk: high
risk_notes: |
  - Higher entropy floors may initially reduce training efficiency
  - Probability floors change action distribution (may affect early convergence)
  - CRITICAL: Current state is completely broken - any fix is better than status quo

depends_on: []
blocks:
  - All training runs (policy freezes without this fix)
  - blueprint-compiler (can't explore blueprints with collapsed policy)

status_notes: |
  DRL EXPERT DIAGNOSIS (2026-01-11):

  Root cause is OP HEAD collapse, not sparse heads:
  1. Op entropy: 0.51 → 0.14 (collapse point just above floor!)
  2. WAIT rate: 92% → 99.9%
  3. Blueprint entropy: 0.125 → 0.000 (zero gradients)
  4. GERMINATE rate: 3.68% → 0.07%
  5. Fossilizations: 47 (Q1) → 0 (Q2-Q4)

  SOLUTION (two-pronged):
  1. HARD FLOOR: Add PROBABILITY_FLOOR_PER_HEAD with op=0.05
  2. SOFT FLOOR: Increase ENTROPY_FLOOR_PER_HEAD op from 0.15 to 0.25
  3. STRONGER PENALTY: Increase blueprint/tempo coef from 0.1 to 0.3

  IMPLEMENT IMMEDIATELY - replaces shaped-delta-clip as priority.
percent_complete: 0
```

**Commentary:**
> 🔴 **CRITICAL.** DRL expert telemetry analysis (2026-01-11) revealed that the
> previous entropy-collapse fix was insufficient. The op head floor (0.15) was
> too close to the collapse point (0.14), allowing degenerate equilibrium.
>
> Key finding: 98% WAIT isn't "learned selectivity" - it's policy freeze.
> Blueprint head collapsed to 0.000 entropy, picking conv_small (0% fossil rate)
> 72% of the time in Q4. The policy stopped learning after batch 40.
>
> Supersedes shaped-delta-clip - the root cause is entropy collapse, not reward inflation.

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
type: ready
created: 2026-01-12
updated: 2026-01-12
location: docs/plans/ready/2026-01-12-post-fossilization-drip-reward-impl.md

urgency: high
value: |
  Prevents seeds from gaming by fossilizing at peak metrics then degrading.
  Enforces long-term accountability for fossilized seeds via drip reward.

complexity: M
risk: medium
risk_notes: |
  - PPO variance could increase if drip magnitude miscalibrated
  - Credit assignment more complex (rewards arrive after action)
  - Must ensure drip normalization prevents early-fossilization gaming

depends_on: []
blocks: []

reviewed_by:
  - reviewer: drl-expert
    date: 2026-01-12
    verdict: approved_with_modifications

status_notes: |
  DRL EXPERT REVIEWED & APPROVED (2026-01-12):

  APPROVED DESIGN:
  - 70/30 split: 30% immediate for credit assignment, 70% drip for accountability
  - Epoch-normalized drip scale prevents early-fossilization gaming
  - Contribution-only for drip (not harmonic mean - improvement fixed at fossilization)
  - Per-epoch counterfactual is correct signal (NOT terminal Shapley)
  - ~85% effective value due to gamma discounting (desirable - slight pessimism)

  MODIFICATIONS FROM ORIGINAL PLAN:
  1. REMOVED: Diminishing returns for multiple fossils (replace with contribution threshold)
  2. ADDED: Asymmetric clipping (+0.1 positive, -0.05 negative to prevent death spirals)
  3. CONFIRMED: No Shapley reconciliation needed (per-epoch counterfactual is Markovian)

  IMPLEMENTATION PLAN: 8 TDD tasks with bite-sized steps
  - Task 1: Add drip config fields
  - Task 2: Add FossilizedSeedDripState dataclass
  - Task 3: Modify compute_basic_reward for drip split
  - Task 4-5: Golden tests and property tests
  - Task 6-7: Update dispatcher and telemetry
  - Task 8: Integrate in VectorizedPPOTrainer
percent_complete: 0
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
│ 🔴 entropy       │    ← FIX FIRST (training broken without this)
│    collapse      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ holding-warning  │    │ telemetry        │    │ counterfactual   │
│ (quick fix)      │    │ domain-sep       │    │ -aux             │
└────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      FOUNDATION READY                                │
└──────────────────────────────────────────────────────────────────────┘
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
│ phase3-          │    │ sanctum-help     │    │ defensive-       │
│ tinystories      │    │ (UX)             │    │ patterns         │
│ (85% done)       │    │                  │    │ (code quality)   │
└──────────────────┘    └──────────────────┘    └──────────────────┘

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
| entropy-collapse | 🔴 CRITICAL | Training completely broken | Implement immediately |
| counterfactual-oracle | HIGH | Goodhart/reward hacking | Probe as observation only, never as reward |
| emrakul-immune | HIGH | Novel architecture, two-timescale learning | Phased rollout, Shapley labels in Phase 1 only |
| phase3-tinystories | HIGH | NaN spikes on graft | Zero-init projections, LayerNorm pre-injection |
| blueprint-compiler | MEDIUM | New blueprints could destabilize | Phase 4 deferred until entropy >0.10 |
| counterfactual-aux | MEDIUM | Auxiliary loss could destabilize PPO | Low coefficient (0.05), warmup, stop-grad |
| telemetry-domain-sep | MEDIUM | Schema migration | Do early before more runs accumulate |
| kasmina2-phase0 | MEDIUM | Cross-domain coordination | Six parallel tracks with clear ownership |
| esika-superstructure | MEDIUM | New coordination layer | Infrastructure only, no intelligence |
| holding-warning | LOW | Simple fix | Terminal actions exempt |
| defensive-patterns | LOW | Code quality only | No behavior change |
| sanctum-help | LOW | User-facing only | N/A |
| karn2 | LOW | User-facing only | N/A |

---

## Recommendations

### Immediate Actions (This Week)

1. **Run reward-efficiency experiment** - Infrastructure is 100% complete:
   ```bash
   uv run python -m esper.scripts.train ppo --dual-ab shaped-vs-simplified --episodes 100
   ```

2. **Run TinyStories baseline** - Implementation is 85% complete:
   ```bash
   uv run python -m esper.scripts.train ppo --task tinystories --episodes 50
   ```

### Short-Term (Next 2 Weeks)

3. **Implement telemetry-domain-sep** - Currently ~15% done (3/9 DRL fields). Break schema now.
4. **Implement counterfactual-aux** - 0% done. Adds ContributionPredictor head.
5. **Implement heuristic-tamiyo** - 0% done. TamiyoDecision needs tempo field.
6. **Analyze reward A/B results** - Declare winner (SHAPED vs SIMPLIFIED).

### Medium-Term (Next Month)

7. **Begin kasmina2-phase0 implementation** - Design complete, simic2 blocker removed.
8. **Begin counterfactual-oracle Phase 1** - Unblocked once reward-efficiency has data.
9. **Blueprint compiler Phase 4** - New curriculum blueprints (ONLY if entropy stable >0.10).

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
| `2026-01-12-post-fossilization-drip-reward.md` | drip-reward |
| `defensive-pattern-fixes.md` | defensive-patterns |
| `2025-12-29-sanctum-help-system.md` | sanctum-help |
| `h-tamiyo-updates.md` | heuristic-tamiyo |
| `phase1-final-exam.md` | reward-efficiency |

### planning/ (Active Design)
| Folder | ID |
|--------|-----|
| `kasmina2/` | kasmina2-phase0 |
| `emrakul1/` | emrakul-immune |
| `kasmina1.5/` | kasmina-multichannel |
| `esika1/` | esika-superstructure |
| `karn2/` | karn2 |
| `tamiyo4/` | tamiyo4 |
| `narset1/` | narset1 |

### concepts/ (Early Ideas)
| File | ID |
|------|-----|
| `emrakul-sketch.md` | emrakul-sketch |
| `counterfactual_oracle.md` | counterfactual-oracle |
| `phase3-tinystories-strategy.md` | phase3-tinystories |
| `scaled_counterfactuals.md` | scaled-counterfactuals |
| `emrakul-submodule-editing-blending-holding.md` | emrakul-submodule-editing (ABANDONED) |

### abandoned/ (Superseded)
| File | ID |
|------|-----|
| `shaped-mode-delta-clipping.md` | shaped-delta-clip (superseded by op-entropy-collapse) |

### completed/ (Historical)
| File/Folder | ID |
|-------------|-----|
| `simic2/` | simic2-phase1, simic2-phase2, simic2-phase3 |
| `2026-01-09-fix-per-head-entropy-collapse.md` | entropy-collapse |
| `2026-01-08-fix-set-alpha-target-turntabling.md` | holding-warning |
| `2026-01-03-diagnostic-panel-metrics-wiring.md` | diagnostic-panel-metrics |
| `2026-01-03-tele-340-lstm-health-wiring.md` | tele-340-lstm-health |
| `2026-01-04-tele-610-episode-stats-wiring.md` | tele-610-episode-stats |
| `2026-01-04-value-function-metrics-wiring.md` | value-function-metrics |
