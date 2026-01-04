# Reward Function Transition Plan: SHAPED → SIMPLIFIED with Pareto Tracking

> **Status:** COMPLETED (2026-01-03)

**Goal:** Transition Tamiyo's reward system from complex SHAPED mode to cleaner SIMPLIFIED mode, with multi-objective Pareto frontier tracking for principled reward engineering.

**Architecture:** Extend `episode_history` to capture multi-objective outcomes (`EpisodeOutcome`), add Pareto frontier computation to Karn analytics, create ablation configs for systematic comparison, and add reward health panel to Sanctum TUI.

**Tech Stack:** Python dataclasses, DuckDB views (Karn MCP), Textual widgets (Sanctum), existing A/B testing infrastructure.

---

## Implementation Summary

| Phase | Task | Status | Evidence |
|-------|------|--------|----------|
| 0 | Stability score design | ✅ | Uses reward variance proxy |
| 0.5 | A/B → A/B/n generalization | ✅ | `config.py`, `vectorized.py`, `train.py` |
| 1 | EpisodeOutcome schema | ✅ | `episode_outcome.py` |
| 1.5 | stage_bonus in store.py | ✅ | `store.py:194` |
| 2 | Wire EpisodeOutcome | ✅ | `vectorized.py` |
| 3 | Pareto frontier | ✅ | `pareto.py` |
| 4 | MCP view | ✅ | `views.py:375` |
| 4.5 | Ablation flags | ✅ | `config.py:132-134` |
| 5 | Ablation configs | ✅ | 6 files in `configs/ablations/` |
| 6-7 | RewardHealthPanel | ✅ | `reward_health.py`, wired in app |
| 8 | EPISODE_OUTCOME event | ✅ | `telemetry.py` |
| 9 | PBRS verification | ✅ | `test_pbrs_verification.py` |

**Total: 13 tasks implemented**

---

## Key Features Delivered

### 1. Multi-Objective Episode Tracking
- `EpisodeOutcome` dataclass captures accuracy, param_ratio, stability_score
- `dominates()` method for Pareto dominance checking
- Telemetry emission via `EPISODE_OUTCOME` event type

### 2. Pareto Frontier Analysis
- `extract_pareto_frontier()` finds non-dominated outcomes
- `compute_hypervolume_2d()` sweep-line algorithm for progress metric
- MCP view for SQL querying

### 3. A/B/n Reward Testing
- Renamed `ab_reward_modes` → `reward_mode_per_env`
- `TrainingConfig.with_reward_split()` builder for ergonomic config
- Support for arbitrary N-way comparisons

### 4. Ablation Configuration
- 6 config files for systematic experiments:
  - `simplified_baseline.json`
  - `no_pbrs.json`
  - `no_terminal.json`
  - `no_anti_gaming.json`
  - `pure_sparse.json`
  - `ab_shaped_vs_simplified.json`

### 5. Sanctum RewardHealthPanel
- PBRS fraction display (10-40% healthy range)
- Anti-gaming trigger rate (<5% healthy)
- Explained variance indicator
- Hypervolume progress tracking

### 6. PBRS Verification
- Tests verify Ng et al. (1999) properties
- Stage potential monotonicity
- Telescoping over episode trajectory
- Property-based tests with Hypothesis

---

## Original Plan Content

[Full original plan preserved below for reference]

---

**Revision Notes (2025-12-24):** Updated based on DRL Expert and Code Reviewer feedback:
- Fixed hypervolume algorithm (corrected sweep-line implementation)
- Fixed stability score computation (uses reward variance proxy, not non-existent `governor.anomalies_detected`)
- Added missing aggregator methods
- Adjusted PBRS healthy range to 10-40%
- Added 3 missing ablation configs from DRL Expert Section 7.2
- Added property-based tests for Pareto operations
- Added PBRS verification tests for SIMPLIFIED mode
- **Generalized A/B → A/B/n**: Renamed `ab_reward_modes` → `reward_mode_per_env`, added `with_reward_split()` builder

**Revision Notes (2025-12-26):** Pre-execution codebase review identified gaps:
- **Added Task 1.5**: `store.py` RewardComponents lacks `stage_bonus` field (sanctum/schema.py has it)
- **Added Task 4.5**: Ablation configs need `disable_pbrs`, `disable_terminal_reward`, `disable_anti_gaming` flags in TrainingConfig
- **Fixed Task 0.5**: Removed incorrect "backwards-compatible property alias" from commit message
- **Updated task count**: 11 → 13 tasks, ~3-3.5 hours estimated

---

## References

- DRL Expert Paper: `docs/research/reward-function-design-for-morphogenetic-controller.md`
- A/B/n infrastructure: `TrainingConfig.reward_mode_per_env` + `with_reward_split()` builder
- PBRS theory: Ng et al. (1999) - Policy invariance under reward transformations
- Hypervolume: Zitzler & Thiele (1999) - Multiobjective Evolutionary Algorithms
