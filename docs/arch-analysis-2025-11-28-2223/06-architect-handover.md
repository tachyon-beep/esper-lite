# Esper V1.0 - Architect Handover Document

## Purpose

This document provides actionable improvement recommendations for a technical architect or senior engineer taking ownership of the Esper codebase. It synthesizes findings from comprehensive architecture analysis into prioritized, concrete next steps.

---

## Current State Assessment

### What's Working Well

**Architecture Strengths** (preserve these patterns):

1. **Domain-Driven Package Design** - The MTG-themed packages (Leyline, Kasmina, Tamiyo, Simic, Nissa) create memorable, well-bounded contexts. Each has single responsibility:
   ```
   Leyline (Contracts) → Kasmina (Mechanics) → Tamiyo (Decisions) → Simic (Learning)
                                    ↑
                              Nissa (Telemetry)
   ```

2. **Contract-First Communication** - Leyline provides shared types (enums, dataclasses, protocols) that enable loose coupling. Subsystems depend on contracts, not implementations.

3. **Hot Path Isolation** - `simic/features.py` imports only from Leyline, enabling future JIT compilation or vectorization without subsystem overhead.

4. **Finite State Machine Validation** - SeedStage FSM with `VALID_TRANSITIONS` dict prevents invalid lifecycle states. Terminal stages properly identified.

5. **Protocol-Based Extensibility** - `TamiyoPolicy`, `BlueprintProtocol`, `OutputBackend` enable swapping implementations without modifying consumers.

### Technical Debt Summary

| Category | Severity | Items | Estimated Effort |
|----------|----------|-------|------------------|
| Error Handling | HIGH | 24 bare excepts, no recovery | 3 days |
| Large Files | MEDIUM | 5 files >600 LOC | 4 days |
| Test Coverage | MEDIUM | 70% of code untested | 4 days |
| Documentation | LOW | Algorithm explanations missing | 2 days |
| Script Stubs | LOW | 2 incomplete entry points | 2 days |

**Total Estimated Effort**: ~15 developer-days for all items

---

## Priority 1: Critical Path Improvements

### 1.1 Error Handling Infrastructure (3 days)

**Problem**: Training crashes on unexpected data/state without recovery. Only 10 files have try-except blocks.

**Solution**: Add structured error handling at key boundaries:

```python
# simic/ppo.py - Training loop protection
class PPOTrainer:
    def train_episode(self, env: VectorizedEnv) -> EpisodeResult:
        try:
            observations = env.reset()
            for step in range(self.max_steps):
                actions = self.policy.select_action(observations)
                observations, rewards, dones, infos = env.step(actions)
                # ... training logic
        except FeatureDimensionError as e:
            logger.error(f"Feature mismatch at step {step}: {e}")
            return EpisodeResult.failed(reason="feature_dimension")
        except InvalidStageTransition as e:
            logger.warning(f"Invalid transition: {e}, rolling back")
            return EpisodeResult.partial(steps_completed=step)
        except Exception as e:
            logger.critical(f"Unhandled error: {e}", exc_info=True)
            raise TrainingAborted(episode=self.current_episode) from e
```

**Files to modify**:
- `simic/ppo.py:train()` - Wrap training loop
- `simic/iql.py:train()` - Wrap training loop
- `kasmina/slot.py:transition_stage()` - Handle invalid transitions
- `simic/features.py:extract_features()` - Validate input dimensions

**Acceptance criteria**:
- [ ] No bare `except:` clauses
- [ ] All training loops wrapped with recovery logic
- [ ] Errors logged with context (epoch, seed, action)
- [ ] Training can resume after non-fatal errors

### 1.2 Orchestrator Decomposition (2 days)

**Problem**: `simic_overnight.py` (859 LOC) is monolithic, mixing episode generation, policy training, and evaluation in single file.

**Solution**: Extract into focused modules:

```
simic/
├── generation.py   # generate_episodes() - Episode data collection
├── training.py     # train_policy() - PPO/IQL training orchestration
├── evaluation.py   # evaluate_policy() - Policy assessment
└── overnight.py    # Slim orchestrator calling above modules
```

**Migration pattern**:
1. Create `simic/generation.py` with `EpisodeGenerator` class
2. Create `simic/training.py` with `PolicyTrainer` class
3. Create `simic/evaluation.py` with `PolicyEvaluator` class
4. Refactor `simic_overnight.py` to import and compose these classes
5. Add unit tests for each class in isolation

**Acceptance criteria**:
- [ ] Each new module <300 LOC
- [ ] `simic_overnight.py` <200 LOC (composition only)
- [ ] Unit tests for each extracted module
- [ ] No behavior change (integration tests pass)

### 1.3 Complete Script Entry Points (2 days)

**Problem**: `scripts/generate.py` and `scripts/evaluate.py` are TODO stubs blocking user workflows.

**Solution**: Implement missing CLIs using existing infrastructure:

**generate.py**:
```python
#!/usr/bin/env python3
"""Generate episode data for offline RL training."""
import argparse
from esper.simic.episodes import EpisodeCollector
from esper.tamiyo.heuristic import HeuristicTamiyo

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--output', type=str, default='data/episodes/')
    args = parser.parse_args()

    policy = HeuristicTamiyo()
    collector = EpisodeCollector(policy=policy)
    collector.collect(n_episodes=args.episodes, output_dir=args.output)

if __name__ == '__main__':
    main()
```

**evaluate.py**:
```python
#!/usr/bin/env python3
"""Evaluate trained policy against heuristic baseline."""
import argparse
from esper.simic.networks import PolicyNetwork
from esper.tamiyo.heuristic import HeuristicTamiyo

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--episodes', type=int, default=10)
    args = parser.parse_args()

    learned = PolicyNetwork.load(args.model)
    baseline = HeuristicTamiyo()
    # ... comparison logic

if __name__ == '__main__':
    main()
```

**Acceptance criteria**:
- [ ] Both scripts runnable from CLI
- [ ] Help text with usage examples
- [ ] Integration with existing modules (no code duplication)
- [ ] README.md updated with usage

---

## Priority 2: Quality Improvements

### 2.1 Consolidate Signal Types (2 days)

**Problem**: Two parallel types (`TrainingSignals` vs `FastTrainingSignals`) create confusion and conversion overhead.

**Current state**:
- `leyline/signals.py` defines both types
- 27 features in both, but different representations
- Conversion functions scattered across codebase

**Solution**: Standardize on `FastTrainingSignals` (NamedTuple for performance):

1. Audit all `TrainingSignals` usages
2. Replace with `FastTrainingSignals` construction
3. Remove `TrainingSignals` dataclass
4. Update docstrings to clarify canonical type

**Acceptance criteria**:
- [ ] Single signal type in Leyline
- [ ] All subsystems using canonical type
- [ ] No conversion functions needed

### 2.2 Subsystem Test Coverage (4 days)

**Problem**: ~30% test coverage concentrated in Leyline and Simic. Kasmina, Tamiyo, Nissa untested.

**Test priority by risk**:

| Subsystem | Risk Level | Test Focus |
|-----------|------------|------------|
| Kasmina | HIGH | Lifecycle transitions, quality gates, alpha blending |
| Tamiyo | MEDIUM | HeuristicTamiyo edge cases, signal tracking |
| Nissa | LOW | Config validation, output backends |

**Test patterns to implement**:

```python
# tests/test_kasmina.py
class TestSeedSlot:
    def test_lifecycle_transition_happy_path(self):
        """Seed can progress DORMANT → GERMINATED → TRAINING → FOSSILIZED."""

    def test_lifecycle_transition_rejects_invalid(self):
        """Invalid transitions raise error with context."""

    def test_quality_gate_blocks_advancement(self):
        """Seed cannot advance without meeting gate criteria."""

    def test_alpha_blending_schedule(self):
        """Alpha increases monotonically from 0 to 1."""
```

**Acceptance criteria**:
- [ ] >80% function coverage per subsystem
- [ ] Edge cases tested (boundary conditions, error paths)
- [ ] Tests run in CI pipeline

### 2.3 Algorithm Documentation (2 days)

**Problem**: PPO/IQL training loops lack inline comments explaining key concepts.

**Documentation needed**:

**PPO (simic/ppo.py)**:
- GAE (Generalized Advantage Estimation) computation
- PPO-Clip objective and epsilon parameter
- Entropy bonus for exploration
- Value function loss computation
- Mini-batch iteration strategy

**IQL (simic/iql.py)**:
- V-network (value function)
- Expectile regression (asymmetric loss)
- Conservative Q-learning (pessimism in offline RL)
- AWR (Advantage Weighted Regression) for policy extraction

**Template for algorithm comments**:
```python
def compute_gae(rewards, values, dones, gamma, gae_lambda):
    """Compute Generalized Advantage Estimation.

    GAE provides variance-bias tradeoff in advantage estimation:
    A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...

    where δ_t = r_t + γV(s_{t+1}) - V(s_t) is TD error.

    Args:
        rewards: Shape (T,) rewards for each timestep
        values: Shape (T+1,) value estimates (includes bootstrap)
        dones: Shape (T,) episode termination flags
        gamma: Discount factor (typically 0.99)
        gae_lambda: GAE lambda (typically 0.95, lower = more bias, less variance)

    Returns:
        advantages: Shape (T,) GAE advantages
        returns: Shape (T,) discounted returns for value target

    Reference: Schulman et al. "High-Dimensional Continuous Control Using
               Generalized Advantage Estimation" (2015)
    """
```

**Acceptance criteria**:
- [ ] All algorithm functions have docstrings with math explanation
- [ ] Paper references where appropriate
- [ ] Hyperparameter sensitivity documented

---

## Priority 3: Long-term Improvements

### 3.1 Performance Benchmarking (2 days)

**Claims to verify**:
1. Feature extraction is O(1) - Measure actual time
2. Telemetry overhead acceptable - Profile with/without
3. Replay buffer scales - Test with 100k+ episodes

**Benchmark suite**:
```python
# benchmarks/bench_features.py
def bench_feature_extraction():
    """Feature extraction should be <1ms per observation."""
    signals = generate_sample_signals(n=1000)

    start = time.perf_counter()
    for s in signals:
        extract_features(s)
    elapsed = time.perf_counter() - start

    assert elapsed / 1000 < 0.001, f"Too slow: {elapsed/1000:.4f}s per extraction"
```

### 3.2 Multi-Dataset Support (3 days)

**Current limitation**: CIFAR-10 hardcoded throughout.

**Generalization path**:
1. Extract dataset loading to `DatasetProvider` interface
2. Implement CIFAR-10, ImageNet providers
3. Parameterize input shape in blueprints
4. Update feature extraction for variable input sizes

### 3.3 Checkpoint/Resume (2 days)

**Current limitation**: Long training runs lost on interruption.

**Solution**:
- Save checkpoints every N episodes
- Include optimizer state, episode count, running stats
- Resume flag in CLI
- Graceful SIGTERM handling

---

## Architecture Evolution Recommendations

### Near-term (1-3 months)

1. **Add configuration layer** - Extract hardcoded values to YAML/TOML config
2. **Implement metrics dashboard** - Real-time training visualization
3. **Add profiling hooks** - Built-in performance measurement

### Medium-term (3-6 months)

1. **Distributed training** - Multi-GPU PPO with gradient synchronization
2. **Experiment tracking** - MLflow/W&B integration for reproducibility
3. **Hyperparameter optimization** - Optuna integration for tuning

### Long-term (6-12 months)

1. **Custom operators** - TorchScript compilation of hot paths
2. **Online evaluation** - A/B testing framework for policy comparison
3. **Production deployment** - Model serving infrastructure

---

## Handover Checklist

### Documentation Complete
- [x] Architecture report (`04-final-report.md`)
- [x] Subsystem catalog (`02-subsystem-catalog.md`)
- [x] C4 diagrams (`03-diagrams.md`)
- [x] Quality assessment (`05-quality-assessment.md`)
- [x] Architect handover (this document)

### Knowledge Transfer Topics
- [ ] Seed lifecycle FSM and quality gates
- [ ] Gradient isolation mechanism
- [ ] Alpha-blending schedule and rationale
- [ ] PPO vs IQL trade-offs
- [ ] Hot path constraints and maintenance

### Access Requirements
- [ ] Repository access
- [ ] GPU compute access
- [ ] CI/CD pipeline access
- [ ] Model storage access

---

## Contact & Questions

**Analysis performed by**: Claude Code (System Archaeologist)
**Date**: 2025-11-28 to 2025-11-29
**Confidence level**: High (85%)

**Outstanding questions for domain expert**:
1. Is the 11-state FSM complete, or are additional states anticipated?
2. What is the expected episode count for production training runs?
3. Are there specific performance targets (training time, memory)?
4. Should Leyline contracts be versioned for backward compatibility?

---

## Appendix: Quick Reference

### Key Files by Task

| Task | Files |
|------|-------|
| Understand contracts | `leyline/stages.py`, `leyline/actions.py` |
| Modify seed lifecycle | `kasmina/slot.py`, `kasmina/blueprints.py` |
| Tune decision policy | `tamiyo/heuristic.py` |
| Train RL policy | `simic/ppo.py`, `simic/iql.py` |
| Debug training | `nissa/tracker.py`, `nissa/output.py` |
| Run experiments | `scripts/train_ppo.sh` |

### Critical Invariants

1. **Hot path isolation**: `simic/features.py` must only import from `leyline`
2. **FSM validity**: All stage transitions must be in `VALID_TRANSITIONS`
3. **Gradient isolation**: Hooks must be cleaned up after blending
4. **Feature dimensions**: Observation space is 27 dimensions (TensorSchema)

### Command Reference

```bash
# Run PPO training
./scripts/train_ppo.sh -e 100 -n 6

# Run tests
PYTHONPATH=src .venv/bin/python -m pytest tests/

# Check types
mypy src/esper/
```

---

**Document Version**: 1.0
**Generated**: 2025-11-29
