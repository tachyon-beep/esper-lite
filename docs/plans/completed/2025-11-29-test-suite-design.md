# Test Suite Design - Esper-Lite (From First Principles)

**Date**: 2025-11-29
**Status**: DESIGN
**Approach**: Quality Engineering methodology applied to RL systems

---

## Executive Summary

This document designs a fit-for-purpose test suite for esper-lite from first principles, informed by quality engineering best practices but adapted for the unique characteristics of reinforcement learning systems.

**Key insight**: RL systems have **strong mathematical invariants** but **inherent stochasticity**. Our test strategy must embrace both properties.

**Target distribution** (adapted for RL):
- **60% Property-Based Unit Tests** (mathematical invariants)
- **25% Integration Tests** (algorithm correctness)
- **10% E2E Pipeline Tests** (training workflows)
- **5% Performance/Regression Tests** (model quality benchmarks)

---

## Part 1: Domain Analysis

### Esper-Lite System Characteristics

**Core Components:**
1. **Ley line**: Contracts (SeedState, TrainingSignals, Action, SeedTelemetry)
2. **Simic**: RL algorithms (PPO, IQL, comparison, networks, buffers, rewards)
3. **Tamiyo**: Decision-making (heuristic policies)
4. **Kasmina**: Architecture discovery/adaptation
5. **Tolaria**: Training infrastructure (models, trainers)
6. **Nissa**: Seed lifecycle management
7. **Datagen**: Synthetic data generation

**Mathematical Properties** (must be tested):
- Reward bounds: `reward ∈ [min, max]`
- Normalization invariants: `E[normalized_x] ≈ 0, Var[normalized_x] ≈ 1`
- Action space completeness: `∀ blueprint → ∃ action`
- Gradient health: `grad_norm ∈ [ε, threshold]`
- Policy improvement: `V_new(s) >= V_old(s)` (monotonic improvement)
- Advantage estimation: `A(s,a) = Q(s,a) - V(s)`
- Feature dimension consistency: `dim(features_ppo) == dim(features_iql)`

**Stochastic Properties** (must handle non-determinism):
- Policy outputs (stochastic by design)
- Environment interactions (episode variance)
- Neural network initialization
- Training dynamics (gradient descent)

### Testing Challenges Unique to RL

| Challenge | Traditional Testing | RL Testing |
|-----------|-------------------|------------|
| **Non-determinism** | Flaky tests are bugs | Stochasticity is feature |
| **Long feedback** | Tests run in ms | Training takes hours |
| **Data requirements** | Simple fixtures | Episodes, models, checkpoints |
| **Correctness** | Binary (pass/fail) | Distributional (mean ± std) |
| **State management** | Stateless functions | Stateful policies, buffers |

**Strategy**: Use **property-based testing** for invariants, **statistical assertions** for stochastic outputs, **snapshot testing** for model states.

---

## Part 2: Test Pyramid Design (RL-Adapted)

### Traditional Test Pyramid

```
       /\
      /E2E\     10% - Full workflows
     /------\
    / Integ  \   20% - Component interaction
   /----------\
  /    Unit    \ 70% - Pure functions
 /--------------\
```

### RL-Optimized Test Pyramid

```
           /\
          /E2E\         5% - Full training pipelines
         /------\
        / Perf   \      5% - Model quality benchmarks
       /----------\
      /Integration\ 25% - Algorithm correctness
     /--------------\
    /   Property     \ 60% - Mathematical invariants
   /------------------\
  / Traditional Unit   \ 5% - Simple utilities
 /----------------------\
```

**Key difference**: Property-based tests form the BASE (not traditional units) because RL has strong mathematical properties.

---

## Part 3: Test Categories (Detailed)

### Category 1: Property-Based Unit Tests (60%)

**Purpose**: Test mathematical invariants that must hold for ALL inputs

**Tools**: `hypothesis` (Python property-based testing)

**Test Scope**:

#### 1.1 Reward Function Properties (15 tests)

```python
# Property: Reward bounds
@given(floats(min_value=-1e6, max_value=1e6))
def test_reward_within_bounds(loss):
    """Reward must be bounded for all losses."""
    reward = compute_reward(loss_delta=loss)
    assert REWARD_MIN <= reward <= REWARD_MAX

# Property: Monotonicity
@given(floats(), floats())
def test_reward_monotonic(loss1, loss2):
    """Better loss (lower) → higher reward."""
    assume(loss1 < loss2)
    r1 = compute_reward(loss_delta=-loss1)
    r2 = compute_reward(loss_delta=-loss2)
    assert r1 >= r2  # Lower loss = higher reward

# Property: Plateau penalties
@given(integers(min_value=0, max_value=100))
def test_plateau_penalty_increases(plateau_epochs):
    """Longer plateaus → larger penalties."""
    penalty = compute_plateau_penalty(plateau_epochs)
    if plateau_epochs == 0:
        assert penalty == 0
    else:
        assert penalty > 0
```

**Coverage targets**:
- All reward computation paths (100%)
- Edge cases auto-discovered by Hypothesis
- Shrinking to minimal failing examples

---

#### 1.2 Normalization Properties (10 tests)

```python
# Property: Running mean/std convergence
@given(lists(floats(allow_nan=False), min_size=100, max_size=1000))
def test_normalization_convergence(values):
    """Normalized values should have mean≈0, std≈1."""
    normalizer = RunningMeanStd()

    for val in values:
        normalizer.update(val)

    normalized = [normalizer.normalize(v) for v in values]

    mean = sum(normalized) / len(normalized)
    variance = sum((x - mean)**2 for x in normalized) / len(normalized)

    assert abs(mean) < 0.1  # Near zero
    assert abs(variance - 1.0) < 0.2  # Near 1

# Property: Normalization is invertible
@given(floats(allow_nan=False, allow_infinity=False))
def test_normalize_denormalize_inverse(value):
    """normalize(denormalize(x)) == x."""
    normalizer = RunningMeanStd(mean=10.0, std=2.0)
    normalized = normalizer.normalize(value)
    denormalized = normalizer.denormalize(normalized)
    assert abs(value - denormalized) < 1e-6
```

---

#### 1.3 Feature Extraction Properties (12 tests)

```python
# Property: Feature dimension consistency
@given(st.data())
def test_feature_dimensions_consistent(data):
    """Features should have consistent dimensions regardless of inputs."""
    snapshot = data.draw(training_snapshots())
    telemetry = data.draw(seed_telemetries()) if snapshot.has_active_seed else None

    features_with = snapshot_to_features(snapshot, use_telemetry=True, seed_telemetry=telemetry)
    features_without = snapshot_to_features(snapshot, use_telemetry=False)

    # Dimensions must be deterministic
    assert len(features_with) == 37  # 27 base + 10 telemetry
    assert len(features_without) == 27

# Property: Telemetry enforcement
@given(training_snapshots(has_active_seed=True))
def test_telemetry_required_when_seed_active(snapshot):
    """ValueError if telemetry required but missing."""
    with pytest.raises(ValueError, match="seed_telemetry is required"):
        snapshot_to_features(snapshot, use_telemetry=True, seed_telemetry=None)

# Property: Feature bounds
@given(training_snapshots())
def test_features_within_bounds(snapshot):
    """All features should be normalized (roughly [-3, 3] for z-scores)."""
    features = snapshot_to_features(snapshot, use_telemetry=False)

    for feat in features:
        assert -10.0 < feat < 10.0  # Generous bounds (allow outliers)
```

---

#### 1.4 Action Space Properties (8 tests)

```python
# Property: Blueprint-action bijection
@given(sampled_from(list(BLUEPRINT_TO_ACTION.keys())))
def test_blueprint_action_round_trip(blueprint_id):
    """blueprint → action → blueprint should be identity."""
    action = blueprint_to_action(blueprint_id)
    recovered_blueprint = action_to_blueprint(action)
    assert recovered_blueprint == blueprint_id

# Property: Action space completeness
def test_all_blueprints_have_actions():
    """Every blueprint should map to exactly one action."""
    blueprints = {BLUEPRINT_CONV_ENHANCE, BLUEPRINT_ATTENTION,
                  BLUEPRINT_NORM, BLUEPRINT_DEPTHWISE}

    for blueprint in blueprints:
        action = blueprint_to_action(blueprint)
        assert isinstance(action, Action)

    # Verify no duplicates
    actions = [blueprint_to_action(b) for b in blueprints]
    assert len(actions) == len(set(actions))
```

---

#### 1.5 Gradient Properties (10 tests)

```python
# Property: Gradient norm is non-negative
@given(st.data())
def test_gradient_norm_nonnegative(data):
    """Gradient norms must be >= 0."""
    model = data.draw(simple_networks())
    loss = data.draw(floats(min_value=0.0, max_value=100.0))

    # Simulate backward pass
    loss.backward()

    collector = SeedGradientCollector()
    stats = collector.collect(model.parameters())

    assert stats['gradient_norm'] >= 0.0

# Property: Health score bounds
@given(st.data())
def test_gradient_health_bounded(data):
    """Health score must be in [0, 1]."""
    model = data.draw(simple_networks())
    # ... collect gradients ...

    assert 0.0 <= stats['gradient_health'] <= 1.0
```

---

### Category 2: Integration Tests (25%)

**Purpose**: Test components working together (algorithms, contracts, data flow)

**Test Scope**:

#### 2.1 Algorithm Integration (20 tests)

```python
class TestPPOIntegration:
    """PPO algorithm correctness tests."""

    def test_ppo_updates_policy(self):
        """PPO should improve policy after training."""
        # Setup
        env = MockEnvironment()
        agent = PPOAgent(state_dim=27, action_dim=7)

        # Collect episodes
        episodes = collect_episodes(env, agent, num_episodes=10)

        # Get initial policy performance
        initial_returns = [ep.total_reward for ep in episodes]

        # Train
        agent.train(episodes)

        # Collect new episodes
        new_episodes = collect_episodes(env, agent, num_episodes=10)
        new_returns = [ep.total_reward for ep in new_episodes]

        # Assert: Mean return should improve (statistical test)
        assert np.mean(new_returns) > np.mean(initial_returns) * 0.9  # Allow 10% variance

    def test_ppo_respects_clip_ratio(self):
        """PPO should clip probability ratios."""
        # ... test that clip_ratio is respected ...

    def test_ppo_value_function_convergence(self):
        """Value function should converge to true returns."""
        # ... test V(s) ≈ E[G_t | s] ...

class TestIQLIntegration:
    """IQL algorithm correctness tests."""

    def test_iql_q_learning(self):
        """IQL should learn Q-values from replay buffer."""
        # ... test Q-learning updates ...

    def test_iql_target_network_sync(self):
        """Target network should sync periodically."""
        # ... test target network updates ...
```

---

#### 2.2 Pipeline Integration (15 tests)

```python
class TestTelemetryPipeline:
    """End-to-end telemetry collection."""

    def test_gradient_collection_in_training_loop(self):
        """Gradients should be collected during training."""
        model = SimpleModel()
        collector = SeedGradientCollector()

        # Training loop
        for batch in batches:
            loss = model(batch)
            loss.backward()

            stats = collector.collect(model.parameters())

            assert 'gradient_norm' in stats
            assert stats['gradient_norm'] > 0  # Non-zero after backward

    def test_telemetry_snapshot_generation(self):
        """TrainingSignals should generate valid snapshots."""
        signals = TrainingSignals()
        signals.metrics.epoch = 10
        signals.metrics.loss = 0.5

        snapshot = create_snapshot(signals)

        assert snapshot.epoch == 10
        assert snapshot.loss == 0.5

class TestActionUnificationPipeline:
    """Cross-module action flow."""

    def test_tamiyo_to_simic_action_flow(self):
        """TamiyoDecision → ActionTaken pipeline."""
        # Tamiyo makes decision
        tamiyo = HeuristicTamiyo(HeuristicPolicyConfig())
        signals = TrainingSignals()
        decision = tamiyo.decide(signals, active_seeds=[])

        # Simic processes action
        action_taken = action_from_decision(decision)

        # Verify action consistency
        assert action_taken.action == decision.action
        assert action_taken.blueprint_id == decision.blueprint_id
```

---

### Category 3: E2E Pipeline Tests (5%)

**Purpose**: Validate full training workflows

**Strategy**: Use **smoke tests** (fast, critical paths only)

```python
class TestTrainingPipeline:
    """End-to-end training workflows."""

    @pytest.mark.slow
    def test_train_ppo_agent_smoke(self, tmp_path):
        """Can train PPO agent for 5 epochs without crashing."""
        # Setup
        config = TrainingConfig(
            epochs=5,
            batch_size=32,
            checkpoint_dir=tmp_path
        )

        # Run training
        trainer = PPOTrainer(config)
        final_metrics = trainer.train()

        # Assertions
        assert final_metrics.epoch == 5
        assert final_metrics.loss < float('inf')  # Finite loss
        assert (tmp_path / "checkpoint_epoch_5.pt").exists()

    @pytest.mark.slow
    def test_head_to_head_comparison_smoke(self):
        """Can run head-to-head comparison without crashing."""
        results = head_to_head_comparison(
            num_episodes=10,
            use_telemetry=True
        )

        assert len(results) == 10
        assert all(r.winner in ['control', 'experimental'] for r in results)
```

---

### Category 4: Performance/Regression Tests (5%)

**Purpose**: Catch performance degradation and model quality regressions

```python
class TestPerformanceRegression:
    """Model quality benchmarks."""

    def test_ppo_converges_on_toy_env(self):
        """PPO should solve CartPole in <100 episodes."""
        env = gym.make('CartPole-v1')
        agent = PPOAgent(state_dim=4, action_dim=2)

        rewards = []
        for ep in range(100):
            episode_reward = run_episode(env, agent)
            rewards.append(episode_reward)

            # Early stopping if solved
            if np.mean(rewards[-10:]) >= 195.0:
                break

        # Assert: Should converge
        assert np.mean(rewards[-10:]) >= 195.0

    @pytest.mark.benchmark
    def test_feature_extraction_performance(self, benchmark):
        """Feature extraction should be <1ms per call."""
        snapshot = create_test_snapshot()

        result = benchmark(snapshot_to_features, snapshot, use_telemetry=False)

        # Benchmark will fail if >1ms
        assert benchmark.stats.mean < 0.001  # 1ms
```

---

## Part 4: Test Data Management

### Strategy: Fixtures for Complex Objects, Factories for Simple Objects

#### Fixtures (tests/fixtures/)

```
tests/fixtures/
├── snapshots/
│   ├── early_training.json      # Epoch 1-5
│   ├── mid_training.json         # Epoch 50
│   └── converged.json            # Epoch 200
├── models/
│   ├── untrained_ppo.pt
│   ├── trained_ppo_epoch50.pt
│   └── trained_iql_epoch100.pt
├── episodes/
│   ├── successful_germination.json
│   ├── failed_cull.json
│   └── plateau_advance.json
└── telemetry/
    ├── healthy_gradients.json
    ├── vanishing_gradients.json
    └── exploding_gradients.json
```

**Usage**:
```python
@pytest.fixture
def converged_snapshot():
    """Realistic snapshot at convergence."""
    with open('tests/fixtures/snapshots/converged.json') as f:
        return TrainingSnapshot(**json.load(f))
```

---

#### Factories (Hypothesis Strategies)

```python
# tests/strategies.py
from hypothesis import strategies as st
from hypothesis.strategies import composite

@composite
def training_snapshots(draw, has_active_seed=None):
    """Generate random but valid TrainingSnapshots."""
    return TrainingSnapshot(
        epoch=draw(st.integers(min_value=0, max_value=1000)),
        loss=draw(st.floats(min_value=0.0, max_value=10.0)),
        plateau_epochs=draw(st.integers(min_value=0, max_value=50)),
        has_active_seed=draw(st.booleans()) if has_active_seed is None else has_active_seed,
        # ... other fields ...
    )

@composite
def seed_telemetries(draw):
    """Generate random but valid SeedTelemetry."""
    return SeedTelemetry(
        gradient_norm=draw(st.floats(min_value=0.0, max_value=100.0)),
        gradient_health=draw(st.floats(min_value=0.0, max_value=1.0)),
        loss_delta=draw(st.floats(min_value=-10.0, max_value=10.0)),
        # ... other fields ...
    )
```

---

### Model Checkpoints

**Challenge**: Models are large (100MB+), can't commit to git

**Solution**: Use pytest fixtures with lazy loading

```python
@pytest.fixture(scope="session")
def trained_ppo_model(tmp_path_factory):
    """Train a small PPO model once per test session."""
    cache_dir = tmp_path_factory.mktemp("models")
    checkpoint_path = cache_dir / "ppo_trained.pt"

    if checkpoint_path.exists():
        # Load cached model
        return torch.load(checkpoint_path)

    # Train minimal model (fast: 30 seconds)
    model = train_minimal_ppo(epochs=10)
    torch.save(model.state_dict(), checkpoint_path)

    return model
```

---

## Part 5: Quality Metrics & Thresholds

### Metric Dashboard

| Metric | Target | Alert Threshold | Action |
|--------|--------|-----------------|--------|
| **Test pass rate** | >99% | <98% | Fix flaky tests immediately |
| **Property test coverage** | 80% of mathematical properties | <70% | Add missing properties |
| **Integration test coverage** | 100% of algorithm paths | <90% | Add missing algorithm tests |
| **Build time** | <10 min | >15 min | Parallelize or optimize |
| **Flakiness rate** | <0.5% | >2% | Use seed-based RNG, not random |
| **Model quality (regression)** | Baseline ±5% | >10% degradation | Investigate model changes |

---

### Coverage Strategy

**NOT using traditional line coverage** (meaningless for RL)

**Instead**:
1. **Property coverage**: % of mathematical invariants tested
2. **Algorithm coverage**: % of RL update rules validated
3. **Integration coverage**: % of cross-module contracts tested
4. **Regression coverage**: % of known bugs with regression tests

---

## Part 6: CI/CD Pipeline

### Progressive Testing Strategy

```yaml
# .github/workflows/test-pipeline.yml

name: Test Pipeline

on: [pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - run: ruff check src/
      - run: mypy src/
    # Target: <30s

  property-tests:
    runs-on: ubuntu-latest
    steps:
      - run: pytest tests/properties/ -v
    # Target: <2 min (Hypothesis generates fast)

  unit-tests:
    needs: lint
    runs-on: ubuntu-latest
    steps:
      - run: pytest tests/unit/ -v --cov=src --cov-report=json
      - run: |
          COVERAGE=$(jq '.totals.percent_covered' coverage.json)
          if (( $(echo "$COVERAGE < 80" | bc -l) )); then
            exit 1
          fi
    # Target: <3 min

  integration-tests:
    needs: unit-tests
    runs-on: ubuntu-latest
    steps:
      - run: pytest tests/integration/ -v
    # Target: <5 min

  e2e-smoke:
    needs: integration-tests
    runs-on: ubuntu-latest
    steps:
      - run: pytest tests/e2e/ -v -m "not slow"
    # Target: <3 min

  # Total PR pipeline: <15 min

on:
  schedule:
    - cron: '0 2 * * *'  # Nightly

jobs:
  full-e2e:
    runs-on: ubuntu-latest
    steps:
      - run: pytest tests/e2e/ -v  # All E2E tests
      - run: pytest tests/benchmarks/ -v  # Performance regression
    # Target: <60 min
```

---

## Part 7: Folder Structure

```
tests/
├── conftest.py                 # Shared fixtures
├── strategies.py               # Hypothesis strategies
├── fixtures/                   # Static test data
│   ├── snapshots/
│   ├── models/
│   ├── episodes/
│   └── telemetry/
├── properties/                 # Property-based tests (60%)
│   ├── test_reward_properties.py
│   ├── test_normalization_properties.py
│   ├── test_feature_properties.py
│   ├── test_action_properties.py
│   └── test_gradient_properties.py
├── integration/                # Integration tests (25%)
│   ├── test_ppo_integration.py
│   ├── test_iql_integration.py
│   ├── test_telemetry_pipeline.py
│   └── test_action_unification.py
├── e2e/                        # End-to-end tests (5%)
│   ├── test_training_pipeline.py
│   └── test_comparison_pipeline.py
├── benchmarks/                 # Performance tests (5%)
│   ├── test_performance_regression.py
│   └── test_model_quality.py
└── unit/                       # Traditional unit tests (5%)
    └── test_utilities.py
```

---

## Part 8: Migration Plan

### Phase 1: Foundation (Week 1-2)

**Goal**: Set up infrastructure

- [x] Install Hypothesis (`pip install hypothesis`)
- [ ] Create `tests/strategies.py` with core Hypothesis strategies
- [ ] Create `tests/conftest.py` with shared fixtures
- [ ] Set up pytest configuration (`pytest.ini`)
- [ ] Add property tests for reward computation (15 tests)
- [ ] Add property tests for normalization (10 tests)

**Deliverable**: 25 property tests passing

---

### Phase 2: Core Properties (Week 3-4)

**Goal**: Test all mathematical invariants

- [ ] Feature extraction properties (12 tests)
- [ ] Action space properties (8 tests)
- [ ] Gradient properties (10 tests)
- [ ] Add Hypothesis to CI/CD

**Deliverable**: 60 property tests passing, CI running <5 min

---

### Phase 3: Integration (Week 5-6)

**Goal**: Test algorithm correctness

- [ ] PPO integration tests (10 tests)
- [ ] IQL integration tests (10 tests)
- [ ] Telemetry pipeline tests (5 tests)
- [ ] Create model fixtures (trained checkpoints)

**Deliverable**: 85 tests total (60 property + 25 integration)

---

### Phase 4: E2E & Performance (Week 7-8)

**Goal**: Add critical path smoke tests

- [ ] Training pipeline smoke tests (3 tests)
- [ ] Comparison pipeline smoke tests (2 tests)
- [ ] Performance regression benchmarks (3 tests)
- [ ] Add nightly CI job for slow tests

**Deliverable**: 93 tests total, full CI/CD pipeline

---

### Phase 5: Quality Gates (Week 9-10)

**Goal**: Enforce quality standards

- [ ] Set up coverage reporting (property coverage)
- [ ] Add quality gates to CI (fail if <80% property coverage)
- [ ] Create quality dashboard (Grafana or equivalent)
- [ ] Document testing guidelines

**Deliverable**: Automated quality enforcement, documented strategy

---

## Part 9: Success Criteria

### After Migration

**Quantitative**:
- 93+ tests (60 property, 25 integration, 8 E2E+perf)
- Property coverage >80% (mathematical invariants)
- Integration coverage >90% (algorithm paths)
- CI pipeline <15 min on PR
- Flakiness rate <0.5% (seed-based RNG)
- Zero production escapes from untested invariants

**Qualitative**:
- New features require property tests for invariants
- Bugs discovered via Hypothesis shrinking
- Developers trust test suite (no manual verification)
- CI catches regressions before merge

---

## Part 10: Anti-Patterns to Avoid

### ❌ Testing Stochastic Outputs Deterministically

**Bad**:
```python
def test_policy_output():
    policy = Policy()
    action = policy.sample([1.0, 0.0, 0.0])
    assert action == 2  # FLAKY! Policy is stochastic
```

**Good**:
```python
@given(st.data())
def test_policy_output_distribution(data):
    """Policy should sample from valid action space."""
    policy = Policy()
    state = data.draw(states())

    action = policy.sample(state)

    # Test property, not exact value
    assert action in range(policy.action_dim)
```

---

### ❌ Not Using Seeds for Reproducibility

**Bad**:
```python
def test_training():
    model = Model()  # Random init
    loss = model.train(data)
    assert loss < 0.5  # FLAKY! Random init causes variance
```

**Good**:
```python
def test_training():
    torch.manual_seed(42)  # Deterministic
    model = Model()
    loss = model.train(data)
    assert loss < 0.5  # Reproducible
```

---

### ❌ Ignoring Numerical Precision

**Bad**:
```python
def test_gradient_norm():
    assert gradient_norm == 1.0  # FAILS due to float precision
```

**Good**:
```python
def test_gradient_norm():
    assert abs(gradient_norm - 1.0) < 1e-6  # Tolerance
```

---

## Conclusion

This test suite design embraces the **mathematical rigor** of RL systems while handling **inherent stochasticity**. By prioritizing property-based testing (60%) over traditional unit tests, we test invariants that must hold across infinite inputs, not just specific examples.

**Key innovations**:
1. Property-based testing for mathematical invariants
2. Statistical assertions for stochastic outputs
3. Model fixture caching for performance
4. Separate property/integration/e2e categories
5. Quality metrics focused on invariant coverage, not line coverage

**Next steps**:
1. Review this design with team
2. Approve migration plan
3. Execute Phase 1 (Foundation)
4. Iterate based on learnings

---

**References**:
- Test Automation Architecture (test-automation-architecture.md)
- Property-Based Testing (property-based-testing.md)
- Test Data Management (test-data-management.md)
- Quality Metrics & KPIs (quality-metrics-and-kpis.md)
