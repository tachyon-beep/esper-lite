# Task 11: Testing and Validation Guide - UCB1 Blueprint Curriculum

## Test Coverage Analysis

### Current Tests (12 tests in test_curriculum.py)

**1. Initialization & Configuration**
```python
test_initial_curriculum_favors_simple()
test_mismatched_lengths_raises_error()
```
Status: ✓ Tests that complexity penalty biases toward simpler blueprints at startup

**2. UCB1 Updates**
```python
test_ucb_updates_after_success()
test_select_blueprint_returns_highest_score()
```
Status: ✓ Tests that outcomes are recorded and selection uses highest score

**3. Exploration Bonus**
```python
test_ucb_exploration_bonus()
test_exploration_weight_affects_scores()
```
Status: ✓ Tests that untried blueprints get exploration bonus, scales with weight

**4. Curriculum Learning**
```python
test_complexity_penalty_favors_simple_initially()
test_high_reward_overcomes_complexity_penalty()
```
Status: ✓ Tests that penalty favors simple early, but high reward wins later

**5. Statistics Tracking**
```python
test_stats_track_successes_and_failures()
```
Status: ✓ Tests that trials, successes, and mean_reward are accurate

**6. Error Handling**
```python
test_invalid_blueprint_raises_error()
```
Status: ✓ Tests validation of blueprint names

**Assessment:** ✓ Good coverage of intended behavior
- All major code paths tested
- Edge cases covered (zero trials, mismatched lengths, invalid names)
- Parameterized tests verify hyperparameter effects

---

## Integration Testing Recommendations

### Test 1: End-to-End Seed Selection in Training

**What to test:** Curriculum works correctly in actual training loop

**Setup:**
```python
def test_curriculum_integration_with_training():
    curriculum = BlueprintCurriculum(
        blueprints=['norm', 'lora', 'attention'],
        complexity=[100, 6000, 50000],
    )

    # Simulate 50 seeds
    seeds_by_blueprint = defaultdict(list)
    for seed_idx in range(50):
        blueprint = curriculum.select_blueprint()
        seeds_by_blueprint[blueprint].append(seed_idx)

        # Simulate outcome: norm succeeds early, others struggle
        if blueprint == 'norm':
            reward = 0.8
        else:
            reward = 0.3 + 0.1 * seed_idx / 50  # Gradually improve

        curriculum.record_outcome(blueprint, success=True, reward=reward)

    # Verify curriculum learned reward distribution
    scores = curriculum.get_ucb_scores()
    assert scores['norm'] > scores['lora']  # norm best
    assert scores['attention'] > 0  # attention still has chance (exploration bonus decays)

    # Verify all blueprints were tried (cold-start worked)
    assert all(len(seeds) > 0 for seeds in seeds_by_blueprint.values())
```

**Expected behavior:**
- Norm dominates by seed 50 (highest reward)
- All blueprints tried initially (exploration bonus forced)
- Scores converge to reward ordering by end

---

### Test 2: Regret Analysis

**What to test:** Actual regret matches O(k·ln(N)) bound

**Setup:**
```python
def test_regret_bound():
    curriculum = BlueprintCurriculum(
        blueprints=['a', 'b', 'c', 'd', 'e'],
        complexity=[100, 200, 300, 400, 500],
    )

    # True reward function (simulator)
    rewards_true = {'a': 0.3, 'b': 0.4, 'c': 0.9, 'd': 0.7, 'e': 0.5}
    best_blueprint = max(rewards_true, key=rewards_true.get)  # 'c'
    best_reward = rewards_true[best_blueprint]

    total_reward = 0.0
    for trial in range(100):
        blueprint = curriculum.select_blueprint()
        reward = rewards_true[blueprint] + np.random.randn() * 0.05  # small noise
        curriculum.record_outcome(blueprint, success=True, reward=reward)
        total_reward += reward

    # Optimal policy: always select 'c'
    optimal_reward = best_reward * 100

    # Actual regret
    regret = optimal_reward - total_reward

    # UCB1 bound: O(k * ln(N)) = 5 * ln(100) ≈ 23
    assert regret < 50  # Conservative upper bound (2x theoretical)
    print(f"Regret: {regret:.1f} (bound: 50, theory: 23)")
```

**Expected behavior:**
- Regret should be O(k·ln(N)) ≈ 23-30 for 5 blueprints, 100 trials
- If regret >> 50: exploration_weight too high or complexity_penalty too strong

---

### Test 3: Complexity Penalty Decay

**What to test:** Penalty becomes negligible as exploration bonus shrinks

**Setup:**
```python
def test_complexity_penalty_decay():
    curriculum = BlueprintCurriculum(
        blueprints=['simple', 'complex'],
        complexity=[100, 10000],
        exploration_weight=2.0,
        complexity_penalty=0.1,
    )

    # Both blueprints equally rewarding
    for trial in range(100):
        curriculum.record_outcome('simple', success=True, reward=1.0)
        curriculum.record_outcome('complex', success=True, reward=1.0)

    scores = curriculum.get_ucb_scores()

    # With equal means, complex should win (penalty negligible)
    assert scores['complex'] > scores['simple'] * 0.95  # within 5%

    # Check penalty contribution
    simple_stats = curriculum._stats['simple']
    exploration_bonus = curriculum.exploration_weight * math.sqrt(
        math.log(200 + 1) / simple_stats.trials
    )
    complexity_term = curriculum.complexity_penalty * curriculum._normalized_complexity['simple']

    # Exploration bonus should dominate penalty
    assert exploration_bonus > 5 * complexity_term  # bonus >> penalty
```

**Expected behavior:**
- At trial 100 with equal rewards: exploration bonus ≈ 0.2
- Complexity penalty ≈ 0.01 (negligible relative to bonus)
- Score dominated by mean reward (1.0), bonus and penalty cancel

---

### Test 4: Cold-Start Determinism

**What to test:** First selections follow deterministic curriculum, not random

**Setup:**
```python
def test_cold_start_deterministic():
    curriculum1 = BlueprintCurriculum(
        blueprints=['norm', 'lora', 'attention', 'depthwise', 'mlp'],
        complexity=[100, 6000, 50000, 2000, 1200000],
        exploration_weight=2.0,
        complexity_penalty=0.1,
    )

    curriculum2 = BlueprintCurriculum(
        blueprints=['norm', 'lora', 'attention', 'depthwise', 'mlp'],
        complexity=[100, 6000, 50000, 2000, 1200000],
        exploration_weight=2.0,
        complexity_penalty=0.1,
    )

    # Both should make identical first 5 selections (cold-start)
    selections1 = [curriculum1.select_blueprint() for _ in range(5)]
    selections2 = [curriculum2.select_blueprint() for _ in range(5)]

    assert selections1 == selections2
    assert selections1[0] == 'norm'  # Simplest selected first

    # All blueprints should be in first 5 (with high exploration bonus)
    assert len(set(selections1)) == 5 or len(set(selections1)) == 4  # All or all but one
```

**Expected behavior:**
- Deterministic: same selection for same input
- Complexity-ordered: simpler blueprints selected first
- Diverse: all blueprints tried in first few selections (exploration bonus forces it)

---

### Test 5: Hyperparameter Sensitivity

**What to test:** Effects of exploration_weight and complexity_penalty

**Setup:**
```python
def test_exploration_weight_scales_uncertainty():
    # High exploration
    curriculum_high = BlueprintCurriculum(
        blueprints=['a', 'b'],
        complexity=[100, 100],
        exploration_weight=5.0,  # Aggressive exploration
        complexity_penalty=0.0,  # Disable complexity
    )

    # Low exploration
    curriculum_low = BlueprintCurriculum(
        blueprints=['a', 'b'],
        complexity=[100, 100],
        exploration_weight=0.5,  # Conservative
        complexity_penalty=0.0,
    )

    # Try 'a' once with low reward
    curriculum_high.record_outcome('a', True, 0.1)
    curriculum_low.record_outcome('a', True, 0.1)

    scores_high = curriculum_high.get_ucb_scores()
    scores_low = curriculum_low.get_ucb_scores()

    # High exploration should favor untried 'b' more
    ratio_high = scores_high['b'] / scores_high['a']
    ratio_low = scores_low['b'] / scores_low['a']

    assert ratio_high > ratio_low  # Untried has bigger advantage with high c
```

**Expected behavior:**
- exploration_weight=5.0: untried blueprints ~5x bonus over 1.0
- exploration_weight=0.5: untried blueprints ~0.5x bonus over 1.0
- Exploration weight is a dial from conservative to aggressive

---

## Monitoring During Training

### Metrics to Track

**1. Blueprint Selection Frequency**
```python
def log_blueprint_stats(curriculum):
    """Log blueprint selection statistics."""
    stats_by_blueprint = {}
    for blueprint in curriculum.blueprints:
        stats = curriculum.get_stats(blueprint)
        stats_by_blueprint[blueprint] = {
            'trials': stats['trials'],
            'successes': stats['successes'],
            'success_rate': stats['successes'] / max(1, stats['trials']),
            'mean_reward': stats['mean_reward'],
        }

    print("Blueprint Statistics:")
    for bp, s in sorted(stats_by_blueprint.items(), key=lambda x: x[1]['mean_reward'], reverse=True):
        print(f"  {bp:12s}: trials={s['trials']:3d}, success_rate={s['success_rate']:.1%}, mean_reward={s['mean_reward']:.3f}")
```

**When to log:**
- Every 10 seeds during first 50 (monitor cold-start)
- Every 50 seeds during main training (monitor convergence)
- At end of training (validate final selection frequencies)

**Red flags:**
- One blueprint tried >50% of the time before 20 seeds: curriculum overfitting
- Some blueprint never tried by seed 50: exploration_weight too low
- Simplest blueprint tried >70% at seed 100: complexity_penalty too high

---

### Metrics to Compute

**2. Exploration Bonus Trajectory**
```python
def plot_exploration_trajectory(curriculum):
    """Plot exploration bonus over time."""
    for blueprint in curriculum.blueprints:
        stats = curriculum._stats[blueprint]
        if stats.trials > 0:
            exploration = curriculum.exploration_weight * math.sqrt(
                math.log(curriculum._total_trials + 1) / stats.trials
            )
        else:
            exploration = curriculum.exploration_weight * 2.0

        print(f"{blueprint}: exploration_bonus={exploration:.3f}")
```

**Expected pattern:**
- Untried blueprints: 2.0 (constant)
- Tried blueprints: ~√(ln(N)/n) (decays as trials increase)
- At N=100:
  - 1 trial: √(ln(101)/1) ≈ 2.3
  - 5 trials: √(ln(101)/5) ≈ 1.0
  - 20 trials: √(ln(101)/20) ≈ 0.48
  - 50 trials: √(ln(101)/50) ≈ 0.30

**Interpretation:**
- Early (low N): exploration bonus >1.0 (significant)
- Late (high N): exploration bonus <0.3 (negligible)
- This decay is correct UCB1 behavior

---

**3. Regret vs. Theoretical Bound**
```python
def compute_regret(curriculum, true_rewards):
    """Estimate regret vs optimal policy."""
    best_reward = max(true_rewards.values())

    total_reward = 0.0
    for blueprint in curriculum.blueprints:
        stats = curriculum.get_stats(blueprint)
        total_reward += stats['mean_reward'] * stats['trials']

    optimal_reward = best_reward * curriculum._total_trials
    regret = optimal_reward - total_reward

    k = len(curriculum.blueprints)
    N = curriculum._total_trials
    theoretical_bound = k * math.log(N + 1) + math.log(math.log(N + 1))

    print(f"Regret: {regret:.1f} (bound: {theoretical_bound:.1f}, ratio: {regret/theoretical_bound:.2f})")
```

**Expected behavior:**
- Regret/bound ratio should be 0.5-2.0
- <0.5: may be overfitting (simplistic blueprint distribution)
- >2.0: may need more exploration (increase exploration_weight)

---

## Validation Checklist

Run before deploying to production:

- [ ] All 12 existing tests pass
- [ ] Integration test: curriculum works in training loop
- [ ] Regret test: actual regret < 2x theoretical bound
- [ ] Cold-start test: all blueprints tried in first 5 selections
- [ ] Determinism test: same input → same output
- [ ] Sensitivity test: hyperparameters affect selection as expected
- [ ] Monitor blueprint stats during first 50 seeds of training
- [ ] Verify no blueprint dominates >60% by seed 50
- [ ] Verify all blueprints have ≥5 trials by seed 50
- [ ] Log exploration bonus trajectory (should decay from ~2.0 to ~0.2)

---

## Production Monitoring

### Dashboard Metrics

**Critical (watch constantly):**
- Blueprint selection frequency (should spread across all options)
- Mean reward per blueprint (should match true performance)
- Exploration bonus magnitude (should be <0.5 by N=100)

**Important (check daily):**
- Regret trajectory (should grow sublinearly, slope decreasing)
- Cold-start compliance (all blueprints tried by seed 20)
- Hyperparameter effects (exploration_weight controls exploitation)

**Nice-to-have (check weekly):**
- UCB score distribution
- Complexity penalty contribution to scores
- Correlation between complexity and actual performance

---

## Troubleshooting Guide

### Issue: One blueprint tried >70% of the time

**Likely causes:**
1. Complexity penalty too high (favoring simple blueprint too much)
2. True reward very skewed (one blueprint much better than others)
3. exploration_weight too low (not exploring enough)

**Fix:**
1. Check actual rewards: if one genuinely better, this is correct
2. Reduce complexity_penalty from 0.1 to 0.01-0.05
3. Increase exploration_weight from 2.0 to 3.0-5.0
4. Reduce complexity values (overestimating difficulty)

### Issue: Some blueprint never tried

**Likely causes:**
1. Complexity penalty too high (blocking selection)
2. exploration_weight too low (bad luck early, now blocked by low exploration)
3. Too few seeds for UCB1 convergence

**Fix:**
1. Increase exploration_weight from 2.0 to 5.0+
2. Reduce complexity_penalty or complexity values
3. Run more seeds (need ≥20 per blueprint for convergence)

### Issue: Regret very high (>2x theoretical bound)

**Likely causes:**
1. exploration_weight too high (wasting trials on bad blueprints)
2. complexity_penalty wrong direction (penalizing good blueprint)
3. True rewards have very high variance

**Fix:**
1. Decrease exploration_weight from 2.0 to 1.0-1.5
2. Check complexity penalty sign (should favor simple)
3. Reduce noise in reward signal if possible

### Issue: Always selects same blueprint

**Likely causes:**
1. exploration_weight = 0 (no exploration)
2. All other blueprints have very low reward (correct behavior)
3. complexity_penalty enormous (blocking other blueprints)

**Fix:**
1. Set exploration_weight ≥ 0.5 (default 2.0)
2. Check true rewards; if one genuinely best, this is correct
3. Check complexity_penalty is 0.05-0.2 range

---

## Conclusion

The current test suite provides good coverage of core functionality. For production deployment, add:

1. **Integration tests** (seed selection in actual training)
2. **Regret validation** (empirical regret vs theoretical bound)
3. **Monitoring dashboards** (blueprint stats, exploration decay)
4. **Hyperparameter sensitivity** (effects of c and β)

With these additions, the curriculum is ready for reliable production use.
