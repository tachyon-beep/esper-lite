"""Property-based tests for simic reward functions.

These tests verify invariants that must hold for ALL valid inputs,
not just specific examples. Property-based testing uses Hypothesis to
generate hundreds of diverse test cases, ensuring comprehensive coverage
of the input space.

## Test Organization

The property tests are organized into five tiers, from fundamental
mathematical properties to high-level warning signals:

### Tier 1: Mathematical Invariants (test_reward_invariants.py)
Fundamental mathematical properties that MUST hold for any valid reward function:
- Scale invariance (rewards scale consistently with episode length)
- Boundedness (rewards stay within valid ranges)
- Terminal state consistency (terminal states have zero potential)
- Monotonicity (curriculum rewards increase with progress)

### Tier 2: Semantic Invariants (test_reward_semantics.py)
Domain-specific properties that ensure rewards match the problem semantics:
- Solved state detection (completion rewards only for actually solved episodes)
- Partial progress recognition (step rewards reflect real progress)
- Efficiency rewards (time penalties for longer solutions)
- Curriculum phase ordering (valid transitions between phases)

### Tier 3: Anti-Gaming Properties (test_reward_antigaming.py)
Properties that prevent agents from exploiting reward function loopholes:
- No trivial loop exploitation (repetitive actions don't accumulate unbounded rewards)
- No false completion rewards (agents can't claim completion without solving)
- No checkpoint camping (staying in curriculum phases doesn't yield infinite rewards)
- Curriculum phase monotonicity (agents can't farm rewards by phase regression)

### Tier 4: PBRS Properties (test_pbrs_properties.py)
Potential-Based Reward Shaping (PBRS) guarantees:
- Policy invariance (shaping doesn't change optimal policy)
- Telescoping sum property (shaping contributions cancel along trajectories)
- Terminal state zero potential (shaping vanishes at episode boundaries)
- Reward decomposition (total reward = base + shaping, with known contributions)

### Tier 5: Warning Signals (test_warning_signals.py)
Statistical properties that indicate potential training issues:
- Reward variance (high variance may cause training instability)
- Reward sparsity (too many zero rewards hinder learning)
- Reward magnitude (extreme values may cause numerical issues)
- Curriculum phase distribution (imbalanced phase coverage may indicate problems)

## Running the Tests

Run all property tests:
```bash
pytest tests/simic/properties/ -m property -v
```

Run with CI profile (more examples, slower but more thorough):
```bash
HYPOTHESIS_PROFILE=ci pytest tests/simic/properties/ -m property -v
```

Run a specific tier:
```bash
pytest tests/simic/properties/test_reward_invariants.py -m property -v
pytest tests/simic/properties/test_reward_semantics.py -m property -v
pytest tests/simic/properties/test_reward_antigaming.py -m property -v
pytest tests/simic/properties/test_pbrs_properties.py -m property -v
pytest tests/simic/properties/test_warning_signals.py -m property -v
```

## Interpreting Results

### Failures
When a property test fails, Hypothesis will:
1. Show the failing example (the specific input that violated the property)
2. Attempt to shrink it to the minimal failing case
3. Save it for replay in `.hypothesis/examples/`

This makes debugging much easier than traditional example-based tests.

### Warnings
Tier 5 tests may emit warnings rather than failures. These indicate
potential issues that should be investigated but don't necessarily
invalidate the reward function.

### Performance
Property tests run many examples (default 100, CI profile 1000).
Expect them to take longer than unit tests. This is intentional and
provides much stronger correctness guarantees.
"""
