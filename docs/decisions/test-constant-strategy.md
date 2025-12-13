# Test Constants Strategy

This document explains why certain test files use hardcoded values that differ from leyline constants. These differences are **intentional** and should not be "fixed" to match leyline defaults.

## Philosophy

Tests need to verify behavior at **boundary conditions**, not just default configurations. Using the same constants everywhere would only test the happy path.

## Intentionally Different Test Values

### 1. STRICT/LENIENT Fixtures (`tests/tamiyo/conftest.py`)

Two test fixtures provide extreme configurations for boundary testing:

| Constant | STRICT Value | Default | LENIENT Value | Purpose |
|----------|--------------|---------|---------------|---------|
| `plateau_epochs_to_germinate` | 1 | 3 | 10 | Test immediate vs delayed germination |
| `min_epochs_before_germinate` | 0 | 5 | 20 | Test early vs late germination gates |
| `cull_after_epochs_without_improvement` | 1 | 5 | 10 | Test aggressive vs patient culling |
| `cull_if_accuracy_drops_by` | 0.5% | 2.0% | 5.0% | Test sensitive vs tolerant drop detection |
| `embargo_epochs_after_cull` | 0 | 5 | 10 | Test no embargo vs long embargo |
| `blueprint_penalty_on_cull` | 1.0 | 2.0 | 3.0 | Test weak vs strong penalties |
| `blueprint_penalty_threshold` | 5.0 | 3.0 | 10.0 | Test low vs high penalty tolerance |
| `stabilization_threshold` | 0.01 | 0.03 | 0.10 | Test tight vs loose convergence |
| `stabilization_epochs` | 5 | 3 | 2 | Test long vs short stability windows |

**Why keep these different:** These fixtures allow property-based tests to explore the full behavior space. A test using `STRICT` config should trigger behaviors immediately; `LENIENT` should rarely trigger them.

### 2. GAE Parameters (`tests/simic/test_tamiyo_buffer.py`)

```python
buffer.compute_advantages_and_returns(gamma=0.99, gae_lambda=0.95)
```

| Parameter | Test Value | Leyline Default | Why Different |
|-----------|------------|-----------------|---------------|
| `gamma` | 0.99 | 0.995 | Tests buffer math with different discount |
| `gae_lambda` | 0.95 | 0.97 | Tests advantage estimation with different bias-variance tradeoff |

**Why keep these different:** The buffer must work correctly with *any* valid GAE parameters, not just the defaults. Using different values ensures the buffer doesn't accidentally depend on specific constants.

### 3. Dropout in Transformer Tests (`tests/integration/test_transformer_integration.py`)

```python
host = TransformerHost(..., dropout=0.0)
```

| Parameter | Test Value | Leyline Default | Why Different |
|-----------|------------|-----------------|---------------|
| `dropout` | 0.0 | 0.1 | Deterministic behavior for reproducible tests |

**Why keep this different:** Dropout introduces randomness. Setting it to 0.0 makes tests deterministic and reproducible. This is standard practice for unit/integration tests.

### 4. Learning Rate in Integration Tests (`tests/integration/test_tamiyo_tolaria.py`)

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
```

| Parameter | Test Value | Leyline Default | Why Different |
|-----------|------------|-----------------|---------------|
| `lr` | 0.01 | 0.001 (host) | Faster convergence for quick tests |

**Why keep this different:** Integration tests don't need production-grade training convergence. A 10x higher learning rate speeds up tests while still exercising the training loop.

### 5. Property-Based Test Strategies (`tests/*/strategies/*.py`)

Hypothesis strategies generate values across ranges:

```python
blueprint_penalty_on_cull=draw(bounded_floats(0.5, 5.0))
blueprint_penalty_decay=draw(bounded_floats(0.1, 0.9))
```

**Why use generated ranges:** Property-based testing explores the parameter space to find edge cases. Fixed defaults would miss bugs that only appear at extreme values.

### 6. Tensor Shape Tests (`tests/simic/test_tamiyo_buffer.py`)

```python
hidden_h=torch.zeros(1, 1, 128)  # Shape: (layers, batch, hidden_dim)
```

**Why keep hardcoded:** These are structural tensor dimensions that must match the model architecture. They're not configuration constants; they're mathematical requirements of LSTM cell state shapes.

## When TO Update Test Constants

Update test hardcoded values when:

1. **Tests verify defaults:** A test that asserts `config.value == X` should import the constant from leyline
2. **Tests drift from reality:** If the test's hardcoded value no longer makes sense given codebase changes
3. **Single source of truth matters:** When a value change in leyline should automatically update the test expectation

## When NOT TO Update Test Constants

Keep test-specific values when:

1. **Testing boundaries:** STRICT/LENIENT fixtures intentionally differ
2. **Testing flexibility:** The code should work with many valid parameter values
3. **Determinism required:** Setting dropout=0, temperature=0, etc.
4. **Speed matters:** Higher learning rates for faster integration tests
5. **Mathematical constraints:** Tensor shapes, enum sizes, etc.

---

*Last updated: 2025-12-14*
*Related: leyline/__init__.py (source of truth for defaults)*
