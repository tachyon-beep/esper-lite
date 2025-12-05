# Kasmina Audit Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix critical bugs, improve PyTorch performance, and enhance RL observation quality identified by the three-specialist audit.

**Architecture:** Targeted fixes to kasmina (host.py, slot.py, isolation.py, blueprints/cnn.py) and simic (features.py, rewards.py) with minimal changes to working code. Each fix is independent and can be committed separately.

**Tech Stack:** Python 3.11+, PyTorch 2.8+, pytest, hypothesis

---

## Phase 1: Critical Bugs (MUST FIX)

### Task 1: Fix MorphogeneticModel.to() Device Handling

**Files:**
- Modify: `src/esper/kasmina/host.py:280-292`
- Test: `tests/test_seed_slot.py` (add new test)

**Issue:** Race condition and redundant seed transfer. The current code calls `super().to()` which already moves all submodules, then tries to move the seed again. Also uses StopIteration exception for control flow.

**Step 1: Write the failing test**

```python
# Add to tests/test_seed_slot.py

def test_morphogenetic_model_to_device_consistency():
    """Verify device transfer doesn't cause inconsistencies."""
    from esper.kasmina.host import CNNHost, MorphogeneticModel

    host = CNNHost(num_classes=10)
    model = MorphogeneticModel(host, device="cpu")

    # Germinate a seed
    model.germinate_seed("norm", "test-seed")
    assert model.seed_slot.seed is not None

    # Transfer to CPU (no-op but exercises the code path)
    model = model.to("cpu")

    # Verify consistency
    assert str(model._device) == "cpu"
    assert model.seed_slot.device == torch.device("cpu")

    # Verify seed is on correct device
    seed_param = next(model.seed_slot.seed.parameters())
    assert seed_param.device == torch.device("cpu")
```

**Step 2: Run test to verify it passes (baseline)**

Run: `PYTHONPATH=src pytest tests/test_seed_slot.py::test_morphogenetic_model_to_device_consistency -v`
Expected: PASS (current code works for CPU-to-CPU)

**Step 3: Fix the implementation**

Replace lines 280-292 in `src/esper/kasmina/host.py`:

```python
def to(self, *args, **kwargs):
    """Override to() to update device tracking after transfer.

    Note: super().to() already moves all registered submodules including
    seed_slot and its seed. We only update our device tracking string.

    Implementation note (PyTorch Expert review): Query device from parameters
    AFTER super().to() completes rather than parsing args. This is simpler,
    correct, and follows PyTorch conventions - query state after mutation
    rather than trying to parse the complex .to() signature which accepts
    device, dtype, tensor, memory_format, and non_blocking in various forms.
    """
    result = super().to(*args, **kwargs)

    # Query actual device from parameters (canonical source of truth)
    try:
        actual_device = next(self.parameters()).device
    except StopIteration:
        # No parameters - keep existing device tracking
        return result

    # Update tracking (seed already moved by super().to())
    self.seed_slot.device = actual_device
    self._device = str(actual_device)

    return result
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_seed_slot.py::test_morphogenetic_model_to_device_consistency -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/kasmina/host.py tests/test_seed_slot.py
git commit -m "fix(kasmina): simplify MorphogeneticModel.to() device handling

Remove redundant seed transfer (super().to() already handles it) and
replace StopIteration exception abuse with reliable arg parsing.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Fix AttentionSeed Zero-Division Risk

**Files:**
- Modify: `src/esper/kasmina/blueprints/cnn.py:75-95`
- Test: `tests/test_simic_networks.py` (add new test)

**Issue:** When `channels < reduction`, `channels // reduction = 0`, causing `nn.Linear(channels, 0)` which crashes.

**Step 1: Write the failing test**

```python
# Add to tests/test_simic_networks.py

def test_attention_seed_small_channels():
    """AttentionSeed should handle small channel counts gracefully."""
    from esper.kasmina.blueprints.cnn import create_attention_seed

    # This should not crash - channels=2 with reduction=4 would give 0 features
    seed = create_attention_seed(channels=2, reduction=4)

    # Verify it works
    x = torch.randn(1, 2, 8, 8)
    out = seed(x)
    assert out.shape == x.shape

    # Also test edge case: channels=1
    seed_tiny = create_attention_seed(channels=1, reduction=4)
    x_tiny = torch.randn(1, 1, 8, 8)
    out_tiny = seed_tiny(x_tiny)
    assert out_tiny.shape == x_tiny.shape
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_simic_networks.py::test_attention_seed_small_channels -v`
Expected: FAIL with RuntimeError or ValueError about 0 features

**Step 3: Fix the implementation**

Replace the `create_attention_seed` function in `src/esper/kasmina/blueprints/cnn.py`:

```python
@BlueprintRegistry.register(
    "attention", "cnn", param_estimate=2000, description="SE-style channel attention"
)
def create_attention_seed(channels: int, reduction: int = 4) -> nn.Module:
    """Channel attention seed (SE-style)."""

    class AttentionSeed(nn.Module):
        def __init__(self, channels: int, reduction: int):
            super().__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            # Ensure reduced dimension is at least 1
            reduced = max(1, channels // reduction)
            self.fc = nn.Sequential(
                nn.Linear(channels, reduced, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(reduced, channels, bias=False),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, c, _, _ = x.size()
            y = self.avg_pool(x).view(b, c)
            y = self.fc(y).view(b, c, 1, 1)
            return x * y.expand_as(x)

    return AttentionSeed(channels, reduction)
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_simic_networks.py::test_attention_seed_small_channels -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/kasmina/blueprints/cnn.py tests/test_simic_networks.py
git commit -m "fix(kasmina): prevent zero-division in AttentionSeed

Ensure reduced dimension is at least 1 when channels < reduction.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Remove Unused pickle Import

**Files:**
- Modify: `src/esper/kasmina/slot.py:15`

**Issue:** `pickle` is imported but never used. Security concern if it were used.

**Step 1: Verify import is unused**

Run: `grep -n "pickle" src/esper/kasmina/slot.py`
Expected: Only line 15 (the import)

**Step 2: Remove the import**

Delete line 15 in `src/esper/kasmina/slot.py`:

```python
# Remove this line:
import pickle
```

**Step 3: Run tests to verify no regression**

Run: `PYTHONPATH=src pytest tests/test_seed_slot.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/esper/kasmina/slot.py
git commit -m "chore(kasmina): remove unused pickle import

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3.5: Fix SeedState Telemetry Type Annotation

**Files:**
- Modify: `src/esper/kasmina/slot.py:179`

**Issue:** Type annotation says `SeedTelemetry` but default is `None`. This breaks the type contract.

**Step 1: Fix the type annotation**

Change line 179 in `src/esper/kasmina/slot.py`:

```python
# From:
telemetry: SeedTelemetry = field(default=None)

# To:
telemetry: SeedTelemetry | None = field(default=None)
```

**Step 2: Run type check**

Run: `PYTHONPATH=src python -m mypy src/esper/kasmina/slot.py --ignore-missing-imports`
Expected: No new errors

**Step 3: Commit**

```bash
git add src/esper/kasmina/slot.py
git commit -m "fix(kasmina): correct SeedState.telemetry type annotation

Type was SeedTelemetry but default is None - should be Optional.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Phase 2: PyTorch Performance Fixes

### Task 4: Optimize GradientIsolationMonitor CUDA Sync

**Files:**
- Modify: `src/esper/kasmina/isolation.py:80-109`
- Test: `tests/test_seed_slot.py` (add benchmark)

**Issue:** Multiple `.item()` calls cause CUDA synchronization per parameter. Should accumulate on GPU and sync once.

**Step 1: Write the test**

```python
# Add to tests/test_seed_slot.py

def test_gradient_isolation_monitor_batch_sync():
    """Verify check_isolation works correctly with batched computation."""
    from esper.kasmina.isolation import GradientIsolationMonitor

    monitor = GradientIsolationMonitor()

    # Create simple modules
    host = torch.nn.Linear(10, 10)
    seed = torch.nn.Linear(10, 10)

    monitor.register(host, seed)

    # Simulate gradients
    for p in host.parameters():
        p.grad = torch.randn_like(p) * 0.01
    for p in seed.parameters():
        p.grad = torch.randn_like(p)

    is_isolated, metrics = monitor.check_isolation()

    # Should detect non-zero host gradients
    assert not is_isolated
    assert metrics["host_grad_norm"] > 0
    assert metrics["seed_grad_norm"] > 0
    assert metrics["violations"] == 1
```

**Step 2: Run test to verify baseline**

Run: `PYTHONPATH=src pytest tests/test_seed_slot.py::test_gradient_isolation_monitor_batch_sync -v`
Expected: PASS

**Step 3: Optimize the implementation**

Replace `check_isolation` method in `src/esper/kasmina/isolation.py`:

```python
@torch.no_grad()
def check_isolation(self) -> tuple[bool, dict]:
    """Check if gradient isolation is maintained.

    Uses batched norm computation to minimize CUDA synchronization points.
    Reduces from O(n_params) CUDA syncs to O(1) by accumulating on GPU.

    Implementation note (PyTorch Expert review): For maximum performance on
    large models, consider using torch._foreach_norm (used internally by
    clip_grad_norm_). The current approach is correct and sufficient for
    typical model sizes.
    """
    # Collect gradients that exist
    host_grads = [p.grad for p in self._host_params if p.grad is not None]
    seed_grads = [p.grad for p in self._seed_params if p.grad is not None]

    # Compute norms with single sync per group
    if host_grads:
        # Stack squared norms, sum, sqrt - single .item() call
        host_norm_sq = sum(g.pow(2).sum() for g in host_grads)
        host_norm = host_norm_sq.sqrt().item()
    else:
        host_norm = 0.0

    if seed_grads:
        seed_norm_sq = sum(g.pow(2).sum() for g in seed_grads)
        seed_norm = seed_norm_sq.sqrt().item()
    else:
        seed_norm = 0.0

    self.host_grad_norm = host_norm
    self.seed_grad_norm = seed_norm

    is_isolated = host_norm < self.threshold

    if not is_isolated:
        self.violations += 1

    return is_isolated, {
        "host_grad_norm": host_norm,
        "seed_grad_norm": seed_norm,
        "isolated": is_isolated,
        "violations": self.violations,
    }
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_seed_slot.py::test_gradient_isolation_monitor_batch_sync -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/kasmina/isolation.py tests/test_seed_slot.py
git commit -m "perf(kasmina): batch CUDA sync in GradientIsolationMonitor

Reduce from O(n_params) CUDA syncs to O(1) by accumulating norms on GPU.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: Fix Shape Probe Cache Device Comparison

**Files:**
- Modify: `src/esper/kasmina/slot.py:546-574`

**Issue:** Uses string comparison for devices which is fragile. Should compare torch.device objects directly.

**Step 1: Write the test**

```python
# Add to tests/test_seed_slot.py

def test_shape_probe_cache_device_comparison():
    """Shape probe cache should use direct device comparison."""
    from esper.kasmina.slot import SeedSlot

    slot = SeedSlot("test", channels=64, device="cpu")

    # Get probe - should create and cache
    probe1 = slot._get_shape_probe("cnn")
    assert probe1.device == torch.device("cpu")

    # Get again - should return cached
    probe2 = slot._get_shape_probe("cnn")
    assert probe1 is probe2  # Same object

    # Different topology - should create new
    probe3 = slot._get_shape_probe("transformer")
    assert probe3.device == torch.device("cpu")
    assert probe1 is not probe3
```

**Step 2: Run test to verify baseline**

Run: `PYTHONPATH=src pytest tests/test_seed_slot.py::test_shape_probe_cache_device_comparison -v`
Expected: PASS

**Step 3: Fix the implementation**

Replace `_get_shape_probe` in `src/esper/kasmina/slot.py`:

```python
def _get_shape_probe(self, topology: str) -> torch.Tensor:
    """Get cached shape probe for topology, creating if needed."""
    cached = self._shape_probe_cache.get(topology)

    if cached is not None:
        cached_device, cached_tensor = cached
        # Use direct device comparison instead of string
        if cached_device == self.device:
            return cached_tensor

    # Create new probe for this topology/device
    if topology == "cnn":
        probe = torch.randn(
            1,
            self.channels,
            CNN_SHAPE_PROBE_SPATIAL,
            CNN_SHAPE_PROBE_SPATIAL,
            device=self.device,
        )
    else:
        probe = torch.randn(
            2,
            TRANSFORMER_SHAPE_PROBE_SEQ_LEN,
            self.channels,
            device=self.device,
        )

    # Store device as torch.device, not string
    self._shape_probe_cache[topology] = (self.device, probe)
    return probe
```

Also update the type hint for `_shape_probe_cache` in `__init__`:

```python
# Change line ~544 from:
self._shape_probe_cache: dict[str, tuple[str, torch.Tensor]] = {}
# To:
self._shape_probe_cache: dict[str, tuple[torch.device, torch.Tensor]] = {}
```

**Also normalize device in `__init__`** (PyTorch Expert requirement):

```python
# In SeedSlot.__init__, line ~528, ensure device is always torch.device:
self.device = torch.device(device) if isinstance(device, str) else device
```

**Also update `SeedSlot.to()` to only clear cache on device change** (PyTorch Expert optimization):

In the `SeedSlot.to()` method, track old device and only clear cache if it changes:

```python
def to(self, *args, **kwargs) -> "SeedSlot":
    """Transfer slot and any active seed to device."""
    old_device = self.device  # Track before move
    super().to(*args, **kwargs)

    # Update device tracking (query from parameters after move)
    try:
        actual_device = next(self.parameters()).device
        self.device = actual_device
    except StopIteration:
        # Infer from args if no parameters
        for arg in args:
            if isinstance(arg, (str, torch.device)):
                self.device = torch.device(arg) if isinstance(arg, str) else arg
                break

    # Only clear cache if device actually changed
    if self.device != old_device:
        self._shape_probe_cache.clear()

    return self
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_seed_slot.py::test_shape_probe_cache_device_comparison -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/kasmina/slot.py tests/test_seed_slot.py
git commit -m "fix(kasmina): use direct device comparison in shape probe cache

String comparison was fragile for equivalent device representations.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Phase 3: RL Observation Quality

### Task 6: Add Blueprint ID to Observation Space (One-Hot Encoding)

**Files:**
- Modify: `src/esper/simic/features.py:67-124`
- Modify: `src/esper/leyline/__init__.py` (update TENSOR_SCHEMA_SIZE from 30 to 35)
- Test: `tests/test_simic_features.py` (add new test)

**Issue:** Agent cannot learn blueprint-specific policies because blueprint_id is not in observations.

**Design Decision (DRL Expert review):** Use **one-hot encoding** instead of ordinal encoding.

Rationale:
- Ordinal encoding (`id / num_blueprints`) imposes artificial ordering on categorical data
- Blueprints (norm, attention, conv_enhance, etc.) have no meaningful ordinal relationship
- One-hot encoding allows clean gradient flow and easier interpretation of policy attention
- With 5 blueprints, this adds 5 features (not 1), but the representation is semantically correct

**Retraining Requirements:**
- Existing models trained on 30-dim observations CANNOT load into 35-dim networks
- Transfer learning strategy: Copy first 30 input weights, initialize new 5 weights to small random
- Warm-start observation normalizer: Initialize blueprint features with mean=0.2, std=0.4 (sparse)

**Step 1: Write the failing test**

```python
# Add to tests/test_simic_features.py

def test_base_features_includes_blueprint_one_hot():
    """Base features should include one-hot blueprint encoding."""
    from esper.simic.features import obs_to_base_features

    obs = {
        'epoch': 10,
        'global_step': 500,
        'train_loss': 1.5,
        'val_loss': 1.6,
        'loss_delta': -0.1,
        'train_accuracy': 60.0,
        'val_accuracy': 58.0,
        'accuracy_delta': 2.0,
        'plateau_epochs': 3,
        'best_val_accuracy': 60.0,
        'best_val_loss': 1.4,
        'loss_history_5': [2.0, 1.8, 1.7, 1.6, 1.5],
        'accuracy_history_5': [40.0, 45.0, 50.0, 55.0, 58.0],
        'has_active_seed': 1.0,
        'seed_stage': 3,
        'seed_epochs_in_stage': 5,
        'seed_alpha': 0.0,
        'seed_improvement': 2.0,
        'available_slots': 0,
        'seed_counterfactual': 0.0,
        'host_grad_norm': 0.5,
        'host_learning_phase': 0.4,
        # New: blueprint encoding (0=none, 1-5=blueprint index)
        'seed_blueprint_id': 2,  # attention blueprint
        'num_blueprints': 5,
    }

    features = obs_to_base_features(obs, max_epochs=25)

    # Should now be 35 features (30 base + 5 one-hot blueprint)
    assert len(features) == 35

    # Blueprint one-hot should be at indices 30-34
    blueprint_one_hot = features[30:35]
    assert blueprint_one_hot == [0.0, 1.0, 0.0, 0.0, 0.0]  # Index 1 is hot (blueprint_id=2, 0-indexed)

    # Test with no active seed (blueprint_id=0)
    obs['seed_blueprint_id'] = 0
    features_no_seed = obs_to_base_features(obs, max_epochs=25)
    assert features_no_seed[30:35] == [0.0, 0.0, 0.0, 0.0, 0.0]  # All zeros
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_simic_features.py::test_base_features_includes_blueprint_one_hot -v`
Expected: FAIL (assertion error on length)

**Step 3: Update the implementation**

Replace `obs_to_base_features` in `src/esper/simic/features.py`:

```python
def obs_to_base_features(obs: dict, max_epochs: int = 200) -> list[float]:
    """Extract V3-style base features (35 dims) with pre-normalization.

    Pre-normalizes features to ~[0, 1] range for early training stability.
    This reduces the burden on RunningMeanStd during the initial warmup phase
    where statistics are poorly estimated.

    Base features capture training state without telemetry:
    - Timing: epoch, global_step (2)
    - Losses: train_loss, val_loss, loss_delta (3)
    - Accuracies: train_accuracy, val_accuracy, accuracy_delta (3)
    - Trends: plateau_epochs, best_val_accuracy, best_val_loss (3)
    - History: loss_history_5 (5), accuracy_history_5 (5)
    - Seed state: has_active_seed, seed_stage, seed_epochs_in_stage,
                  seed_alpha, seed_improvement, seed_counterfactual (6)
    - Slots: available_slots (1)
    - Host state: host_grad_norm, host_learning_phase (2)
    - Blueprint: one-hot encoding (5) [NEW - DRL Expert recommendation]

    Total: 35 features

    Args:
        obs: Observation dictionary from TrainingSnapshot.to_dict()
        max_epochs: Maximum training epochs (for normalization)

    Returns:
        List of 35 floats, pre-normalized to ~[0, 1] range
    """
    # Blueprint one-hot encoding (DRL Expert recommendation)
    # blueprint_id: 0=none, 1=first blueprint, 2=second, etc.
    # One-hot avoids imposing artificial ordinal relationships on categorical data
    blueprint_id = obs.get('seed_blueprint_id', 0)
    num_blueprints = obs.get('num_blueprints', 5)
    blueprint_one_hot = [0.0] * num_blueprints
    if blueprint_id > 0 and blueprint_id <= num_blueprints:
        blueprint_one_hot[blueprint_id - 1] = 1.0  # 1-indexed to 0-indexed

    return [
        # Timing features
        float(obs['epoch']) / max_epochs,                     # [0, 1]
        float(obs['global_step']) / (max_epochs * 100),       # ~[0, 1] assuming ~100 batches/epoch
        # Loss features (safe already clips to 10.0, divide for ~[0, 1])
        safe(obs['train_loss'], 10.0) / 10.0,                 # ~[0, 1]
        safe(obs['val_loss'], 10.0) / 10.0,                   # ~[0, 1]
        safe(obs['loss_delta'], 0.0, max_val=5.0) / 5.0,      # ~[-1, 1]
        # Accuracy features (already [0, 100] -> [0, 1])
        obs['train_accuracy'] / 100.0,                        # [0, 1]
        obs['val_accuracy'] / 100.0,                          # [0, 1]
        safe(obs['accuracy_delta'], 0.0, max_val=50.0) / 50.0,  # ~[-1, 1]
        # Trend features
        float(obs['plateau_epochs']) / 20.0,                  # ~[0, 1] typical max ~20
        obs['best_val_accuracy'] / 100.0,                     # [0, 1]
        safe(obs['best_val_loss'], 10.0) / 10.0,              # ~[0, 1]
        # History features
        *[safe(v, 10.0) / 10.0 for v in obs['loss_history_5']],       # ~[0, 1]
        *[v / 100.0 for v in obs['accuracy_history_5']],              # [0, 1]
        # Seed state features
        float(obs['has_active_seed']),                        # Already 0/1
        float(obs['seed_stage']) / 7.0,                       # Stages 0-7 -> [0, 1]
        float(obs['seed_epochs_in_stage']) / 50.0,            # ~[0, 1] typical max ~50
        obs['seed_alpha'],                                    # Already [0, 1]
        safe(obs['seed_improvement'], 0.0, max_val=10.0) / 10.0,  # [-1, 1] clamped
        float(obs['available_slots']),                        # Usually 0-2, small scale ok
        safe(obs.get('seed_counterfactual', 0.0), 0.0, max_val=10.0) / 10.0,  # [-1, 1] clamped
        # Host state features
        safe(obs.get('host_grad_norm', 0.0), 0.0, max_val=10.0) / 10.0,  # [0, 1] clamped
        obs.get('host_learning_phase', 0.0),                 # Already [0, 1]
        # Blueprint features (NEW - one-hot encoding)
        *blueprint_one_hot,                                   # 5 features, exactly one is 1.0 or all zeros
    ]
```

Also update `TENSOR_SCHEMA_SIZE` in `src/esper/leyline/__init__.py` from 30 to 35.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_simic_features.py::test_base_features_includes_blueprint_one_hot -v`
Expected: PASS

**Step 5: Run full test suite to check for regressions**

Run: `PYTHONPATH=src pytest tests/ -v --tb=short`
Expected: All tests pass (some may need observation dict updates)

**Step 6: Commit**

```bash
git add src/esper/simic/features.py src/esper/leyline/__init__.py tests/test_simic_features.py
git commit -m "feat(simic): add blueprint_id to observation space

Enables agent to learn blueprint-specific policies. Feature is
normalized by blueprint count for stability.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: Reduce CULL PBRS Penalty for Late Stages

**Files:**
- Modify: `src/esper/simic/rewards.py:553-609`
- Test: `tests/test_simic_rewards.py` (add new test)

**Issue:** PBRS penalty for culling BLENDING+ seeds (~-1.65) may be too strong, preventing necessary interventions on failing seeds.

**Step 1: Write the test**

```python
# Add to tests/test_simic_rewards.py

def test_cull_shaping_late_stage_not_excessive():
    """CULL penalty for late-stage failing seeds should not be excessive."""
    from esper.simic.rewards import _cull_shaping, SeedInfo, RewardConfig

    config = RewardConfig()

    # Failing seed in BLENDING stage
    seed_info = SeedInfo(
        stage=4,  # BLENDING
        improvement_since_stage_start=-2.0,  # Clearly failing
        total_improvement=-1.0,
        epochs_in_stage=5,
        seed_params=2000,
        previous_stage=3,
        seed_age_epochs=8,
    )

    shaping = _cull_shaping(seed_info, config)

    # Should be positive (reward for culling failing seed) or only mildly negative
    # The old behavior gave ~-1.65 which is too harsh
    assert shaping > -1.0, f"CULL penalty too harsh for failing seed: {shaping}"
```

**Step 2: Run test to verify current behavior**

Run: `PYTHONPATH=src pytest tests/test_simic_rewards.py::test_cull_shaping_late_stage_not_excessive -v`
Expected: May FAIL depending on current values

**Step 3: Adjust the implementation**

In `_cull_shaping` function in `src/esper/simic/rewards.py`, scale the PBRS penalty by seed health **only for late stages**:

```python
def _cull_shaping(seed_info: SeedInfo | None, config: RewardConfig) -> float:
    """Compute shaping for CULL action.

    CULL is incentivized for failing seeds but FOSSILIZED seeds cannot be culled.
    Attempting to cull a FOSSILIZED seed is a wasted action (no-op) and penalized.

    Age penalty prevents "germinate then immediately cull" anti-pattern.
    PBRS penalty is scaled by seed health FOR LATE STAGES ONLY - this preserves
    full PBRS incentives for early-stage decisions while allowing exits from
    failing late-stage seeds.

    Note on PBRS Deviation (DRL Expert review):
        The health_factor scaling intentionally deviates from pure potential-based
        reward shaping (Ng et al., 1999) which would preserve optimal policy guarantees.
        We accept this deviation because:
        1. Failing seeds trapped in late stages represent a pathological case
        2. The 0.3 floor prevents gaming by intentionally tanking seeds
        3. Other shaping terms (base_shaping, param_recovery) are already non-PBRS
        4. Early stages (< BLENDING) retain full PBRS to preserve learning signal
    """
    if seed_info is None:
        return config.cull_no_seed_penalty

    improvement = seed_info.improvement_since_stage_start
    stage = seed_info.stage
    seed_params = seed_info.seed_params
    seed_age = seed_info.seed_age_epochs

    # FOSSILIZED seeds cannot be culled - they are permanent by design.
    # Attempting to cull is a wasted action. Heavy penalty to discourage.
    if stage == STAGE_FOSSILIZED:
        return -1.0  # Wasted action penalty

    # Age penalty: culling a very young seed wastes the germination investment.
    # This prevents the "germinate then immediately cull" anti-pattern.
    # Scale: -0.3 per epoch missing from minimum age
    if seed_age < MIN_CULL_AGE and stage in (STAGE_GERMINATED, STAGE_TRAINING):
        age_penalty = -0.3 * (MIN_CULL_AGE - seed_age)  # -0.9 at age 0, -0.6 at age 1
        return age_penalty  # Return early - don't add other bonuses to young culls

    # Base shaping: reward culling failing seeds, penalize culling promising ones
    if improvement < config.cull_failing_threshold:
        base_shaping = config.cull_failing_bonus
    elif improvement < 0:
        base_shaping = config.cull_acceptable_bonus
    else:
        # Scale penalty with improvement - culling +14% seed should hurt more than +1%
        improvement_penalty = -0.1 * max(0, improvement)
        # Scale penalty with stage - culling at SHADOWING hurts more than at TRAINING
        stage_penalty = -0.5 * max(0, stage - 3)  # TRAINING=3, so penalty starts at BLENDING
        base_shaping = config.cull_promising_penalty + improvement_penalty + stage_penalty

    # Param recovery bonus: incentivize culling bloated seeds to free resources
    # +0.1 per 10K params, capped at 0.5
    param_recovery_bonus = min(0.5, (seed_params / 10_000) * config.cull_param_recovery_weight)

    # Terminal PBRS correction: account for potential loss from destroying the seed
    current_obs = {
        "has_active_seed": 1,
        "seed_stage": stage,
        "seed_epochs_in_stage": seed_info.epochs_in_stage,
    }
    phi_current = compute_seed_potential(current_obs)

    # Health discount: failing seeds in LATE STAGES get reduced PBRS penalty
    # (DRL Expert recommendation: only apply for stage >= BLENDING to preserve
    # early-stage PBRS incentives where the penalty is smaller anyway)
    health_factor = 1.0
    if improvement < 0 and stage >= STAGE_BLENDING:
        # Scale from 1.0 (improvement=0) to 0.3 (improvement=-3 or worse)
        health_factor = max(0.3, 1.0 + improvement / 3.0)

    # PBRS: gamma * phi(next) - phi(current) where next = no seed (phi=0)
    pbrs_correction = 0.99 * 0.0 - phi_current  # = -phi_current
    terminal_pbrs = config.seed_potential_weight * pbrs_correction * health_factor

    return base_shaping + param_recovery_bonus + terminal_pbrs
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_simic_rewards.py::test_cull_shaping_late_stage_not_excessive -v`
Expected: PASS

**Step 5: Run reward tests to check for regressions**

Run: `PYTHONPATH=src pytest tests/test_simic_rewards.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/esper/simic/rewards.py tests/test_simic_rewards.py
git commit -m "fix(simic): scale CULL PBRS penalty by seed health

Failing seeds now have reduced PBRS penalty, allowing necessary
interventions even in late lifecycle stages.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Phase 4: Code Quality (Low Priority)

### Task 8: Extract Cache Invalidation Helper

**Files:**
- Modify: `src/esper/kasmina/blueprints/registry.py`

**Issue:** Same try/except block for cache invalidation appears 3 times.

**Step 1: Refactor to helper function**

Add helper and update callers in `src/esper/kasmina/blueprints/registry.py`:

```python
def _invalidate_action_cache(topology: str | None = None) -> None:
    """Invalidate Leyline action enum cache for topology.

    Best-effort operation that fails silently during import cycles.
    """
    try:
        from esper.leyline import actions as leyline_actions
    except ImportError:
        return

    try:
        if topology is None:
            leyline_actions._action_enum_cache.clear()
        else:
            leyline_actions._action_enum_cache.pop(topology, None)
    except AttributeError:
        pass  # Cache doesn't exist yet


class BlueprintRegistry:
    # ... existing code ...

    @classmethod
    def register(cls, name: str, topology: str, param_estimate: int, description: str = ""):
        def decorator(factory: Callable[[int], nn.Module]):
            key = f"{topology}:{name}"
            cls._blueprints[key] = BlueprintSpec(
                name=name,
                topology=topology,
                factory=factory,
                param_estimate=param_estimate,
                description=description,
            )
            _invalidate_action_cache(topology)
            return factory
        return decorator

    @classmethod
    def unregister(cls, topology: str, name: str) -> None:
        key = f"{topology}:{name}"
        cls._blueprints.pop(key, None)
        _invalidate_action_cache(topology)

    @classmethod
    def reset(cls) -> None:
        cls._blueprints.clear()
        _invalidate_action_cache()
```

**Step 2: Run tests**

Run: `PYTHONPATH=src pytest tests/ -k blueprint -v`
Expected: PASS

**Step 3: Commit**

```bash
git add src/esper/kasmina/blueprints/registry.py
git commit -m "refactor(kasmina): extract cache invalidation helper

DRY cleanup - same try/except was repeated 3 times.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 9: Add force_alpha Thread Safety Warning

**Files:**
- Modify: `src/esper/kasmina/slot.py:622-647`

**Issue:** The `force_alpha` context manager is not thread-safe but lacks documentation.

**Step 1: Update docstring**

```python
@contextmanager
def force_alpha(self, value: float):
    """Temporarily override alpha for counterfactual evaluation.

    Used for differential validation to measure true seed contribution
    by comparing real output (current alpha) vs host-only (alpha=0).

    Warning:
        NOT THREAD-SAFE. Do not use during concurrent forward passes
        or with DataParallel/DistributedDataParallel. Use model.eval()
        and single-threaded validation only.

        Nested calls are NOT supported - the inner override will be
        clobbered when the outer context exits.

    Args:
        value: Alpha value to force (typically 0.0 for host-only baseline)

    Yields:
        Context where alpha is temporarily overridden
    """
    if self.state is None:
        yield
        return

    prev_alpha = self.state.alpha
    self.state.alpha = value
    try:
        yield
    finally:
        self.state.alpha = prev_alpha
```

**Step 2: Commit**

```bash
git add src/esper/kasmina/slot.py
git commit -m "docs(kasmina): add thread safety warning to force_alpha

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Verification Checklist

After all tasks, run the full test suite:

```bash
PYTHONPATH=src pytest tests/ -v --tb=short
```

Expected: All tests pass

Run type checking:

```bash
PYTHONPATH=src python -m mypy src/esper/kasmina/ --ignore-missing-imports
```

Expected: No errors (or only pre-existing ones)

---

## Summary

| Phase | Tasks | Priority |
|-------|-------|----------|
| 1 | 1-3, 3.5 | Critical bugs + type fix |
| 2 | 4-5 | PyTorch performance |
| 3 | 6-7 | RL observation quality |
| 4 | 8-9 | Code quality |

Total: 10 tasks, ~50-70 minutes implementation time

Each task is independently committable and testable.

**Note on Task 6:** Observation space changes from 30 to 35 dimensions. Existing trained models are incompatible and require retraining or transfer learning.

---

## Specialist Sign-Offs

- **Code Reviewer**: APPROVED (after adding Task 3.5 for type annotation)
- **PyTorch Expert**: APPROVED with refinements:
  - Task 1: Simplified to post-move device query (simpler, follows PyTorch conventions)
  - Task 4: Added note about `torch._foreach_norm` for large models
  - Task 5: Added optimization to only clear cache on actual device change
- **DRL Expert**: APPROVED with refinements:
  - Task 6: Changed to one-hot encoding (avoids ordinal assumption on categorical data)
  - Task 6: Added retraining requirements documentation
  - Task 7: Gate health_factor by stage >= BLENDING (preserves early-stage PBRS)
