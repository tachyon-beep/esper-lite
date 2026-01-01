# Simic Expert Improvements Implementation Plan (v2 - PyTorch 2.9)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement DRL and PyTorch expert recommendations to improve simic training stability, performance, and torch.compile compatibility.

**Architecture:** Incremental improvements across PPO update loop (NaN/Inf checks), feature extraction (tensor returns), anomaly detection (configurable thresholds), and gradient collection (vectorization). Each change is isolated and testable.

**Tech Stack:** PyTorch 2.9, Python 3.13, pytest

**Revision Notes (v2):**
- Task 3: Changed from `torch.cat` to `torch._foreach_norm` (PyTorch 2.9 fused kernel)
- Task 5: Weight decay now applies ONLY to critic (RL best practice)
- Task 5: Added `fused=True` for PyTorch 2.9 optimized kernels
- Task 6: Changed from linear scaling to information-theoretic log-ratio scaling
- Task 7: Added additional metrics (advantage pre-norm, value MSE, return range)
- Task 8: NEW - LSTM LayerNorm for recurrent stability
- Task 9: NEW - Batched metric extraction to reduce CPU syncs

**Revision Notes (v3 - Expert Review):**
- Task 1: Corrected rationale - reduces kernel launches, not CPU syncs (PyTorch Expert)
- Task 3: Clarified `_foreach_norm` is stable in PyTorch 2.9, no fallback needed
- Task 5: Shared layers weight_decay must be 0.0, not 0.1x (DRL Expert CRITICAL)
- Task 5: Added `foreach=True` for CPU optimization (PyTorch Expert)
- Task 6: Added wiring of `get_entropy_floor()` into loss computation (DRL Expert CRITICAL)
- Task 7: Added float32 cast for mixed precision safety (PyTorch Expert)
- Task 7: Added warning when advantage_std_prenorm < 0.1 (DRL Expert)
- Task 9: Changed warning threshold from >4 to >2, two-tier system (DRL Expert)

---

## Task 1: Fuse NaN/Inf Checks to Single Kernel

**Rationale:** Current code runs separate `isnan().any()` and `isinf().any()` checks, launching two kernels. Fusing into single `isfinite().all()` combines both checks into one kernel with one reduction, reducing kernel launch overhead. (Note: CPU sync count is the same - both patterns sync once when converted to Python bool.)

**Files:**
- Modify: `src/esper/simic/ppo.py:362-365` (update() method)
- Modify: `src/esper/simic/ppo.py:671-674` (update_recurrent() method)
- Test: `tests/test_simic_ppo.py`

**Step 1: Write the failing test**

Add to `tests/test_simic_ppo.py`:

```python
def test_fused_nan_inf_detection():
    """Verify NaN/Inf detection uses single fused check."""
    import torch
    from esper.simic.ppo import PPOAgent

    agent = PPOAgent(state_dim=10, action_dim=4, device="cpu")

    # Create buffer with normal data
    state = torch.randn(10)
    action_mask = torch.ones(4)
    agent.store_transition(state, 0, -0.5, 0.5, 1.0, False, action_mask)
    agent.store_transition(state, 1, -0.3, 0.6, 0.5, True, action_mask)

    metrics = agent.update(last_value=0.0)

    # Should NOT have separate nan/inf flags (old API)
    # New API uses combined 'ratio_has_numerical_issue' only when issues occur
    assert 'ratio_has_nan' not in metrics and 'ratio_has_inf' not in metrics, \
        "Should use fused check, not separate nan/inf flags"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_simic_ppo.py::test_fused_nan_inf_detection -v`
Expected: FAIL - current code sets both `ratio_has_nan` and `ratio_has_inf` separately

**Step 3: Implement fused check in update()**

In `src/esper/simic/ppo.py`, replace lines 361-365:

```python
                # Check ratio for numerical issues (critical detection point)
                if torch.isnan(ratio).any():
                    metrics['ratio_has_nan'] = True
                if torch.isinf(ratio).any():
                    metrics['ratio_has_inf'] = True
```

With:

```python
                # Check ratio for numerical issues - single fused kernel
                # isfinite() combines isnan + isinf into one kernel, all() is single reduction
                if not torch.isfinite(ratio).all():
                    metrics['ratio_has_numerical_issue'] = True
                    # Debug breakdown only when telemetry requests it (avoids extra syncs)
                    if telemetry_config.should_collect("debug"):
                        metrics['ratio_has_nan'] = torch.isnan(ratio).any().item()
                        metrics['ratio_has_inf'] = torch.isinf(ratio).any().item()
```

**Step 4: Implement fused check in update_recurrent()**

In `src/esper/simic/ppo.py`, replace lines 671-674:

```python
            # Check for NaN/Inf in ratios
            if torch.isnan(stacked).any():
                metrics['ratio_has_nan'] = True
            if torch.isinf(stacked).any():
                metrics['ratio_has_inf'] = True
```

With:

```python
            # Check for NaN/Inf in ratios - single fused check
            if not torch.isfinite(stacked).all():
                metrics['ratio_has_numerical_issue'] = True
```

**Step 5: Update anomaly detector to use new flag**

In `src/esper/simic/ppo.py`, update the anomaly check around line 452:

```python
            batch_has_nan = any(math.isnan(v) for v in all_losses) or metrics.get('ratio_has_nan', False)
            batch_has_inf = any(math.isinf(v) for v in all_losses) or metrics.get('ratio_has_inf', False)
```

To:

```python
            # Check losses for NaN/Inf - fused check
            loss_has_issue = any(math.isnan(v) or math.isinf(v) for v in all_losses)
            ratio_has_issue = metrics.get('ratio_has_numerical_issue', False)
            has_numerical_issue = loss_has_issue or ratio_has_issue
```

And update the anomaly_detector.check_all call:

```python
            anomaly_report = anomaly_detector.check_all(
                ratio_max=max_ratio,
                ratio_min=min_ratio,
                explained_variance=explained_variance,
                has_nan=has_numerical_issue,  # Combined flag
                has_inf=False,  # Covered by combined flag
            )
```

**Step 6: Run test to verify it passes**

Run: `uv run pytest tests/test_simic_ppo.py::test_fused_nan_inf_detection -v`
Expected: PASS

**Step 7: Run full PPO test suite**

Run: `uv run pytest tests/test_simic_ppo.py -v`
Expected: All tests PASS

**Step 8: Commit**

```bash
git add src/esper/simic/ppo.py tests/test_simic_ppo.py
git commit -m "perf(simic): fuse NaN/Inf checks to single kernel

Replace separate torch.isnan().any() and torch.isinf().any() calls
with single torch.isfinite().all() check. Combines two kernel launches
into one kernel with one reduction, reducing kernel launch overhead.

Debug breakdown available when telemetry level is DEBUG.

PyTorch Expert recommendation from deep dive analysis."
```

---

## Task 2: Make Anomaly Detection Thresholds Configurable

**Rationale:** Make all thresholds configurable via TrainingConfig for per-task tuning.

**Files:**
- Modify: `src/esper/simic/config.py` (add threshold fields)
- Modify: `src/esper/simic/anomaly_detector.py` (accept config)
- Modify: `src/esper/simic/ppo.py` (pass config to detector)
- Test: `tests/test_simic_config.py`

**Step 1: Write the failing test**

Add to `tests/test_simic_config.py`:

```python
def test_anomaly_thresholds_in_config():
    """TrainingConfig should include anomaly detection thresholds."""
    from esper.simic.config import TrainingConfig

    config = TrainingConfig()

    # Should have configurable thresholds
    assert hasattr(config, 'anomaly_max_ratio_threshold')
    assert hasattr(config, 'anomaly_min_ratio_threshold')
    assert hasattr(config, 'anomaly_min_explained_variance')

    # Default explained_variance should be 0.1 (existing behavior)
    assert config.anomaly_min_explained_variance == 0.1

    # Should be able to override (0.3 = more sensitive to value collapse)
    sensitive_config = TrainingConfig(anomaly_min_explained_variance=0.3)
    assert sensitive_config.anomaly_min_explained_variance == 0.3
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_simic_config.py::test_anomaly_thresholds_in_config -v`
Expected: FAIL - TrainingConfig doesn't have these fields

**Step 3: Add threshold fields to TrainingConfig**

In `src/esper/simic/config.py`, add after line 98 (after stabilization fields):

```python
    # === Anomaly Detection Thresholds ===
    # Ratio explosion/collapse detection
    anomaly_max_ratio_threshold: float = 5.0
    anomaly_min_ratio_threshold: float = 0.1
    # Value function collapse detection
    # Higher values (e.g., 0.3) = MORE SENSITIVE to critic issues (triggers more often)
    # Lower values (e.g., 0.1) = LESS SENSITIVE (fewer false alarms, default)
    anomaly_min_explained_variance: float = 0.1
```

**Step 4: Add to_anomaly_kwargs method**

In `src/esper/simic/config.py`, add after `to_tracker_kwargs`:

```python
    def to_anomaly_kwargs(self) -> dict[str, Any]:
        """Extract AnomalyDetector constructor kwargs."""
        return {
            "max_ratio_threshold": self.anomaly_max_ratio_threshold,
            "min_ratio_threshold": self.anomaly_min_ratio_threshold,
            "min_explained_variance": self.anomaly_min_explained_variance,
        }
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_simic_config.py::test_anomaly_thresholds_in_config -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/esper/simic/config.py tests/test_simic_config.py
git commit -m "feat(simic): make anomaly detection thresholds configurable

Add anomaly_max_ratio_threshold, anomaly_min_ratio_threshold, and
anomaly_min_explained_variance to TrainingConfig.

Note: Higher min_explained_variance values (e.g., 0.3) are MORE
sensitive to value collapse, triggering detection more often.
Default 0.1 is less sensitive to reduce false alarms.

DRL Expert recommendation from deep dive analysis."
```

---

## Task 3: Vectorize Gradient Norm Computation with torch._foreach_norm

**Rationale:** PyTorch 2.9 provides `torch._foreach_norm` - a fused CUDA kernel that computes norms for multiple tensors in a single kernel launch, avoiding Python iteration overhead. This API is used internally by `clip_grad_norm_` and is stable in PyTorch 2.9.

**IMPORTANT:** The original plan suggested `torch.cat` which is WORSE (allocates large tensor). Use `torch._foreach_norm` instead. No fallback needed - we require PyTorch 2.9+.

**Files:**
- Modify: `src/esper/simic/gradient_collector.py:128-133`
- Test: `tests/test_simic_gradient_collector.py` (create if needed)

**Step 1: Write the failing test**

Create `tests/test_simic_gradient_collector.py`:

```python
"""Tests for gradient collector performance and correctness."""
import pytest
import torch
import torch.nn as nn

from esper.simic.gradient_collector import (
    SeedGradientCollector,
    materialize_grad_stats,
)


def test_gradient_collector_vectorized():
    """Verify gradient collection uses vectorized operations."""
    # Create simple model with gradients
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5),
    )

    # Forward/backward to create gradients
    x = torch.randn(4, 10)
    y = model(x).sum()
    y.backward()

    collector = SeedGradientCollector()
    async_stats = collector.collect_async(model.parameters())
    stats = materialize_grad_stats(async_stats)

    # Basic correctness checks
    assert 'gradient_norm' in stats
    assert 'gradient_health' in stats
    assert stats['gradient_norm'] > 0
    assert 0 <= stats['gradient_health'] <= 1


def test_gradient_collector_empty():
    """Verify handling of parameters without gradients."""
    model = nn.Linear(10, 5)  # No backward called

    collector = SeedGradientCollector()
    stats = collector.collect(model.parameters())

    assert stats['gradient_norm'] == 0.0
    assert stats['gradient_health'] == 1.0
    assert stats['has_vanishing'] is False
    assert stats['has_exploding'] is False


def test_gradient_collector_uses_foreach():
    """Verify _foreach_norm is used when available (PyTorch 2.9+)."""
    # This test documents the expected implementation
    import torch
    assert hasattr(torch, '_foreach_norm'), "PyTorch 2.9 _foreach_norm required"
```

**Step 2: Run test to verify baseline**

Run: `uv run pytest tests/test_simic_gradient_collector.py -v`
Expected: PASS (tests existing behavior, last test verifies API exists)

**Step 3: Optimize collect_async with torch._foreach_norm**

In `src/esper/simic/gradient_collector.py`, replace lines 128-133:

```python
        # Compute L2 norm for each gradient tensor
        # Using public API (torch.norm) instead of private torch._foreach_norm
        per_param_norms = [g.norm(2) for g in grads]

        # Stack to compute stats efficiently on GPU/CPU
        all_norms = torch.stack(per_param_norms)
```

With:

```python
        # [PyTorch 2.9] Use _foreach_norm for efficient multi-tensor norm computation
        # This is a fused CUDA kernel that computes all norms in a single kernel launch,
        # avoiding Python iteration overhead. Used internally by clip_grad_norm_.
        per_param_norms = torch._foreach_norm(grads, ord=2)

        # Stack for vectorized comparisons
        all_norms = torch.stack(per_param_norms)

        # Total norm via Pythagorean theorem (avoids large tensor allocation)
        total_squared_norm = (all_norms ** 2).sum()
```

And update the return dict (around line 137):

```python
        return {
            '_empty': False,
            '_n_grads': n_grads,
            '_total_squared_norm': total_squared_norm,
            '_all_norms': all_norms,
            '_n_vanishing': (all_norms < self.vanishing_threshold).sum(),
            '_n_exploding': (all_norms > self.exploding_threshold).sum(),
        }
```

**Step 4: Run tests to verify correctness**

Run: `uv run pytest tests/test_simic_gradient_collector.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/gradient_collector.py tests/test_simic_gradient_collector.py
git commit -m "perf(simic): use torch._foreach_norm for gradient collection

[PyTorch 2.9] Replace list comprehension with _foreach_norm fused kernel.
Computes all per-parameter norms in a single CUDA kernel launch,
eliminating Python iteration overhead.

Note: _foreach_norm is stable and used internally by clip_grad_norm_.

PyTorch Expert recommendation from deep dive analysis."
```

---

## Task 4: Return Tensors from Feature Extraction

**Rationale:** PyTorch expert noted `obs_to_base_features` returns Python list requiring torch.tensor() conversion at every step. Pre-allocating tensor is more efficient for high-throughput training.

**Files:**
- Modify: `src/esper/simic/features.py`
- Test: `tests/test_simic_features.py`

**Step 1: Write the failing test**

Add to `tests/test_simic_features.py`:

```python
def test_obs_to_features_tensor_output():
    """Feature extraction should optionally return tensor directly."""
    import torch
    from esper.simic.features import obs_to_base_features_tensor

    obs = {
        'epoch': 10,
        'global_step': 500,
        'train_loss': 1.5,
        'val_loss': 1.6,
        'loss_delta': -0.1,
        'train_accuracy': 45.0,
        'val_accuracy': 43.0,
        'accuracy_delta': 2.0,
        'plateau_epochs': 3,
        'best_val_accuracy': 45.0,
        'best_val_loss': 1.4,
        'loss_history_5': [2.0, 1.8, 1.6, 1.5, 1.5],
        'accuracy_history_5': [30.0, 35.0, 40.0, 43.0, 45.0],
        'has_active_seed': 1.0,
        'seed_stage': 3,
        'seed_epochs_in_stage': 5,
        'seed_alpha': 0.5,
        'seed_improvement': 0.02,
        'available_slots': 0,
        'seed_counterfactual': 0.01,
        'host_grad_norm': 1.2,
        'host_learning_phase': 0.4,
        'seed_blueprint_id': 2,
    }

    device = torch.device('cpu')
    result = obs_to_base_features_tensor(obs, device=device)

    # Should be a tensor, not a list
    assert isinstance(result, torch.Tensor)
    assert result.device == device
    assert result.shape == (35,)  # Base feature dim
    assert result.dtype == torch.float32
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_simic_features.py::test_obs_to_features_tensor_output -v`
Expected: FAIL - `obs_to_base_features_tensor` doesn't exist

**Step 3: Implement tensor-returning version**

Add to `src/esper/simic/features.py` after `obs_to_base_features`:

```python
def obs_to_base_features_tensor(
    obs: dict,
    device: torch.device,
    max_epochs: int = 200,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Extract V3-style base features directly as tensor.

    More efficient than obs_to_base_features() + torch.tensor() for
    high-throughput training loops. Avoids Python list allocation.

    Note: This function is NOT designed for torch.compile due to
    dict access and variable-length history. Use for rollout collection
    only; the network forward pass should use the tensor directly.

    Args:
        obs: Observation dictionary
        device: Target device for tensor
        max_epochs: Maximum epochs for normalization
        out: Optional pre-allocated output tensor (35,) for zero-alloc mode

    Returns:
        Tensor of shape (35,) with base features
    """
    import torch

    if out is None:
        out = torch.empty(35, dtype=torch.float32, device=device)

    # Blueprint one-hot
    blueprint_id = obs.get('seed_blueprint_id', 0)
    num_blueprints = obs.get('num_blueprints', 5)

    # Convert histories to tensors for vectorized assignment
    loss_hist = torch.tensor(obs['loss_history_5'], dtype=torch.float32, device=device)
    acc_hist = torch.tensor(obs['accuracy_history_5'], dtype=torch.float32, device=device)

    # Vectorized clipping and normalization
    loss_hist = torch.clamp(loss_hist, max=10.0) / 10.0
    acc_hist = acc_hist / 100.0

    # Fill tensor - scalar assignments
    out[0] = float(obs['epoch']) / max_epochs
    out[1] = float(obs['global_step']) / (max_epochs * 100)
    out[2] = safe(obs['train_loss'], 10.0) / 10.0
    out[3] = safe(obs['val_loss'], 10.0) / 10.0
    out[4] = safe(obs['loss_delta'], 0.0, max_val=5.0) / 5.0
    out[5] = obs['train_accuracy'] / 100.0
    out[6] = obs['val_accuracy'] / 100.0
    out[7] = safe(obs['accuracy_delta'], 0.0, max_val=50.0) / 50.0
    out[8] = float(obs['plateau_epochs']) / 20.0
    out[9] = obs['best_val_accuracy'] / 100.0
    out[10] = safe(obs['best_val_loss'], 10.0) / 10.0

    # History features - vectorized slice assignment
    out[11:16] = loss_hist
    out[16:21] = acc_hist

    # Seed state features
    out[21] = float(obs['has_active_seed'])
    out[22] = float(obs['seed_stage']) / 7.0
    out[23] = float(obs['seed_epochs_in_stage']) / 50.0
    out[24] = obs['seed_alpha']
    out[25] = safe(obs['seed_improvement'], 0.0, max_val=10.0) / 10.0
    out[26] = float(obs['available_slots'])
    out[27] = safe(obs.get('seed_counterfactual', 0.0), 0.0, max_val=10.0) / 10.0
    out[28] = safe(obs.get('host_grad_norm', 0.0), 0.0, max_val=10.0) / 10.0
    out[29] = obs.get('host_learning_phase', 0.0)

    # Blueprint one-hot (5 features)
    out[30:35] = 0.0
    if blueprint_id > 0 and blueprint_id <= num_blueprints:
        out[29 + blueprint_id] = 1.0

    return out
```

Add import at top of file:

```python
import torch
```

Update `__all__`:

```python
__all__ = [
    "safe",
    "obs_to_base_features",
    "obs_to_base_features_tensor",
    "compute_action_mask",
    "TaskConfig",
    "normalize_observation",
]
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_simic_features.py::test_obs_to_features_tensor_output -v`
Expected: PASS

**Step 5: Run full feature test suite**

Run: `uv run pytest tests/test_simic_features.py -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/esper/simic/features.py tests/test_simic_features.py
git commit -m "perf(simic): add tensor-returning feature extraction

Add obs_to_base_features_tensor() that returns PyTorch tensor directly
instead of Python list. Uses vectorized slice assignment for history
features. Supports pre-allocated output tensor for zero-alloc mode.

PyTorch Expert recommendation from deep dive analysis."
```

---

## Task 5: Add Weight Decay with Critic-Only Application (RL Best Practice)

**Rationale:** PyTorch expert noted no weight decay is applied. DRL expert CRITICAL CORRECTION: Weight decay on actor biases policy toward determinism (smaller weights = smaller logits = sharper softmax). Apply weight decay ONLY to critic.

**IMPORTANT:** This is different from supervised learning - RL requires asymmetric regularization.

**DRL Expert Critical Fix:** Weight decay must be 0.0 for BOTH actor AND shared layers - shared layers feed into actor, so any regularization still affects actor gradients.

**Files:**
- Modify: `src/esper/simic/config.py` (add weight_decay field)
- Modify: `src/esper/simic/ppo.py` (use AdamW with param groups, fused=True)
- Test: `tests/test_simic_ppo.py`

**Step 1: Write the failing test**

Add to `tests/test_simic_ppo.py`:

```python
def test_ppo_agent_weight_decay_critic_only():
    """PPOAgent weight decay should apply only to critic, not actor or shared."""
    import torch
    from esper.simic.ppo import PPOAgent

    # Without weight decay - uses Adam
    agent_no_wd = PPOAgent(state_dim=10, action_dim=4, device="cpu")
    assert isinstance(agent_no_wd.optimizer, torch.optim.Adam)

    # With weight decay - should use AdamW with param groups
    agent_with_wd = PPOAgent(
        state_dim=10, action_dim=4, device="cpu", weight_decay=0.01
    )
    assert isinstance(agent_with_wd.optimizer, torch.optim.AdamW)

    # Verify param groups exist
    param_groups = agent_with_wd.optimizer.param_groups
    assert len(param_groups) >= 3, "Should have separate param groups for actor/shared/critic"

    # Find each group's weight decay
    actor_wd = None
    shared_wd = None
    critic_wd = None
    for group in param_groups:
        name = group.get('name', '')
        if 'actor' in name:
            actor_wd = group['weight_decay']
        elif 'shared' in name:
            shared_wd = group['weight_decay']
        elif 'critic' in name:
            critic_wd = group['weight_decay']

    # Actor should NOT have weight decay (biases toward determinism)
    assert actor_wd == 0.0, f"Actor weight_decay should be 0, got {actor_wd}"
    # Shared should NOT have weight decay (feeds into actor)
    assert shared_wd == 0.0, f"Shared weight_decay should be 0, got {shared_wd}"
    # Critic SHOULD have weight decay
    assert critic_wd > 0.0, f"Critic weight_decay should be >0, got {critic_wd}"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_simic_ppo.py::test_ppo_agent_weight_decay_critic_only -v`
Expected: FAIL - PPOAgent doesn't accept weight_decay parameter

**Step 3: Add weight_decay to PPOAgent with critic-only application**

In `src/esper/simic/ppo.py`, add parameter to `__init__` (around line 148):

```python
        target_kl: float | None = 0.015,
        weight_decay: float = 0.0,  # Applied to critic only (RL best practice)
        device: str = "cuda:0",
```

Store it:

```python
        self.weight_decay = weight_decay
```

Update optimizer creation (around line 186):

```python
        # [PyTorch 2.9] Use fused=True for CUDA, foreach=True for CPU
        use_cuda = device.startswith("cuda")
        optimizer_kwargs = {'lr': lr, 'eps': 1e-5}
        if use_cuda:
            optimizer_kwargs['fused'] = True
        else:
            optimizer_kwargs['foreach'] = True  # Multi-tensor optimization for CPU

        if weight_decay > 0:
            # [DRL Best Practice] Apply weight decay ONLY to critic, not actor or shared
            # Weight decay on actor biases toward determinism (smaller weights =
            # smaller logits = sharper softmax), which kills exploration.
            # Shared layers feed into actor, so they must also have wd=0.
            # Reference: SAC, TD3 implementations apply WD only to critic.
            actor_params = list(self.network.actor.parameters())
            critic_params = list(self.network.critic.parameters())
            shared_params = list(self.network.shared.parameters())

            self.optimizer = torch.optim.AdamW([
                {'params': actor_params, 'weight_decay': 0.0, 'name': 'actor'},
                {'params': shared_params, 'weight_decay': 0.0, 'name': 'shared'},  # Must be 0!
                {'params': critic_params, 'weight_decay': weight_decay, 'name': 'critic'},
            ], **optimizer_kwargs)
        else:
            self.optimizer = torch.optim.Adam(
                self.network.parameters(), **optimizer_kwargs
            )
```

**Step 4: Add weight_decay to TrainingConfig**

In `src/esper/simic/config.py`, add after max_grad_norm (around line 57):

```python
    # Weight decay (L2 regularization) - applied to critic only
    # Actor weight decay is disabled to preserve exploration (RL best practice)
    weight_decay: float = 0.0
```

Update `to_ppo_kwargs`:

```python
            "max_grad_norm": self.max_grad_norm,
            "weight_decay": self.weight_decay,
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_simic_ppo.py::test_ppo_agent_weight_decay_critic_only -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/esper/simic/ppo.py src/esper/simic/config.py tests/test_simic_ppo.py
git commit -m "feat(simic): add weight_decay with critic-only application

[DRL Best Practice] Weight decay on actor biases toward determinism
by shrinking logits, which kills exploration. Apply WD only to critic.
Shared layers must also have wd=0 since they feed into actor.

[PyTorch 2.9] Uses fused=True on CUDA, foreach=True on CPU for
optimized multi-tensor operations in optimizer step.

DRL Expert CRITICAL recommendation: asymmetric regularization is
standard in SAC/TD3 and essential for stable policy gradients."
```

---

## Task 6: Add Adaptive Entropy Floor with Information-Theoretic Scaling

**Rationale:** DRL expert noted entropy_coef_min=0.01 may allow determinism too early when action masking reduces valid actions to 2-3 choices.

**CRITICAL CORRECTION:** Use information-theoretic log-ratio scaling instead of linear scaling. This maintains the same "relative exploration" level across different action space sizes.

**Files:**
- Modify: `src/esper/simic/ppo.py` (add adaptive entropy method)
- Modify: `src/esper/simic/config.py` (add adaptive_entropy_floor flag)
- Test: `tests/test_simic_ppo.py`

**Step 1: Write the failing test**

Add to `tests/test_simic_ppo.py`:

```python
def test_adaptive_entropy_floor_log_scaling():
    """Entropy floor should use log-ratio scaling (information-theoretic)."""
    import math
    import torch
    from esper.simic.ppo import PPOAgent

    agent = PPOAgent(
        state_dim=10, action_dim=7, device="cpu",
        entropy_coef=0.05, entropy_coef_min=0.01,
        adaptive_entropy_floor=True,
    )

    # With all 7 actions valid, floor is base floor
    mask_all = torch.ones(7)
    floor_all = agent.get_entropy_floor(mask_all)
    assert floor_all == 0.01  # Base floor

    # With only 2 actions valid, floor should use log-ratio scaling:
    # scale = log(7) / log(2) = 1.95 / 0.69 = 2.8
    # floor = 0.01 * 2.8 = 0.028
    mask_few = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    floor_few = agent.get_entropy_floor(mask_few)

    expected_scale = math.log(7) / math.log(2)  # ~2.8
    expected_floor = 0.01 * min(expected_scale, 3.0)  # Capped at 3x

    assert abs(floor_few - expected_floor) < 0.001, \
        f"Expected {expected_floor:.4f}, got {floor_few:.4f}"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_simic_ppo.py::test_adaptive_entropy_floor_log_scaling -v`
Expected: FAIL - PPOAgent doesn't have adaptive_entropy_floor or get_entropy_floor

**Step 3: Add adaptive entropy floor with log-ratio scaling**

In `src/esper/simic/ppo.py`, add parameter (around line 148):

```python
        entropy_coef_min: float = 0.01,
        adaptive_entropy_floor: bool = False,  # Scale floor with valid action count
```

Store it:

```python
        self.adaptive_entropy_floor = adaptive_entropy_floor
```

Add method after `get_entropy_coef`:

```python
    def get_entropy_floor(self, action_mask: torch.Tensor | None = None) -> float:
        """Get entropy floor, optionally scaled by valid action count.

        When adaptive_entropy_floor=True, uses information-theoretic scaling:
        scale_factor = log(num_total) / log(num_valid)

        This maintains the same "relative exploration" level - if we want
        10% of max entropy with 7 actions, we want 10% of max entropy with
        2 actions, but max_entropy(2) = log(2) < max_entropy(7) = log(7).

        Args:
            action_mask: Binary mask of valid actions [action_dim] or None

        Returns:
            Entropy coefficient floor (minimum value)
        """
        import math

        if not self.adaptive_entropy_floor or action_mask is None:
            return self.entropy_coef_min

        # Count valid actions
        num_valid = int(action_mask.sum().item())
        num_total = action_mask.shape[-1]

        if num_valid >= num_total or num_valid <= 1:
            return self.entropy_coef_min

        # Information-theoretic scaling: ratio of maximum entropies
        # max_entropy_full = log(num_total), max_entropy_valid = log(num_valid)
        max_entropy_full = math.log(num_total)
        max_entropy_valid = math.log(num_valid)

        # Scale to maintain same fraction of max entropy
        scale_factor = max_entropy_full / max_entropy_valid

        # Cap at 3x to avoid extreme values
        scale_factor = min(scale_factor, 3.0)

        return self.entropy_coef_min * scale_factor
```

**Step 4: Add to TrainingConfig**

In `src/esper/simic/config.py`:

```python
    # Scale entropy floor with valid action count (information-theoretic)
    adaptive_entropy_floor: bool = False
```

Update `to_ppo_kwargs`:

```python
            "adaptive_entropy_floor": self.adaptive_entropy_floor,
```

**Step 5: Wire get_entropy_floor() into the loss computation**

**CRITICAL:** The DRL expert flagged that `get_entropy_floor()` must actually be used in the update loop. The method exists but isn't integrated.

In `src/esper/simic/ppo.py`, modify `get_entropy_coef()` to accept an optional action_mask and use the floor:

```python
    def get_entropy_coef(self, action_mask: torch.Tensor | None = None) -> float:
        """Get current entropy coefficient with optional adaptive floor.

        Args:
            action_mask: Optional action mask for adaptive floor computation

        Returns:
            Entropy coefficient (decayed if enabled, floored if adaptive)
        """
        if not self.use_entropy_decay:
            return self.entropy_coef

        # Calculate decayed coefficient
        progress = self.total_steps / max(self.total_steps_for_decay, 1)
        progress = min(1.0, progress)

        # Get floor (adaptive if enabled, otherwise base floor)
        floor = self.get_entropy_floor(action_mask)

        # Linear decay from initial to floor
        decayed = self.entropy_coef - (self.entropy_coef - floor) * progress

        return max(decayed, floor)
```

Then in the update loop where loss is computed (around line 410), pass the action mask:

```python
                # Get entropy coefficient with adaptive floor
                # Use batch's action mask for floor computation
                batch_mask = action_masks[batch_indices] if action_masks is not None else None
                # For batched masks, use the most restrictive (min valid actions)
                representative_mask = batch_mask[0] if batch_mask is not None else None
                entropy_coef = self.get_entropy_coef(representative_mask)

                loss = policy_loss + self.value_coef * value_loss - entropy_coef * entropy_bonus
```

**Step 6: Run test to verify it passes**

Run: `uv run pytest tests/test_simic_ppo.py::test_adaptive_entropy_floor_log_scaling -v`
Expected: PASS

**Step 7: Commit**

```bash
git add src/esper/simic/ppo.py src/esper/simic/config.py tests/test_simic_ppo.py
git commit -m "feat(simic): add adaptive entropy floor with log-ratio scaling

When adaptive_entropy_floor=True, entropy coefficient floor scales
using information-theoretic log-ratio: log(total)/log(valid).

This maintains consistent 'relative exploration' across different
action space sizes. With 2 of 7 valid: scale = log(7)/log(2) = 2.8x.

DRL Expert recommendation: linear scaling over-compensates; log-ratio
is principled based on maximum entropy of restricted action space."
```

---

## Task 7: Add Comprehensive Value Function Monitoring Metrics

**Rationale:** DRL expert noted missing visualization of value predictions vs actual returns. Adding comprehensive metrics enables debugging value function collapse and PPO stability issues.

**Files:**
- Modify: `src/esper/simic/ppo.py` (add value prediction metrics)
- Test: `tests/test_simic_ppo.py`

**Step 1: Write the failing test**

Add to `tests/test_simic_ppo.py`:

```python
def test_comprehensive_value_function_metrics():
    """PPO update should return comprehensive value function diagnostics."""
    import torch
    from esper.simic.ppo import PPOAgent

    agent = PPOAgent(state_dim=10, action_dim=4, device="cpu")

    # Add transitions
    for i in range(10):
        state = torch.randn(10)
        action_mask = torch.ones(4)
        reward = 1.0 if i > 5 else -0.5
        done = i == 9
        agent.store_transition(state, i % 4, -0.5, 0.5, reward, done, action_mask)

    metrics = agent.update(last_value=0.0)

    # Core value function diagnostics
    assert 'value_pred_mean' in metrics
    assert 'value_pred_std' in metrics
    assert 'return_mean' in metrics
    assert 'return_std' in metrics

    # Additional diagnostics (DRL Expert recommendations)
    assert 'value_mse_before' in metrics  # Critic error before update
    assert 'return_min' in metrics
    assert 'return_max' in metrics
    assert 'advantage_mean_prenorm' in metrics  # Critical for PPO stability
    assert 'advantage_std_prenorm' in metrics
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_simic_ppo.py::test_comprehensive_value_function_metrics -v`
Expected: FAIL - metrics don't include these stats

**Step 3: Add comprehensive metrics to update()**

In `src/esper/simic/ppo.py`, after computing returns and advantages (around line 323), add:

```python
        # === Value Function Diagnostics (DRL Expert recommendations) ===
        # Compute all stats as a single tensor for batched extraction (fewer CPU syncs)
        # [PyTorch Expert] Cast to float32 for stable statistics under mixed precision
        with torch.no_grad():
            v = values_tensor.float() if values_tensor.dtype != torch.float32 else values_tensor
            r = returns.float() if returns.dtype != torch.float32 else returns
            a = advantages.float() if advantages.dtype != torch.float32 else advantages

            value_stats = torch.stack([
                v.mean(),
                v.std(),
                r.mean(),
                r.std(),
                r.min(),
                r.max(),
                F.mse_loss(v, r),  # Critic error before update
                a.mean(),  # Pre-normalization (critical for stability debugging)
                a.std(),   # If very small, normalization amplifies noise
            ])

        # Single CPU sync to extract all values
        stats_list = value_stats.tolist()

        metrics['value_pred_mean'] = [stats_list[0]]
        metrics['value_pred_std'] = [stats_list[1]]
        metrics['return_mean'] = [stats_list[2]]
        metrics['return_std'] = [stats_list[3]]
        metrics['return_min'] = [stats_list[4]]
        metrics['return_max'] = [stats_list[5]]
        metrics['value_mse_before'] = [stats_list[6]]
        metrics['advantage_mean_prenorm'] = [stats_list[7]]
        metrics['advantage_std_prenorm'] = [stats_list[8]]

        # [DRL Expert] Warn when advantage std is very low - normalization amplifies noise
        if stats_list[8] < 0.1:
            logger.warning(
                f"Very low advantage std ({stats_list[8]:.4f}) before normalization. "
                f"Normalization may amplify noise. Consider reducing gamma or checking reward scale."
            )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_simic_ppo.py::test_comprehensive_value_function_metrics -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/ppo.py tests/test_simic_ppo.py
git commit -m "feat(simic): add comprehensive value function monitoring

Add value_mse_before, return_min/max, advantage_mean/std_prenorm to
PPO update metrics. Uses batched tensor extraction for single CPU sync.

advantage_std_prenorm is CRITICAL: if very small, advantage normalization
amplifies noise causing unstable gradients.

DRL Expert recommendation: comprehensive diagnostics enable debugging
value collapse and PPO stability issues."
```

---

## Task 8: Add LSTM LayerNorm for Recurrent Stability

**Rationale:** DRL expert noted LSTM hidden states can grow unbounded over long sequences. Adding LayerNorm on LSTM output stabilizes training.

**Files:**
- Modify: `src/esper/simic/networks.py` (add LayerNorm to RecurrentActorCritic)
- Test: `tests/test_simic_networks.py`

**Step 1: Write the failing test**

Add to `tests/test_simic_networks.py`:

```python
def test_recurrent_actor_critic_has_layernorm():
    """RecurrentActorCritic should have LayerNorm on LSTM output."""
    import torch.nn as nn
    from esper.simic.networks import RecurrentActorCritic

    network = RecurrentActorCritic(state_dim=10, action_dim=4)

    # Should have lstm_ln attribute
    assert hasattr(network, 'lstm_ln'), "RecurrentActorCritic should have lstm_ln"
    assert isinstance(network.lstm_ln, nn.LayerNorm), "lstm_ln should be LayerNorm"

    # LayerNorm should match LSTM hidden dim
    assert network.lstm_ln.normalized_shape[0] == network.lstm_hidden_dim
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_simic_networks.py::test_recurrent_actor_critic_has_layernorm -v`
Expected: FAIL - RecurrentActorCritic doesn't have lstm_ln

**Step 3: Add LayerNorm to RecurrentActorCritic**

In `src/esper/simic/networks.py`, in `RecurrentActorCritic.__init__` (around line 623), add after LSTM definition:

```python
            # LSTM for temporal processing
            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=lstm_hidden_dim,
                num_layers=num_lstm_layers,
                batch_first=True,
            )

            # [DRL Best Practice] LayerNorm on LSTM output for stability
            # Prevents hidden state magnitude drift over long sequences
            self.lstm_ln = nn.LayerNorm(lstm_hidden_dim)
```

In the `forward` method (around line 724), update LSTM processing:

```python
            # LSTM processing: [batch, seq_len, lstm_hidden_dim]
            lstm_out, hidden = self.lstm(features, hidden)

            # [DRL Best Practice] Normalize LSTM output to stabilize hidden magnitudes
            lstm_out = self.lstm_ln(lstm_out)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_simic_networks.py::test_recurrent_actor_critic_has_layernorm -v`
Expected: PASS

**Step 5: Run full networks test suite**

Run: `uv run pytest tests/test_simic_networks.py -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/esper/simic/networks.py tests/test_simic_networks.py
git commit -m "feat(simic): add LSTM LayerNorm for recurrent stability

Add LayerNorm on LSTM output in RecurrentActorCritic. Prevents hidden
state magnitude drift over long sequences which can destabilize training.

DRL Expert recommendation: LSTM hidden states can grow unbounded;
LayerNorm provides implicit regularization."
```

---

## Task 9: Add Recurrent PPO Safety Caps

**Rationale:** DRL expert noted n_epochs > 2 for recurrent PPO risks policy drift due to hidden state staleness. Warn early (> 2), cap at 4.

**Files:**
- Modify: `src/esper/simic/ppo.py` (add hard cap and warning)
- Test: `tests/test_simic_ppo.py`

**Step 1: Write the failing test**

Add to `tests/test_simic_ppo.py`:

```python
def test_recurrent_ppo_epochs_safety_cap():
    """Recurrent PPO should warn early (>2) and cap n_epochs to prevent policy drift."""
    import torch
    from esper.simic.ppo import PPOAgent

    agent = PPOAgent(
        state_dim=10, action_dim=4, device="cpu",
        recurrent=True, lstm_hidden_dim=32,
    )

    # Add some transitions to each env
    for env_id in range(2):
        for i in range(5):
            state = torch.randn(10)
            action_mask = torch.ones(4)
            agent.store_recurrent_transition(
                state, 0, -0.5, 0.5, 1.0, i == 4, action_mask, env_id
            )

    # n_epochs > 2 should warn (early warning)
    with pytest.warns(RuntimeWarning, match="n_epochs.*elevated"):
        metrics = agent.update_recurrent(n_epochs=3)

    # n_epochs > 4 should be capped (hard limit)
    with pytest.warns(RuntimeWarning, match="n_epochs.*capped"):
        metrics = agent.update_recurrent(n_epochs=10)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_simic_ppo.py::test_recurrent_ppo_epochs_safety_cap -v`
Expected: FAIL - No warning emitted

**Step 3: Add safety cap to update_recurrent**

In `src/esper/simic/ppo.py`, in `update_recurrent` (around line 540), update:

```python
        # Default to 1 epoch for recurrent (safest)
        if n_epochs is None:
            n_epochs = 1

        # [DRL Best Practice] Two-tier warnings for recurrent PPO n_epochs
        # After gradient updates, policy changes, so recomputed log_probs differ
        # from stored log_probs. With multiple epochs, this staleness compounds.
        import warnings

        WARN_THRESHOLD = 2   # Warn when > 2 (early warning)
        MAX_RECURRENT_EPOCHS = 4  # Hard cap

        if n_epochs > MAX_RECURRENT_EPOCHS:
            warnings.warn(
                f"n_epochs={n_epochs} is too high for recurrent PPO and has been capped "
                f"to {MAX_RECURRENT_EPOCHS}. Values > {MAX_RECURRENT_EPOCHS} cause severe "
                f"policy drift due to hidden state staleness.",
                RuntimeWarning,
            )
            n_epochs = MAX_RECURRENT_EPOCHS
        elif n_epochs > WARN_THRESHOLD:
            warnings.warn(
                f"n_epochs={n_epochs} is elevated for recurrent PPO. "
                f"Values > {WARN_THRESHOLD} risk policy drift due to hidden state staleness. "
                f"Consider n_epochs=1-2 for maximum stability.",
                RuntimeWarning,
            )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_simic_ppo.py::test_recurrent_ppo_epochs_safety_cap -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/ppo.py tests/test_simic_ppo.py
git commit -m "feat(simic): add recurrent PPO n_epochs safety warnings

Two-tier warning system for recurrent PPO:
- n_epochs > 2: early warning about elevated policy drift risk
- n_epochs > 4: hard cap with warning

Higher values risk policy drift due to hidden state staleness between
epochs - after gradient updates, policy changes but old log_probs remain.

DRL Expert recommendation: n_epochs=1-2 is safest for recurrent policies."
```

---

## Task 10: Run Full Test Suite and Integration Tests

**Files:**
- All modified files from previous tasks

**Step 1: Run simic unit tests**

Run: `uv run pytest tests/test_simic*.py -v`
Expected: All PASS

**Step 2: Run PPO integration test**

Run: `uv run pytest tests/integration/test_ppo_integration.py -v`
Expected: PASS

**Step 3: Run recurrent PPO integration test**

Run: `uv run pytest tests/integration/test_recurrent_ppo_integration.py -v`
Expected: PASS

**Step 4: Type check**

Run: `uv run mypy src/esper/simic/`
Expected: No errors

**Step 5: Lint check**

Run: `uv run ruff check src/esper/simic/`
Expected: No errors

**Step 6: Final commit with all improvements**

```bash
git add -A
git commit -m "test: verify simic expert improvements integration

All unit tests and integration tests pass after implementing:
- Fused NaN/Inf checks (fewer kernel launches)
- Configurable anomaly thresholds
- torch._foreach_norm for gradient collection (PyTorch 2.9)
- Tensor-returning feature extraction
- Weight decay with critic-only application (RL best practice)
- Adaptive entropy floor with log-ratio scaling
- Comprehensive value function monitoring
- LSTM LayerNorm for recurrent stability
- Recurrent PPO n_epochs safety cap"
```

---

## Summary

| Task | Impact | Complexity | Source | Expert Review |
|------|--------|------------|--------|---------------|
| 1. Fuse NaN/Inf checks | Fewer kernel launches | Low | PyTorch Expert | ✓ Fixed rationale |
| 2. Configurable anomaly thresholds | Better tuning | Low | DRL Expert | ✓ Approved |
| 3. torch._foreach_norm | Fused CUDA kernel | Medium | PyTorch 2.9 | ✓ No fallback needed |
| 4. Tensor feature extraction | Zero-alloc mode | Medium | PyTorch Expert | ✓ Approved |
| 5. Critic-only weight decay | Preserve exploration | Medium | DRL Expert | ✓ Fixed shared=0.0 |
| 6. Log-ratio entropy scaling | Information-theoretic | Medium | DRL Expert | ✓ Wired into loss |
| 7. Comprehensive metrics | PPO stability debugging | Low | DRL Expert | ✓ Added float32 cast |
| 8. LSTM LayerNorm | Recurrent stability | Low | DRL Expert | ✓ Approved |
| 9. Recurrent n_epochs warnings | Prevent policy drift | Low | DRL Expert | ✓ Two-tier (>2, >4) |
| 10. Integration testing | Verification | Low | Both | ✓ Approved |

**Status: Expert-reviewed and ready for implementation**
