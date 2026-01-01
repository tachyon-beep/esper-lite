# Kasmina Optimizations Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement high-priority PyTorch 2.9 optimizations and correctness fixes for the kasmina module based on specialist review findings.

**Architecture:** Performance optimizations (channels_last, FlexAttention caching) applied at host/blueprint level with optional constructor parameters. Fixes address safety (.reshape) and documentation (compile strategy).

**Tech Stack:** PyTorch 2.9, Python 3.13, pytest

**Review Status:** Approved by DRL Expert, PyTorch Expert, and Code Reviewer (with incorporated fixes)

---

## Phase 1: High-Priority Performance (ENH-1)

### Task 1: Add channels_last Support to CNNHost

**Files:**
- Modify: `src/esper/kasmina/host.py:70-80`
- Test: `tests/kasmina/test_channels_last.py` (create)

**Step 1: Write the failing test for channels_last memory format**

```python
# tests/kasmina/test_channels_last.py
"""Tests for channels_last memory format support in CNNHost."""

import pytest
import torch

from esper.kasmina.host import CNNHost


class TestChannelsLastFormat:
    """Test channels_last memory format optimization."""

    def test_cnnhost_accepts_channels_last_input(self):
        """CNNHost should process channels_last tensors without error."""
        host = CNNHost(num_classes=10, n_blocks=3, base_channels=32)
        x = torch.randn(2, 3, 32, 32).to(memory_format=torch.channels_last)

        output = host(x)

        assert output.shape == (2, 10)

    def test_cnnhost_with_memory_format_parameter(self):
        """CNNHost with memory_format converts input automatically."""
        host = CNNHost(
            num_classes=10,
            n_blocks=3,
            base_channels=32,
            memory_format=torch.channels_last,
        )
        x = torch.randn(2, 3, 32, 32)  # contiguous format

        output = host(x)

        assert output.shape == (2, 10)

    def test_cnnhost_default_memory_format_is_none(self):
        """CNNHost defaults to None (no automatic conversion)."""
        host = CNNHost(num_classes=10, n_blocks=3, base_channels=32)

        # memory_format is always defined, defaults to None
        assert host.memory_format is None

    def test_channels_last_numerical_equivalence(self):
        """channels_last should produce numerically identical outputs."""
        torch.manual_seed(42)

        host_default = CNNHost(num_classes=10, n_blocks=3, base_channels=32)
        host_cl = CNNHost(
            num_classes=10,
            n_blocks=3,
            base_channels=32,
            memory_format=torch.channels_last,
        )

        # Sync weights
        host_cl.load_state_dict(host_default.state_dict())

        x = torch.randn(2, 3, 32, 32)
        torch.testing.assert_close(host_default(x), host_cl(x), rtol=1e-5, atol=1e-5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_channels_last_gpu_performance(self):
        """Verify channels_last works on GPU (where perf benefit occurs)."""
        host = CNNHost(
            num_classes=10,
            n_blocks=3,
            base_channels=32,
            memory_format=torch.channels_last,
        ).cuda()
        x = torch.randn(4, 3, 32, 32).cuda()

        output = host(x)

        assert output.shape == (4, 10)
        assert output.device.type == "cuda"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/kasmina/test_channels_last.py -v`
Expected: FAIL with "unexpected keyword argument 'memory_format'"

**Step 3: Implement channels_last support in CNNHost**

Edit `src/esper/kasmina/host.py`:

```python
class CNNHost(nn.Module):
    """CNN host with dynamic blocks and injection points after each block (except the first).

    Mirrors TransformerHost's pattern: a ModuleList of blocks, a ModuleDict of slots keyed
    by block index, and a simple looped forward that applies slots as identities when unused.

    Args:
        num_classes: Number of output classes.
        n_blocks: Number of conv blocks (minimum 2).
        base_channels: Base channel count, doubled each block.
        memory_format: Optional memory format for input conversion. Use torch.channels_last
            for 20-40% speedup on NVIDIA Tensor Cores (Ampere+). Seed modules should be
            compatible with channels_last format for optimal performance. Default: None.
    """

    def __init__(
        self,
        num_classes: int = 10,
        n_blocks: int = 3,
        base_channels: int = 32,
        memory_format: torch.memory_format | None = None,
    ):
        super().__init__()
        if n_blocks < 2:
            raise ValueError("CNNHost requires at least 2 blocks to expose an injection point")

        self.n_blocks = n_blocks
        self.base_channels = base_channels
        self.memory_format = memory_format

        # Build blocks with doubling channels each stage
        blocks: list[nn.Module] = []
        in_c = 3
        for i in range(n_blocks):
            out_c = base_channels * (2 ** i)
            blocks.append(ConvBlock(in_c, out_c))
            in_c = out_c
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.MaxPool2d(2, 2)

        # Slots after each block except the first
        self._slot_indices = tuple(range(1, n_blocks))
        self._slot_keys = tuple(f"block{idx + 1}_post" for idx in self._slot_indices)
        self.slots = nn.ModuleDict({k: nn.Identity() for k in self._slot_keys})

        # Classifier maps final channels → logits
        self.classifier = nn.Linear(in_c, num_classes)

    # ... (keep @property injection_points, register_slot, unregister_slot unchanged)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert to specified memory format if configured
        if self.memory_format is not None:
            x = x.to(memory_format=self.memory_format)

        slot_idx = 0
        for idx, block in enumerate(self.blocks):
            x = self.pool(block(x))
            # Use pre-computed _slot_indices instead of string formatting
            if idx in self._slot_indices:
                x = self.slots[self._slot_keys[slot_idx]](x)
                slot_idx += 1

        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.classifier(x)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/kasmina/test_channels_last.py -v`
Expected: PASS (all 5 tests)

**Step 5: Run full test suite to check for regressions**

Run: `uv run pytest tests/kasmina/ -v --tb=short`
Expected: All existing tests still pass

**Step 6: Commit**

```bash
git add src/esper/kasmina/host.py tests/kasmina/test_channels_last.py
git commit -m "feat(kasmina): add channels_last memory format support to CNNHost

Optional memory_format parameter enables 20-40% speedup on NVIDIA Tensor Cores
by converting input to channels_last format. Defaults to None (opt-in behavior
to avoid unintended changes to existing training runs).

Refs: kasmina-specialist-review.md ENH-1"
```

---

## Phase 2: Correctness Fixes (FIX-1, FIX-2, FIX-3)

### Task 2: Replace .view() with .reshape() in AttentionSeed

**Files:**
- Modify: `src/esper/kasmina/blueprints/cnn.py:93`
- Test: `tests/kasmina/test_blueprints_cnn.py` (verify existing tests pass)

**Step 1: Apply the safer .reshape() fix**

This is a defensive change. While AdaptiveAvgPool2d outputs are always contiguous,
`.reshape()` is safer and has zero overhead when tensors are contiguous.

Edit `src/esper/kasmina/blueprints/cnn.py:91-95`:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    b, c, _, _ = x.size()
    y = self.avg_pool(x).reshape(b, c)  # reshape handles non-contiguous safely
    y = self.fc(y).reshape(b, c, 1, 1)  # Also fix the second view
    return x * y.expand_as(x)
```

**Step 2: Run existing tests to verify no regression**

Run: `uv run pytest tests/kasmina/test_blueprints_cnn.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add src/esper/kasmina/blueprints/cnn.py
git commit -m "fix(kasmina): use reshape instead of view in AttentionSeed

.reshape() handles both contiguous and non-contiguous tensors safely,
returning a view when possible and a copy when necessary.

Refs: kasmina-specialist-review.md FIX-1"
```

---

### Task 3: Document torch.compile Strategy at Module Level

**Files:**
- Modify: `src/esper/kasmina/slot.py:1-6` (module docstring)

**Step 1: No test needed (documentation only)**

This is a documentation enhancement, not a behavioral change.

**Step 2: Add module-level compile strategy documentation**

Edit `src/esper/kasmina/slot.py` module docstring (lines 1-6):

```python
"""Kasmina Slot - Seed lifecycle management.

The SeedSlot manages a single seed module through its lifecycle:
germination -> training -> blending -> fossilization/culling.

torch.compile Strategy
----------------------
The SeedSlot.forward() method is decorated with @torch.compiler.disable because:

1. Stage-dependent control flow (self.state.stage) creates separate compiled graphs
   for each execution path (TRAINING vs BLENDING vs FOSSILIZED), wasting compilation time
2. The conditional isolation logic (isolate_gradients) and alpha-based blending paths
   would require recompilation on every stage transition
3. Stage transitions are infrequent; compilation overhead far exceeds runtime benefit

The underlying tensor operations (ste_forward, blend_with_isolation in isolation.py)
ARE compile-compatible and traced when called from compiled code paths outside SeedSlot.
This means the actual math is optimized; only the dispatch logic is interpreted.

This is an intentional architectural decision, not a workaround.
"""
```

**Step 3: Verify module still imports correctly**

Run: `python -c "from esper.kasmina.slot import SeedSlot; print('OK')"`
Expected: "OK"

**Step 4: Commit**

```bash
git add src/esper/kasmina/slot.py
git commit -m "docs(kasmina): document torch.compile strategy in slot module

Explains why SeedSlot.forward() uses @torch.compiler.disable and confirms
underlying tensor ops (ste_forward, blend_with_isolation) are compile-compatible.

Refs: kasmina-specialist-review.md FIX-2"
```

---

### Task 4: Document Entropy Coefficient Normalization

**Files:**
- Modify: `src/esper/simic/ppo.py` (add comment near entropy_coef)

**Step 1: Locate entropy_coef definition**

The entropy_coef is in PPOConfig dataclass. Add clarifying comment.

**Step 2: Add documentation comment**

Find the `entropy_coef` field in PPOConfig and add:

```python
# Entropy coefficient operates on NORMALIZED entropy [0, 1], not raw nats.
# MaskedCategorical normalizes by log(num_valid_actions), so:
#   0.05 on normalized scale ≈ 0.1 on raw nats with 7 actions
#   0.05 on normalized scale ≈ 0.035 on raw nats with 2 actions
# This keeps exploration pressure consistent across action mask sizes.
entropy_coef: float = 0.05
```

**Step 3: Verify no syntax errors**

Run: `python -m py_compile src/esper/simic/ppo.py`
Expected: No output (success)

**Step 4: Commit**

```bash
git add src/esper/simic/ppo.py
git commit -m "docs(simic): clarify entropy_coef operates on normalized entropy

MaskedCategorical normalizes entropy by log(num_valid_actions), producing
values in [0, 1]. Documents that entropy_coef=0.05 is calibrated for this scale.

Refs: kasmina-specialist-review.md FIX-3"
```

---

## Phase 3: FlexAttention Optimization (ENH-2)

### Task 5: Add Block Mask Caching to FlexAttentionSeed

**Files:**
- Modify: `src/esper/kasmina/blueprints/transformer.py:119-150`
- Test: `tests/kasmina/test_flex_attention_cache.py` (create)

**Step 1: Write failing test for block mask caching**

```python
# tests/kasmina/test_flex_attention_cache.py
"""Tests for FlexAttention block mask caching."""

import pytest
import torch

# Skip entire module if FlexAttention not available
pytest.importorskip("torch.nn.attention.flex_attention")

from esper.kasmina.blueprints.transformer import create_flex_attention_seed


class TestFlexAttentionCache:
    """Test block mask caching in FlexAttentionSeed."""

    def test_flex_attention_seed_creates_cache(self):
        """FlexAttentionSeed should have block mask cache after forward."""
        seed = create_flex_attention_seed(dim=64, n_head=4)
        x = torch.randn(2, 16, 64)

        _ = seed(x)

        # Cache is always initialized, check it has entries after forward
        assert len(seed._block_mask_cache) > 0

    def test_flex_attention_seed_reuses_cache(self):
        """FlexAttentionSeed should reuse cached mask for same seq_len."""
        seed = create_flex_attention_seed(dim=64, n_head=4)
        x = torch.randn(2, 16, 64)

        _ = seed(x)
        cache_after_first = dict(seed._block_mask_cache)

        _ = seed(x)  # Same seq_len
        cache_after_second = dict(seed._block_mask_cache)

        # Cache should have same entries (reused, not recreated)
        assert cache_after_first.keys() == cache_after_second.keys()

    def test_flex_attention_seed_different_seq_lens(self):
        """FlexAttentionSeed should cache different seq_lens separately."""
        seed = create_flex_attention_seed(dim=64, n_head=4)

        _ = seed(torch.randn(2, 16, 64))
        _ = seed(torch.randn(2, 32, 64))

        # Should have 2 cache entries
        assert len(seed._block_mask_cache) == 2

    def test_flex_attention_seed_cache_limit(self):
        """FlexAttentionSeed cache should not grow unbounded."""
        seed = create_flex_attention_seed(dim=64, n_head=4)

        # Create many different seq_lens
        for seq_len in range(8, 128, 4):
            _ = seed(torch.randn(1, seq_len, 64))

        # Cache should be bounded (LRU with max 8 entries)
        assert len(seed._block_mask_cache) <= 8

    def test_flex_attention_output_unchanged(self):
        """Cached version should produce same output as uncached."""
        seed = create_flex_attention_seed(dim=64, n_head=4)
        torch.manual_seed(42)
        x = torch.randn(2, 16, 64)

        # First call (creates cache)
        out1 = seed(x.clone())

        # Second call (uses cache)
        out2 = seed(x.clone())

        torch.testing.assert_close(out1, out2)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/kasmina/test_flex_attention_cache.py -v`
Expected: FAIL with "has no attribute '_block_mask_cache'"

**Step 3: Implement block mask caching**

Edit `src/esper/kasmina/blueprints/transformer.py` (replace FlexAttentionSeed class):

```python
# FlexAttention blueprint - conditionally registered
if _HAS_FLEX_ATTENTION:
    from collections import OrderedDict
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from torch.nn.attention.flex_attention import BlockMask

    @BlueprintRegistry.register(
        "flex_attention", "transformer", param_estimate=55000,
        description="FlexAttention with cached causal mask (PyTorch 2.5+)"
    )
    def create_flex_attention_seed(dim: int, n_head: int = 4) -> nn.Module:
        """FlexAttention seed with block mask caching for efficiency."""

        class FlexAttentionSeed(nn.Module):
            _CACHE_MAX_SIZE = 8  # LRU cache limit

            def __init__(self, dim: int, n_head: int):
                super().__init__()
                self.n_head = n_head
                self.head_dim = dim // n_head

                self.qkv = nn.Linear(dim, 3 * dim)
                self.proj = nn.Linear(dim, dim)
                nn.init.zeros_(self.proj.weight)
                nn.init.zeros_(self.proj.bias)

                # LRU cache: (seq_len, device_str) -> BlockMask
                # Use str(device) for reliable hashing across device creation paths
                self._block_mask_cache: OrderedDict[tuple[int, str], "BlockMask"] = OrderedDict()

            @torch.compiler.disable
            def _get_block_mask(self, seq_len: int, device: torch.device):
                """Get or create cached block mask for causal attention.

                Note: @torch.compiler.disable because cache mutation would cause
                graph breaks. The flex_attention call itself is still compiled.
                """
                # Use device string for reliable hashing
                key = (seq_len, str(device))

                if key in self._block_mask_cache:
                    # Move to end (most recently used)
                    self._block_mask_cache.move_to_end(key)
                    return self._block_mask_cache[key]

                # Create new mask
                def causal_mask_fn(b, h, q_idx, kv_idx):
                    return q_idx >= kv_idx

                block_mask = create_block_mask(
                    causal_mask_fn,
                    B=None,
                    H=None,
                    Q_LEN=seq_len,
                    KV_LEN=seq_len,
                    device=device,
                )

                # Add to cache with LRU eviction
                self._block_mask_cache[key] = block_mask
                if len(self._block_mask_cache) > self._CACHE_MAX_SIZE:
                    self._block_mask_cache.popitem(last=False)  # Remove oldest

                return block_mask

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                b, t, c = x.shape

                qkv = self.qkv(x).reshape(b, t, 3, self.n_head, self.head_dim)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]

                # Use cached block mask for causal attention
                block_mask = self._get_block_mask(t, x.device)
                out = flex_attention(q, k, v, block_mask=block_mask)

                out = out.transpose(1, 2).reshape(b, t, c)
                return x + self.proj(out)

        return FlexAttentionSeed(dim, n_head)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/kasmina/test_flex_attention_cache.py -v`
Expected: PASS (all 5 tests)

**Step 5: Run existing FlexAttention tests for regressions**

Run: `uv run pytest tests/kasmina/ -v -k flex`
Expected: All tests pass

**Step 6: Commit**

```bash
git add src/esper/kasmina/blueprints/transformer.py tests/kasmina/test_flex_attention_cache.py
git commit -m "perf(kasmina): add block mask caching to FlexAttentionSeed

Pre-computes and caches causal BlockMask per (seq_len, device) using LRU
eviction (max 8 entries). Avoids recomputing mask every forward pass.

Key implementation details:
- Uses str(device) for reliable cache key hashing
- @torch.compiler.disable on cache method to avoid graph breaks
- Cache excluded from state_dict (recomputed on load)

Expected 10-20% attention speedup for repeated sequence lengths.

Refs: kasmina-specialist-review.md ENH-2"
```

---

## Phase 4: Experiment Setup (EXP-1, EXP-2)

### Task 6: Create GAE Lambda Ablation Experiment Config

**Files:**
- Create: `experiments/gae_lambda_ablation.py`
- Create: `experiments/__init__.py` (if needed)

**Step 1: Create experiments directory structure**

```bash
mkdir -p experiments
touch experiments/__init__.py
```

**Step 2: Create ablation experiment script**

```python
# experiments/gae_lambda_ablation.py
"""GAE Lambda Ablation Experiment (EXP-1)

Tests hypothesis: Higher lambda improves seed lifecycle credit assignment.

Metrics to track (per DRL Expert review):
- episode_return_mean: Average episode returns
- episode_return_variance: Variance in returns (primary metric)
- fossilization_rate: successful / total seeds
- avg_seed_lifetime: Average epochs per seed
- explained_variance: V(s) quality - critical for credit assignment
- advantage_std_prenorm: Before normalization
- policy_gradient_norm: Gradient magnitude
- value_loss: Value function convergence
- approx_kl: Policy stability
"""

from dataclasses import dataclass, field

# Import when running, not at module level to avoid circular imports
# from esper.simic.ppo import PPOConfig
# from esper.simic.training import train_ppo


@dataclass
class GaeLambdaExperiment:
    """Configuration for GAE lambda ablation.

    Extended range per DRL Expert recommendation:
    - 0.85: Lower bound to characterize inflection point
    - 0.90, 0.95, 0.98: Standard range
    - 1.0: Pure Monte Carlo control (tests if GAE variance reduction is needed)
    """

    name: str = "gae_lambda_ablation"
    seeds: tuple[int, ...] = (42, 123, 456)
    lambda_values: tuple[float, ...] = (0.85, 0.90, 0.95, 0.98, 1.0)
    max_epochs: int = 50
    num_envs: int = 4

    # Metrics to track per DRL Expert review
    metrics_to_track: tuple[str, ...] = field(default_factory=lambda: (
        "episode_return_mean",
        "episode_return_variance",
        "fossilization_rate",
        "avg_seed_lifetime",
        "explained_variance",
        "advantage_std_prenorm",
        "policy_gradient_norm",
        "value_loss",
        "approx_kl",
    ))

    def get_config_for_lambda(self, gae_lambda: float):
        """Create PPOConfig with specified lambda."""
        from esper.simic.ppo import PPOConfig

        return PPOConfig(
            gae_lambda=gae_lambda,
            # Keep other params at defaults for controlled comparison
            lr=3e-4,
            gamma=0.99,
            clip_ratio=0.2,
            entropy_coef=0.05,
        )

    def run_single(self, gae_lambda: float, seed: int) -> dict:
        """Run single experiment trial."""
        raise NotImplementedError("Implement with actual training loop")

    def run_all(self) -> list[dict]:
        """Run full ablation grid."""
        results = []
        for lambda_val in self.lambda_values:
            for seed in self.seeds:
                result = self.run_single(lambda_val, seed)
                result["gae_lambda"] = lambda_val
                result["seed"] = seed
                results.append(result)
        return results


if __name__ == "__main__":
    exp = GaeLambdaExperiment()
    print(f"GAE Lambda Ablation: {exp.lambda_values}")
    print(f"Seeds: {exp.seeds}")
    print(f"Total trials: {len(exp.lambda_values) * len(exp.seeds)}")
    print(f"Metrics: {exp.metrics_to_track}")
```

**Step 3: Verify script is valid Python**

Run: `python -m py_compile experiments/gae_lambda_ablation.py`
Expected: No output (success)

**Step 4: Commit**

```bash
git add experiments/
git commit -m "exp(simic): add GAE lambda ablation experiment scaffold

EXP-1: Tests hypothesis that higher lambda improves seed lifecycle credit
assignment. Extended grid per DRL Expert review:
- lambda in {0.85, 0.90, 0.95, 0.98, 1.0} x 3 seeds
- 1.0 (pure MC) tests if GAE variance reduction is needed for this domain
- Tracks 9 metrics including explained_variance and advantage_std_prenorm

Refs: kasmina-specialist-review.md EXP-1"
```

---

### Task 7: Create Reward Comparison Experiment Config

**Files:**
- Create: `experiments/reward_comparison.py`

**Step 1: Create reward comparison experiment script**

```python
# experiments/reward_comparison.py
"""Reward Function Comparison Experiment (EXP-2)

Tests hypothesis: Counterfactual attribution reduces variance compared to
accuracy-based reward.

Three-arm experiment per DRL Expert recommendation:
- CONTRIBUTION: Counterfactual-based (current default)
- ACCURACY: Raw accuracy delta
- SPARSE: Only reward on fossilize/cull (tests if dense shaping helps)

Metrics to track:
- policy_loss_variance: Primary metric
- explained_variance: Value function quality
- fossilization_quality: Contribution of fossilized seeds
- reward_mean, reward_std: Track reward scale across modes
- counterfactual_correlation: Even for ACCURACY mode, track counterfactual
- final_accuracy: Ultimate outcome metric
"""

from dataclasses import dataclass, field
from enum import Enum, auto


class RewardMode(Enum):
    """Reward function variants to compare."""
    CONTRIBUTION = auto()  # Counterfactual-based (current default)
    ACCURACY = auto()       # Raw accuracy delta
    SPARSE = auto()         # Only terminal rewards (fossilize/cull)


@dataclass
class RewardComparisonExperiment:
    """Configuration for reward function comparison.

    Three-arm design per DRL Expert recommendation to test:
    1. Does counterfactual reduce variance vs accuracy-based?
    2. Does dense shaping help at all vs sparse terminal rewards?
    """

    name: str = "reward_comparison"
    seeds: tuple[int, ...] = (42, 123, 456)
    reward_modes: tuple[RewardMode, ...] = (
        RewardMode.CONTRIBUTION,
        RewardMode.ACCURACY,
        RewardMode.SPARSE,
    )
    max_epochs: int = 50
    num_envs: int = 4

    # Metrics to track per DRL Expert review
    metrics_to_track: tuple[str, ...] = field(default_factory=lambda: (
        "policy_loss_variance",
        "explained_variance",
        "fossilization_quality",
        "reward_mean",
        "reward_std",
        "reward_min",
        "reward_max",
        "counterfactual_correlation",  # Track even for non-counterfactual modes
        "final_accuracy",
        "fossilization_rate",
    ))

    def get_reward_config(self, mode: RewardMode) -> dict:
        """Get reward configuration for mode."""
        if mode == RewardMode.CONTRIBUTION:
            return {
                "use_counterfactual": True,
                "contribution_weight": 5.0,
                "use_dense_shaping": True,
            }
        elif mode == RewardMode.ACCURACY:
            return {
                "use_counterfactual": False,
                "accuracy_weight": 5.0,
                "use_dense_shaping": True,
            }
        else:  # SPARSE
            return {
                "use_counterfactual": True,  # Still compute for tracking
                "contribution_weight": 5.0,
                "use_dense_shaping": False,  # Only terminal rewards
            }

    def run_single(self, mode: RewardMode, seed: int) -> dict:
        """Run single experiment trial."""
        raise NotImplementedError("Implement with actual training loop")

    def run_all(self) -> list[dict]:
        """Run full comparison."""
        results = []
        for mode in self.reward_modes:
            for seed in self.seeds:
                result = self.run_single(mode, seed)
                result["reward_mode"] = mode.name
                result["seed"] = seed
                results.append(result)
        return results


if __name__ == "__main__":
    exp = RewardComparisonExperiment()
    print(f"Reward Comparison: {[m.name for m in exp.reward_modes]}")
    print(f"Seeds: {exp.seeds}")
    print(f"Total trials: {len(exp.reward_modes) * len(exp.seeds)}")
    print(f"Metrics: {exp.metrics_to_track}")
```

**Step 2: Add import test for experiment scripts**

```python
# experiments/test_experiments.py
"""Basic import tests for experiment configs."""

def test_gae_lambda_experiment_importable():
    from experiments.gae_lambda_ablation import GaeLambdaExperiment
    exp = GaeLambdaExperiment()
    assert exp.name == "gae_lambda_ablation"
    assert len(exp.lambda_values) == 5  # Extended range


def test_reward_comparison_experiment_importable():
    from experiments.reward_comparison import RewardComparisonExperiment, RewardMode
    exp = RewardComparisonExperiment()
    assert exp.name == "reward_comparison"
    assert RewardMode.SPARSE in exp.reward_modes  # Three-arm design
```

**Step 3: Verify scripts are valid Python**

Run: `python -m py_compile experiments/reward_comparison.py`
Expected: No output (success)

**Step 4: Commit**

```bash
git add experiments/reward_comparison.py experiments/test_experiments.py
git commit -m "exp(simic): add reward function comparison experiment scaffold

EXP-2: Three-arm experiment per DRL Expert review:
- CONTRIBUTION: Counterfactual-based (current default)
- ACCURACY: Raw accuracy delta
- SPARSE: Terminal rewards only (tests if dense shaping helps)

Tracks 10 metrics including reward statistics and counterfactual correlation
even for non-counterfactual modes to enable post-hoc analysis.

Refs: kasmina-specialist-review.md EXP-2"
```

---

## Summary

**Total Tasks:** 7
**Review Status:** Approved with incorporated fixes

| Phase | Task | Type | Priority | Review Fixes Applied |
|-------|------|------|----------|---------------------|
| 1 | channels_last support | Enhancement | HIGH | Removed hasattr, added numerical equivalence test, fixed commit msg |
| 2 | .view() → .reshape() | Fix | LOW | Simplified test approach |
| 2 | Compile strategy docs | Fix | LOW | Clarified specialization vs breaks |
| 2 | Entropy coef docs | Fix | MEDIUM | None needed |
| 3 | FlexAttention caching | Enhancement | MEDIUM | str(device) key, @compiler.disable, proper types, removed hasattr |
| 4 | GAE lambda experiment | Experiment | HIGH | Extended lambda range, added metrics |
| 4 | Reward comparison experiment | Experiment | HIGH | Added SPARSE arm, extended metrics |

**Verification Commands:**

After all tasks complete:
```bash
# Full test suite
uv run pytest tests/ -v --tb=short

# Type checking
uv run mypy src/esper/kasmina/ src/esper/simic/

# Lint
uv run ruff check src/esper/

# Experiment imports
python -c "from experiments.gae_lambda_ablation import GaeLambdaExperiment; print('OK')"
python -c "from experiments.reward_comparison import RewardComparisonExperiment; print('OK')"
```
