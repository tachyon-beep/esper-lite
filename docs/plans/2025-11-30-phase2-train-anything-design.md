# Phase 2 Design: Train Anything Foundation

**Status:** Approved
**Date:** 2025-11-30
**Authors:** Claude (with DRL and PyTorch expert review)

## Overview

Phase 2 extends Esper from single-task (CIFAR-10 classification with CNN) to multi-task support (also TinyStories language modeling with Transformer). The goal is to establish a "train anything" foundation without over-abstracting.

### Key Principles

1. **Loss-primary rewards** — Loss is universal; accuracy is optional telemetry
2. **HostProtocol abstraction** — Hosts are pluggable; Kasmina doesn't know their internals
3. **Blueprint plugin system** — Adding blueprints automatically extends action space
4. **No legacy code** — Clean breaks, no compatibility shims

---

## 1. HostProtocol Abstraction

### Interface

```python
from typing import Protocol
from dataclasses import dataclass
from torch import Tensor

@dataclass(frozen=True, slots=True)
class InjectionPoint:
    """Describes where a seed can be injected."""
    after_layer_idx: int  # Injection occurs AFTER this layer (-1 = before first)
    position: str         # "post_block", "post_attn", "post_mlp"
    dim: int              # Channel/embedding dimension

class HostProtocol(Protocol):
    """Contract for graftable host networks."""

    @property
    def injection_points(self) -> dict[str, InjectionPoint]:
        """Map of point_id -> injection point metadata."""
        ...

    def forward_through_layer(self, x: Tensor, last_layer_idx: int) -> Tensor:
        """Forward pass through layers [0, last_layer_idx] inclusive."""
        ...

    def forward_from_layer(self, x: Tensor, first_layer_idx: int) -> Tensor:
        """Forward pass from layer first_layer_idx to output."""
        ...

    def forward(self, x: Tensor) -> Tensor:
        """Full forward pass (no seeds active)."""
        ...
```

### Implementations

**HostCNN** (existing, adapted):
- Single injection point after block2: `{"block2_post": InjectionPoint(1, "post_block", 64)}`
- `forward_through_layer(x, 1)` runs blocks 0-1
- `forward_from_layer(h, 2)` runs block 2 + classifier

**TransformerHost** (new):
- One injection point per layer: `{"layer_i_post_block": InjectionPoint(i, "post_block", n_embd)}`
- GPT-style decoder: tok_emb, pos_emb, n_layer TransformerBlocks, ln_f, lm_head
- Weight tying: `tok_emb.weight = head.weight`
- Pre-norm architecture for stable seed grafting

---

## 2. Loss-Primary Rewards

### Rationale

Loss is universal (every differentiable model has it). Accuracy is classification-specific. By keying rewards off loss_delta, we support any task type without branching.

### Reward Function

```python
def compute_shaped_reward(
    action: int,
    loss_delta: float,      # current_loss - previous_loss (negative = improvement)
    val_loss: float,
    seed_info: SeedInfo | None,
    epoch: int,
    max_epochs: int,
    config: RewardConfig,
) -> float:
    reward = 0.0

    # Normalize loss_delta for cross-task comparability
    normalized_delta = loss_delta / config.typical_loss_delta_std

    # Clip to handle architecture-change spikes
    clipped = clip(normalized_delta, -config.max_loss_delta, config.max_loss_delta)

    # Asymmetric: forgive temporary regressions
    if clipped > 0:
        clipped *= config.regression_penalty_scale  # 0.5

    reward += (-clipped) * config.loss_delta_weight

    # Stage bonuses (unchanged from Phase 1)
    if seed_info:
        reward += stage_bonus(seed_info.stage, config)

    # Terminal bonus: normalized by achievable range
    if epoch == max_epochs:
        improvement = config.baseline_loss - val_loss
        normalized = clamp(improvement / config.achievable_range, 0, 1)
        reward += normalized * config.terminal_loss_weight

    return reward
```

### Configuration

```python
@dataclass
class RewardConfig:
    loss_delta_weight: float = 5.0          # Reduced from 10 per DRL expert
    max_loss_delta: float = 5.0             # After normalization
    regression_penalty_scale: float = 0.5   # Asymmetric clipping
    typical_loss_delta_std: float = 0.1     # Task-specific, set via TaskConfig
    baseline_loss: float = 2.3              # Task-specific
    target_loss: float = 0.3                # Task-specific (achievable)
    terminal_loss_weight: float = 1.0

    @property
    def achievable_range(self) -> float:
        return self.baseline_loss - self.target_loss
```

### Observation Space Changes

Remove accuracy fields from core observations. New TensorSchema (23 features):
- Core state: epoch, global_step
- Loss metrics: train_loss, val_loss, loss_delta
- Loss history: 5 slots
- Best tracking: best_val_loss
- Seed state: has_active_seed, seed_stage, seed_epochs_in_stage, seed_alpha, seed_improvement, available_slots

Accuracy retained only in `TrainingSignals` for human telemetry, not policy input.

---

## 3. Blueprint Plugin System

### Registry

```python
@dataclass(frozen=True, slots=True)
class BlueprintSpec:
    name: str
    topology: str                           # "cnn" or "transformer"
    factory: Callable[[int], nn.Module]     # dim -> seed module
    param_estimate: int
    description: str = ""

class BlueprintRegistry:
    _blueprints: dict[str, BlueprintSpec] = {}

    @classmethod
    def register(cls, name: str, topology: str, param_estimate: int, description: str = ""):
        def decorator(factory):
            cls._blueprints[f"{topology}:{name}"] = BlueprintSpec(...)
            return factory
        return decorator

    @classmethod
    def list_for_topology(cls, topology: str) -> list[BlueprintSpec]:
        return sorted([s for s in cls._blueprints.values() if s.topology == topology],
                      key=lambda s: s.param_estimate)
```

### CNN Blueprints (existing)

| Name | Params | Description |
|------|--------|-------------|
| norm | ~100 | BatchNorm only |
| attention | ~2k | SE-style channel attention |
| depthwise | ~4.8k | Depthwise-separable conv |
| conv_enhance | ~74k | Heavy conv block |

### Transformer Blueprints (new)

| Name | Params | Description |
|------|--------|-------------|
| norm | ~800 | LayerNorm only |
| lora | ~6k | Low-rank adapter (rank=8) |
| attention | ~50k | Additional self-attention head |
| mlp | ~1.2M | Additional MLP (4x expansion) |

### Dynamic Action Space

```python
def build_action_enum(topology: str) -> type[IntEnum]:
    blueprints = BlueprintRegistry.list_for_topology(topology)
    members = {"WAIT": 0}
    for i, spec in enumerate(blueprints, start=1):
        members[f"GERMINATE_{spec.name.upper()}"] = i
    members["ADVANCE"] = len(blueprints) + 1
    members["CULL"] = len(blueprints) + 2
    return IntEnum(f"{topology.title()}Action", members)
```

---

## 4. TransformerHost Implementation

```python
class TransformerHost(nn.Module):
    def __init__(
        self,
        vocab_size: int = 50257,
        n_embd: int = 384,
        n_head: int = 6,
        n_layer: int = 6,
        block_size: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_layer = n_layer
        self.n_embd = n_embd

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            TransformerBlock(n_embd, n_head, block_size, dropout)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying
        self.tok_emb.weight = self.head.weight

        # Injection points
        self._injection_points = {
            f"layer_{i}_post_block": InjectionPoint(i, "post_block", n_embd)
            for i in range(n_layer)
        }

    @property
    def injection_points(self) -> dict[str, InjectionPoint]:
        return self._injection_points

    def forward_through_layer(self, x: Tensor, last_layer_idx: int) -> Tensor:
        B, T = x.shape
        h = self.tok_emb(x) + self.pos_emb(torch.arange(T, device=x.device))
        h = self.drop(h)
        for i in range(last_layer_idx + 1):
            h = self.layers[i](h)
        return h

    def forward_from_layer(self, h: Tensor, first_layer_idx: int) -> Tensor:
        for i in range(first_layer_idx, self.n_layer):
            h = self.layers[i](h)
        return self.head(self.ln_f(h))

    def forward(self, x: Tensor) -> Tensor:
        h = self.forward_through_layer(x, self.n_layer - 1)
        return self.head(self.ln_f(h))
```

---

## 5. TinyStories Integration

### Dataset

```python
class TinyStoriesDataset(Dataset):
    def __init__(self, split: str, block_size: int = 256, max_samples: int | None = None):
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        dataset = load_dataset("roneneldan/TinyStories", split=split)

        # Tokenize and chunk
        all_tokens = []
        for example in dataset:
            all_tokens.extend(self.tokenizer.encode(example["text"]))

        self.examples = [
            all_tokens[i:i + block_size + 1]
            for i in range(0, len(all_tokens) - block_size - 1, block_size)
        ]

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        tokens = torch.tensor(self.examples[idx])
        return tokens[:-1], tokens[1:]  # x, y shifted by 1
```

### TaskConfig

```python
TINYSTORIES_TASK = TaskConfig(
    task_type="lm",
    topology="transformer",
    baseline_loss=10.8,             # ln(50257)
    target_loss=3.5,                # ~33 perplexity
    typical_loss_delta_std=0.15,
)
```

### Training Loop

No changes to `train_epoch_*` functions. Only `validate_and_get_metrics` gains a `task_type` parameter to compute perplexity for LM tasks.

---

## 6. Testing Strategy

### Unit Tests
- HostProtocol conformance for both hosts
- Blueprint registry and action space derivation
- Reward computation with clipping and normalization

### Integration Tests
- MorphogeneticModel with both host types
- Seed lifecycle on both topologies
- Gradient flow through grafted seeds

### Behavioral Tests
- Tamiyo learns on CIFAR (Phase 1 preserved)
- Tamiyo learns on TinyStories (Phase 2 validated)
- No regression in Phase 1 performance

### Property Tests
- Reward bounded across input space
- Monotonicity: more improvement = more reward

---

## 7. Migration Plan

### Files to Modify

| File | Changes |
|------|---------|
| `kasmina/host.py` | Add `HostProtocol`, adapt `HostCNN`, add `TransformerHost` |
| `kasmina/blueprints.py` | Split into `blueprints/cnn.py` and `blueprints/transformer.py` |
| `kasmina/blueprints/__init__.py` | Add `BlueprintRegistry` |
| `leyline/actions.py` | Add `build_action_enum()`, per-topology enums |
| `leyline/signals.py` | Remove `ACCURACY_*` from TensorSchema |
| `leyline/tasks.py` | New file with `TaskConfig` |
| `simic/rewards.py` | Loss-primary refactor, add normalization/clipping |
| `tolaria/trainer.py` | Add `task_type` parameter to validation |
| `utils/data.py` | Add `TinyStoriesDataset`, `load_tinystories()` |

### Files to Add

| File | Purpose |
|------|---------|
| `kasmina/blueprints/cnn.py` | CNN blueprint classes |
| `kasmina/blueprints/transformer.py` | Transformer blueprint classes |
| `kasmina/blueprints/registry.py` | Plugin registry |
| `leyline/tasks.py` | TaskConfig definitions |

### Breaking Changes

- `TensorSchema` feature count: 27 → 23 (accuracy fields removed)
- Reward function signature: `acc_delta` → `loss_delta`
- Action enum: global `Action` → per-topology `CNNAction`/`TransformerAction`

---

## 8. Expert Review Notes

### DRL Expert (Approved)

**High priority:** Terminal bonus must use achievable-range normalization, not raw loss comparison.

**Medium priority:**
- Remove accuracy from observations to avoid spurious correlations
- Add per-task `typical_loss_delta_std` for cross-task comparability

**Low priority:**
- Consider 1-epoch grace period after germination (no regression penalty)
- Scale entropy coefficient by action space size

### PyTorch Expert (Approved)

**Medium priority:**
- Rename `forward_to_layer` → `forward_through_layer` for clarity
- Add topology validation in `germinate_seed()` to prevent cross-topology blueprints

**Low priority:**
- Use `torch.compile(dynamic=True)` for variable sequence lengths
- Ensure causal masking in TransformerBlock

**Deferred:**
- Pre-allocation for FSDP (not needed for Phase 2 single-GPU)

---

## 9. Success Criteria

1. **CIFAR-10 preserved:** Tamiyo achieves same mean return as Phase 1 (within 10%)
2. **TinyStories functional:** Tamiyo improves over random policy on LM task
3. **No legacy code:** Zero backwards-compatibility shims
4. **Plugin extensibility:** Adding a new blueprint requires only a decorated class
5. **Tests pass:** All unit, integration, and behavioral tests green
