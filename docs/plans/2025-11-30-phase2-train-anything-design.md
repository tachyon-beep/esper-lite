# Phase 2 Design: Train Anything Foundation

**Status:** Approved (v3 — with expert feedback incorporated)
**Date:** 2025-11-30
**Authors:** Claude (with DRL and PyTorch expert review)

## Overview

Phase 2 extends Esper from single-task (CIFAR-10 classification with CNN) to multi-task support (also TinyStories language modeling with Transformer). The goal is to establish a "train anything" foundation without over-abstracting.

### Key Principles

1. **Loss-primary rewards** — Loss is universal; accuracy is optional telemetry
2. **Host owns topology** — Hosts control their forward pass and call slots internally
3. **Kasmina plants, doesn't orchestrate** — Kasmina discovers injection points and registers slots; hosts do the wiring
4. **Blueprint plugin system** — Adding blueprints automatically extends action space
5. **No legacy code** — Clean breaks, no compatibility shims

---

## 1. HostProtocol Abstraction

### Design Philosophy

The "Kasmina as morphogenetic plane" model:
- **Hosts own their wiring internally** — they control when and where slots are called
- **Kasmina just plants SeedSlots** — it discovers injection points and registers modules
- **MorphogeneticModel calls `host.forward()`** — no partial-forward orchestration

This keeps the abstraction clean: hosts are plugins that declare growth sites and handle their own topology.

### Interface

```python
from typing import Protocol

class HostProtocol(Protocol):
    """Contract for graftable host networks.

    Hosts declare where seeds can be planted (injection_points),
    accept seed modules (register_slot), and handle their own
    forward pass including calling any attached slots.
    """

    @property
    def injection_points(self) -> dict[str, int]:
        """Map of slot_id -> channel/embedding dimension.

        Example:
            {"block2_post": 64}  # CNN
            {"layer_0_post_block": 384, "layer_1_post_block": 384, ...}  # Transformer
        """
        ...

    def register_slot(self, slot_id: str, slot: nn.Module) -> None:
        """Attach a seed module at the specified injection point.

        The host is responsible for calling this slot during forward().
        Implementation must move slot to correct device.

        Raises:
            ValueError: If slot_id is not a valid injection point.
        """
        ...

    def unregister_slot(self, slot_id: str) -> None:
        """Remove a seed module from the specified injection point.

        Resets the slot to identity (no-op) behavior.

        Raises:
            ValueError: If slot_id is not a valid injection point.
        """
        ...

    def forward(self, x: Tensor) -> Tensor:
        """Full forward pass, including any attached slots.

        The host calls registered slots at appropriate points internally.
        """
        ...
```

### Why This Design

- **Simple** — Four methods, clear responsibilities
- **Host-controlled** — Topology stays inside the host, not leaked to Kasmina
- **Compile-friendly** — Uses ModuleDict + Identity pattern (see implementation)
- **Extensible** — Complex topologies (U-Net, MoE) can implement the same interface

### What's Deferred to Phase 3+

- `forward_through_layer`/`forward_from_layer` for partial routing (recursive seeds)
- Multi-slot coordination across injection points
- Cross-host seed migration

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
    total_params: int,      # Fossilized + active seed params
    host_params: int,       # Baseline host params
    config: RewardConfig,
) -> float:
    """Compute shaped reward for seed lifecycle control.

    Sign convention:
        loss_delta < 0 means loss improved (went down)
        reward > 0 means good outcome
    """
    reward = 0.0

    # === PRIMARY: Loss improvement ===
    # Normalize for cross-task comparability
    normalized_delta = loss_delta / config.typical_loss_delta_std

    # Clip to handle architecture-change spikes
    clipped = clip(normalized_delta, -config.max_loss_delta, config.max_loss_delta)

    # Asymmetric: forgive temporary regressions (e.g., post-germination spikes)
    if clipped > 0:
        clipped *= config.regression_penalty_scale  # 0.5

    reward += (-clipped) * config.loss_delta_weight

    # === SECONDARY: Compute rent ===
    # Penalize excess parameters proportionally (blueprint-agnostic cost)
    # Note: Consider rent-free grace period for newly germinated seeds (see expert feedback)
    if host_params > 0 and total_params > 0:
        params_ratio = total_params / host_params
        reward -= config.compute_rent_weight * params_ratio

    # === TERTIARY: Stage bonuses (PBRS-compatible) ===
    # Uses potential-based shaping: gamma * Phi(s') - Phi(s)
    if seed_info is not None:
        reward += compute_pbrs_stage_bonus(seed_info, config)

    # === TERMINAL: Normalized improvement from baseline ===
    # Note: This is sparse — see expert feedback on potential-based leaking
    if epoch == max_epochs:
        improvement = config.baseline_loss - val_loss
        normalized = clamp(improvement / config.achievable_range, 0.0, 1.0)
        reward += normalized * config.terminal_loss_weight

    return reward


def compute_pbrs_stage_bonus(
    seed_info: SeedInfo,
    config: RewardConfig,
    gamma: float = 0.99,
) -> float:
    """PBRS-compatible stage bonus using potential function.

    Potential Phi(s) increases monotonically toward FOSSILIZED.
    Bonus = gamma * Phi(s') - Phi(s), computed as delta from previous step.

    This ensures shaping doesn't change optimal policy (Ng et al., 1999).
    """
    # Potential values per stage (monotonically increasing)
    STAGE_POTENTIALS = {
        SeedStage.DORMANT: 0.0,
        SeedStage.GERMINATED: 1.0,
        SeedStage.TRAINING: 2.0,
        SeedStage.BLENDING: 3.0,
        SeedStage.SHADOWING: 4.0,
        SeedStage.PROBATIONARY: 5.0,
        SeedStage.FOSSILIZED: 6.0,  # Highest (terminal success)
    }

    current_potential = STAGE_POTENTIALS.get(seed_info.stage, 0.0)
    previous_potential = STAGE_POTENTIALS.get(seed_info.previous_stage, 0.0)

    # PBRS: F(s, s') = gamma * Phi(s') - Phi(s)
    return config.stage_potential_weight * (gamma * current_potential - previous_potential)
```

### Observation Normalization

**Critical:** Observations span wildly different scales. Apply per-feature normalization:

```python
def normalize_observation(obs: dict, config: TaskConfig) -> dict:
    """Normalize observations for stable PPO training."""
    return {
        # Progress features: normalize to [0, 1]
        'epoch': obs['epoch'] / config.max_epochs,
        'global_step': obs['global_step'] / config.max_steps,

        # Loss features: normalize relative to task baseline
        'train_loss': (obs['train_loss'] - config.target_loss) / config.achievable_range,
        'val_loss': (obs['val_loss'] - config.target_loss) / config.achievable_range,
        'loss_delta': obs['loss_delta'] / config.typical_loss_delta_std,

        # Already normalized
        'plateau_epochs': min(obs['plateau_epochs'] / 10.0, 1.0),
        'alpha': obs['alpha'],  # Already [0, 1]

        # Seed state: discrete, leave as-is
        'has_active_seed': obs['has_active_seed'],
        'seed_stage': obs['seed_stage'] / 7.0,  # Normalize by max stage
        ...
    }
```

### Configuration

```python
@dataclass
class RewardConfig:
    # Loss delta scaling
    loss_delta_weight: float = 5.0
    max_loss_delta: float = 5.0             # After normalization
    regression_penalty_scale: float = 0.5   # Asymmetric clipping
    typical_loss_delta_std: float = 0.1     # Task-specific, set via TaskConfig

    # Compute rent
    compute_rent_weight: float = 0.05       # Penalize excess params

    # Stage bonuses (PBRS-compatible)
    stage_potential_weight: float = 0.1     # Scaling for PBRS bonus

    # Terminal bonus
    baseline_loss: float = 2.3              # Task-specific (random init loss)
    target_loss: float = 0.3                # Task-specific (achievable loss)
    terminal_loss_weight: float = 1.0

    @property
    def achievable_range(self) -> float:
        return self.baseline_loss - self.target_loss
```

### Task Configs

```python
CIFAR10_CONFIG = TaskConfig(
    task_type="classification",
    topology="cnn",
    baseline_loss=2.3,              # ln(10) at random init
    target_loss=0.3,                # ~90% accuracy
    typical_loss_delta_std=0.05,
    max_epochs=25,
)

TINYSTORIES_CONFIG = TaskConfig(
    task_type="lm",
    topology="transformer",
    baseline_loss=10.8,             # ln(50257) at random init
    target_loss=3.5,                # ~33 perplexity (good for small model)
    typical_loss_delta_std=0.15,
    max_epochs=50,
)
```

---

## 3. Observation Space

### Design

Remove accuracy fields from policy input. Loss is the universal signal.

**TensorSchema (23 features):**

| Index | Feature | Normalization |
|-------|---------|---------------|
| 0 | epoch | / max_epochs |
| 1 | global_step | / max_steps |
| 2-4 | train_loss, val_loss, loss_delta | Task-relative (see above) |
| 5 | plateau_epochs | / 10, clamped |
| 6-7 | best_val_loss, loss_improvement | Task-relative |
| 8-12 | loss_history_5 | Task-relative |
| 13-18 | Seed state | Various (see normalize_observation) |

**What's removed:** `train_accuracy`, `val_accuracy`, `accuracy_delta`, `accuracy_history_5`

**What's kept for telemetry:** Accuracy logged in `TrainingSignals` for human interpretability, but not fed to policy.

---

## 4. Blueprint Plugin System

### Registry

```python
@dataclass(frozen=True, slots=True)
class BlueprintSpec:
    name: str
    topology: str                           # "cnn" or "transformer"
    factory: Callable[[int], nn.Module]     # dim -> seed module
    param_estimate: int
    description: str = ""

    def actual_param_count(self, dim: int) -> int:
        """Compute actual param count for given dimension."""
        module = self.factory(dim)
        return sum(p.numel() for p in module.parameters())


class BlueprintRegistry:
    _blueprints: dict[str, BlueprintSpec] = {}

    @classmethod
    def register(cls, name: str, topology: str, param_estimate: int, description: str = ""):
        """Decorator to register a blueprint class."""
        def decorator(factory):
            cls._blueprints[f"{topology}:{name}"] = BlueprintSpec(
                name=name,
                topology=topology,
                factory=factory,
                param_estimate=param_estimate,
                description=description,
            )
            return factory
        return decorator

    @classmethod
    def list_for_topology(cls, topology: str) -> list[BlueprintSpec]:
        """All blueprints for a topology, sorted by param count."""
        return sorted(
            [s for s in cls._blueprints.values() if s.topology == topology],
            key=lambda s: s.param_estimate,
        )
```

### CNN Blueprints

| Name | Params | Description |
|------|--------|-------------|
| norm | ~100 | BatchNorm only |
| attention | ~2k | SE-style channel attention |
| depthwise | ~4.8k | Depthwise-separable conv |
| conv_enhance | ~74k | Heavy conv block |

### Transformer Blueprints

| Name | Params | Description |
|------|--------|-------------|
| norm | ~800 | LayerNorm only |
| lora | ~6k | Low-rank adapter (rank=8) |
| attention | ~50k | Additional self-attention head |
| mlp | ~1.2M | Additional MLP (4x expansion) |

### Dynamic Action Space

```python
def build_action_enum(topology: str) -> type[IntEnum]:
    """Build action enum from registered blueprints."""
    blueprints = BlueprintRegistry.list_for_topology(topology)

    members = {"WAIT": 0}
    for i, spec in enumerate(blueprints, start=1):
        members[f"GERMINATE_{spec.name.upper()}"] = i
    members["ADVANCE"] = len(blueprints) + 1
    members["CULL"] = len(blueprints) + 2

    return IntEnum(f"{topology.title()}Action", members)
```

### Phase 2 Limitation (Explicit)

> **Note:** For Phase 2 we use a single-slot, flat action enum. This works for single injection point control.
>
> Multi-slot designs (Phase 4+) will require a **factored policy** over:
> - Head A: slot selection
> - Head B: blueprint selection
> - Head C: blend method selection
> - Head D: lifecycle op (WAIT/GERMINATE/ADVANCE/CULL)
>
> This prevents action explosion (10 slots × 8 blueprints × 3 blends × 4 ops = 960 actions).

---

## 5. Host Implementations

### Implementation Pattern (torch.compile-safe)

Both hosts use the **ModuleDict + Identity** pattern for compile compatibility:

1. Pre-register all injection points with `nn.Identity()` modules
2. Use `nn.ModuleDict` so slots appear in `parameters()` and `state_dict()`
3. Always call slots in forward (Identity compiles away to no-op)
4. No conditional branching on slot presence

### HostCNN

```python
class HostCNN(nn.Module):
    """CNN host with single injection point after block2."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.block1 = ConvBlock(3, 32)
        self.block2 = ConvBlock(32, 64)
        self.block3 = ConvBlock(64, 128)
        self.pool = nn.MaxPool2d(2, 2)
        self.classifier = nn.Linear(128, num_classes)

        # Injection points with compile-friendly ModuleDict
        self._slot_keys = ("block2_post",)
        self.slots = nn.ModuleDict({k: nn.Identity() for k in self._slot_keys})

    @property
    def injection_points(self) -> dict[str, int]:
        return {"block2_post": 64}

    def register_slot(self, slot_id: str, slot: nn.Module) -> None:
        if slot_id not in self.slots:
            raise ValueError(f"Unknown injection point: {slot_id}")
        device = next(self.parameters()).device
        self.slots[slot_id] = slot.to(device)

    def unregister_slot(self, slot_id: str) -> None:
        if slot_id not in self.slots:
            raise ValueError(f"Unknown injection point: {slot_id}")
        self.slots[slot_id] = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(self.block1(x))
        x = self.pool(self.block2(x))

        # Always call slot (Identity is no-op when empty)
        x = self.slots["block2_post"](x)

        x = self.pool(self.block3(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.classifier(x)
```

### TransformerHost

```python
class TransformerHost(nn.Module):
    """GPT-style decoder with injection points after each layer."""

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
        self.block_size = block_size

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(n_embd, n_head, block_size, dropout)
            for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying: tok_emb is master (convention from GPT-2)
        self.head.weight = self.tok_emb.weight
        assert self.head.weight.data_ptr() == self.tok_emb.weight.data_ptr()

        # Injection points with compile-friendly ModuleDict
        self._slot_keys = tuple(f"layer_{i}_post_block" for i in range(n_layer))
        self.slots = nn.ModuleDict({k: nn.Identity() for k in self._slot_keys})

    @property
    def injection_points(self) -> dict[str, int]:
        return {k: self.n_embd for k in self._slot_keys}

    def register_slot(self, slot_id: str, slot: nn.Module) -> None:
        if slot_id not in self.slots:
            raise ValueError(f"Unknown injection point: {slot_id}")
        device = self.tok_emb.weight.device
        self.slots[slot_id] = slot.to(device)

    def unregister_slot(self, slot_id: str) -> None:
        if slot_id not in self.slots:
            raise ValueError(f"Unknown injection point: {slot_id}")
        self.slots[slot_id] = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        B, T = x.shape
        assert T <= self.block_size, f"Sequence length {T} exceeds block_size {self.block_size}"

        # Embeddings
        pos = torch.arange(T, device=x.device)
        h = self.drop(self.tok_emb(x) + self.pos_emb(pos))

        # Transformer layers with slot injection
        for i, layer in enumerate(self.layers):
            h = layer(h)
            h = self.slots[self._slot_keys[i]](h)  # Always call, Identity is no-op

        # Output
        h = self.ln_f(h)
        return self.head(h)
```

### TransformerBlock

```python
class TransformerBlock(nn.Module):
    """Pre-norm transformer block with causal attention."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
```

---

## 6. TinyStories Integration

### Dataset

```python
class TinyStoriesDataset(Dataset):
    """TinyStories for causal language modeling.

    Note: For large-scale training, use streaming mode from HuggingFace
    datasets to avoid loading entire corpus into RAM.
    """

    def __init__(
        self,
        split: str = "train",
        block_size: int = 256,
        max_samples: int | None = None,
        streaming: bool = False,  # Use for large-scale
    ):
        self.block_size = block_size
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if streaming:
            # Streaming mode for memory efficiency
            self._init_streaming(split, max_samples)
        else:
            # In-memory mode for small-scale experiments
            self._init_inmemory(split, max_samples)

    def _init_inmemory(self, split: str, max_samples: int | None):
        dataset = load_dataset("roneneldan/TinyStories", split=split)
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        # Tokenize per-story, chunk into blocks (not flattening entire corpus)
        self.examples = []
        for example in dataset:
            tokens = self.tokenizer.encode(example["text"])
            # Chunk this story into block_size sequences
            for i in range(0, len(tokens) - self.block_size, self.block_size):
                self.examples.append(tokens[i:i + self.block_size + 1])

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        tokens = torch.tensor(self.examples[idx], dtype=torch.long)
        x = tokens[:-1]   # input
        y = tokens[1:]    # target (shifted by 1)
        return x, y
```

### Training Loop

No changes to `train_epoch_*` functions. Only `validate_and_get_metrics` adds a `task_type` parameter to compute perplexity for LM tasks.

---

## 7. MorphogeneticModel Updates

```python
class MorphogeneticModel(nn.Module):
    """Wraps a host with Kasmina seed management.

    Kasmina's role:
    - Discover injection points from host
    - Create and manage SeedSlots
    - Register slots with host
    - Delegate forward() entirely to host
    """

    def __init__(self, host: HostProtocol, device: str = "cpu"):
        super().__init__()
        self.host = host
        self._device = device

        # For Phase 2: single slot at first injection point
        first_point = next(iter(host.injection_points.keys()))
        first_dim = host.injection_points[first_point]

        self.seed_slot = SeedSlot(
            slot_id=first_point,
            channels=first_dim,
            device=device,
        )

    def forward(self, x: Tensor) -> Tensor:
        # Host owns the forward pass entirely
        return self.host(x)

    def germinate_seed(self, blueprint_id: str, seed_id: str) -> None:
        """Germinate a seed and register it with the host."""
        self.seed_slot.germinate(
            blueprint_id=blueprint_id,
            seed_id=seed_id,
            host_module=self.host,
        )
        # Register the slot's forward with the host
        self.host.register_slot(self.seed_slot.slot_id, self.seed_slot)

    def cull_seed(self) -> None:
        """Cull the current seed and unregister from host."""
        self.seed_slot.cull()
        self.host.unregister_slot(self.seed_slot.slot_id)
```

---

## 8. Testing Strategy

### Unit Tests
- HostProtocol conformance for both HostCNN and TransformerHost
- Blueprint registry and action space derivation
- Reward computation with clipping, rent, and normalization
- PBRS stage bonus correctness

### Integration Tests
- MorphogeneticModel with both host types
- Seed lifecycle: germinate → train → blend → fossilize
- Gradient flow through grafted seeds
- torch.compile compatibility (no graph breaks)

### Behavioral Tests
- Tamiyo learns on CIFAR (Phase 1 preserved)
- Tamiyo learns on TinyStories (Phase 2 validated)
- No regression in Phase 1 performance (within 10%)

### Property Tests
- Reward bounded across input space
- Monotonicity: more improvement → more reward
- Compute rent increases with params
- PBRS: stage transitions don't change cumulative potential

---

## 9. Migration Plan

### Files to Modify

| File | Changes |
|------|---------|
| `kasmina/host.py` | Add `HostProtocol`, adapt `HostCNN`, add `TransformerHost` |
| `kasmina/slot.py` | Ensure `forward()` returns identity when inactive |
| `kasmina/blueprints/` | Split into `cnn.py`, `transformer.py`, `registry.py` |
| `leyline/actions.py` | Add `build_action_enum()`, per-topology enums |
| `leyline/signals.py` | Remove `ACCURACY_*` from TensorSchema (27→23 features) |
| `simic/rewards.py` | Loss-primary refactor with explicit compute rent and PBRS |
| `simic/features.py` | Add observation normalization |
| `tolaria/trainer.py` | Add `task_type` parameter to validation |
| `utils/data.py` | Add `TinyStoriesDataset`, `load_tinystories()` |

### Files to Add

| File | Purpose |
|------|---------|
| `kasmina/blueprints/cnn.py` | CNN blueprint classes |
| `kasmina/blueprints/transformer.py` | Transformer blueprint classes |
| `kasmina/blueprints/registry.py` | Plugin registry |
| `kasmina/protocol.py` | HostProtocol definition |
| `leyline/tasks.py` | TaskConfig definitions |

### Breaking Changes

- `TensorSchema` feature count: 27 → 23 (accuracy fields removed)
- Reward function signature: `acc_delta` → `loss_delta`, adds `total_params`/`host_params`
- Action enum: global `Action` → per-topology `CNNAction`/`TransformerAction`
- Host interface: must implement `register_slot()` and `unregister_slot()`

---

## 10. Success Criteria

1. **CIFAR-10 preserved:** Tamiyo achieves same mean return as Phase 1 (within 10%)
2. **TinyStories functional:** Tamiyo improves over random policy on LM task
3. **No legacy code:** Zero backwards-compatibility shims
4. **Plugin extensibility:** Adding a new blueprint requires only a decorated class
5. **Compute rent visible:** Depthwise preference emerges under rent pressure
6. **Tests pass:** All unit, integration, and behavioral tests green
7. **torch.compile works:** No graph breaks in forward pass

---

## Appendix A: Expert Review Feedback

### DRL Expert Review (v2)

**Approved with recommendations.**

#### Critical Issues Addressed

1. **Terminal bonus timing:** The terminal bonus fires only at `epoch == max_epochs`, which creates sparse reward that GAE struggles with.
   - *Status:* Documented as known limitation. Consider potential-based leaking in future iteration.
   - *Mitigation:* Stage bonuses provide denser signal throughout training.

2. **Observation normalization:** Observations span wildly different scales (epoch: 0-200, loss: 0.3-10.8).
   - *Status:* Added `normalize_observation()` function with per-feature normalization strategy.

3. **Stage bonus PBRS compatibility:** Original design didn't show PBRS-compliant implementation.
   - *Status:* Added `compute_pbrs_stage_bonus()` with explicit potential function.

#### Recommendations Incorporated

- Compute rent calibration: Monitor param ratios empirically. Consider rent-free grace period (noted in code).
- Observation normalization: Added per-feature normalization strategy.
- PBRS stage bonus: Implemented with monotonic potential toward FOSSILIZED.

#### Open Questions (To Address in Implementation)

1. What's the GAE λ and discount γ? (Current: λ=0.95, γ=0.99)
2. How are policies shared across topologies? (Phase 2: Separate policies per topology)
3. What's typical seed lifetime? (To measure empirically)

### PyTorch Expert Review (v2)

**Approved with recommendations.**

#### Critical Issues Addressed

1. **Dict lookup breaks torch.compile:** `if slot_id in self.attached_slots` causes graph breaks.
   - *Status:* Switched to `nn.ModuleDict` + `nn.Identity()` pattern. Always call slots.

2. **attached_slots is plain dict:** Slots wouldn't appear in `parameters()` or `state_dict()`.
   - *Status:* Changed to `nn.ModuleDict`.

#### Issues Addressed

3. **Weight tying order:** Should be `self.head.weight = self.tok_emb.weight` (tok_emb as master).
   - *Status:* Fixed with assertion.

4. **Missing unregister_slot:** Need way to remove seeds.
   - *Status:* Added to HostProtocol and implementations.

5. **Missing device handling:** `register_slot` should move slot to correct device.
   - *Status:* Added `.to(device)` in both host implementations.

#### Recommendations Incorporated

- ModuleDict + Identity pattern for compile safety
- Weight tying with tok_emb as master + assertion
- Device handling in register_slot
- unregister_slot in protocol

#### Implementation Notes

- Changing slots invalidates compiled graph (unavoidable — dynamic module swapping changes computation)
- Zero-init seed output projections for stable integration (already in blueprint designs)

---

## Appendix B: Design Decisions Log

| Decision | Options Considered | Choice | Rationale |
|----------|-------------------|--------|-----------|
| Host-Kasmina responsibility | Kasmina orchestrates vs Host owns | Host owns | Matches "morphogenetic plane" model, simpler abstraction |
| Reward signal | Accuracy vs Loss | Loss | Universal across task types |
| Slot registration | Dynamic creation vs Pre-allocated | Pre-allocated Identity | torch.compile compatibility |
| Action space | Unified vs Per-topology | Per-topology | Different blueprints per host type |
| Observation normalization | Running vs Explicit | Explicit per-feature | More predictable, task-config driven |
| Stage bonuses | Additive vs PBRS | PBRS | Preserves optimal policy |
