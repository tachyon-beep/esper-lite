# Slot Transformer Architecture for Scalable Seed Management

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Consult `yzmir-pytorch-engineering` and `yzmir-neural-architectures` skills for transformer implementation details.

**Goal:** Replace flat observation concatenation with a Slot Transformer Encoder that provides weight sharing across slots, natural variable slot count support, and learned slot-slot interactions via self-attention. This enables scaling from 3 slots to 50+ slots without linear parameter growth.

**Architecture:** Four-phase incremental delivery: (1) SlotTransformerEncoder module, (2) SlotTransformerActorCritic network, (3) PolicyBundle integration and factory, (4) Benchmarking and cutover. Each phase is independently testable.

**Tech Stack:** PyTorch 2.x, torch.compile compatible, nn.TransformerEncoder, learnable positional encoding

---

## Problem Statement

### Current Architecture (Flat Concatenation)

```
Observation = [BASE_FEATURES: 23] + [SLOT_0: 39] + [SLOT_1: 39] + ... + [SLOT_N-1: 39]

Total dimension = 23 + num_slots × 39
```

| Slots | Dimension | Parameters in feature_net (64 hidden) |
|-------|-----------|---------------------------------------|
| 3 | 140 | 9,024 |
| 10 | 413 | 26,496 |
| 50 | 1,973 | 126,336 |

**Issues:**
1. **O(n) input dimension** - Linear growth in observation size
2. **No weight sharing** - Network learns "slot 0 semantics" separately from "slot 5 semantics"
3. **Fixed positional assumptions** - Slot positions baked into input layer weights
4. **Scaffolding features are approximations** - `upstream_alpha`, `downstream_alpha`, `interaction_sum` are scalar aggregates of pairwise relationships that self-attention learns end-to-end

### Proposed Architecture (Slot Transformer)

```
Base Features [B,T,23] ──► GlobalEncoder ──► [B,T,64] ─────────────┐
                                                                    │
Slots [B,T,N,39] ──► SlotEmbedding ──► +PosEnc ──► TransformerEnc ─┼─► CLS [B,T,64]
                                              │                     │
                                              └─► slot_repr [B,T,N,64]
                                                        │           │
                                                        │           ▼
                                                        │      Fusion ──► LSTM ──► Heads
                                                        │                    │
                                                        └────► CrossAttention ◄──┘
                                                                    │
                                                                    ▼
                                                            slot_logits [B,T,N]
```

**Benefits:**
- **O(1) parameters per slot** - Same projection weights for all slots
- **Learned slot-slot interactions** - Self-attention replaces hand-crafted scaffolding features
- **Variable slot counts** - Attention masking handles any N without recompilation
- **Interpretable attention** - Visualize which slots the agent considers together

---

## Phase 1: SlotTransformerEncoder Module

**Files:** `src/esper/tamiyo/networks/slot_transformer.py` (new)

### Task 1.1: SlotPositionalEncoding

Create learnable row/column positional embeddings for grid-based slot positions.

```python
class SlotPositionalEncoding(nn.Module):
    """Factorized row/column positional encoding for slot grid.

    Supports generalization to unseen grid configurations by learning
    separate row and column embeddings that combine additively.
    """

    def __init__(self, max_rows: int = 8, max_cols: int = 8, embed_dim: int = 64):
        super().__init__()
        self.row_embed = nn.Embedding(max_rows, embed_dim // 2)
        self.col_embed = nn.Embedding(max_cols, embed_dim // 2)

    def forward(self, row_ids: Tensor, col_ids: Tensor) -> Tensor:
        """Return positional embeddings [num_slots, embed_dim]."""
        return torch.cat([self.row_embed(row_ids), self.col_embed(col_ids)], dim=-1)
```

**Test:** `tests/tamiyo/networks/test_slot_transformer.py::TestSlotPositionalEncoding`
- Verify output shape matches embed_dim
- Verify different positions produce different embeddings
- Verify embeddings are differentiable

### Task 1.2: SlotTransformerEncoder

Core transformer encoder with CLS token aggregation and per-slot outputs.

```python
class SlotTransformerEncoder(nn.Module):
    """Transformer encoder for slot feature processing.

    Processes slots as tokens with self-attention for slot-slot interactions.
    Returns both aggregated (CLS) and per-slot representations.
    """

    def __init__(
        self,
        slot_dim: int = 39,  # SLOT_FEATURE_SIZE
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        max_slots: int = 64,
        dropout: float = 0.1,
    ):
        ...

    def forward(
        self,
        slot_features: Tensor,      # [B, T, N, slot_dim]
        slot_active_mask: Tensor,   # [B, T, N] bool
        row_ids: Tensor,            # [N]
        col_ids: Tensor,            # [N]
    ) -> tuple[Tensor, Tensor]:
        """
        Returns:
            cls_repr: [B, T, embed_dim] aggregated slot representation
            slot_repr: [B, T, N, embed_dim] per-slot representations
        """
```

**Key implementation details:**
1. Merge B and T dimensions for transformer: `[B*T, N+1, D]`
2. Add CLS token at position 0
3. Use `src_key_padding_mask` to mask inactive slots
4. Pre-LN transformer (norm_first=True) for training stability
5. GELU activation in FFN

**Test:** `tests/tamiyo/networks/test_slot_transformer.py::TestSlotTransformerEncoder`
- Verify output shapes
- Verify inactive slots don't affect CLS representation (masked properly)
- Verify gradient flow to all slot embeddings
- Verify torch.compile compatibility with `fullgraph=True`

### Task 1.3: torch.compile Compatibility Testing

Create dedicated compile tests:

```python
def test_slot_transformer_compiles_fullgraph():
    """Verify no graph breaks in SlotTransformerEncoder."""
    encoder = SlotTransformerEncoder()
    compiled = torch.compile(encoder, fullgraph=True)

    # Run forward pass
    slot_features = torch.randn(2, 4, 5, 39)
    mask = torch.ones(2, 4, 5, dtype=torch.bool)
    row_ids = torch.tensor([0, 0, 1, 1, 2])
    col_ids = torch.tensor([0, 1, 0, 1, 0])

    cls_out, slot_out = compiled(slot_features, mask, row_ids, col_ids)
    assert cls_out.shape == (2, 4, 64)
```

---

## Phase 2: SlotTransformerActorCritic Network

**Files:** `src/esper/tamiyo/networks/slot_transformer.py` (extend)

### Task 2.1: Feature Extraction Helper

Add method to split flat observation back into structured components:

```python
def _extract_structured_features(
    self,
    state: Tensor,  # [B, T, state_dim]
) -> tuple[Tensor, Tensor, Tensor]:
    """Split flat state into base features, slot features, and active mask.

    This maintains backward compatibility with existing observation format
    while enabling structured processing internally.
    """
    B, T, _ = state.shape
    N = self.num_slots

    base = state[:, :, :BASE_FEATURE_SIZE]
    slot_flat = state[:, :, BASE_FEATURE_SIZE:BASE_FEATURE_SIZE + N * SLOT_FEATURE_SIZE]
    slot_features = slot_flat.view(B, T, N, SLOT_FEATURE_SIZE)
    slot_active = slot_features[:, :, :, 0] > 0.5  # is_active flag

    return base, slot_features, slot_active
```

### Task 2.2: Slot Selection via Cross-Attention

Replace fixed Linear slot_head with cross-attention mechanism:

```python
# Query from temporal representation, keys from slot representations
self.slot_query = nn.Linear(lstm_hidden_dim, embed_dim)
self.slot_key = nn.Linear(embed_dim, embed_dim)

def _compute_slot_logits(
    self,
    temporal_repr: Tensor,  # [B, T, H]
    slot_repr: Tensor,      # [B, T, N, D]
) -> Tensor:
    """Compute slot selection logits via scaled dot-product attention."""
    query = self.slot_query(temporal_repr)  # [B, T, D]
    keys = self.slot_key(slot_repr)         # [B, T, N, D]

    # [B, T, D] @ [B, T, D, N] -> [B, T, N]
    logits = torch.einsum('btd,btnd->btn', query, keys)
    return logits / math.sqrt(self.embed_dim)
```

**Rationale:** This allows the network to score each slot based on learned query-key compatibility, rather than fixed positional weights. The LSTM hidden state "asks" which slots are relevant.

### Task 2.3: Full SlotTransformerActorCritic

Complete network with same API as `FactoredRecurrentActorCritic`:

```python
class SlotTransformerActorCritic(nn.Module):
    """Actor-Critic with Slot Transformer encoder.

    Drop-in replacement for FactoredRecurrentActorCritic with identical
    forward() signature and return format.
    """

    def __init__(
        self,
        state_dim: int,
        slot_config: SlotConfig,
        embed_dim: int = 64,
        lstm_hidden_dim: int = 128,
        lstm_layers: int = 1,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
    ):
        ...

    def forward(
        self,
        state: Tensor,
        hidden: tuple[Tensor, Tensor] | None = None,
        slot_mask: Tensor | None = None,
        **action_masks,
    ) -> dict[str, Tensor]:
        """Same return format as FactoredRecurrentActorCritic."""
```

**Test:** `tests/tamiyo/networks/test_slot_transformer.py::TestSlotTransformerActorCritic`
- Verify forward pass produces all expected keys
- Verify shapes match FactoredRecurrentActorCritic output
- Verify evaluate_actions works for PPO
- Verify checkpoint save/load roundtrip

### Task 2.4: Weight Initialization

Match existing orthogonal initialization scheme:

```python
def _init_weights(self) -> None:
    """Orthogonal init matching FactoredRecurrentActorCritic."""
    for module in self.modules():
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # Smaller init for output layers (policy stability)
    for head in self._action_heads():
        nn.init.orthogonal_(head[-1].weight, gain=0.01)

    # Standard init for value head
    nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)

    # LSTM forget gate bias = 1 (learn to remember by default)
    for name, param in self.lstm.named_parameters():
        if "bias" in name:
            n = param.size(0)
            param.data[n // 4 : n // 2].fill_(1.0)
```

---

## Phase 3: PolicyBundle Integration

**Files:** `src/esper/tamiyo/policy/transformer_bundle.py` (new), `src/esper/tamiyo/policy/factory.py` (modify)

### Task 3.1: TransformerPolicyBundle

Create PolicyBundle implementation wrapping SlotTransformerActorCritic:

```python
@register_policy("transformer")
class TransformerPolicyBundle:
    """Transformer-based PolicyBundle for scalable slot management.

    Uses self-attention for slot-slot interactions and cross-attention
    for slot selection. Supports variable slot counts via masking.
    """

    def __init__(
        self,
        feature_dim: int,
        slot_config: SlotConfig,
        device: str | torch.device = "cpu",
        compile_mode: str = "off",
        hidden_dim: int = 128,
        embed_dim: int = 64,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
    ):
        ...
```

**Must implement full PolicyBundle protocol** (see `protocol.py`):
- `network` property
- `slot_config` property
- `hidden_dim` property
- `compile()` method
- `get_initial_hidden()` method
- `forward()` method
- `evaluate_actions()` method

### Task 3.2: Factory Extension

Add transformer option to `create_policy`:

```python
def create_policy(
    policy_type: str,  # "lstm" | "transformer"
    state_dim: int | None = None,
    slot_config: SlotConfig | None = None,
    ...
    # Transformer-specific
    transformer_layers: int = 2,
    transformer_heads: int = 4,
    embed_dim: int = 64,
) -> PolicyBundle:
    """Create a PolicyBundle of the specified type."""
```

### Task 3.3: TrainingConfig Extension

Add network selection to training configuration:

```python
@dataclass
class TrainingConfig:
    ...
    network_type: str = "lstm"  # "lstm" | "transformer"
    transformer_layers: int = 2
    transformer_heads: int = 4
    embed_dim: int = 64
```

Update `vectorized.py` to respect `network_type`:

```python
policy = create_policy(
    policy_type=config.network_type,  # Was hardcoded "lstm"
    state_dim=state_dim,
    slot_config=slot_config,
    ...
)
```

---

## Phase 4: Benchmarking and Cutover

**Files:** `scripts/benchmark_architectures.py` (new)

### Task 4.1: Benchmark Script

Create comprehensive benchmark comparing LSTM vs Transformer:

```python
"""Benchmark LSTM vs Transformer architectures across slot counts."""

SLOT_CONFIGS = [3, 10, 25, 50]
METRICS = [
    "forward_time_ms",
    "backward_time_ms",
    "memory_mb",
    "compile_time_s",
    "parameters",
]

def benchmark_architecture(
    network_type: str,
    num_slots: int,
    batch_size: int = 32,
    seq_len: int = 64,
    n_iterations: int = 100,
) -> dict[str, float]:
    ...
```

### Task 4.2: Training Comparison

Run short training comparisons on each configuration:

| Config | Architecture | Episodes | Final Return | Train Time |
|--------|--------------|----------|--------------|------------|
| 3 slots | LSTM | 500 | ? | ? |
| 3 slots | Transformer | 500 | ? | ? |
| 10 slots | LSTM | 500 | ? | ? |
| 10 slots | Transformer | 500 | ? | ? |

### Task 4.3: Cutover Decision

**If Transformer wins on 10+ slots:**
1. Make `network_type="transformer"` the default
2. Add deprecation warning to LSTM path
3. After 1 release cycle, remove LSTM per No Legacy Code Policy

**If LSTM wins:**
1. Keep LSTM as default
2. Document Transformer as experimental option for large slot counts
3. Revisit when scaling requirements change

---

## Implementation Notes

### torch.compile Considerations

1. **Use `dynamic=True`** if supporting variable slot counts at runtime
2. **Prefer `reduce-overhead`** mode for transformer (more fusion opportunities)
3. **Isolate mask construction** with `@torch.compiler.disable` if causing graph breaks
4. **FlexAttention** (PyTorch 2.5+) optional optimization for custom attention patterns

### Memory Optimization

For very large slot counts (50+), consider:

1. **Gradient checkpointing** in transformer layers
2. **Mixed precision** (bfloat16) for attention
3. **Chunked attention** if sequence length becomes problematic

### Attention Visualization

Add optional attention weight extraction for interpretability:

```python
def forward(..., return_attention: bool = False):
    ...
    if return_attention:
        return outputs, attention_weights
```

Useful for debugging which slots the agent considers together.

---

## File Checklist

| File | Action | Phase |
|------|--------|-------|
| `src/esper/tamiyo/networks/slot_transformer.py` | Create | 1-2 |
| `tests/tamiyo/networks/test_slot_transformer.py` | Create | 1-2 |
| `src/esper/tamiyo/policy/transformer_bundle.py` | Create | 3 |
| `tests/tamiyo/policy/test_transformer_bundle.py` | Create | 3 |
| `src/esper/tamiyo/policy/factory.py` | Modify | 3 |
| `src/esper/tamiyo/policy/__init__.py` | Modify | 3 |
| `src/esper/simic/training/config.py` | Modify | 3 |
| `src/esper/simic/training/vectorized.py` | Modify | 3 |
| `scripts/benchmark_architectures.py` | Create | 4 |

---

## Success Criteria

1. **Functional parity**: Transformer produces valid actions for all slot configurations
2. **No regression on 3-slot**: Performance within 10% of LSTM baseline
3. **Scaling win**: Transformer uses <50% memory of LSTM at 50 slots
4. **Compile clean**: `fullgraph=True` succeeds with no graph breaks
5. **All tests pass**: Including existing PPO integration tests

---

## References

- PyTorch TransformerEncoder: https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html
- Set Transformer paper: https://arxiv.org/abs/1810.00825 (for alternatives)
- FlexAttention: https://pytorch.org/blog/flexattention/ (PyTorch 2.5+)
