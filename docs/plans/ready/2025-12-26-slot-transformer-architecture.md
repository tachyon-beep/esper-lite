# Slot Transformer Architecture for Scalable Seed Management

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Consult `yzmir-pytorch-engineering` and `yzmir-neural-architectures` skills for transformer implementation details.

---

## Expert Review Summary (2026-01-12)

| Reviewer | Verdict | Key Findings |
|----------|---------|--------------|
| **PyTorch Expert** | CONDITIONAL GO | Attention mask semantics inverted (PyTorch: `True=ignore`); CLS token needs explicit mask entry; pre-allocate buffers to avoid graph breaks; Xavier init for transformer layers |
| **DRL Expert** | GO w/ modifications | Contribution predictor must use `slot_repr` not `lstm_out`; cross-attention must include `slot_active_mask`; value head architecture should match 4-layer depth |

**Status:** Plan updated with all critical findings. Ready for implementation.

---

**Goal:** Replace flat observation concatenation with a Slot Transformer Encoder that provides weight sharing across slots, natural variable slot count support, and learned slot-slot interactions via self-attention. This enables scaling from 3 slots to 50+ slots without linear parameter growth.

**Architecture:** Five-phase incremental delivery: (1) SlotTransformerEncoder module, (2) SlotTransformerActorCritic network, (3) PolicyBundle integration and factory, (4) Telemetry and Sanctum integration, (5) Benchmarking and cutover. Each phase is independently testable.

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

**⚠️ CRITICAL (PyTorch Expert Review 2026-01-12):**

6. **Attention mask semantics INVERTED**: PyTorch uses `True = ignore/pad`, not `True = valid`.
   Must invert: `src_key_padding_mask = ~slot_active_mask`
7. **CLS token must be in padding mask**: Prepend `False` (valid) for CLS position:
   ```python
   cls_mask = torch.zeros(B * T, 1, dtype=torch.bool, device=device)
   full_padding_mask = torch.cat([cls_mask, ~slot_active_mask.view(B*T, N)], dim=1)
   ```
8. **Pre-allocate CLS mask as buffer** to avoid graph breaks:
   ```python
   self.register_buffer("_cls_valid", torch.zeros(1, 1, dtype=torch.bool), persistent=False)
   ```

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
    slot_active_mask: Tensor,  # [B, T, N] bool - CRITICAL: must mask inactive
) -> Tensor:
    """Compute slot selection logits via scaled dot-product attention."""
    # PyTorch Expert: Scale BEFORE dot product to prevent overflow
    query = self.slot_query(temporal_repr) / math.sqrt(self.embed_dim)  # [B, T, D]
    keys = self.slot_key(slot_repr)         # [B, T, N, D]

    # [B, T, D] @ [B, T, D, N] -> [B, T, N]
    logits = torch.einsum('btd,btnd->btn', query, keys)

    # DRL Expert: MUST mask inactive slots for valid action selection
    logits = logits.masked_fill(~slot_active_mask, MASKED_LOGIT_VALUE)
    return logits
```

**⚠️ CRITICAL (DRL Expert Review 2026-01-12):**
- Cross-attention MUST include `slot_active_mask` parameter
- Without masking, policy may select inactive slots for FOSSILIZE/PRUNE
- Use `MASKED_LOGIT_VALUE` from leyline (same as MaskedCategorical)

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

    # Value head: gain=0.01 to start predictions near zero
    nn.init.orthogonal_(self.value_head[-1].weight, gain=0.01)

    # Contribution predictor: gain=0.1 for [-10, +10] target range
    nn.init.orthogonal_(self.contribution_predictor[-1].weight, gain=0.1)
    nn.init.zeros_(self.contribution_predictor[-1].bias)

    # LSTM forget gate bias = 1 (learn to remember by default)
    for name, param in self.lstm.named_parameters():
        if "bias" in name:
            n = param.size(0)
            param.data[n // 4 : n // 2].fill_(1.0)
```

### Task 2.5: Op-Conditioned Value Head (Q(s, op))

**CRITICAL UPDATE (2026-01):** The current `FactoredRecurrentActorCritic` uses Q(s, op) not V(s).
The value function is conditioned on the selected operation to learn distinct expected returns per action.

```python
def __init__(self, ...):
    ...
    # Value head input: temporal repr + one-hot op action
    # This allows learning Q(s, GERMINATE) ≠ Q(s, PRUNE) for same state
    value_input_dim = lstm_hidden_dim + self.num_ops  # 128 + 6 = 134
    self.value_head = nn.Sequential(
        nn.Linear(value_input_dim, head_hidden),  # 134 -> 256
        nn.LayerNorm(head_hidden),
        nn.ReLU(),
        nn.Linear(head_hidden, head_hidden // 2),  # 256 -> 128
        nn.LayerNorm(head_hidden // 2),
        nn.ReLU(),
        nn.Linear(head_hidden // 2, 1),  # 128 -> 1
    )

def _compute_value(self, lstm_out: Tensor, op: Tensor) -> Tensor:
    """Compute Q(s, op) value conditioned on operation."""
    op_one_hot = F.one_hot(op, num_classes=self.num_ops).to(lstm_out)
    value_input = torch.cat([lstm_out, op_one_hot], dim=-1)
    return self.value_head(value_input).squeeze(-1)
```

For transformer: fuse temporal LSTM output with slot CLS token before op-conditioning.

### Task 2.6: Contribution Predictor Auxiliary Head

**CRITICAL UPDATE (2026-01):** The current network includes an auxiliary head for predicting
seed contributions. This is used for counterfactual proxy training.

**⚠️ CRITICAL (DRL Expert Review 2026-01-12):**
For transformer architecture, contribution predictor **MUST** use `slot_repr` (per-slot
representations), NOT `lstm_out`. Using global LSTM output for per-slot predictions defeats
the entire purpose of the transformer architecture—we want the predictor to leverage the
rich per-slot features that self-attention produces.

```python
def __init__(self, ...):
    ...
    # Auxiliary head: predict seed contributions
    # For LSTM: input is lstm_out broadcast to each slot
    # For Transformer: input is slot_repr (per-slot representations from self-attention)
    # Dropout prevents shortcut learning
    self.contribution_predictor = nn.Sequential(
        nn.Linear(embed_dim, head_hidden),  # embed_dim for slot_repr input
        nn.LayerNorm(head_hidden),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(head_hidden, head_hidden // 2),
        nn.LayerNorm(head_hidden // 2),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(head_hidden // 2, 1),  # One scalar per slot
    )

def predict_contributions(
    self,
    state: Tensor,
    hidden: tuple[Tensor, Tensor] | None = None,
    stop_gradient: bool = True,  # Prevent aux task from shaping features
) -> Tensor:
    """Predict per-slot contributions from state.

    For transformer architecture: uses slot_repr [B, T, N, D] -> [B, T, N]
    Each slot's prediction comes from its own rich representation.
    """
    # Get per-slot features from transformer encoder
    _, slot_repr = self._forward_slot_transformer(state, hidden)  # [B, T, N, D]
    if stop_gradient:
        slot_repr = slot_repr.detach()
    # Apply predictor to each slot's representation
    return self.contribution_predictor(slot_repr).squeeze(-1)  # [B, T, N]
```

**Rationale:** Each slot's contribution prediction should be based on that slot's learned
representation (which includes context from self-attention with other slots), not a global
summary. This enables the predictor to learn slot-specific features like "this slot attends
heavily to the host → higher contribution".

### Task 2.7: ResidualLSTM Integration

**CRITICAL UPDATE (2026-01):** The current implementation uses `ResidualLSTM` (from
`factored_lstm.py`) with residual connections, layer normalization, and skip connections
for training stability. The transformer architecture should use this instead of vanilla `nn.LSTM`.

```python
from esper.tamiyo.networks.factored_lstm import ResidualLSTM

# In __init__:
self.lstm = ResidualLSTM(
    input_size=fusion_dim,  # Fused base + slot CLS
    hidden_size=lstm_hidden_dim,
    num_layers=lstm_layers,
    dropout=0.0,  # Single layer = no dropout needed
)
```

### Task 2.8: BlueprintEmbedding for Obs V3

**CRITICAL UPDATE (2026-01):** The observation includes blueprint indices per slot.
These need learned embeddings rather than raw indices.

```python
from esper.tamiyo.networks.factored_lstm import BlueprintEmbedding

# In __init__:
self.blueprint_embedding = BlueprintEmbedding(
    num_blueprints=NUM_BLUEPRINTS,
    embed_dim=DEFAULT_BLUEPRINT_EMBED_DIM,  # 4
)

# In forward:
# blueprint_indices: [batch, seq, num_slots] with -1 for inactive
bp_embeds = self.blueprint_embedding(blueprint_indices)  # [batch, seq, slots, 4]
# Concatenate with slot features before transformer
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

## Phase 4: Telemetry and Sanctum Integration

**Files:** `src/esper/leyline/telemetry.py` (modify), `src/esper/karn/sanctum/` (modify)

### Task 4.1: Extend PPOUpdatePayload for Transformer Metrics

Add optional transformer-specific fields to telemetry:

```python
@dataclass
class PPOUpdatePayload:
    ...
    # Existing fields unchanged

    # OPTIONAL - Transformer attention diagnostics (None if LSTM)
    attention_entropy_mean: float | None = None  # Mean entropy of attention distributions
    attention_entropy_min: float | None = None   # Lowest entropy (sharpest attention)
    slot_attention_distribution: list[float] | None = None  # Per-slot attention mass
    network_type: str = "lstm"  # "lstm" | "transformer" for TUI routing
```

**Rationale:**
- `attention_entropy_mean`: Low = sharp attention (good for focused decisions), High = diffuse (exploring)
- `slot_attention_distribution`: Which slots get most attention - useful for debugging slot selection
- `network_type`: Allows Sanctum to adapt display based on architecture

### Task 4.2: Emit Attention Metrics in PPO Update

Modify `PPOAgent._compute_diagnostics()` to extract attention weights when using transformer:

```python
def _compute_diagnostics(self, ...) -> dict:
    diagnostics = {...}  # Existing

    # Transformer-specific metrics
    if hasattr(self.policy.network, 'get_attention_stats'):
        attn_stats = self.policy.network.get_attention_stats()
        diagnostics.update({
            "attention_entropy_mean": attn_stats["entropy_mean"],
            "attention_entropy_min": attn_stats["entropy_min"],
            "slot_attention_distribution": attn_stats["slot_distribution"],
            "network_type": "transformer",
        })
    else:
        diagnostics["network_type"] = "lstm"

    return diagnostics
```

### Task 4.3: TamiyoBrain Architecture Indicator

Add network type badge to TamiyoBrain status banner:

```python
# In _render_status_banner()
network_badge = "🔄 LSTM" if state.network_type == "lstm" else "⚡ XFMR"
self._append_text(network_badge, style="dim")
```

**Location:** Status banner, right-aligned after existing metrics.

### Task 4.4: TamiyoBrain Layout Restructure for Attention Heatmap

Restructure TamiyoBrain's middle section to place the slot attention heatmap alongside the head entropy/gradient heatmaps. Move slot summary from bottom to middle-right.

**Current layout (vertical stack):**
```
Row 4: Head entropy heatmap (full width)
Row 5: Head gradient heatmap (full width)
...
Row 10: Slot summary (full width, bottom)
```

**New layout (60:40 horizontal split in middle rows):**
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Status Banner + Sparklines + Gauges (Rows 1-3)                              │
├───────────────────────────────────────────┬─────────────────────────────────┤
│ Head Entropy Heatmap (60%)                │ Slot Attention Heatmap (40%)    │
│ ┌─────────────────────────────────────┐   │ ┌─────────────────────────────┐ │
│ │ slot bpnt styl temp atgt aspd acrv op│  │ │     r0c0 r0c1 r1c0         │ │
│ │  ▓▓   ██   ░░   ▓▓   ░░   ██   ░░  ▓▓│  │ │r0c0  ██   ▓▓   ░░          │ │
│ └─────────────────────────────────────┘   │ │r0c1  ▓▓   ██   ▓▓          │ │
│ Head Gradient Heatmap                      │ │r1c0  ░░   ▓▓   ██          │ │
│ ┌─────────────────────────────────────┐   │ └─────────────────────────────┘ │
│ │ slot bpnt styl temp atgt aspd acrv op│  │ Slot Summary:                   │
│ │  ██   ▓▓   ██   ░░   ▓▓   ░░   ██  ░░│  │ 🌱2 📈1 🔀0 🪨0              │
│ └─────────────────────────────────────┘   │ → GERMINATE available           │
├───────────────────────────────────────────┴─────────────────────────────────┤
│ Action Distribution Bar + Action Sequence + Return History (Rows 6-8)       │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Implementation in `_render_horizontal_layout()`:**

```python
def _render_middle_section(self, snapshot: SanctumSnapshot) -> Table:
    """Render head heatmaps (left 60%) alongside slot attention (right 40%)."""
    middle = Table.grid(expand=True)
    middle.add_column("heads", width=60, ratio=60)  # 60%
    middle.add_column("slots", width=40, ratio=40)  # 40%

    # Left column: head entropy + head gradient (stacked)
    heads_content = Table.grid()
    heads_content.add_row(self._render_head_heatmap())
    heads_content.add_row(self._render_head_gradient_heatmap())

    # Right column: slot attention heatmap + slot summary (stacked)
    slots_content = Table.grid()
    if snapshot.tamiyo.network_type == "transformer":
        slots_content.add_row(self._render_attention_heatmap())
    slots_content.add_row(self._render_slot_summary())

    middle.add_row(heads_content, slots_content)
    return middle
```

**Attention Heatmap Widget (embedded in TamiyoBrain):**

```python
def _render_attention_heatmap(self) -> Text:
    """Render slot-slot attention matrix as ASCII heatmap.

    Only rendered when network_type == "transformer".
    Shows which slots attend to which other slots.
    """
    weights = self._snapshot.tamiyo.slot_attention_weights
    if not weights:
        return Text("(no attention data)", style="dim")

    # Build heatmap using block characters
    # ██ = high attention, ▓▓ = medium, ░░ = low, ·· = minimal
    ...
```

**CSS updates for 60:40 split:**

```css
/* In styles.tcss - TamiyoBrain middle section */
.tamiyo-middle-section {
    layout: horizontal;
}

.tamiyo-heads-column {
    width: 60%;
}

.tamiyo-slots-column {
    width: 40%;
    border-left: solid $primary-lighten-2;
    padding-left: 1;
}
```

**Benefits of this layout:**
1. Slot attention visible at same glance as head entropy (correlated metrics)
2. Removes bottom slot summary row (saves vertical space)
3. Natural grouping: "what the heads are doing" | "what the slots are doing"
4. Extra width for attention heatmap accommodates larger slot counts

### Task 4.5: SanctumSnapshot Schema Extension

Extend schema to carry transformer metrics:

```python
@dataclass
class TamiyoState:
    ...
    # Existing fields

    # Transformer metrics (None if LSTM)
    network_type: str = "lstm"
    attention_entropy: float | None = None
    slot_attention_weights: list[float] | None = None
```

### Task 4.6: Aggregator Updates

Update `SanctumAggregator` to populate transformer fields from PPOUpdatePayload:

```python
def _handle_ppo_update(self, payload: PPOUpdatePayload) -> None:
    ...
    # Existing handling

    # Transformer metrics
    self._tamiyo.network_type = payload.network_type
    if payload.attention_entropy_mean is not None:
        self._tamiyo.attention_entropy = payload.attention_entropy_mean
        self._tamiyo.slot_attention_weights = payload.slot_attention_distribution
```

---

## Phase 5: Benchmarking and Cutover

**Files:** `scripts/benchmark_architectures.py` (new)

### Task 5.1: Benchmark Script

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

### Task 5.2: Training Comparison

Run short training comparisons on each configuration:

| Config | Architecture | Episodes | Final Return | Train Time |
|--------|--------------|----------|--------------|------------|
| 3 slots | LSTM | 500 | ? | ? |
| 3 slots | Transformer | 500 | ? | ? |
| 10 slots | LSTM | 500 | ? | ? |
| 10 slots | Transformer | 500 | ? | ? |

### Task 5.3: Cutover Decision

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
| `src/esper/leyline/telemetry.py` | Modify | 4 |
| `src/esper/simic/agent/ppo.py` | Modify | 4 |
| `src/esper/karn/sanctum/schema.py` | Modify | 4 |
| `src/esper/karn/sanctum/aggregator.py` | Modify | 4 |
| `src/esper/karn/sanctum/widgets/tamiyo_brain.py` | Modify | 4 |
| `src/esper/karn/sanctum/styles.tcss` | Modify | 4 |
| `tests/karn/sanctum/test_tamiyo_brain.py` | Modify | 4 |
| `scripts/benchmark_architectures.py` | Create | 5 |

---

## Success Criteria

1. **Functional parity**: Transformer produces valid actions for all slot configurations
2. **No regression on 3-slot**: Performance within 10% of LSTM baseline
3. **Scaling win**: Transformer uses <50% memory of LSTM at 50 slots
4. **Compile clean**: `fullgraph=True` succeeds with no graph breaks
5. **Telemetry complete**: Attention metrics flow from PPO → Nissa → Karn → Sanctum
6. **TUI adaptation**: TamiyoBrain shows network type badge; attention heatmap available
7. **All tests pass**: Including existing PPO integration tests and new telemetry tests

---

## References

- PyTorch TransformerEncoder: https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html
- Set Transformer paper: https://arxiv.org/abs/1810.00825 (for alternatives)
- FlexAttention: https://pytorch.org/blog/flexattention/ (PyTorch 2.5+)
