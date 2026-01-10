# Blueprint Compiler Appendix: Future Curriculum Blueprints

> **Parent Plan:** `2026-01-09-blueprint-compiler-and-curriculum-seeds.md`
> **Status:** Deferred to Phase 3 (post-compiler infrastructure)

**Expert Review Status:** v1 - Reviewed by DRL and PyTorch specialists

**Expert Sign-offs:**
- ✅ DRL: 6/7 blueprints approved, Reparam needs redesign (45k params, lesson not reward-capturable)
- ✅ PyTorch: 2 critical fixes (Shift, Laplacian), 2 minor fixes (ECA, Shuffle), 1 design concern (Reparam)

This appendix catalogs the 7 additional curriculum blueprints identified during design review. These are **not** included in the main implementation plan but are documented here for future work.

---

## Index Allocation

| Index | Blueprint | Phase | Status |
|-------|-----------|-------|--------|
| 0-7 | Legacy CNN (noop, conv_light, attention, norm, depthwise, bottleneck, conv_small, conv_heavy) | - | Existing |
| 8-12 | Legacy Transformer (lora, lora_large, mlp_small, mlp, flex_attention) | - | Existing |
| 13-16 | Phase 2 CNN (dilated, asymmetric, coord, gated) | 2 | In main plan |
| **17-23** | **Phase 3 CNN (this appendix)** | 3 | **Deferred** |

---

## Phase 3 Blueprints

### 1. ECA (Efficient Channel Attention) — Index 17

**Curriculum Lesson:** Global context without full SE overhead; 1D conv as a cheap cross-channel mixer.

**Trade-off:** Very cheap but limited expressiveness compared to full attention.

**Implementation Sketch:**
```python
@BlueprintRegistry.register(
    "eca", "cnn",
    param_estimate=200,
    action_index=17,
    description="Efficient Channel Attention (1D conv across channels)"
)
def create_eca_seed(dim: int, kernel_size: int = 3, layer_scale: float = 1e-3, **kwargs: Any) -> nn.Module:
    """ECA: squeeze → 1D conv → sigmoid gating.

    Reference: ECA-Net (Wang et al., 2020)
    """
    class ECASeed(nn.Module):
        def __init__(self, channels: int, k: int, layer_scale_init: float):
            super().__init__()
            # Adaptive kernel size based on channels (ECA paper formula)
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
            self.ls = LayerScale(channels, init_value=layer_scale_init)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, c, h, w = x.shape
            # Global average pool → [B, C, 1, 1] → [B, 1, C]
            # Use reshape (not view) for robustness with non-contiguous tensors
            y = self.gap(x).reshape(b, 1, c)
            # 1D conv across channel dimension
            y = self.conv(y)  # [B, 1, C]
            y = torch.sigmoid(y).reshape(b, c, 1, 1)
            return x + self.ls(x * y - x)  # Residual of the attention delta

    return ECASeed(dim, kernel_size, layer_scale)
```

---

### 2. Spatial Attention — Index 18

**Curriculum Lesson:** "Where to look" via spatial (not channel) gating.

**Trade-off:** Complements channel attention; together they form CBAM-style dual attention.

**Implementation Sketch:**
```python
@BlueprintRegistry.register(
    "spatial_attn", "cnn",
    param_estimate=1500,
    action_index=18,
    description="Spatial attention (max+avg pool → conv → sigmoid)"
)
def create_spatial_attn_seed(dim: int, layer_scale: float = 1e-3, **kwargs: Any) -> nn.Module:
    """Spatial attention from CBAM (Woo et al., 2018)."""
    class SpatialAttnSeed(nn.Module):
        def __init__(self, channels: int, layer_scale_init: float):
            super().__init__()
            self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
            self.ls = LayerScale(channels, init_value=layer_scale_init)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Channel-wise max and avg → [B, 2, H, W]
            max_pool = x.max(dim=1, keepdim=True)[0]
            avg_pool = x.mean(dim=1, keepdim=True)
            spatial_desc = torch.cat([max_pool, avg_pool], dim=1)
            # Spatial attention map
            attn = torch.sigmoid(self.conv(spatial_desc))  # [B, 1, H, W]
            return x + self.ls(x * attn - x)

    return SpatialAttnSeed(dim, layer_scale)
```

---

### 3. Shift — Index 19

**Curriculum Lesson:** Zero-parameter spatial mixing via channel shifts (ShiftNet).

**Trade-off:** Free FLOPs but requires careful channel grouping; limited receptive field.

**Implementation Sketch:**
```python
@BlueprintRegistry.register(
    "shift", "cnn",
    param_estimate=4200,  # Only from the 1x1 conv
    action_index=19,
    description="Channel shift + 1x1 conv (zero-cost spatial mixing)"
)
def create_shift_seed(dim: int, layer_scale: float = 1e-3, **kwargs: Any) -> nn.Module:
    """Shift channels spatially, then mix with 1x1 conv.

    Reference: Shift (Wu et al., 2018)
    """
    class ShiftSeed(nn.Module):
        def __init__(self, channels: int, layer_scale_init: float):
            super().__init__()
            self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
            self.gn = nn.GroupNorm(get_num_groups(channels), channels)
            self.ls = LayerScale(channels, init_value=layer_scale_init)
            # Pre-compute channel splits for forward pass
            g = channels // 5
            self.splits = [g, g, g, g, channels - 4 * g]  # 4 directions + identity

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Split channels into 5 groups: 4 shift directions + identity
            # Use functional approach (no clone + in-place) for torch.compile compatibility
            g = self.splits[0]

            # Zero-pad and slice for each shift direction
            left = F.pad(x[:, :g], (0, 1, 0, 0))[:, :, :, 1:]        # shift left
            right = F.pad(x[:, g:2*g], (1, 0, 0, 0))[:, :, :, :-1]    # shift right
            up = F.pad(x[:, 2*g:3*g], (0, 0, 0, 1))[:, :, 1:, :]      # shift up
            down = F.pad(x[:, 3*g:4*g], (0, 0, 1, 0))[:, :, :-1, :]   # shift down
            identity = x[:, 4*g:]  # remaining channels unchanged

            out = torch.cat([left, right, up, down, identity], dim=1)
            y = self.gn(self.conv(out))
            return x + self.ls(y)

    return ShiftSeed(dim, layer_scale)
```

**Note:** The in-place clone pattern may need adjustment for torch.compile compatibility.

---

### 4. Shuffle — Index 20

**Curriculum Lesson:** Cross-group information flow via channel shuffling (ShuffleNet).

**Trade-off:** Enables cheap group convs to communicate; shuffle itself is free.

**Implementation Sketch:**
```python
@BlueprintRegistry.register(
    "shuffle", "cnn",
    param_estimate=2500,
    action_index=20,
    description="Group conv + channel shuffle (ShuffleNet pattern)"
)
def create_shuffle_seed(dim: int, groups: int = 4, layer_scale: float = 1e-3, **kwargs: Any) -> nn.Module:
    """Group conv followed by channel shuffle.

    Reference: ShuffleNet (Zhang et al., 2018)
    """
    class ShuffleSeed(nn.Module):
        def __init__(self, channels: int, groups: int, layer_scale_init: float):
            super().__init__()
            # Validate divisibility for channel shuffle
            if channels % groups != 0:
                raise ValueError(f"channels ({channels}) must be divisible by groups ({groups})")
            self.groups = groups
            self.gconv = nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                                   groups=groups, bias=False)
            self.gn = nn.GroupNorm(get_num_groups(channels), channels)
            self.ls = LayerScale(channels, init_value=layer_scale_init)

        def channel_shuffle(self, x: torch.Tensor) -> torch.Tensor:
            b, c, h, w = x.shape
            # [B, C, H, W] → [B, G, C//G, H, W] → [B, C//G, G, H, W] → [B, C, H, W]
            x = x.view(b, self.groups, c // self.groups, h, w)
            x = x.transpose(1, 2).contiguous()
            return x.view(b, c, h, w)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.gconv(x)
            y = self.channel_shuffle(y)
            y = self.gn(y)
            return x + self.ls(y)

    return ShuffleSeed(dim, groups, layer_scale)
```

---

### 5. Reparam (Re-parameterizable Conv) — Index 21

> ⚠️ **NEEDS REDESIGN** — DRL expert flagged: 45k params is 10-20× larger than other Phase 3
> blueprints, and the core lesson (train/inference structural difference) cannot be captured
> by standard RL loss rewards. Consider simplifying to 3x3 + identity only (~15k params) or
> deferring until an explicit "inference efficiency" reward is implemented.

**Curriculum Lesson:** Training-time multi-branch → inference-time single conv fusion.

**Trade-off:** Complex training graph but fuses to efficient inference; teaches structural reparameterization.

**Implementation Sketch (Simplified - GroupNorm version):**
```python
@BlueprintRegistry.register(
    "reparam", "cnn",
    param_estimate=15000,  # Reduced from 45k (removed 1x1 branch)
    action_index=21,
    description="Re-parameterizable 3x3 (simplified RepVGG-style)"
)
def create_reparam_seed(dim: int, layer_scale: float = 1e-3, **kwargs: Any) -> nn.Module:
    """Simplified multi-branch conv (3x3 + identity only).

    Reference: RepVGG (Ding et al., 2021) - simplified for RL learnability

    Training: 3x3 + identity (parallel branches)
    Inference: Can be fused to single 3x3 conv

    Note: Uses GroupNorm (not BatchNorm) for consistency with other seeds
    and to avoid running stats drift during dynamic instantiation.
    """
    class ReparamSeed(nn.Module):
        def __init__(self, channels: int, layer_scale_init: float):
            super().__init__()
            self.conv3x3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
            # Use GroupNorm for consistency with project standard (no running stats)
            num_groups = get_num_groups(channels)
            self.gn3x3 = nn.GroupNorm(num_groups, channels)
            self.gn_id = nn.GroupNorm(num_groups, channels)
            self.ls = LayerScale(channels, init_value=layer_scale_init)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Simplified: 3x3 branch + identity branch
            y = self.gn3x3(self.conv3x3(x)) + self.gn_id(x)
            return x + self.ls(y)

    return ReparamSeed(dim, layer_scale)
```

**Design Notes:**
- Original RepVGG uses BatchNorm for fusion compatibility, but GroupNorm is used here for consistency with the morphogenetic training regime (no running stats drift)
- If true reparameterization fusion is needed at deployment, a separate "deployment mode" conversion would be required
- The 1x1 branch was removed to reduce param count and improve RL learnability

---

### 6. MixDW (Mixed Depthwise) — Index 22

**Curriculum Lesson:** Multi-scale depthwise via mixed kernel sizes (MixConv).

**Trade-off:** Richer multi-scale features; slightly more complex than single kernel DW.

**Implementation Sketch:**
```python
@BlueprintRegistry.register(
    "mix_dw", "cnn",
    param_estimate=6000,
    action_index=22,
    description="Mixed depthwise conv (3x3 + 5x5 parallel)"
)
def create_mix_dw_seed(dim: int, layer_scale: float = 1e-3, **kwargs: Any) -> nn.Module:
    """Mixed depthwise convolution with multiple kernel sizes.

    Reference: MixConv (Tan & Le, 2019)
    """
    class MixDWSeed(nn.Module):
        def __init__(self, channels: int, layer_scale_init: float):
            super().__init__()
            # Split channels between kernel sizes
            self.c_split = channels // 2
            self.dw3 = nn.Conv2d(self.c_split, self.c_split, kernel_size=3,
                                 padding=1, groups=self.c_split, bias=False)
            self.dw5 = nn.Conv2d(channels - self.c_split, channels - self.c_split,
                                 kernel_size=5, padding=2,
                                 groups=channels - self.c_split, bias=False)
            self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
            self.gn = nn.GroupNorm(get_num_groups(channels), channels)
            self.ls = LayerScale(channels, init_value=layer_scale_init)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x1, x2 = x.split([self.c_split, x.size(1) - self.c_split], dim=1)
            y = torch.cat([self.dw3(x1), self.dw5(x2)], dim=1)
            y = self.gn(self.pw(y))
            return x + self.ls(y)

    return MixDWSeed(dim, layer_scale)
```

---

### 7. Laplacian (Multi-scale Pyramid) — Index 23

**Curriculum Lesson:** Explicit multi-scale decomposition via Laplacian pyramid.

**Trade-off:** Strong inductive bias for scale-awareness; heavier than single-scale ops.

**Implementation Sketch:**
```python
@BlueprintRegistry.register(
    "laplacian", "cnn",
    param_estimate=15000,
    action_index=23,
    description="Laplacian pyramid decomposition (multi-scale processing)"
)
def create_laplacian_seed(dim: int, layer_scale: float = 1e-3, **kwargs: Any) -> nn.Module:
    """Process at multiple scales via Laplacian pyramid.

    Decomposes input into low-freq (downsampled) and high-freq (residual),
    processes each, then reconstructs.
    """
    class LaplacianSeed(nn.Module):
        def __init__(self, channels: int, layer_scale_init: float):
            super().__init__()
            # Low-frequency path (downscale → process → upscale)
            self.down = nn.AvgPool2d(2)
            self.low_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
            # Note: nn.Upsample removed - use F.interpolate with explicit size for torch.compile

            # High-frequency path (detail residual)
            self.high_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

            self.gn = nn.GroupNorm(get_num_groups(channels), channels)
            self.ls = LayerScale(channels, init_value=layer_scale_init)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h, w = x.shape[2:]

            # Laplacian decomposition
            # Always use F.interpolate with explicit target size (torch.compile safe)
            low = self.down(x)
            low_up = F.interpolate(low, size=(h, w), mode='bilinear', align_corners=False)
            high = x - low_up  # High-frequency residual

            # Process each scale
            low_out = F.interpolate(
                F.relu(self.low_conv(low)),
                size=(h, w),
                mode='bilinear',
                align_corners=False
            )
            high_out = F.relu(self.high_conv(high))

            # Reconstruct
            y = self.gn(low_out + high_out)
            return x + self.ls(y)

    return LaplacianSeed(dim, layer_scale)
```

---

## Implementation Notes

### Prerequisites
- Phase 2 blueprints (indices 13-16) must be implemented first
- BlueprintCompiler infrastructure must be complete
- LayerScale helper must be available

### Testing Strategy
Each blueprint should have:
1. **Shape preservation test** - `y.shape == x.shape`
2. **Learnability test** - optimizer step changes output
3. **Near-identity at init test** - `(y - x).abs().mean() < threshold`
4. **Param estimate accuracy test** - actual vs. estimated within 20%
5. **torch.compile fullgraph test** - no graph breaks under compilation

**PyTorch Expert Recommended Tests:**
```python
@pytest.mark.parametrize("seed_name", ["eca", "spatial_attn", "shift", "shuffle", "reparam", "mix_dw", "laplacian"])
def test_phase3_seed_compiles_fullgraph(seed_name: str):
    """Phase 3 seeds must compile without graph breaks."""
    seed = BlueprintRegistry.create("cnn", seed_name, dim=64)
    compiled_seed = torch.compile(seed, fullgraph=True)

    x = torch.randn(2, 64, 32, 32)
    y = compiled_seed(x)
    assert y.shape == x.shape


def test_laplacian_handles_odd_sizes():
    """LaplacianSeed must handle odd spatial dimensions without graph breaks."""
    seed = BlueprintRegistry.create("cnn", "laplacian", dim=64)
    compiled_seed = torch.compile(seed, fullgraph=True)

    for size in [(33, 33), (31, 47), (17, 17)]:
        x = torch.randn(2, 64, *size)
        y = compiled_seed(x)
        assert y.shape == x.shape
```

### Ordering Recommendations

**DRL Expert Recommended Order** (curriculum progression):

1. **eca** (17) — Simplest, introduces efficient 1D-conv attention
2. **spatial_attn** (18) — Builds on attention, introduces spatial dimension
3. **shift** (19) — Zero-param spatial mixing (simpler concept than shuffle)
4. **shuffle** (20) — Cross-group communication builds on understanding of grouped convs
5. **mix_dw** (22) — Multi-scale via kernel mixing
6. **laplacian** (23) — Explicit multi-scale decomposition (capstone)
7. **reparam** (21) — Only if redesigned; may not fit standard RL rewards

**Curriculum Coherence:**
- **Attention efficiency** (1-2): ECA, Spatial — lightweight attention variants
- **Spatial mixing efficiency** (3-4): Shift, Shuffle — zero-param or grouped approaches
- **Multi-scale efficiency** (5-6): MixDW, Laplacian — scale-aware processing
- **Structural efficiency** (7): Reparam — train/inference optimization (problematic for RL)

---

## Future Considerations

### Transformer Phase 3 Blueprints (not specified yet)
The transformer topology may also benefit from additional blueprints:
- **RoPE injection** — Rotary position embeddings
- **SwiGLU** — Gated MLP variant
- **Grouped Query Attention** — GQA pattern
- **Mixture of Experts** — Sparse activation

These would use indices 24+ to maintain global uniqueness.

### Dynamic Index Allocation
If the blueprint count grows significantly, consider:
- Reserving index ranges per topology (e.g., CNN: 0-99, Transformer: 100-199)
- Or using a topology prefix in a compound index scheme

For now, sequential global allocation is sufficient.
