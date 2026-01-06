Short answer: **2×3 or 2×5 is a “few dozen lines + tests” change** (add a second “surface” per block: pre-pool and post-pool). A *true* **3×5** that means “three independent chains” is closer to a sprint (you need multi-lane topology and a merge).

Right now your CNNHost only exposes **one boundary per block**:

```py
slot_id = format_slot_id(0, i)   # always row 0
layer_range = (i, i+1)
```

…and `forward_to_segment()` assumes “segment = block boundary”. To get a 2×N grid without rewriting MorphogeneticModel, the trick is:

* keep MorphogeneticModel’s linear pass (it already follows `host.injection_specs()` order)
* make CNNHost expose **two boundaries per block** in a deterministic order
* make CNNHost routing understand “boundary within a block”

## Option 1: 2×N grid via pre/post-pool surfaces (recommended)

This gives you:

* **rows = surface type** (row 0 = pre-pool, row 1 = post-pool)
* **cols = block index** (0..n_blocks-1)

So `n_blocks=3` → **2×3**, `n_blocks=5` → **2×5**.

### What to change

### 1) Change `injection_specs()` to emit two specs per block

Pre-pool always exists. Post-pool exists only where pooling exists (`idx < pool_layers`).

```py
def injection_specs(self) -> list["InjectionSpec"]:
    from esper.leyline import InjectionSpec
    from esper.leyline.slot_id import format_slot_id

    specs: list[InjectionSpec] = []
    for i in range(self.n_blocks):
        block = self.blocks[i]
        assert isinstance(block, ConvBlock)

        # Row 0: PRE_POOL boundary (after block, before pooling)
        specs.append(InjectionSpec(
            slot_id=format_slot_id(0, i),
            channels=block.conv.out_channels,
            position=(2*i + 1) / (2*self.n_blocks),
            layer_range=(i, i + 1),
        ))

        # Row 1: POST_POOL boundary (after pooling), only if pooling applies
        if i < self._pool_layers:
            specs.append(InjectionSpec(
                slot_id=format_slot_id(1, i),
                channels=block.conv.out_channels,
                position=(2*i + 2) / (2*self.n_blocks),
                layer_range=(i, i + 1),
            ))
    return specs
```

### 2) Replace `_segment_to_block` with `_segment_to_boundary`

You need to know both:

* which block index `i`
* which surface row (0 = pre, 1 = post)

```py
@functools.cached_property
def _segment_to_boundary(self) -> dict[str, tuple[int, int]]:
    # returns (block_idx, surface_row)
    mapping: dict[str, tuple[int, int]] = {}
    for spec in self.injection_specs():
        # slot_id is r{row}c{col}
        row = int(spec.slot_id[1])  # or parse properly; better to use a helper
        col = int(spec.slot_id[3:]) # ditto
        mapping[spec.slot_id] = (col, row)
    return mapping
```

But don’t parse strings in the hot path. Better: store row/col in `InjectionSpec` long-term. For now, you can parse once during cached_property initialisation and it’s fine.

### 3) Rewrite `forward_to_segment()` as a boundary-walker

The easiest robust method is to precompute a linear “boundary timeline” of operations, then execute from start boundary index to end boundary index.

Conceptually per block:

* step A: apply block → reaches PRE_POOL boundary
* step B (if pooled): apply pool → reaches POST_POOL boundary

Implementation sketch:

```py
@functools.cached_property
def _boundary_timeline(self) -> tuple[list[tuple[str, int]], dict[str, int]]:
    """
    timeline: list of (op, block_idx)
      op in {"block", "pool"}
    boundary_index: slot_id -> timeline index where that boundary is reached
    """
    from esper.leyline.slot_id import format_slot_id
    timeline: list[tuple[str, int]] = []
    boundary_index: dict[str, int] = {}

    for i in range(self.n_blocks):
        # After block => PRE_POOL boundary r0c{i}
        timeline.append(("block", i))
        boundary_index[format_slot_id(0, i)] = len(timeline) - 1

        if i < self._pool_layers:
            # After pool => POST_POOL boundary r1c{i}
            timeline.append(("pool", i))
            boundary_index[format_slot_id(1, i)] = len(timeline) - 1

    return timeline, boundary_index


def forward_to_segment(self, segment: str, x: torch.Tensor, from_segment: str | None = None) -> torch.Tensor:
    timeline, boundary_index = self._boundary_timeline

    # memory_format conversion unchanged
    if from_segment is None and self._memory_format == torch.channels_last:
        x = x.to(memory_format=torch.channels_last)

    start_idx = -1 if from_segment is None else boundary_index[from_segment]
    end_idx = boundary_index[segment]

    if from_segment is not None and start_idx >= end_idx:
        raise ValueError("Cannot route backwards or to same boundary")

    # Execute operations to move from start boundary to end boundary
    for k in range(start_idx + 1, end_idx + 1):
        op, i = timeline[k]
        if op == "block":
            x = self.blocks[i](x)
        else:  # "pool"
            x = self.pool(x)
    return x
```

### 4) Rewrite `forward_from_segment()` similarly

Continue from the boundary index to the end of the backbone, then classify.

```py
def forward_from_segment(self, segment: str, x: torch.Tensor) -> torch.Tensor:
    timeline, boundary_index = self._boundary_timeline
    start_idx = boundary_index[segment]

    for k in range(start_idx + 1, len(timeline)):
        op, i = timeline[k]
        if op == "block":
            x = self.blocks[i](x)
        else:
            x = self.pool(x)

    x = F.adaptive_avg_pool2d(x, 1).flatten(1)
    return self.classifier(x)
```

### 5) MorphogeneticModel doesn’t need changes

It already does:

```py
self._slot_order = [spec.slot_id for spec in host.injection_specs()]
```

So once injection_specs returns `r0c0, r1c0, r0c1, r1c1, ...` you automatically get a deterministic linear traversal across the 2×N grid.

**Effort estimate:** closer to “evening + tests” than “6-week sprint”. Biggest work is careful tests and ensuring no backwards routing.

**WEP:** Highly likely you can get a 2×3 / 2×5 working quickly with this approach.

---

## Option 2: True 3×5 grid (three chains) is a bigger change

If you mean “three independent chains of depth 5” (rows = lanes, cols = depth), you’re talking about a **multi-lane CNNHost**:

* build 3 towers (each a ModuleList of blocks)
* expose injection_specs for each lane: `r{lane}c{depth}`
* define a merge (sum / concat+1×1 / gated merge) before classifier
* update forward_to_segment routing to handle lane-local routing and a consistent merge point

That’s more engineering because you now have:

* multiple activation streams
* a merge semantics (and that affects what “segment” means)
* more invariants to test and keep deterministic

**Effort estimate:** “sprint” is credible.

**WEP:** Likely to take longer than you want for a quick slot scaling experiment.

---

## Recommendation

* If your goal is **“more slots now”**, do **2×N via pre/post pool surfaces**. It’s conceptually meaningful (resolution-aware) and mechanically simple.
* If your goal is **“multiple independent slot chains”** as a precursor to Esika/Narset partitioning, then plan a multi-lane host as an explicit milestone.
