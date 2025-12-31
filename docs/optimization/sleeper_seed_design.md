# Design: "Sleeper Seed" Static Graph Architecture
**Date:** January 1, 2026
**Author:** PyTorch 2.9 Specialist Agent
**Reference:** `docs/optimization/hot_path_analysis.md`

## 1. Concept

The primary barrier to full graph compilation (`torch.compile(mode="reduce-overhead")`) in the current architecture is the **Dynamic Topology** caused by `ModuleDict` mutation during `germinate_seed` and `prune_seed`.

The "Sleeper Seed" architecture replaces dynamic instantiation with **Static Pre-allocation** and **Dynamic Masking**.

## 2. Architecture

### 2.1 The Container
Instead of `ModuleDict`, we use a `nn.ModuleList` of fixed size $K$ (Capacity) per slot.

```python
class StaticSlotContainer(nn.Module):
    def __init__(self, capacity=4, ...):
        super().__init__()
        # Pre-allocate K seeds. They are "physically" present but "logically" dead.
        self.seeds = nn.ModuleList([SeedSlot(...) for _ in range(capacity)])
        
        # Buffer to track active status (Persistent state)
        # 1.0 = Active, 0.0 = Sleeper
        self.register_buffer("active_mask", torch.zeros(capacity))
        
        # Track blueprint types (Integer ID)
        # 0 = None, 1 = CNN-A, 2 = CNN-B, ...
        self.register_buffer("blueprint_ids", torch.zeros(capacity, dtype=torch.long))

    def forward(self, x):
        # Accumulate output from all ACTIVE seeds
        out = torch.zeros_like(x)
        
        for i, seed in enumerate(self.seeds):
            # Branchless execution (ideal for CUDA Graphs)
            # The mask multiplication happens *after* the seed forward to ensure
            # constant compute graph, OR we use specialized kernels.
            # For purely static graphs, we must run the forward.
            
            # Optimization: If we trust dynamic control flow within compiled graph:
            # if self.active_mask[i] > 0.5:
            #     out += seed(x)
            
            # "PyTorch 2.9" Static Approach (No Graph Breaks):
            seed_out = seed(x)
            mask = self.active_mask[i].view(1, 1, 1, 1) # Broadcast
            out = out + (seed_out * mask)
            
        return out
```

### 2.2 Lifecycle Operations (Virtual Surgery)

**Germinate:**
Instead of `new Seed()`, we:
1.  Find the first index `i` where `active_mask[i] == 0`.
2.  Reset parameters of `self.seeds[i]` (in-place).
3.  Set `active_mask[i] = 1.0`.

**Prune:**
Instead of `del dict[key]`, we:
1.  Identify index `i`.
2.  Set `active_mask[i] = 0.0`.
3.  (Optional) Zero out parameters to prevent gradient accumulation, though `mask` handles the forward pass.

### 2.3 Challenges & Solutions

**Challenge 1: Parameter Initialization overhead**
*   **Current:** `Germinate` allocates new memory.
*   **Static:** Re-initialization (filling existing tensor with random) is faster and doesn't fragment memory. It can be captured in the graph if using RNG states correctly.

**Challenge 2: Different Blueprints**
*   **Problem:** `SeedSlot` isn't generic; we have different architectures (CNN vs Attention).
*   **Solution:** The `Sleeper` pool must contain pre-allocated instances of *supported* blueprints. 
    *   *Option A (Homogeneous):* All seeds are the same "SuperNet".
    *   *Option B (Heterogeneous):* Capacity=4 means "2 CNN seeds + 2 Attn seeds". We only activate the correct type.

**Challenge 3: Compute Waste**
*   **Problem:** Running forward pass on "Sleeper" seeds and multiplying by 0 is wasteful.
*   **Solution:** `torch.compile` is smart enough to optimize `x * 0`. However, the *layers* inside the seed might still run.
*   **Advanced Solution:** Use **Streams**.
    *   Active seeds run in main stream.
    *   Sleeper seeds don't run (using `if` logic inside the graph).
    *   `torch.cond` (PyTorch 2.1+) allows data-dependent control flow that compiles to GPU predicates, avoiding CPU round-trip.

## 3. Transition Strategy

1.  **Phase 1:** Implement `StaticSlotContainer` but keep Python logic. Manually manage the masks.
2.  **Phase 2:** Use `torch.where` or `mask` for the forward pass.
3.  **Phase 3:** Wrap the entire training step in `torch.compile`.

## 4. Expected Gains

*   **Graph Capture:** 100% of the training loop (including lifecycle) becomes a single CUDA graph.
*   **Zero CPU Overhead:** The "CPU Trench" is completely filled.
*   **Throughput:** Estimated 2x-5x speedup for small-batch RL where CPU scheduling is the bottleneck.
