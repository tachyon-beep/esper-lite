# PyTorch SDPA and PyG GAT — Compatibility Notes (Prototype)

Purpose
- Summarise feasibility and constraints for adopting PyTorch scaled dot‑product attention (SDPA) and CUDA Graphs with PyTorch Geometric’s graph attention in Tamiyo’s hetero‑GNN.

Context
- Tamiyo uses `torch_geometric.nn.GATConv` inside a heterogenous `HeteroConv` stack (GraphSAGE → GAT). Edge attributes (`edge_dim>0`) are used on several relations.
- PyTorch SDPA (`torch.nn.functional.scaled_dot_product_attention`) and memory‑efficient attention kernels are optimised for dense, regular sequence attention (B×H×Q×K), not sparse, irregular neighbourhood attention.

Takeaways
- SDPA is not a drop‑in replacement for GATConv today:
  - GAT operates on sparse adjacency with per‑edge attention; SDPA expects dense key/query/value blocks over regular dimensions.
  - Mapping neighbourhood attention to SDPA requires batching variable‑length neighbourhoods into padded blocks (or NestedTensor) and reconstructing graph structure post‑attention; this increases memory and complexity and complicates use of `edge_attr`.
  - Edge attributes (common in Tamiyo) further deviate from SDPA’s QK^T formulation and require custom projections or biasing that SDPA does not natively expose per edge.
- CUDA Graphs capture: forward capture of the hetero‑GNN is admissible only when tensor shapes and adjacency patterns remain static. In Tamiyo, graph size and edges vary step‑to‑step, making capture brittle; recapture costs often outweigh savings.

Recommended Posture (Prototype)
- Keep GATConv for graph attention. Optimise inference via:
  - `torch.compile(dynamic=True, mode="reduce-overhead")` with eager fallback (Implemented).
  - CUDA warm‑up on a tiny hetero‑graph to reduce first‑step variance; export `tamiyo.gnn.compile_warm_ms` (Implemented).
  - TF32 (`set_float32_matmul_precision('high')`) and autocast (bfloat16) on CUDA (Implemented).
  - Vectorise graph‑builder hot paths (Implemented) to reduce per‑step overhead before kernels.
- Defer CUDA Graphs capture unless shapes and adjacency are provably static across many steps.

When to Re‑evaluate
- If PyG adds native SDPA‑backed attention or block‑sparse attention kernels compatible with `edge_attr`.
- If NestedTensor + SDPA can stably cover ragged neighbourhoods without prohibitive padding/overhead.
- If future workloads have long runs with fixed graph structure allowing CUDA Graph capture to amortise recapture costs.

Alternatives to Explore (Future)
- Replace GATConv with PyG `TransformerConv` if attention needs to be closer to transformer‑style (still sparse, but may evolve towards SDPA backends in future releases).
- Block‑sparse/dilated attention kernels tailored to KNN or fixed‑degree neighbourhoods (research area; not mainstream in PyG today).

Operational Notes
- Continue to prioritise compile + autocast/TF32 and data‑movement reductions (pinned memory, non‑blocking transfers) before considering attention kernel swaps.
- Keep perf tests opt‑in and tied to realistic graphs; prefer stability over speculative kernel changes in the prototype.

