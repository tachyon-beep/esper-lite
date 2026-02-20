# Phase 3 Strategy: The Second Domain Pivot (TinyStories)

**Status:** DRAFT
**Owner:** Strategy Partner
**Date:** 2025-12-19

---

## 1. Strategic Context

Esper has proven "Morphogenetic AI" on ResNet architectures (CIFAR-10). The core hypothesis—that an RL agent can grow a network by observing gradients—has been validated in the visual domain.

**The Risk:** The current policy (`Tamiyo`) might be overfitting to *convolutional* dynamics (spatial hierarchies, channel pruning).

**The Pivot:** We must demonstrate that the **same** morphogenetic principles apply to **Transformers** (Language Modeling). We will use the **TinyStories** dataset (Eldan & Li, 2023) as the "University" environment for this curriculum.

**Why TinyStories?**
- **Fast Iteration:** Trains in minutes/hours, not days.
- **Structural Richness:** Requires grammar, consistency, and reasoning, unlike synthetic tasks.
- **Architectural Relevance:** GPT-style blocks are the lingua franca of modern AI.

---

## 2. Architecture: The Transformer Host

We need a new host implementation that conforms to `HostProtocol` but manages a Transformer backbone.

### 2.1 The Host (`TransformerHost`)
- **Backbone:** GPT-2 style decoder-only transformer.
- **Config:** `n_layer=4`, `n_head=4`, `n_embd=256` (Base) → grows to `n_layer=8+`.
- **Injection Points:**
  - **Pre-Attention:** Inject `AttentionSeed` (extra heads).
  - **Post-FFN:** Inject `FFNSeed` (extra MLP width or blocks).
  - **Residual Stream:** Inject `BlockSeed` (entire new layers).

### 2.2 New Blueprints
Kasmina's "Stem Cells" need new DNA to differentiate into transformer components.

| Blueprint | Role | Description |
|-----------|------|-------------|
| **AttentionSeed** | "New Eyes" | Adds a new attention head (Query/Key/Value projections) to an existing layer. |
| **FFNSeed** | "New Muscle" | Adds parallel MLP capacity (wider intermediate dimension) or a sparsely gated expert. |
| **ResidualSeed** | "New Depth" | Inserts a zero-initialized identity block that evolves into a full transformer layer. |

---

## 3. The Curriculum

We will not train a 7B model. We will train a "gardener" that knows *when* to add capacity to a small model.

### 3.1 Task: `tinystories`
- **Vocab:** ~30k (standard tokenizer) or custom small vocab.
- **Objective:** Causal Language Modeling (Next Token Prediction).
- **Metric:** Perplexity (PPL) and Validation Loss.

### 3.2 Experiments
1.  **Baseline (Static):** Train a fixed 4-layer GPT. Record Wall-clock vs Loss.
2.  **Growth (Heuristic):** Use `Tamiyo` (rule-based) to add heads/layers when loss plateaus.
3.  **Growth (PPO):** Train `Simic` to optimize `(-Loss - Rent)`.

---

## 4. Implementation Stages

### Stage 1: The Transformer Skeleton
- Implement `TransformerHost` in `src/esper/tolaria/model/transformer.py`.
- Ensure it exposes `HostProtocol` (injection specs).
- Verify standard training loop works (no growth).

### Stage 2: The New Seeds
- Implement `AttentionSeed` and `FFNSeed` in `src/esper/kasmina/blueprints/`.
- **Constraint:** Must support "Zero Initialization" (start as identity function) to prevent shock when grafted.

### Stage 3: The TinyStories Dataset
- Create `src/esper/tolaria/data/tinystories.py`.
- Support streaming from HuggingFace `datasets` or local pre-tokenized shards.

### Stage 4: Integration
- Create `presets/tinystories.json` config.
- Run `train heuristic --task tinystories` to verify mechanics.

---

## 5. Success Criteria

1.  **Mechanics:** Can we graft a new Head/Layer into a running GPT without NaN spikes?
2.  **Efficiency:** Does the growing model achieve lower loss *per parameter-second* than a large static model?
3.  **Transfer:** (Stretch) Does a policy trained on CIFAR-10 growth generalize to TinyStories? (Likely no, but valuable to measure).

---

## 6. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| **Gradient Instability** | Use LayerNorm pre-injection. Zero-init output projections of new seeds. |
| **Tokenizer Overhead** | Pre-tokenize data. Use simple character-level or small BPE for initial tests. |
| **Slow Training** | TinyStories is small, but Transformers are heavy. Use `flash-attention` if available. |
