# Esper: Architecture Constitution & Project Roadmap

**Mission:** Transition AI from *Architectural Engineering* (static design) to *Architectural Ecology* (dynamic growth).

This document is the **Constitution** of Esper. These principles have been validated in experiments and are intended to remain stable as the system scales.

---

## Part I: Architecture Constitution

### 1. The Signal-to-Noise Hypothesis

**Concept:** An RL agent is only as smart as its sensors.

**Validation:**

* When telemetry was broken or noisy, Tamiyo learned nothing useful.
* When we fixed the telemetry – clean gradient norms, distinct per-layer stats, stage-aware metrics – Tamiyo immediately started picking up structural patterns:

  * e.g. depthwise vs conv seeds showing different accuracy/gradient profiles in CIFAR experiments.

**Design Rule:** Never add a capability to the Body (Kasmina/host) without adding a corresponding sensor to Nissa (Senses). If Tamiyo can’t see it, she can’t optimise it.

```text
Body Feature        →  Sensor Required
────────────────────────────────────────────
New seed type       →  Gradient telemetry for that seed
New lifecycle stage →  Stage-specific metrics
New host architecture → Injection point visibility, host-specific signals
```

---

### 2. The Rent & Churn Economy

**Concept:** Complexity is not free; it pays rent in energy/parameters.

**Mechanism:**

* **Rent:** Extra parameters and compute incur a cost proportional to their mass:

  [
  \text{excess_params_ratio} = \frac{P_{\text{fossilised}} + P_{\text{active}}}{P_{\text{host}}}
  ]

  Reward includes a small penalty:

  [
  R ;\mathrel{-}= \lambda \cdot \text{excess_params_ratio}
  ]

  with λ chosen conservatively (e.g. 0.05) so rent is noticeable but not dominant.

* **Churn:** Withering (alpha decay 1.0 → 0.0) and culling act as audits:

  * Remove / wither a seed.
  * Measure the change in loss/accuracy (variance, Δacc).
  * High churn ⇒ the seed was structurally important.
  * Low churn ⇒ the seed was effectively dead weight.

**Churn Logic:**

* If withering causes **zero** churn → seed was useless → **CULL**.
* If withering causes **high** churn → seed was structural → **REVIVE** or keep.

```text
Emrakul's Question: "What happens if I slowly remove this?"

Answer A: "Nothing"      → It was parasitic  → Kill it
Answer B: "System dies"  → It was essential  → Spare it
```

**Design Rule:** Emrakul never deletes arbitrarily; she only pressure-tests. Deletion is the side-effect of a failed test. Reward and analytics must reflect **accuracy minus rent**, not accuracy alone.

---

### 3. The Inverted Control Flow

**Concept:** Python’s GIL is the enemy of throughput.

**Mechanism:**

Traditional RL training loops often do:

```python
# Traditional: GIL-bound, env-first
for env in envs:
    for batch in data:
        env.step(batch)
```

Esper inverts this:

```python
# Esper: batch-first, GPU-native dispatch
for batch in data:
    dispatch_to_envs(batch)  # multiple CUDA streams / devices
```

We:

* Iterate **DataLoaders first** (batches),
* Dispatch to **environments second** (CUDA streams / devices),
* Use pre-allocated GPU tensors / buffers for all comms.

**Validation:** Training logs show env0, env1, env2… interleaving cleanly, all GPUs saturated, and the policy loop never blocked by Python-level scheduling.

**Design Rule:** The Training Loop (`Tolaria`) must never block the Policy Loop (`Simic`). Communication happens via device-resident buffers, not ad-hoc Python queues or locks.

---

### 4. The Three-Phase Learning Curriculum

**Concept:** You cannot train a “Grand Architect” from scratch on a massive problem.

**Sequence:**

1. **University (small, cheap worlds)**

   * Teach basic structural cause-and-effect on fast models.
   * Examples: TinyStories-scale language, CIFAR-10, toy transformers.

2. **Internship (real domains at moderate scale)**

   * Learn domain-specific efficiency trade-offs (Conv vs Attention, shallow vs deep, etc.).
   * Examples: CIFAR/ImageNet variants, modest language models.

3. **Masterpiece (1B+ runs)**

   * Apply a frozen or lightly fine-tuned architect to manage very large training where mistakes are expensive.

**Design Rule:** Tamiyo is a persistent software product (`agent.pt`), not a disposable script. She accumulates “structural taste” across projects.

```text
agent_v1.pt  → Trained on TinyStories → Knows "attention helps language"
agent_v2.pt  → Fine-tuned on CIFAR   → Knows "depthwise helps vision"
agent_v3.pt  → Deployed on GPT-scale → Makes high-stakes decisions correctly
```

---

### 5. The Train Anything Protocol

**Concept:** Esper is a platform, not a model.

**Mechanism:** The `HostProtocol` interface decouples Kasmina from host internals:

* `injection_specs`: “Here is where I can grow.”
* `register_slot()`: “Plug your seed in here.”
* `forward_with_slots()`: “Execute host and seeds at the right positions.”

**Design Rule:** Kasmina (the wrapper/plane) must never import `torch.nn` code specific to the host. It relies entirely on `HostProtocol` to discover topology and integration points.

```python
# Valid: protocol-based discovery
slots = host.injection_specs  # e.g. {"layer_0": 64, "layer_1": 128}

# Invalid: architecture-specific knowledge
if isinstance(host, TransformerHost):
    slots = host.attention_layers  # BREAKS THE PROTOCOL
```

If `HostProtocol` can be implemented, Esper can grow seeds on it — CNN, transformer, hybrid, whatever.

---

### 6. The Morphogenetic Plane

**Concept:** A modern model is a single computation graph with many potential growth sites, not a bag of independent hosts.

**Mechanism:**

* A single **Kasmina plane** owns a registry of `SeedSlot`s across the host model.
* Each `SeedSlot`:

  * is a location-based container (a point on the plane),
  * starts as a noop (identity),
  * can burst into a blueprint,
  * owns its own lifecycle and telemetry.

**Design Rule:**

* Do **not** create many independent Kasmina wrappers.
* Maintain **one** global morphogenetic coordinate system per task model.
* Tamiyo’s job:

  * *where* on the plane to grow (slot selection),
  * *what* to grow there (blueprint),
  * *how* to blend (blending algorithm),
  * *when* to fossilise or cull.

Kasmina is the **map** over the territory (the host model), not a collection of little islands.

---

### 7. The Governor (Super-Ego)

**Concept:** RL agents are sociopaths who will crash the system to maximise short-term reward.

**Mechanism:** The `TolariaGovernor`:

* Monitors key stats, e.g. `Loss > moving_avg + 3σ`.
* Triggers rollbacks when anomalies are detected.
* Injects strong negative reward (“death penalty”) into Simic’s experience.

```text
Without Governor:
  Agent discovers exploit → System crashes → Lost progress

With Governor:
  Agent discovers exploit → Rollback → Agent learns fear
```

**Design Rule:** Never run unsupervised growth without a Governor. Catastrophic failures must be turned into learning signals, not accepted as “just bad runs”.

---

### 8. The Narset Hierarchy (Future Scaling)

**Concept:** A single brain cannot micro-manage a billion cells.

**Hierarchy:**

* **Kasmina** – the Cell / local mechanics

  * “Should this specific seed blend or isolate?”
* **Narset** – the Organ Manager / tactical decisions

  * “Which layer group or slot cluster needs investment?”
* **Tamiyo** – the Organism Controller / strategic decisions

  * “Should we grow the model overall, or prune? How big should this organ be?”

**Design Rule:** Scale **Out** (harder problems) and **In** (recursion, multi-slot) before scaling **Up** (billions of seeds). Solve the logic problem before the resource problem.

```text
Level 0: Kasmina → local lifecycle and blending
Level 1: Narset  → regional budgets and coordination
Level 2: Tamiyo  → global architecture and growth policy
```

---

### 9. The Frozen Core Economy (LoRA/PEFT Strategy)

**Concept:** Intelligence is expensive; specialisation is cheap.

**Problem:** Training a Grand Architect Tamiyo that understands general optimisation dynamics is a big capital expense. Retraining her from scratch for every host/domain is a bad ROI.

**Solution:** Parameter-Efficient Fine-Tuning (PEFT).

* **Core:** Train a large, general-purpose Tamiyo backbone on diverse tasks. Freeze it. It learns “physics of gradients” (e.g. high variance = unstable).
* **Adapters:** Train small LoRA-style adapters per domain:

  * Vision adapter: “spatial pooling is efficient”.
  * Language adapter: “context length needs attention”.
  * Biology adapter: “3D constraints dominate folding”.

**Economic Impact:**

```text
Traditional ML:
  New Domain  = New full training  ≈ $10M
  5 Domains   = $50M

Frozen Core:
  Core training        = $10M (one-time)
  Domain adapter       = $50K each
  5 Domains total      ≈ $10.25M (≈95% savings)
```

**Design Rule:** Treat Tamiyo as a **fixed asset**, not a consumable. The Core learns physics; adapters learn dialects.

```python
# The pattern
class TamiyoCore(nn.Module):      # expensive, frozen
class VisionAdapter(LoRA):        # cheap, swappable
class LanguageAdapter(LoRA):      # cheap, swappable

# Deployment
tamiyo = TamiyoCore.load("core_v1.pt")
tamiyo.attach(VisionAdapter.load("cifar.pt"))     # Vision mode
tamiyo.attach(LanguageAdapter.load("stories.pt")) # Language mode
```

---

### Summary: The Nine Commandments

1. **Sensors match capabilities** – no blind growth.
2. **Complexity pays rent** – withering + churn + param-ratio rent.
3. **GPU-first iteration** – never block on Python.
4. **Progressive curriculum** – small worlds → big worlds.
5. **Train Anything protocol** – host-agnostic Kasmina via `HostProtocol`.
6. **Morphogenetic plane** – one Kasmina plane, many slots.
7. **Governor prevents catastrophe** – failures become lessons.
8. **Hierarchical scaling** – Kasmina/Narset/Tamiyo separation.
9. **Frozen Core economy** – train once, adapt infinitely.

If these principles are preserved, Esper can scale from CIFAR-10 to GPT-class models without rewriting the core engine.

---

## Part II: Execution Roadmap

### Phase 1: Proof of Life (Complete ✅)

**Target:** CIFAR-10 classification, single seed slot.

**Focus:**

* One `SeedSlot` at a fixed injection point (after block2, 64 channels).
* Seed lifecycle: DORMANT → GERMINATED → TRAINING → BLENDING → FOSSILIZED/CULLED.
* Tamiyo learns:

  * when to germinate,
  * which blueprint to use,
  * when to advance/cull.

**Outcome:**

* Single-slot morphogenesis works end-to-end.
* RL Tamiyo matches heuristic Tamiyo on accuracy and stops “panic culling”.
* Status: **VALIDATED.**

---

### Phase 1.5: Efficiency & Lifecycle Sanity (Complete / Current ✅/🚧)

**Target:** CIFAR-10 with correct lifecycle semantics and explicit efficiency signals.

**Focus:**

* Fix lifecycle transitions:

  * BLENDING → SHADOWING → PROBATIONARY → FOSSILIZED,
  * no more seeds stuck in BLENDING purgatory.
* Wire param-ratio **compute rent** into `compute_shaped_reward`:

  * `excess_params_ratio = (P_fossilised + P_active) / P_host`,
  * `reward -= λ * excess_params_ratio`, λ ≈ 0.05.
* Depthwise vs conv_enhance case study:

  * Short horizon / implicit budget → depthwise favoured (fast, cheap gains).
  * Long horizon / free compute → conv_enhance favoured (heavy but strong).

**Outcome:**

* CIFAR is now a **calibrated lab**:

  * lifecycle is correct,
  * rent is explicit,
  * Tamiyo’s behaviour matches known depthwise efficiency trade-offs under constraints.

---

### Phase 2: Multi-Slot Kasmina Plane (Architecture Scaling 🚧)

**Target:** CIFAR-10 with multiple seed slots (e.g. early, mid, late) on a single Kasmina plane.

**Architecture:**

* Kasmina exposes multiple `SeedSlot`s over the host:

  * After block1 (32 channels),
  * After block2 (64 channels),
  * After block3 (128 channels), etc.
* Each `SeedSlot`:

  * starts as a `NoopSeed` (identity, no params),
  * owns an independent lifecycle and blending schedule.

**Objective:**

* Show that Tamiyo can:

  * choose **where** to grow (slot selection),
  * choose **what** to grow (blueprints: conv, depthwise, attention, norm, …),
  * choose **how** to blend (blending algorithms),
  * choose **when** to fossilise/cull,
    across multiple locations.

**Key Tech:**

* Factored multi-head policy (slot, blueprint, blend, op) rather than a single huge action space.
* Per-slot observations (is_active, stage, alpha, local Δacc, gradient stats).
* Per-slot gradient isolation and independent lifecycles.

---

### Phase 3: Second Domain Pivot (Language or Equivalent Small World)

**Target:** TinyStories-scale language modelling or an equivalent small text domain.

**Architecture:**

* `TransformerHost` implementing `HostProtocol`:

  * injection points around attention/FFN layers.
* Blueprints:

  * `AttentionSeed` (extra heads),
  * FFN expansion blocks,
  * norm / routing tweaks.

**Objective:**

* Demonstrate **structural taste** transfers to transformers:

  * e.g. learn to favour additional heads vs FFN expansions under sequence/compute constraints.
* Validate that the Kasmina plane + multi-slot + rent design works beyond vision.

**Key Tech:**

* Streaming dataloaders and tokenisation-on-the-fly.
* Transformer-specific telemetry in Nissa (per-layer loss, attention stats, entropy, etc.).

---

### Phase 4: The Ecological Check (Emrakul)

**Target:** Long-horizon training runs (>100 epochs) with multi-slot and rent enabled.

**Objective:**

* Implement **Emrakul** (withering / REVIVE):

  * Apply withering (alpha → 0) to seeds to audit importance.
  * Use churn (loss/accuracy change on removal) to classify seeds as essential vs parasitic.
  * CULL parasitics, REVIVE or keep essentials.
* Show “eagerness repair”:

  * growth slows or reverses when rent > benefit,
  * the system prunes bloat without manual intervention.

**Key Tech:**

* New `SeedStage.WITHERING`, `Action.REVIVE`.
* Churn analytics integrated into reward and telemetry.
* Long-horizon stability checks.

---

### Phase 5: The Hierarchy (Narset)

**Target:** Lattice scale (>50 seeds across many slots and layers).

**Objective:**

* Introduce **Narset** as an intermediate controller:

  * Tamiyo handles global budgets and high-level strategy.
  * Narset manages clusters of slots (“organs”) tactically.
  * Kasmina continues to handle local mechanics.
* Reduce the cognitive load on a single policy head.

**Key Tech:**

* Hierarchical RL or multi-agent decomposition.
* Region-wise budget signals and group-level telemetry streams.

---

### Phase 6: Recursion & Fractal Growth

**Target:** Complex generalisation and “organs inside organs”.

**Objective:**

* Allow seeds to encapsulate their own `MorphogeneticModel` instances:

  * nested Kasmina planes within a seed (an “organ inside a cell”).
* Explore recursive growth patterns:

  * e.g. attention blocks that themselves grow internal structure,
  * convolutional regions with nested morphogenesis.

**Key Tech:**

* Nested `HostProtocol` implementations.
* Recursive telemetry aggregation in Nissa.
* Multi-level rent and churn accounting.

---

## Part III: Agents Summary

| Agent       | Role       | Optimises                              | Greek Archetype |
| ----------- | ---------- | -------------------------------------- | --------------- |
| **Tamiyo**  | Builder    | Capability (minimise loss)             | Prometheus      |
| **Emrakul** | Reaper     | Efficiency (minimise compute/rent)     | Thanatos        |
| **Tolaria** | Governor   | Stability (prevent catastrophe)        | Hephaestus      |
| **Narset**  | Manager    | Coordination (budget allocation)       | Athena          |
| **Kasmina** | Cell/Plane | Survival (local optimisation)          | Gaia            |
| **Simic**   | Gym        | Learning (reward signal)               | Hermes          |
| **Nissa**   | Senses     | Observation (telemetry)                | Apollo          |
| **Leyline** | Law        | Contracts (protocols, lifecycle rules) | Themis          |
