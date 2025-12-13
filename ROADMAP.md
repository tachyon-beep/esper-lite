# Esper: Architecture Constitution & Project Roadmap

**Mission:** Transition AI from *Architectural Engineering* (static design) to *Architectural Ecology* (dynamic growth).

This document is the **Constitution** of Esper. These principles have been validated in experiments and are intended to remain stable as the system scales.

---

## Part I: Architecture Constitution

### 1. The Signal-to-Noise Hypothesis

**Concept:** An RL agent is only as smart as its sensors.

**Validation:**

* When telemetry was broken or noisy, the agent learned nothing useful.
* When we fixed the telemetry – clean gradient norms, distinct per-layer stats, stage-aware metrics – the agent immediately started picking up structural patterns.

**Design Rule:** Never add a capability to the Body (Kasmina/host) without adding a corresponding sensor to Nissa (Senses). If the policy can't see it, it can't optimise it.

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

* **Rent:** Extra parameters and compute incur a cost proportional to their mass. Reward includes a small penalty scaled by parameter overage.

* **Sparse Reward Experiment:** We now support three reward modes to test Goodhart risk in reward shaping:
  - **SHAPED** (default): Dense counterfactual rewards at every timestep
  - **SPARSE**: Terminal-only rewards based on final accuracy
  - **MINIMAL**: Sparse + early-cull penalty for wasted compute

**Design Rule:** Reward and analytics must reflect **accuracy minus rent**, not accuracy alone.

---

### 3. The Inverted Control Flow

**Concept:** Python's GIL is the enemy of throughput.

**Mechanism:**

We iterate **DataLoaders first** (batches), dispatch to **environments second** (CUDA streams / devices), and use pre-allocated GPU tensors / buffers for all communication.

**Design Rule:** The Training Loop (`Tolaria`) must never block the Policy Loop (`Simic`). Communication happens via device-resident buffers, not ad-hoc Python queues or locks.

---

### 4. The Three-Phase Learning Curriculum

**Concept:** You cannot train a "Grand Architect" from scratch on a massive problem.

**Sequence:**

1. **University (small, cheap worlds)** – Teach basic structural cause-and-effect on fast models (TinyStories, CIFAR-10).
2. **Internship (real domains at moderate scale)** – Learn domain-specific efficiency trade-offs.
3. **Masterpiece (1B+ runs)** – Apply a frozen or lightly fine-tuned architect to manage very large training.

**Design Rule:** The policy is a persistent software product, not a disposable script. It accumulates "structural taste" across projects.

---

### 5. The Train Anything Protocol

**Concept:** Esper is a platform, not a model.

**Mechanism:** The `HostProtocol` interface decouples Kasmina from host internals:

* `injection_specs`: "Here is where I can grow."
* `register_slot()`: "Plug your seed in here."
* `forward_with_slots()`: "Execute host and seeds at the right positions."

**Design Rule:** Kasmina must never import `torch.nn` code specific to the host. It relies entirely on `HostProtocol` to discover topology and integration points.

---

### 6. The Morphogenetic Plane

**Concept:** A modern model is a single computation graph with many potential growth sites.

**Mechanism:**

* A single **Kasmina plane** owns a registry of `SeedSlot`s across the host model.
* Each `SeedSlot` is a location-based container that starts as a noop (identity) and can burst into a blueprint.

**Design Rule:** Maintain **one** global morphogenetic coordinate system per task model.

---

### 7. The Governor (Super-Ego)

**Concept:** RL agents are sociopaths who will crash the system to maximise short-term reward.

**Mechanism:** The `TolariaGovernor` monitors key stats, triggers rollbacks when anomalies are detected, and injects strong negative reward into the experience buffer.

**Design Rule:** Never run unsupervised growth without a Governor. Catastrophic failures must be turned into learning signals.

---

### 8. The Biological Hierarchy (Future Scaling)

**Concept:** A single brain cannot micro-manage a billion cells.

**Hierarchy:**

* **Kasmina** – Stem Cells / pluripotent mechanics (can differentiate into any blueprint)
* **Tamiyo** – Brain/Cortex / strategic control (high-level decision making)
* **Narset** – Endocrine System / hormonal coordination (resource allocation signals between organ clusters)
* **Emrakul** – Immune System / phagocytes (identifies and removes parasitic or damaged components)

**Design Rule:** Scale **Out** (harder problems) and **In** (recursion, multi-slot) before scaling **Up** (billions of seeds).

---

### 9. The Frozen Core Economy (LoRA/PEFT Strategy)

**Concept:** Intelligence is expensive; specialisation is cheap.

**Solution:** Parameter-Efficient Fine-Tuning (PEFT).

* **Core:** Train a large, general-purpose backbone on diverse tasks. Freeze it.
* **Adapters:** Train small LoRA-style adapters per domain.

**Design Rule:** Treat the policy as a **fixed asset**, not a consumable. The Core learns physics; adapters learn dialects.

---

### Summary: The Nine Commandments

1. **Sensors match capabilities** – no blind growth.
2. **Complexity pays rent** – param-ratio rent + sparse reward experiments.
3. **GPU-first iteration** – never block on Python.
4. **Progressive curriculum** – small worlds → big worlds.
5. **Train Anything protocol** – host-agnostic Kasmina via `HostProtocol`.
6. **Morphogenetic plane** – one Kasmina plane, many slots.
7. **Governor prevents catastrophe** – failures become lessons.
8. **Hierarchical scaling** – Kasmina/Narset/Tamiyo separation.
9. **Frozen Core economy** – train once, adapt infinitely.

---

## Part II: Execution Roadmap

### Phase 1: Proof of Life ✅

**Target:** CIFAR-10 classification, single seed slot.

**Outcome:** Single-slot morphogenesis works end-to-end. RL policy matches heuristic on accuracy.

**Status:** COMPLETE

---

### Phase 1.5: Efficiency & Lifecycle Sanity ✅

**Target:** CIFAR-10 with correct lifecycle semantics and explicit efficiency signals.

**Changes:**

* Simplified lifecycle: DORMANT → GERMINATED → TRAINING → BLENDING → PROBATIONARY → FOSSILIZED/CULLED
* Removed SHADOWING stage (unnecessary complexity)
* Wired param-ratio compute rent into reward
* Added counterfactual validation for seed contribution

**Status:** COMPLETE

---

### Phase 2: Multi-Slot Kasmina Plane ✅

**Target:** CIFAR-10 with multiple seed slots (early, mid, late) on a single Kasmina plane.

**Implementation:**

* Kasmina exposes multiple `SeedSlot`s over the host at different depth positions
* Factored action space: per-slot lifecycle operations via `FactoredActions`
* Per-slot observations including stage, alpha, local accuracy delta, gradient stats
* Per-slot gradient isolation and independent lifecycles

**Key Features:**

* `--slots early mid late` CLI flag to enable multiple injection points
* `--max-seeds` and `--max-seeds-per-slot` for seed budget control
* Counterfactual reward attribution per slot

**Status:** COMPLETE

---

### Phase 2.5: Reward Shaping Research ✅

**Target:** Investigate Goodhart risk in dense reward shaping.

**Implementation:**

* Three reward modes: `shaped`, `sparse`, `minimal`
* Sparse mode: terminal-only rewards (tests credit assignment)
* Minimal mode: sparse + early-cull penalty (discourages wasted compute)
* Configurable reward scaling and parameter budgets

**Usage:**
```bash
# Test sparse reward credit assignment
python -m esper.scripts.train ppo --reward-mode sparse --sparse-scale 2.0

# Test minimal mode with early-cull penalty
python -m esper.scripts.train ppo --reward-mode minimal
```

**Status:** COMPLETE

---

### Phase 3: Second Domain Pivot 🔮

**Target:** TinyStories-scale language modelling or equivalent small text domain.

**Architecture:**

* `TransformerHost` implementing `HostProtocol`
* Injection points around attention/FFN layers
* Blueprints: `AttentionSeed` (extra heads), FFN expansion blocks

**Objective:** Demonstrate structural taste transfers to transformers.

**Status:** PLANNED

---

### Phase 4: The Immune System (Emrakul) 🔮

**Target:** Long-horizon training runs (>100 epochs) with multi-slot and rent enabled.

**Biological Analogy:** Emrakul acts as the immune system's phagocytes – identifying and removing parasitic or damaged components that consume resources without contributing value.

**Objective:**

* Implement withering (alpha → 0) to audit seed importance
* Use churn (loss/accuracy change on removal) to classify seeds as essential vs parasitic
* Show "eagerness repair": growth slows or reverses when rent > benefit

**Status:** PLANNED

---

### Phase 5: The Endocrine System (Narset) 🔮

**Target:** Lattice scale (>50 seeds across many slots and layers).

**Biological Analogy:** Narset acts as the endocrine system – sending hormonal signals to coordinate resource allocation across organ clusters without micro-managing individual cells.

**Objective:** Introduce Narset as intermediate controller for tactical slot management, managing budgets and coordination between slot clusters.

**Status:** PLANNED

---

### Phase 6: Recursion & Fractal Growth 🔮

**Target:** Complex generalisation and "organs inside organs".

**Objective:** Allow seeds to encapsulate their own `MorphogeneticModel` instances.

**Status:** PLANNED

---

## Part III: System Components

| Component   | Biological Role     | Description                                        | Status |
| ----------- | ------------------- | -------------------------------------------------- | ------ |
| **Kasmina** | Stem Cells          | Pluripotent slots that differentiate into blueprints | Active |
| **Leyline** | DNA/Genome          | Shared types, enums, tensor schemas (genetic code) | Active |
| **Tamiyo**  | Brain/Cortex        | Heuristic decision logic, strategic control        | Active |
| **Tolaria** | Metabolism          | PyTorch training loops, energy conversion          | Active |
| **Simic**   | Evolution           | RL infrastructure (PPO, rewards), adaptation       | Active |
| **Nissa**   | Sensory Organs      | Telemetry hub, observability                       | Active |
| **Karn**    | Memory              | Research telemetry, analytics, historical records  | Active |
| **Emrakul** | Immune System       | Efficiency auditing, removes parasitic components  | Future |
| **Narset**  | Endocrine System    | Hormonal coordination, resource allocation signals | Future |
