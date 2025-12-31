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
````

---

### 2. The Rent & Churn Economy

**Concept:** Complexity is not free; it pays rent in energy/parameters.

**Mechanism:**

* **Rent:** Extra parameters and compute incur a cost proportional to their mass. Reward includes a small penalty scaled by parameter overage.

* **Sparse Reward Experiment:** We now support multiple reward modes to test Goodhart risk in reward shaping:

  * **SHAPED**: Dense counterfactual rewards at every timestep
  * **SIMPLIFIED**: Fewer components; cleaner gradients for temporal credit assignment (preferred challenger during exams)
  * **SPARSE**: Terminal-only rewards based on final accuracy
  * **MINIMAL**: Sparse + early-cull penalty for wasted compute (optional / may be folded into SIMPLIFIED depending on the current experiment set)

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
2. **Complexity pays rent** – param-ratio rent + reward shaping experiments.
3. **GPU-first iteration** – never block on Python.
4. **Progressive curriculum** – small worlds → big worlds.
5. **Train Anything protocol** – host-agnostic Kasmina via `HostProtocol`.
6. **Morphogenetic plane** – one Kasmina plane, many slots.
7. **Governor prevents catastrophe** – failures become lessons.
8. **Hierarchical scaling** – Kasmina/Narset/Tamiyo separation.
9. **Frozen Core economy** – train once, adapt infinitely.

---

## Appendix A: Roadmap Philosophy — Build Up, Build In, Build Out

Esper evolves along three complementary vectors:

1. **Build Up (New kinds of work):** Expand the *type* of tasks Esper can morphogenetically manage

   * Example: **TinyStories / transformer language modelling** as a second domain pivot

2. **Build In (New capability inside the platform):** Improve the *internal machinery* so the same work becomes more reliable, longer-horizon, and more correct

   * Example: **Tamiyo Next (Obs V3 + Policy V2)** is a textbook “Build In”

3. **Build Out (More of the same):** Increase scale factors: more slots, more seeds, longer horizons, more environments

   * Example: **100 slots**, **50 max seeds per environment**, deeper injection lattices

**Why Phase Alpha is tracked separately from Phase 3:**
Phase 3 (TinyStories) is a **Build Up** milestone: domain expansion is the critical path.
Phase Alpha (Slot Transformer architecture) is a **Build In / Build Out enabler**: we know we’ll need it for large-slot scaling, but it is not a hard prerequisite for demonstrating transformer-domain morphogenesis on small slot counts. Keeping it as a parallel track prevents “architecture perfection” from blocking the domain pivot, while still keeping the scaling work visible and planned.

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

* Simplified lifecycle: DORMANT → GERMINATED → TRAINING → BLENDING → HOLDING → FOSSILISED/PRUNED
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
* `--max-seeds` for seed budget control
* Counterfactual reward attribution per slot

**Status:** COMPLETE
**Note:** Phase 3 is gated on a Phase 2 “graduation” check (see Phase 2.5 gates below).

---

### Phase 2.5: Reward Shaping Research ✅ (Historical) + Tamiyo Next ✅ (Implemented) + Gates 🔒 (Pending)

Phase 2.5 is where we harden the “brain + sensors + reward” stack before moving to the transformer domain pivot.

This phase has three sub-tracks that share the same purpose: ensure the agent can **learn cleanly and efficiently** before we spend transformer-scale compute.

#### Phase 2.5A: Reward Shaping Research ✅

**Target:** Investigate Goodhart risk in dense reward shaping.

**Implementation:**

* Reward modes to explore reward shaping trade-offs:

  * `shaped` (dense)
  * `sparse` (terminal-only)
  * `minimal` (sparse + early-cull penalty; may be folded into simplified variants depending on the exam set)
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

#### Phase 2.5B: Tamiyo Next (Obs V3 + Policy V2) ✅

**Target:** Fix value aliasing, reduce observation redundancy, and support long-horizon scaffolding behaviour.

**Delivered Capabilities:**

* Obs V3: compact obs + blueprint embeddings (113 dims + 12 blueprint embed = 125 total network input)
* Policy V2: 512/512 feature+LSTM, 150-step horizon
* Q(s,op) critic: action-conditioned value baseline
* Differential entropy coefficients by head (protect sparse heads from collapse)

**Q-values telemetry (2025-12-31):** ✅ COMPLETE
- Op-conditioned Q(s,op) values wired end-to-end from PPO → telemetry → UI
- Sanctum HealthStatusPanel displays Q-values with variance diagnostic
- Detects if critic ignores op conditioning (Q-variance < 0.01 = critical)

**Status:** IMPLEMENTED (Q-values telemetry ✅ 2025-12-31)
**Operational status:** Waiting on Phase 2.5 gates below before proceeding to Phase 3.

---

#### Phase 2.5 Gates: “Phase 1 Final Exam — The Reward Efficiency Protocol” 🔒

**Objective:** Validate the “Rent & Churn Economy” and select the winning reward signal for Phase 3 (Transformers).

**Context:**
We have successfully trained `cifar_blind`, a model that achieves ~60% accuracy on CIFAR-10 with only +10% parameter growth using a heuristic/random strategy. This sets the **Baseline for Competence**.

For the RL agent (`Simic`) to justify its existence, it must outperform this baseline not just in accuracy, but in **structural efficiency** (getting more accuracy *per unit of growth*).

We have observed that the current 7-component `SHAPED` reward might be creating an “unlearnable landscape” due to conflicting signals (e.g., attribution vs. rent).

##### The Cohorts

| Cohort             | Description        | Reward Function                                         | Hypothesis                                                            |
| :----------------- | :----------------- | :------------------------------------------------------ | :-------------------------------------------------------------------- |
| **Control**        | `cifar_blind`      | Heuristic / Random                                      | **Baseline.** The floor for performance.                              |
| **A (Shaped)**     | Current Default    | 7-component (PBRS + Attribution + Warnings + Rent...)   | **Over-engineered.** likely to cause confusion/instability.           |
| **B (Simplified)** | **The Challenger** | 3-component (PBRS + Intervention Cost + Terminal Bonus) | **Optimal.** Cleanest gradient for temporal credit assignment.        |
| **C (Sparse)**     | Hard Mode          | Terminal Accuracy - Rent                                | **The Truth.** Hardest to learn, but theoretically perfect alignment. |

##### Configuration

* **Task:** `cifar_blind` topology (ResNet host + 2 seed slots)
* **Duration:** 100 Episodes
* **Envs:** 8–12 concurrent environments (split evenly across cohorts)
* **Seed Budget:** Max 2 active seeds

##### Success Metrics (ROI, not just profit)

1. **Accuracy ROI:** (Final Accuracy − Baseline Accuracy) / Added Parameters
2. **Decision Decisiveness:** entropy trends
3. **Lifecycle Efficiency:** ratio of `FOSSILISED` to `PRUNED` seeds

##### Pass Criteria

* **Essential:** Cohort B (Simplified) outperforms Cohort A (Shaped) in Accuracy ROI
* **Essential:** Cohort B outperforms Control (`cifar_blind`) in Final Accuracy
* **Stretch:** Cohort C (Sparse) learns anything better than random chance

##### Execution Plan

1. Implement `SIMPLIFIED` reward (per the reward A/B testing plan)
2. Configure split runs:

   * Run 1: shaped-vs-simplified (4 vs 4)
   * Run 2: simplified-vs-sparse (4 vs 4)
3. Analyse with Overwatch/TUI: watch entropy collapse and decision quality in real time
4. Verdict: select the winning reward mode as default for **Phase 3 (TinyStories)**

**Gate status:** PENDING (must pass before Phase 3 start)

---

### Phase 3: Second Domain Pivot 🔮 (Build Up)

**Target:** TinyStories-scale language modelling or equivalent small text domain.

**Architecture:**

* `TransformerHost` implementing `HostProtocol`
* Injection points around attention/FFN layers
* Blueprints: `AttentionSeed` (extra heads), FFN expansion blocks

**Objective:** Demonstrate structural taste transfers to transformers.

**Prerequisites / Gates:**

* Phase 2.5 reward exam passed (default reward selected)
* Tamiyo Next stability: end-to-end runs without catastrophic collapse; explained variance not persistently negative; sparse head entropy doesn’t collapse

**Status:** PLANNED (blocked on Phase 2.5 gates)

---

### Phase α (Alpha Track): Slot Transformer Architecture 🔮 (Build In / Build Out Enabler)

**Purpose:** Replace flat observation concatenation with a Slot Transformer Encoder that provides:

* weight sharing across slots
* variable slot counts (masking)
* learned slot–slot interactions (self-attention)

**Why it’s a separate track:**
We know we will need it for large-slot scaling (“Build Out”), but it is not strictly required to prove the Phase 3 domain pivot at small slot counts. Keeping it parallel avoids blocking Phase 3 while still progressing the scalability foundation.

**Status:** PLANNED (parallel to Phase 3; can begin whenever Phase 2.5 stabilisation bandwidth allows)

---

### Phase 4: The Immune System (Emrakul) 🔮 (Build In)

**Target:** Long-horizon training runs (>100 epochs) with multi-slot and rent enabled.

**Biological Analogy:** Emrakul acts as the immune system's phagocytes – identifying and removing parasitic or damaged components that consume resources without contributing value.

**Objective:**

* Implement withering (alpha → 0) to audit seed importance
* Use churn (loss/accuracy change on removal) to classify seeds as essential vs parasitic
* Show "eagerness repair": growth slows or reverses when rent > benefit

**Status:** PLANNED

---

### Phase 5: The Endocrine System (Narset) 🔮 (Build Out + Build In)

**Target:** Lattice scale (>50 seeds across many slots and layers).

**Biological Analogy:** Narset acts as the endocrine system – sending hormonal signals to coordinate resource allocation across organ clusters without micro-managing individual cells.

**Objective:** Introduce Narset as intermediate controller for tactical slot management, managing budgets and coordination between slot clusters.

**Explicit Build Out Goals (examples):**

* **100 slots** on a single morphogenetic plane (maskable / variable)
* **50 max seeds per environment** with rent and churn enforced
* Maintain throughput with GPU-first control flow (no Python bottleneck regression)

**Status:** PLANNED

---

### Phase 6: Recursion & Fractal Growth 🔮 (Build Up + Build In)

**Target:** Complex generalisation and "organs inside organs".

**Objective:** Allow seeds to encapsulate their own `MorphogeneticModel` instances.

**Status:** PLANNED

---

## Part III: System Components

| Component   | Biological Role  | Description                                          | Status |
| ----------- | ---------------- | ---------------------------------------------------- | ------ |
| **Kasmina** | Stem Cells       | Pluripotent slots that differentiate into blueprints | Active |
| **Leyline** | DNA/Genome       | Shared types, enums, tensor schemas (genetic code)   | Active |
| **Tamiyo**  | Brain/Cortex     | Heuristic decision logic, strategic control          | Active |
| **Tolaria** | Metabolism       | PyTorch training loops, energy conversion            | Active |
| **Simic**   | Evolution        | RL infrastructure (PPO, rewards), adaptation         | Active |
| **Nissa**   | Sensory Organs   | Telemetry hub, observability                         | Active |
| **Karn**    | Memory           | Research telemetry, analytics, historical records    | Active |
| **Emrakul** | Immune System    | Efficiency auditing, removes parasitic components    | Future |
| **Narset**  | Endocrine System | Hormonal coordination, resource allocation signals   | Future |
