# Esper: Project Roadmap

**Mission:** To transition Artificial Intelligence from *Architectural Engineering* (static design) to *Architectural Ecology* (dynamic growth).

---

## 🏛️ Part I: The Design Constitution

*This section defines the immutable laws of the Esper architecture. Violate them only with explicit justification and team consensus.*

### 1. The Signal-to-Noise Hypothesis

**Concept:** An RL agent is only as smart as its sensors.
**Design Rule:** Never add a feature to the Body (`Kasmina`) without adding a corresponding sensor to Senses (`Nissa`). If Tamiyo can't see it, she can't optimize it.

### 2. The Rent & Churn Economy

**Concept:** Complexity is not free; it pays rent in energy.
**Mechanism:** Withering (Alpha decay $1.0 \to 0.0$) creates **Churn** (Loss Variance).
**Design Rule:** Emrakul never deletes; she only pressure-tests. Deletion is the side effect of a failed test.

### 3. The Inverted Control Flow

**Concept:** Python's GIL is the enemy of throughput.
**Mechanism:** Iterate DataLoaders first, dispatch to Environments second.
**Design Rule:** The Training Loop (`Tolaria`) must never block the Policy Loop (`Simic`). Communicate via pre-allocated GPU tensors.

### 4. The "Train Anything" Protocol

**Concept:** Esper is a platform, not a model wrapper.
**Mechanism:** The `HostProtocol` Interface (`injection_specs`, `register_slot`).
**Design Rule:** `Kasmina` must never import host-specific code. It relies entirely on the Protocol to discover topology.

### 5. The Governor (Super-Ego)

**Concept:** RL agents are sociopaths who will crash the system for short-term gain.
**Mechanism:** `TolariaGovernor` triggers statistical anomaly detection ($Loss > \mu + 3\sigma$).
**Design Rule:** Never run Unsupervised Growth without a Governor. Convert crashes into negative-reward learning signals.

### 6. The Three-Phase Curriculum

**Concept:** You cannot train a "Grand Architect" from scratch on a massive problem.
**Sequence:** University (TinyStories) $\to$ Internship (Vision) $\to$ Masterpiece (1B+).
**Design Rule:** Tamiyo is a persistent software product (`agent.pt`), not a disposable script.

### 7. The Narset Hierarchy

**Concept:** A single brain cannot micro-manage a billion cells.
**Hierarchy:** Kasmina (Cell) $\to$ Narset (Organ/Tactical) $\to$ Tamiyo (Organism/Strategic).
**Design Rule:** Scale "Out" (Harder Problems) and "In" (Recursion) before scaling "Up" (Billions of seeds).

### 8. The Frozen Core Economy

**Concept:** Intelligence is expensive; Specialization is cheap.
**Mechanism:** Train a frozen Tamiyo Core on general dynamics, use LoRA Adapters for domain specifics.
**Design Rule:** Treat the Agent as a Fixed Asset.

---

## 🗺️ Part II: Execution Roadmap

### Phase 1: The Proof of Life (Complete ✅)

* **Target:** CIFAR-10 Classification.
* **Validation:** Tamiyo learned to stop "panic culling" and discovered that Depthwise Convolutions > Standard Convolutions for low-res efficiency.
* **Status:** **VALIDATED.**

### Phase 2: The Language Pivot (Current 🚧)

* **Target:** TinyStories (Language Modeling).
* **Architecture:** `TransformerHost` (NanoGPT style) + `AttentionSeed`.
* **Objective:** Prove "Structural Taste" in a new domain. Tamiyo must learn to prioritize Attention Heads over Convolutions.
* **Key Tech:** Streaming DataLoaders, Tokenization-on-the-fly.

### Phase 3: The Ecological Check

* **Target:** Sustained training runs (>100 epochs).
* **Objective:** Implement **Emrakul**. Demonstrate "Eagerness Repair" where the system autonomously prunes bloat.
* **Key Tech:** `SeedStage.WITHERING`, `Action.REVIVE`.

### Phase 4: The Hierarchy (Narset)

* **Target:** Lattice scale (>50 seeds).
* **Objective:** Decentralize decision making to handle larger graphs.

### Phase 5: The Recursion (Fractal Growth)

* **Target:** Complex Generalization.
* **Objective:** Allow seeds to contain `MorphogeneticModel` instances (Organs).

---

## 🎭 The Cast (Agents Summary)

| Agent | Role | Optimizes | Greek Archetype |
|-------|------|-----------|-----------------|
| **Tamiyo** | Builder | Capability (minimize loss) | Prometheus |
| **Emrakul** | Reaper | Efficiency (minimize compute) | Thanatos |
| **Tolaria** | Governor | Stability (prevent catastrophe) | Hephaestus |
| **Narset** | Manager | Coordination (budget allocation) | Athena |
| **Kasmina** | Cell | Survival (local optimization) | Gaia |
| **Simic** | Gym | Learning (reward signal) | Hermes |
| **Nissa** | Senses | Observation (telemetry) | Apollo |
| **Leyline** | Law | Contracts (protocols) | Themis |
