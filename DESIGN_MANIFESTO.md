# Esper Design Manifesto

**Version:** 1.0
**Purpose:** Foundational architectural principles that govern all implementation decisions.

This document captures the "Constitution" of Esper. These principles were validated through experimentation and must be preserved as the system scales.

---

## 1. The Signal-to-Noise Hypothesis

**Concept:** An RL agent is only as smart as its sensors.

**Validation:** When telemetry was broken/noisy, Tamiyo learned nothing. When telemetry was fixed (clean gradient norms, distinct layer stats), Tamiyo immediately discovered the efficiency of Depthwise Convolutions over Attention for spatial tasks.

**Design Rule:** Never add a feature to the model (Body) without adding a corresponding sensor to Nissa (Senses). If Tamiyo can't see it, she can't optimize it.

```
Body Feature  →  Sensor Required
─────────────────────────────────
New seed type →  Gradient telemetry for that seed
New stage     →  Stage-specific metrics
New host      →  Injection point visibility
```

---

## 2. The Rent & Churn Economy

**Concept:** Complexity is not free; it pays rent in energy.

**Mechanism:** Withering (Alpha decay 1.0 → 0.0) is the audit mechanism.

**The Signal:** Churn (Loss Variance) is the currency.
- If Withering causes zero Churn → seed was useless → **CULL**
- If Withering causes high Churn → seed was structural → **REVIVE**

**Design Rule:** Emrakul never deletes; she only pressure-tests. Deletion is the side effect of a failed test.

```
Emrakul's Question: "What happens if I slowly remove this?"
Answer A: "Nothing"     → It was parasitic  → Kill it
Answer B: "System dies" → It was essential  → Spare it
```

---

## 3. The Inverted Control Flow

**Concept:** Python's GIL is the enemy of throughput.

**Mechanism:** Iterate DataLoaders first (Batches), then dispatch to Environments second (CUDA Streams).

**Validation:** Training logs show perfect interleaving of env0, env1, etc., confirming multiple GPUs saturated simultaneously without Python thread blocking.

**Design Rule:** The Training Loop (Tolaria) must never block the Policy Loop (Simic). They communicate via pre-allocated GPU tensors.

```
Traditional:  for env in envs: env.step()      # Sequential, GIL-bound
Esper:        for batch in data: dispatch(envs) # Parallel, GPU-native
```

---

## 4. The Three-Phase Learning Curriculum

**Concept:** You cannot train a Grand Architect from scratch on a massive problem.

**Sequence:**
1. **University (TinyStories):** Learn basic structural cause-and-effect on cheap, fast models
2. **Internship (CIFAR/ImageNet):** Learn domain-specific efficiency (Conv vs Attention)
3. **Masterpiece (1B+ Run):** Apply frozen/fine-tuned agent to manage massive training where mistakes are expensive

**Design Rule:** Tamiyo is a persistent software product (`agent.pt`), not a disposable script. She accumulates wisdom across projects.

```
agent_v1.pt  →  Trained on TinyStories  →  Knows "attention helps language"
agent_v2.pt  →  Fine-tuned on CIFAR    →  Knows "depthwise helps vision"
agent_v3.pt  →  Deployed on GPT-scale  →  Makes $1M decisions correctly
```

---

## 5. The Train Anything Protocol

**Concept:** Esper is a platform, not a model.

**Mechanism:** The HostProtocol Interface.
- `injection_specs`: "Here is where I can grow"
- `register_slot()`: "Plug your seed in here"
- `forward()`: "Execute with slots at correct positions"

**Design Rule:** Kasmina (The Wrapper) must never import torch.nn code specific to the Host. It relies entirely on the Protocol to discover topology.

```python
# Valid: Protocol-based discovery
slots = host.injection_specs  # {"layer_0": 64, "layer_1": 128}

# Invalid: Architecture-specific knowledge
if isinstance(host, TransformerHost):
    slots = host.attention_layers  # BREAKS THE PROTOCOL
```

---

## 6. The Governor (Super-Ego)

**Concept:** RL agents are sociopaths who will crash the system to maximize short-term reward.

**Mechanism:** The TolariaGovernor.
- **Panic Trigger:** Statistical anomaly detection (Loss > Moving Avg + 3σ)
- **Reaction:** Hard Rollback to RAM snapshot
- **Education:** Inject synthetic Death Penalty (-100 Reward) into Simic's memory

**Design Rule:** Never run Unsupervised Growth without a Governor. It converts catastrophic failures (crashes) into learning opportunities.

```
Without Governor:  Agent discovers exploit → System crashes → Lost progress
With Governor:     Agent discovers exploit → Rollback → Agent learns fear
```

---

## 7. The Narset Hierarchy (Future Scaling)

**Concept:** A single brain cannot micro-manage a billion cells.

**Hierarchy:**
- **Kasmina:** The Cell (Local survival, gradient isolation)
- **Narset:** The Organ Manager (Tactical decisions, local budget)
- **Tamiyo:** The Organism Controller (Strategic decisions, global allocation)

**Design Rule:** Scale "Out" (Harder Problems) and "In" (Recursion) before scaling "Up" (Billions of seeds). Solve the logic problem before the resource problem.

```
Level 0: Kasmina  →  "Should this specific seed blend or isolate?"
Level 1: Narset   →  "Which layer group needs investment?"
Level 2: Tamiyo   →  "Should we grow the model or prune it?"
```

---

## 8. The Frozen Core Economy (LoRA/PEFT Strategy)

**Concept:** Intelligence is expensive; Specialization is cheap.

**The Problem:** Training a Grand Architect Tamiyo to master general gradient dynamics (the "Physics of Learning") is a massive capital investment. Retraining her for every new architecture or domain destroys that ROI.

**The Solution:** Parameter-Efficient Fine-Tuning (PEFT).
- **The Core:** Train a large, general-purpose Tamiyo backbone on diverse tasks. This backbone is FROZEN. It knows universal truths like "High Gradient Variance = Instability."
- **The Adapters:** Train lightweight LoRA Adapters (Rank-8/16) for specific domains:
  - *Vision Adapter:* Knows "Spatial pooling is efficient"
  - *Language Adapter:* Knows "Context length requires Attention"
  - *Biology Adapter:* Knows "Folding requires 3D constraints"

**Economic Impact:** Amortize the cost of the "University" phase across infinite "Internships." Never depreciate the core IP; only accrete new capabilities.

```
Traditional ML:
  New Domain = New Training = $10M
  5 Domains = $50M

Frozen Core:
  Core Training = $10M (one-time)
  Adapter = $50K each
  5 Domains = $10.25M (95% savings)
```

**Design Rule:** Treat Tamiyo as a Fixed Asset, not a Consumable. The Core learns physics; Adapters learn dialects.

```python
# The pattern
class TamiyoCore(nn.Module):      # $10M investment, FROZEN
class VisionAdapter(LoRA):        # $50K, swappable
class LanguageAdapter(LoRA):      # $50K, swappable

# Deployment
tamiyo = TamiyoCore.load("core_v1.pt")
tamiyo.attach(VisionAdapter.load("cifar.pt"))     # Vision mode
tamiyo.attach(LanguageAdapter.load("stories.pt")) # Language mode
```

---

## Summary: The Eight Commandments

1. **Sensors match capabilities** - No blind growth
2. **Complexity pays rent** - Withering audits value
3. **GPU-first iteration** - Never block on Python
4. **Progressive curriculum** - Simple → Complex
5. **Protocol over implementation** - Train anything
6. **Governor prevents catastrophe** - Failures become lessons
7. **Hierarchical scaling** - Logic before resources
8. **Frozen Core economy** - Train once, adapt infinitely

If these principles are preserved, Esper can scale from CIFAR-10 to GPT-6 without rewriting the engine.

---

## Agents Summary

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

---

*This document is the architectural constitution. Violate it only with explicit justification and team consensus.*
