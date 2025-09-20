---
title: FOUNDATIONAL PARADIGMS ENABLING LOCAL EVOLUTION
source: /home/john/esper-lite/docs/paper/draft_paper.md
source_lines: 227-256
split_mode: consolidated
chapter: 3
coauthors:
  - John Morrissey
  - Codex CLI (OpenAI)
generated_by: scripts/split_paper.py
---

# Foundational Paradigms Enabling Local Evolution
Morphogenetic architectures are made viable by the convergence of several foundational paradigms in neural network design and training methodology. This section outlines the structural, algorithmic, and procedural principles that provide the enabling substrate for seed-driven local adaptation within frozen models.
## 3.1 Modular Neural Network Design
Modularisation is a prerequisite for effective structural grafting and localised adaptation. The seed mechanism assumes that the host model is either explicitly modular—composed of clearly defined, independently evaluable components—or at least structurally decomposable through interface analysis and activation tracing.
Benefits of modular design in this context include:
• Isolation of failure points – Modules exhibiting performance degradation or bottleneck characteristics can be individually identified as targets for seed placement.
• Constrained surface area for germination – Seeds can be inserted at clearly defined interfaces (e.g., between encoder layers, projection steps, or decoder blocks), minimising disruption.
• Reduction in parameter entanglement – Modularity encourages weight segregation, making it less likely that local changes will result in emergent global drift.
Where explicit modular design is not available, implicit modularity may still emerge through dropout regularisation, sparse activation, or low‑rank decomposition. These modular affordances are critical, as they define the discrete locations for both the telemetry monitoring and the targeted intervention performed by the Tamiyo policy controller.
## 3.2 Dynamic Neural Networks
Dynamic neural network architectures allow for the creation, insertion, or reconfiguration of structural elements during training or inference. Morphogenetic architectures exploit a constrained subset of this flexibility: static base, dynamic insert. Unlike general dynamic networks where topology may evolve globally, the morphogenetic regime maintains a fixed global structure while permitting controlled local change.
Characteristics inherited from dynamic models:
• Deferred instantiation – Seeds may remain unmaterialised until needed.
• Conditional execution – A seed's internal operations are conditional on its current state in the lifecycle.
• Runtime adaptation – Structure is not fixed at compile time and may vary across instances.
However, morphogenetic systems intentionally restrict this flexibility. Dynamic growth is not used for adaptive computation or routing (e.g., Mixture‑of‑Experts), but is reserved strictly for structural evolution in response to commands from the Tamiyo policy controller.
This distinction matters operationally: morphogenetic systems must remain auditably stable in deployment. No runtime topological change is permitted after a seed has been fossilised. Dynamism is constrained to the formal, multi‑stage seed lifecycle during the training regime.
## 3.3 Continual Learning and Forgetting Constraints
The seed mechanism exists in tension with both continual learning goals and catastrophic forgetting risks. Because the base model is frozen, the system avoids the most common form of interference—destructive global weight update—but still faces challenges:
• Interface drift – The functional boundary between the frozen model and an active seed may shift as the seed trains, altering outputs in uncontrolled ways.
• Gradient leakage – Improper backpropagation isolation may cause unintended parameter updates or optimisation feedback loops.
• Redundant capacity masking – A seed may learn to replicate behaviours already embedded in the frozen base, offering no real extension of capability.
To mitigate these risks, morphogenetic architectures apply constraints and mechanisms such as:
• Strict gradient masking for all frozen parameters during seed training.
• A robust, multi-stage validation lifecycle, including BLENDING, SHADOWING, and PROBATIONARY states, to ensure any new module is integrated smoothly and verified against systemic regressions before being made permanent.
• A co-evolutionary discovery process, where the Karn agent is explicitly rewarded for finding functionally novel blueprints, and the Tamiyo controller can leverage auxiliary losses to select modules that add genuinely new capabilities rather than replicating existing ones.
• Monitoring of output drift at seed boundaries through metrics like cosine similarity to detect and flag unacceptable deviations.
Methods from continual learning, such as Elastic Weight Consolidation (EWC), may be repurposed to relax freezing in select cases, allowing slight upstream adaptation under penalty. However, this extends beyond the seed‑only regime and introduces auditability complexity.
In the strict morphogenetic case, forgetting is avoided by design: the base model does not change. The remaining challenge is ensuring that new growth is genuinely additive and does not unintentionally overwrite, mask, or disrupt existing functions—a challenge the agent‑based framework is explicitly designed to address.
