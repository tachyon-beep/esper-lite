---
title: TECHNIQUES FOR GRAFTING AND PRECISE EDITING
split_mode: consolidated
chapter: 4
coauthors:
  - John Morrissey
  - Codex CLI (OpenAI)
generated_by: scripts/split_paper.py
---

# Techniques for Grafting and Precise Editing
Morphogenetic architectures require structural and procedural mechanisms that allow new modules—introduced through germination—to be inserted into an otherwise static model without compromising stability, gradient discipline, or functional continuity. The choice of which mechanism to use is determined by the architectural blueprint (Germinal Module) selected by the Tamiyo controller for a given germination event. This section outlines the primary techniques that enable this process.
These techniques are not mutually exclusive. A sophisticated blueprint discovered by Karn might specify a hybrid approach: e.g., a structural graft via neural surgery that is initialised as a near-identity adapter, whose final weights are loaded from a pre-trained state.
## 4.1 Neural Network Surgery
Neural network surgery refers to the manual or automated insertion, modification, or pruning of components within an existing network topology, typically without altering the surrounding architecture. Fine‑grained surgical techniques aim to introduce desired changes while minimising side effects on the model's existing capabilities.
In the morphogenetic setting, surgery is initiated by a seed module at the command of the Tamiyo policy controller and must preserve the following invariants:
• Input/output shape consistency: All inserted components must preserve tensor dimensions and types expected by the surrounding architecture.
• Functional continuity: The initial behaviour of the grafted component should approximate an identity or pass‑through function to avoid performance collapse. This is a critical principle of minimal impact initialisation.
• Gradient isolation: During training of the grafted component, gradients must not propagate into the frozen base model.
Common surgery patterns for germination include:
• Residual Grafting: Inserting a residual block in parallel with an existing connection, initialised such that the new path returns zero or near‑zero output.
• Intermediate Injection: Splitting a linear or convolutional layer mid‑flow to insert an additional transformation, typically with identity initialisation.
• Layer Substitution: Replacing an existing module with a seed‑wrapped variant, where the original function is recoverable via parameter configuration.
The Tamiyo controller acts as the high‑level orchestrator for this process, while the seed module serves as the local execution mechanism, ensuring that insertion is minimal, reversible where possible, and auditable. For a visual comparison of techniques, see Figures (09-tables-and-figures.md).
## 4.2 Adapter Layers
Adapter layers are lightweight, often bottlenecked modules inserted between existing layers to introduce trainable capacity with minimal overhead. Originally popularised for parameter‑efficient fine‑tuning in transformer models (Houlsby et al., 2019), adapters provide a natural grafting mechanism for morphogenetic growth. In our framework, an adapter can be considered a minimal form of a Germinal Module—a simple but effective blueprint that Karn can discover and Tamiyo can deploy for low‑cost capacity increases.
Key characteristics relevant to seed-driven architectures:
• Shape preservation: Adapters are designed to preserve tensor shape between layers.
• Near‑identity initialisation: Adapter weights are often initialised to approximate an identity function, minimising disruption upon insertion.
• Low parameter count: Suitable for seed‑scope training budgets and hardware‑constrained environments.
In a morphogenetic context, adapters can be used as the structural basis for a germinated module, which may later evolve into a more complex sub‑network. Their internal activations can also be monitored by the seed to provide telemetry to Tamiyo, indicating a need for further growth.
## 4.3 Germinal Module (GM) Injection
A Germinal Module (GM) is the core unit of knowledge transfer in the morphogenetic framework. It is a pre‑trained, validated, and often compressed architectural blueprint discovered by the Karn agent in its competitive crucible environment. By winning head‑to‑head evaluations for performance and efficiency, GMs represent a library of proven solutions to common sub‑problems (cf. Rusu et al., 2016), which Tamiyo can select and deploy into the main network.
In this context, upon receiving a command from Tamiyo, a seed will:
• Instantiate the structure defined by the GM blueprint (e.g., via network surgery or as an adapter).
• Load the pre‑trained parameters from the specified Germinal Module (GM).
• Resume fine‑tuning or adaptation locally according to its internal lifecycle rules, if allowed.
The integration of a Germinal Module (GM) must respect the same constraints as other seed‑based grafts: structural compatibility, gradient isolation, and non‑disruptive insertion. This allows morphogenetic systems to blend structural growth with prior knowledge reuse—achieving both adaptability and efficiency. The primary advantage of the GM approach is the ability to encapsulate validated functionality in a highly compressed format, making it ideal for low‑bandwidth or storage‑constrained environments (Du et al., 2025).
For the CIFAR‑10 experiment outlined in Section 7.3, a Germinal Module was created by training a standalone residual MLP and then applying aggressive quantisation and pruning (cf. typical compression pipelines). Results:

| Module Version                          | Trainable Parameters | Size on Disk | CIFAR‑10.1 Accuracy Δ |
|-----------------------------------------|----------------------|--------------|-----------------------|
| From‑Scratch Seed (FP32)                | 50k                  | 200 KB       | +0.90%                |
| Germinal Module (INT8, 4:1 pruned)      | 50k (effective)      | 15 KB        | +0.75%                |

Through quantisation and pruning, the GM’s storage footprint was reduced by over 90%, while retaining over 80% of the performance gain of the uncompressed, from‑scratch module. This demonstrates a favourable trade‑off, validating GMs as a core technique for efficient, targeted capability transfer.
## 4.4 Comparative Summary

| Technique                    | Insertion Type              | Initial Behaviour            | Parameter Origin                 | Gradient Scope          | Best Use Case                                      |
|-----------------------------|-----------------------------|------------------------------|----------------------------------|-------------------------|----------------------------------------------------|
| Neural Surgery               | Structural (layer/branch)   | Near‑identity or no‑op       | From scratch or copied           | Seed‑local only         | Custom architectures; structural flexibility       |
| Adapter Layer                | Bottleneck insert           | Identity approximation       | From scratch                     | Seed‑local only         | Transformer/MLP backbones; low parameter growth    |
| Germinal Module (GM) Injection | Pre‑trained module       | Task‑optimised               | Discovered & validated by Karn   | Load‑and‑freeze or fine‑tune | Task reuse; constrained retraining environments |
