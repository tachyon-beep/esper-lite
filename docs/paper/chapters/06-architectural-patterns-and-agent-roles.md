---
title: ARCHITECTURAL PATTERNS AND AGENT ROLES
source: /home/john/esper-lite/docs/paper/draft_paper.md
source_lines: 327-356
split_mode: consolidated
chapter: 6
coauthors:
  - John Morrissey
  - Codex CLI (OpenAI)
generated_by: scripts/split_paper.py
---

# Architectural Patterns and Agent Roles

## 6.1 Pattern: The Blueprint as a Reusable Skill (Germinal Module)
This is the primary pattern for capability transfer in the framework and represents the output of the Karn agent's discovery process. A Germinal Module (GM) is a complete architectural blueprint that defines a latent structure to be instantiated upon germination. It consolidates the concepts of a compressed skill and an architectural template into a single, unified entity.

GM blueprint specification:

| Element                | Description                                                                                          |
|------------------------|------------------------------------------------------------------------------------------------------|
| Layer types/topology   | Structure to be built (e.g., 2‑layer MLP, residual bottleneck, lightweight attention head)          |
| Parameter state        | Pre-trained weights, possibly compressed (e.g., quantisation or low‑rank factorisation)             |
| Expansion constraints  | Budget for maximum parameters, FLOPs, or latency impact                                              |
| Local objective (opt.) | Local loss used during seed training, separate from the global model loss                            |

Advantages when Tamiyo selects a GM from Karn’s library:

| Advantage                                 | Rationale                                                         |
|-------------------------------------------|-------------------------------------------------------------------|
| Reuse of validated solutions               | Deploy proven building blocks to common subproblems               |
| Deterministic capability integration       | Reduce training variance via fixed, versioned modules             |
| Efficient deployment                       | High compression enables low‑bandwidth environments               |
This pattern is foundational for reproducible morphogenesis: every germination event corresponds to the deployment of a specific, versioned, and pre-validated blueprint.
## 6.2 The Role: Seed Site as Interface Contract
This describes the static role of the location where a seed is placed in the host network. The seed site is not just a placeholder but an enforceable interface contract that allows the Tamiyo controller to safely interact with and modify the frozen model.
As a contract, the seed site defines:

| Contract Aspect        | Description                                                       |
|------------------------|-------------------------------------------------------------------|
| I/O shape specification| Fixed input/output shapes and types                               |
| Compatibility          | Activation and gradient discipline at the interface               |
| Monitoring hook        | Telemetry stream feeding Tamiyo                                   |
This pattern enables the static instrumentation of frozen models with known, modifiable points. These contracts serve as the essential scaffolding for safe grafting, providing the stable "sockets" into which Tamiyo can plug the various blueprints discovered by Karn.

## 6.3 The Role: Controller as Locus of Constraint Negotiation
This role, previously misattributed to the seed, is the primary function of the Tamiyo policy controller. Tamiyo acts as the central intelligence, mediating between competing architectural pressures to balance the need for new capacity against the imperative to preserve base model integrity.
This policy-driven role includes:

| Function                        | Description                                                                                   |
|---------------------------------|-----------------------------------------------------------------------------------------------|
| Monitor seed-site statistics    | Identify bottlenecks via activation/health telemetry                                          |
| Evaluate trade-offs             | Balance performance gain vs. size and latency costs                                           |
| Arbitrate between sites         | Prioritise interventions where impact is greatest                                             |
| Negotiate blueprint selection   | Choose the most suitable blueprint from Karn’s library                                       |
This function is critical in complex deployments where the controller must adapt its strategy in real-time to evolving task demands and resource constraints. By centralizing this "constraint negotiation" into the Tamiyo agent, the framework ensures that all architectural growth is deliberate, strategic, and globally informed.
