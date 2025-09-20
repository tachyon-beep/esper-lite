---
title: Preface
source: /home/john/esper-lite/docs/paper/draft_paper.md
source_lines: 1-141
split_mode: consolidated
coauthors:
  - John Morrissey
  - Codex CLI (OpenAI)
generated_by: scripts/split_paper.py
---

# Preface

MORPHOGENETIC ARCHITECTURES
A FORMAL FRAMEWORK FOR LOCALIZED STRUCTURAL EVOLUTION IN FROZEN NEURAL NETWORKS

Author: John Morrissey
Co-author: Codex CLI (OpenAI)
Date: 20 September 2025
Status: Conceptual Draft Only, Results are placeholders and should not be relied upon.
Version 3.0RC1

## Contents
- [Introduction](01-introduction.md)
- [Conceptual Foundations](02-conceptual-foundations.md)
- [Foundational Paradigms Enabling Local Evolution](03-foundational-paradigms-enabling-local-evolution.md)
- [Techniques for Grafting and Precise Editing](04-techniques-for-grafting-and-precise-editing.md)
- [Failure Handling and Risk Containment](05-failure-handling-and-risk-containment.md)
- [Architectural Patterns and Agent Roles](06-architectural-patterns-and-agent-roles.md)
- [Prototype Implementation and Micro-Demonstration](07-prototype-implementation-and-micro-demonstration.md)
- [Controller Training: The Tamiyo Curriculum](08-controller-training-the-tamiyo-curriculum.md)
- [Tables and Figures](09-tables-and-figures.md)
- [Evaluation Criteria and Safety Constraints](10-evaluation-criteria-and-safety-constraints.md)
- [Future Work and Research Directions](11-future-work-and-research-directions.md)
- [Deployment Pathway and Strategic Vision](12-deployment-pathway-and-strategic-vision.md)
- [Citations](13-citations.md)
- Appendices
  - [Appendix A: Prototype Code – Full-Fidelity Managed Germination](appendix-a-prototype-code-full-fidelity-managed-germination.md)
  - [Appendix B: Diagnostic Tooling and Control](appendix-b-diagnostic-tooling-and-control.md)
  - [Appendix C: Bibliography / Reading List](appendix-c-bibliography-reading-list.md)

## Abstract
This document outlines the formal groundwork and technical scaffolding for a class of neural architectures capable of localised, seed-driven structural evolution within frozen host networks. The approach—referred to as morphogenetic architecture—enables the introduction of trainable components that can independently develop new capabilities in response to local failure signals or performance deficits.
The central concept is that of a seed: a compact, parameter-initialised tensor or module with the capacity to 'germinate'—that is, instantiate and integrate new trainable subnetworks into a frozen model context. This strategy allows targeted increases in representational or task-specific capacity without retraining the global model.
Seed-driven structural growth is proposed as a minimally invasive method to evolve capacity-constrained models in safety-critical, memory-limited, or field-deployed conditions. This is of particular interest for low-parameter systems (<10M), edge hardware applications, and environments where full-model retraining is not feasible.
This document serves as a reference design for an MVP architecture implementing these principles and includes technical background, prior art survey, architectural constraints, training constraints, evaluation strategies, and a prototype demonstration.
## Writing Conventions
This document uses the following terminological and structural conventions:

| Style            | Usage                                                                                 |
|------------------|---------------------------------------------------------------------------------------|
| Monospaced       | Code, tensor names, explicit symbolic interfaces                                      |
| Boldface         | Terminology defined in the Introduction or of critical importance in context          |
| Italics          | Emphasis or contrast within definitions                                               |
| Capitalised Terms| Named constructs such as Seed, Germination, Germinal Module (GM) as defined types     |
| “Quoted Terms”   | Metaphoric or analogical references not meant to imply literal biological function    |

## Frequently Used Definitions

| Term                 | Definition                                                                                                                       |
|----------------------|----------------------------------------------------------------------------------------------------------------------------------|
| Seed                 | A compact, initialised tensor or module with latent capacity to instantiate trainable substructures when triggered.              |
| Germination          | The process by which a seed instantiates one or more parameterised submodules in response to local learning signals.             |
| Frozen Base          | A host network or model which remains static during seed training or evaluation.                                                 |
| Germinal Module (GM) | A pre-trained, compressed module or sub-network representing reusable, transferrable functionality.                              |
| Interface Contract   | A defined set of shape, gradient, and activation constraints at a given connection point in the model architecture.              |
| Structural Grafting  | The act of introducing new modules into an existing architecture, typically while preserving legacy behaviour.                   |
| Morphogenetic Policy | A rule or control procedure that determines when, where, and how seeds are permitted to germinate.                               |
| Tamiyo               | Reinforcement-learned policy network governing germination triggers, locations, and the configuration of germinating seeds.      |
| Kasmina              | Lightweight heuristics package embedded in each seed, handling local training, blending, and validation.                         |
| Karn                 | Evolutionary seed architect that proposes new blueprint variants in response to Tamiyo’s feedback and system telemetry.          |

## Document Version and Metadata
Document Title: Morphogenetic Architectures: Localised Evolution in Neural Networks – Seed-Bursting MVP and Reference Design
Version: 2.0a (Draft)
Date: 14 June 2025
Author(s): John Morrissey, Gemini AI, DeepSeek AI
Status: Draft for internal review
Review Cycle: First-stage pre-submission.
Audience: Research collaborators, system engineers, model designers involved in modular learning architecture design

Document Type: Technical Design Specification and Reference Overview

## Document Scope
This document defines and clarifies the following:
• The theoretical and operational rationale for seed-based localised evolution in neural networks.
• Architectural and training constraints required to support seed-driven adaptation without catastrophic interference.
• Techniques for modular insertion, interface preservation, and controlled learning at the graft site.
• A prototype implementation architecture and minimal working demo under constrained compute.
• Evaluation methodologies for performance, safety, and reproducibility.
The intended use case is systems where global retraining is constrained, or where long-lived model deployments require modular augmentation without centralisation or external synchronisation. This includes embedded AI, autonomous systems, and constrained hardware inference contexts.
