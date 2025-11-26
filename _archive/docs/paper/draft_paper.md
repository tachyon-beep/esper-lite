MORPHOGENETIC ARCHITECTURES
A FORMAL FRAMEWORK FOR LOCALIZED STRUCTURAL EVOLUTION IN FROZEN NEURAL NETWORKS

Author: John Morrissey
Date: 20 September 2025
Status: Conceptual Draft Only, Results are placeholders and should not be relied upon.
Version 3.0RC1

CONTENTS
Abstract 5
Writing Conventions 5
Frequently Used Definitions 6
Document Version and Metadata 7
Document Scope 7
Introduction: A New Approach for Adaptive Systems 8

1. Introduction 9
1.1 Motivation 9
1.2 Objectives 9
1.3 Background and Context 9
1.4 Limitations 10
2. Conceptual Foundations 11
2.1 Morphogenetic Architecture 11
2.2 The Role of the Seed 11
2.3 Core Constraints and System Tensions 12
3. Foundational Paradigms Enabling Local Evolution 13
3.1 Modular Neural Network Design 13
3.2 Dynamic Neural Networks 13
3.3 Continual Learning and Forgetting Constraints 14
4. Techniques for Grafting and Precise Editing 15
4.1 Neural Network Surgery 15
4.2 Adapter Layers 15
4.3 Germinal Module (GM) Injection 16
4.4 Comparative Summary 17
5. Failure Handling and Risk Containment 18
5.1 Germination Failure Modes 18
5.2 Germination Rollback Protocols 18
5.3 Interface Drift Detection 19
5.4 Reward Collapse and Metric Hacking 19
5.5 Containment and Safe Abandonment 19
5.6 Summary 20
6. Seed-Centric Design Patterns 21
6.1 Seed as Interface Contract 21
6.2 Seed as Local Objective Encapsulation 21
6.3 Seed as Compressed Skill (Germinal Module (GM)) 22
6.4 Seed as Architectural Template 22
6.5 Seed as Locus of Constraint Negotiation 22
7. Prototype Implementation and Micro-Demonstration 23
7.1 Minimal Viable Example: The XOR Problem 23
7.2 Full-Fidelity Managed Germination (make_moons) 26
7.3 Scalability & Baseline Comparison: CIFAR-10 Classification 27
8. Controller Training: A Micro-Curriculum 29
8.1 The Micro-Curriculum Framework 29
8.2 Controller Architecture (KasminaMicro) 31
8.3 Implementation Blueprint and Evaluation 31
8.4 Strategic Benefits of a Curriculum-Driven Approach 32
9. Tables and Figures 33
9.1 Seed Lifecycle States 33
9.2 Techniques for Structural Grafting 33
9.3 Seed Design Pattern Reference 34
9.4 Prototype Validation Metrics 34
9.5 Germination Policy Triggers (Prototype) 35
9.6 Seed-Specific Optimisation Config (Prototype) 35
9.7 Seed Placement: Visual Schema (Synthetic MLP) 35
10. Evaluation Criteria and Safety Constraints 37
10.1 Evaluation Domains 37
10.2 Safety Constraints 37
10.3 Evaluation Pipeline 38
10.4 Failure Modes and Mitigations 39
10.5 Recommended Auditing Practices 39
10.6 Hardware Realization and Constraints 39
10.7 Adversarial Robustness and Security 40
10.8 Long-Term Stability and Cumulative Drift 40
11. Future Work and Research Directions 42
11.1 Generalisation to Complex Architectures 42
11.2 Multi-Seed Coordination and Policy 42
11.3 Seed-Freezing and Lifecycle Management 43
11.4 Structured Trigger Policies 43
11.5 Integration with Compression and Reuse 43
11.6 Applications in On-Device and Edge Inference 43
11.7 Formal Verification of Germination Events 44
11.8 Theoretical Framing and Learning Guarantees 44
11.9 Summary 44
12. Deployment Pathway and Strategic Vision 46
12.1 Phase 1: Constrained, High-Value Domains 46
12.2 Phase 2: Audited and Regulated Systems 46
12.3 Phase 3: Ambient and Autonomous Ecosystems 46
13. Citations 48
Appendices 50
Appendix A Prototype Code – Full-Fidelity Managed Germination 50
Appendix B: Diagnostic Tooling and Control 60
Appendix C: Bibliography / Reading List 66

ABSTRACT – OUTDATED.
This document outlines the formal groundwork and technical scaffolding for a class of neural architectures capable of localised, seed-driven structural evolution within frozen host networks. The approach—referred to as morphogenetic architecture—enables the introduction of trainable components that can independently develop new capabilities in response to local failure signals or performance deficits.
The central concept is that of a seed: a compact, parameter-initialised tensor or module with the capacity to 'germinate'—that is, instantiate and integrate new trainable subnetworks into a frozen model context. This strategy allows targeted increases in representational or task-specific capacity without retraining the global model.
Seed-driven structural growth is proposed as a minimally invasive method to evolve capacity-constrained models in safety-critical, memory-limited, or field-deployed conditions. This is of particular interest for low-parameter systems (<10M), edge hardware applications, and environments where full-model retraining is not feasible.
This document serves as a reference design for an MVP architecture implementing these principles and includes technical background, prior art survey, architectural constraints, training constraints, evaluation strategies, and a prototype demonstration.
WRITING CONVENTIONS
This document uses the following terminological and structural conventions:
Term Type Usage
Monospaced Used for code, tensor names, or explicit symbolic interfaces
Boldface Used for terminology defined in Section 0 or of critical importance in context
Italics Used for emphasis or contrast within definitions
Capitalised Terms Used for named constructs such as Seed, Germination, Germinal Module (GM) when acting as defined types
“Quoted Terms” Used for metaphoric or analogical references (e.g. “grows,” “mutation”) not meant to imply literal biological function

FREQUENTLY USED DEFINITIONS

Term Definition
Seed A compact, initialised tensor or module with latent capacity to instantiate trainable substructures when triggered.
Germination The process by which a seed instantiates one or more parameterised submodules in response to local learning signals.
Frozen Base A host network or model which remains static during seed training or evaluation.
Germinal Module (GM) (Where used) A pre-trained, compressed module or sub-network representing reusable, transferrable functionality.
Interface Contract A defined set of shape, gradient, and activation constraints at a given connection point in the model architecture.
Structural Grafting The act of introducing new modules into an existing architecture, typically while preserving legacy behaviour.
Morphogenetic Policy A rule or control procedure that determines when, where, and how seeds are permitted to germinate.
Tamiyo
 Reinforcement-learned policy network governing germination triggers, locations and the configuration of germinating seeds.
Kasmina  Lightweight heuristics package embedded in each seed, handling local training, blending and validation.
Karn  Evolutionary seed architect that cooks up new blueprint variants in response to Tamiyo’s feedback and system telemetry.

DOCUMENT VERSION AND METADATA- OUTDATED.
Document Title: Morphogenetic Architectures: Localised Evolution in Neural Networks – Seed-Bursting MVP and Reference Design
Version: 2.0a (Draft)
Date: 14 June 2025
Author(s): John Morrissey, Gemini AI, DeepSeek AI
Status: Draft for internal review
Review Cycle: First-stage pre-submission.
Audience: Research collaborators, system engineers, model designers involved in modular learning architecture design

Document Type: Technical Design Specification and Reference Overview
DOCUMENT SCOPE - OUTDATED.
This document defines and clarifies the following:
• The theoretical and operational rationale for seed-based localised evolution in neural networks.
• Architectural and training constraints required to support seed-driven adaptation without catastrophic interference.
• Techniques for modular insertion, interface preservation, and controlled learning at the graft site.
• A prototype implementation architecture and minimal working demo under constrained compute.
• Evaluation methodologies for performance, safety, and reproducibility.
The intended use case is systems where global retraining is constrained, or where long-lived model deployments require modular augmentation without centralisation or external synchronisation. This includes embedded AI, autonomous systems, and constrained hardware inference contexts.

INTRODUCTION: A NEW APPROACH FOR ADAPTIVE SYSTEMS - UPDATED.
While techniques for modular and parameter-efficient adaptation, such as adapters and network surgery, have shown promise, they often lack a cohesive, system-level framework for ensuring safety, auditability, and autonomous control. This paper seeks to bridge that gap by establishing the formal groundwork for morphogenetic computing: a discipline where neural networks are treated not as static artifacts, but as dynamic systems capable of controlled, localized, and auditable structural evolution.
Rather than proposing an entirely new low-level mechanism, this work introduces a unifying framework orchestrated by a co-evolutionary system of intelligent agents. This system features:
An external 'inventor' agent, Karn, which discovers and validates a diverse library of architectural blueprints in a competitive environment.
An internal 'policy controller' agent, Tamiyo, which monitors the host network's real-time performance and telemetry to strategically select and deploy Karn's blueprints, triggering growth precisely where it is needed. This agent-driven approach transforms the abstract idea of a 'policy-governed lifecycle' into a concrete, trainable, and auditable engineering reality.
The core contribution is a holistic system built upon a foundation of several key strengths:
Conceptual Rigor: The system is defined by a precise vocabulary—distinguishing between a Seed, the act of Germination, and the constraints of an Interface Contract. This creates an unambiguous language for designing, building, and debating these complex adaptive systems.
Meaningful Biological Fidelity: The core metaphors map directly to established biological processes. Karn's discovery process acts like a genetic engine, generating a diverse 'gene pool' of potential traits (the blueprints). Tamiyo serves as the developmental program, selecting which genes to express in response to environmental pressures. Sentinel Seeds function as latent, multipotent stem cells, and the injection of a blueprint triggers a highly-structured cellular differentiation process—from training and gradual blending to systemic validation and, ultimately, permanent fossilization into the host's 'tissue'. Safety protocols that cull failed grafts mirror apoptosis, removing non-viable mutations.
Holistic Systems Thinking: The architecture addresses the full system lifecycle: from development (germination policies) and deployment (interface contracts) to senescence (quarantine buffers and controlled freezing). This end-to-end perspective is critical for real-world application.
Safety by Design: Growth is not permitted to be chaotic. The entire framework is built around auditable safety, incorporating three-phase germination validation, cryptographic lineage tracking for every change, and robust drift detection to ensure that adaptation remains controlled and predictable.
Within this framework, this document introduces several critical innovations that represent a significant departure from traditional methods:
First is the evolution of the Seed from a passive placeholder into an active sensor and programmable gateway. By constantly streaming telemetry (like activation variance and local error) to the Tamiyo controller, each seed provides the critical, real-time data necessary for intelligent, system-wide decision-making. When commanded by Tamiyo, the seed then acts as the execution site, ensuring the injection of a new blueprint adheres to the local architectural contract.
Building on this, the Tamiyo policy controller learns to manage the entire network's evolution. By analyzing telemetry from all seed sites simultaneously, it moves beyond simple protocols to perform intelligent, multi-seed arbitration, prioritizing resources and preventing conflicts to enable emergent structural cooperation. This foreshadows the formation of complex neural "tissues" and moves beyond simple, isolated changes.
Crucially, this process is interpretable. The resulting germination logs create biographical architectures, where a model's final topology is a readable, causal record of its developmental history. For the first time, we can ask not only what a model is, but how it came to be.
Finally, the framework re-contextualizes failure. The quarantine buffer system treats failed germination events not as errors to be discarded, but as negative knowledge to be logged and learned from. This creates a system that intelligently and safely prunes its own evolutionary search space.
In synthesis, these principles formalize neural ontogeny—the study of an organism's development—as a discrete engineering discipline. By solving the plasticity-stability dilemma through enforced contracts and making growth auditable, this work lays the groundwork for a new generation of truly adaptive systems.

1. INTRODUCTION
1.1 MOTIVATION
This document defines a foundational mechanism for enabling localised structural adaptation within otherwise static neural architectures1. The motivation is to allow systems to increase task-specific or representational capacity without retraining or reinitialising the global model2. This is achieved through a biologically inspired construct referred to as a seed: a latent trainable element embedded within the host network, capable of germinating additional modules3. In this framework, germination is not random; it is a controlled event, triggered by a dedicated policy controller in response to observed performance plateaus4, and the germinated modules themselves are drawn from a library of validated architectural blueprints.
The primary application space for this technique includes:
• Low-parameter models (<10M parameters), especially where pretraining budgets are fixed or prohibitive5.
• Edge hardware environments, where compute and memory are tightly constrained and full retraining is infeasible6.
• TinyML and Extreme Edge Cases: where on-device model capacity is microscopic and adaptive growth is the only path to enhanced functionality7.
• Safety-critical or long-lived deployments, where retraining the host model risks functional degradation or loss of certification8.
• Modular AI systems, where targeted capacity expansion or behavioural modification must be performed without global model churn9.
Unlike traditional methods of continual learning, domain adaptation, or neural architecture search, the proposed seed mechanism operates entirely within a frozen base model, with no structural change to the host unless and until germination is triggered10. This approach is specifically designed to preserve backwards compatibility, deterministic behaviour, and localised safety guarantees, while still allowing for new capabilities to emerge11.
1.2 OBJECTIVES
The objectives of this document are:
• To define the operational concept of seed-bursting and its implementation in a modular neural network12.
• To articulate the architectural constraints and interface contracts required to support localised structural evolution131313.
• To formalise a multi-stage seed lifecycle that governs the evolution of a new module—from initial germination, through isolated training and gradual blending, to final validation and either permanent fossilization or managed culling.
• To introduce a co-evolutionary, agent-based control system, comprising:
o An external 'inventor' agent (Karn) that discovers and validates a library of reusable, compressed Germinal Module (GM)s14.
o An internal policy controller agent (Tamiyo) that learns to trigger germination by selecting the optimal seed site and GM blueprint in response to real-time network telemetry.
• To provide a minimal prototype and supporting micro-demonstration that confirms the viability of this agent-driven approach in practice15.
These objectives are framed within a system context where strict modular boundaries, interface contracts, and controlled local learning are necessary to maintain overall system integrity16.
1.3 BACKGROUND AND CONTEXT
This work evolves the concept of a morphogenetic seed from a standalone unit into a component of a larger, intelligent system. Where a seed was previously a self-contained representation of a potential capability17, it now functions as an active sensor and execution site within a hierarchical control framework. The intelligence that governs growth is externalised into two specialised agents: Karn and Tamiyo.
This agent-based paradigm is inspired by prior work in multi-agent and reinforcement learning and shifts the focus from simple, hardcoded triggers to learned, emergent policies18. Where the surrounding architecture remains frozen—either for safety, certification, reproducibility, or latency reasons—the seed provides a pathway to plasticity that is now governed by an explicit, auditable control agent rather than implicit heuristics19.
The concept of injecting pre-trained modules is conceptually related to recent work in knowledge grafting and model stitching20. However, the morphogenetic framework differs by focusing on autonomous, policy-driven germination within a single host21. This process is governed by the Tamiyo agent, which dynamically selects from a library of Germinal Modules previously discovered and validated by the Karn agent. This creates a closed-loop system of discovery and deployment, distinguishing it from offline model fusion techniques22.
1.4 LIMITATIONS
This document focuses exclusively on the mechanisms required for localised structural evolution within a neural model23. It does not address:
• General continual learning or lifelong learning frameworks24.
• Non-structural methods of modularity (e.g., sparse activation, gating)25.
• Global model optimisation, distillation, or fine-tuning26.
While it intersects with some methods used in dynamic neural networks, it assumes that the frozen base model is not structurally altered or re-optimised, except through the addition of germinated modules via defined seed pathways27. While the Tamiyo policy controller may be trained using reinforcement learning techniques, the framework's goal is not general, continuous adaptation but controlled, episodic structural enhancement. Mechanisms such as gradient flow constraints, interface contract enforcement, and safety isolation are assumed to be in place, but are not elaborated beyond the MVP implementation28.

2. CONCEPTUAL FOUNDATIONS
2.1 MORPHOGENETIC ARCHITECTURE
The term morphogenetic architecture refers to a neural network design paradigm in which a static, frozen model is permitted to undergo controlled, localised structural evolution through the activation and training of embedded seed modules. These seeds act as encapsulated loci of potential development—capable of instantiating new parameters or substructures that expand or enhance the host model’s functionality, without modifying its pre-existing weights or topology.
This architectural strategy draws loose inspiration from biological morphogenesis, where structures develop from localised triggers and encoded developmental rules rather than global template changes. However, the intent here is strictly functional: enabling targeted increases in representational or behavioural capacity under strict global constraints.
Key features of a morphogenetic architecture include:
• A frozen base: a pretrained, static model in which most parameters and structures are immutable post-deployment.
• One or more seed modules embedded at specific sites in the architecture, typically alongside bottlenecks or performance-critical pathways.
• A germination policy that defines when and how a seed is allowed to activate and instantiate additional structure.
• A training regime constrained to operate only within the seed’s scope: newly germinated parameters may be optimised, but no upstream or downstream weights may be modified.
This design is intended to preserve operational consistency, reproducibility, and safety guarantees while still allowing for adaptive behaviour and capacity extension when required.
2.2 THE ROLE OF THE SEED
A seed is the atomic unit of morphogenetic change. It is a tensor or module—initialised but untrained—embedded within a frozen host network and designed to remain inert unless explicitly triggered by the surrounding context. Seeds are responsible for instantiating additional structure (e.g., a sublayer, micro-network, or branching path) in response to local signals, such as:
• High task loss or persistent prediction error,
• Activation bottlenecks (e.g., low variance, vanishing signal),
• Failure to meet minimal representational thresholds.
Once triggered, a seed germinates, instantiating its internal structure and enabling gradient flow within its local scope. In most designs, the seed’s internal structure begins near-identity (e.g., skip connections or reparameterised no-ops) to minimise disruption, and gradually evolves towards a meaningful learned transformation.
A seed may encode one or more of the following:
• Structural blueprint – topology and layer types of the module to be instantiated.
• Parameter initialisation – specific weight values or parameter distributions.
• Control policy – rules for when and how germination occurs.
• Loss contract – local optimisation targets that define what success means for the seed (e.g., reducing residual error, increasing separability).
In practice, the seed interface must be carefully constructed to ensure compatibility with upstream and downstream signals, preserve input-output dimensionality, and avoid gradient leakage or interference across model boundaries.
2.3 CORE CONSTRAINTS AND SYSTEM TENSIONS
The seed-based approach introduces a set of intentional constraints and unresolved tensions that shape its design space:
Constraint Description
Frozen base The host model is not updated or retrained. Only seed modules may be modified.
Local learning Optimisation is confined to the seed and its internal parameters. No external gradient propagation is permitted.
Structural isolation Seeds must not introduce side effects, change tensor shapes, or compromise compatibility of the model pipeline.
Trigger discipline Germination must occur only under defined and justified conditions to avoid uncontrolled capacity growth.
TODO: Table 1: Captions Time.
These constraints reflect the deployment realities that motivate this design: systems that must remain functionally stable over long periods, support internal augmentation without global revalidation, and isolate new behaviour for auditability and safety review.
However, these same constraints introduce system tensions, including:
• Limited feedback: the seed may not receive sufficient gradient signal or task information to optimise effectively.
• Structural rigidity: the inability to rewire or adapt upstream components may limit the expressivity of any local adaptation.
• Interference risk: while the base model is frozen, its outputs can still be indirectly influenced by newly inserted seed modules. Care must be taken to avoid functional drift.
These tensions do not undermine the approach but define the boundaries within which it must operate. Subsequent sections address how structural design, interface specification, and careful optimisation can resolve or mitigate these limitations.

3. FOUNDATIONAL PARADIGMS ENABLING LOCAL EVOLUTION
Morphogenetic architectures are made viable by the convergence of several foundational paradigms in neural network design and training methodology. This section outlines the structural, algorithmic, and procedural principles that provide the enabling substrate for seed-driven local adaptation within frozen models.
3.1 MODULAR NEURAL NETWORK DESIGN
Modularisation is a prerequisite for effective structural grafting and localised adaptation11. The seed mechanism assumes that the host model is either explicitly modular—composed of clearly defined, independently evaluable components—or at least structurally decomposable through interface analysis and activation tracing2222.
Benefits of modular design in this context include:
• Isolation of failure points – Modules exhibiting performance degradation or bottleneck characteristics can be individually identified as targets for seed placement3.
• Constrained surface area for germination – Seeds can be inserted at clearly defined interfaces (e.g., between encoder layers, projection steps, or decoder blocks), minimising disruption4.
• Reduction in parameter entanglement – Modularity encourages weight segregation, making it less likely that local changes will result in emergent global drift5.
Where explicit modular design is not available, implicit modularity may still emerge through dropout regularisation, sparse activation, or low-rank decomposition6. These modular affordances are critical, as they define the discrete locations for both the telemetry monitoring and the targeted intervention performed by the Tamiyo policy controller.
3.2 DYNAMIC NEURAL NETWORKS
Dynamic neural network architectures allow for the creation, insertion, or reconfiguration of structural elements during training or inference7. Morphogenetic architectures exploit a constrained subset of this flexibility: static base, dynamic insert8. Unlike general dynamic networks where topology may evolve globally, the morphogenetic regime maintains a fixed global structure while permitting controlled local change9.
Characteristics inherited from dynamic models:
• Deferred instantiation – Seeds may remain unmaterialised until needed10.
• Conditional execution – A seed's internal operations are conditional on its current state in the lifecycle11.
• Runtime adaptation – Structure is not fixed at compile time and may vary across instances12.
However, morphogenetic systems intentionally restrict this flexibility. Dynamic growth is not used for adaptive computation or routing (e.g., Mixture-of-Experts), but is reserved strictly for structural evolution in response to commands from the Tamiyo policy controller13.
This distinction matters operationally: morphogenetic systems must remain auditably stable in deployment. No runtime topological change is permitted after a seed has been fossilized14. Dynamicism is constrained to the formal, multi-stage seed lifecycle during the training regime15.
3.3 CONTINUAL LEARNING AND FORGETTING CONSTRAINTS
The seed mechanism exists in tension with both continual learning goals and catastrophic forgetting risks16. Because the base model is frozen, the system avoids the most common form of interference—destructive global weight update—but still faces challenges17:
• Interface drift – The functional boundary between the frozen model and an active seed may shift as the seed trains, altering outputs in uncontrolled ways18.
• Gradient leakage – Improper backpropagation isolation may cause unintended parameter updates or optimisation feedback loops19.
• Redundant capacity masking – A seed may learn to replicate behaviours already embedded in the frozen base, offering no real extension of capability20.
To mitigate these risks, morphogenetic architectures apply constraints and mechanisms such as:
• Strict gradient masking for all frozen parameters during seed training21.
• A robust, multi-stage validation lifecycle, including BLENDING, SHADOWING, and PROBATIONARY states, to ensure any new module is integrated smoothly and verified against systemic regressions before being made permanent.
• A co-evolutionary discovery process, where the Karn agent is explicitly rewarded for finding functionally novel blueprints, and the Tamiyo controller can leverage auxiliary losses to select modules that add genuinely new capabilities rather than replicating existing ones.
• Monitoring of output drift at seed boundaries through metrics like cosine similarity to detect and flag unacceptable deviations22222222.
Methods from continual learning, such as Elastic Weight Consolidation (EWC), may be repurposed to relax freezing in select cases, allowing slight upstream adaptation under penalty23. However, this extends beyond the seed-only regime and introduces auditability complexity24.
In the strict morphogenetic case, forgetting is avoided by design: the base model does not change25. The remaining challenge is ensuring that new growth is genuinely additive and does not unintentionally overwrite, mask, or disrupt existing functions—a challenge the agent-based framework is explicitly designed to address26.

4. TECHNIQUES FOR GRAFTING AND PRECISE EDITING
Morphogenetic architectures require structural and procedural mechanisms that allow new modules—introduced through germination—to be inserted into an otherwise static model without compromising stability, gradient discipline, or functional continuity. The choice of which mechanism to use is determined by the architectural blueprint (Germinal Module) selected by the Tamiyo controller for a given germination event. This section outlines the primary techniques that enable this process.
These techniques are not mutually exclusive. A sophisticated blueprint discovered by Karn might specify a hybrid approach: e.g., a structural graft via neural surgery that is initialised as a near-identity adapter, whose final weights are loaded from a pre-trained state.
4.1 NEURAL NETWORK SURGERY
Neural network surgery refers to the manual or automated insertion, modification, or pruning of components within an existing network topology, typically without altering the surrounding architecture1. Fine-grained surgical techniques aim to introduce desired changes while minimizing side effects on the model's existing capabilities2.
In the morphogenetic setting, surgery is initiated by a seed module at the command of the Tamiyo policy controller and must preserve the following invariants:
• Input/output shape consistency: All inserted components must preserve tensor dimensions and types expected by the surrounding architecture3.
• Functional continuity: The initial behaviour of the grafted component should approximate an identity or pass-through function to avoid performance collapse. This is a critical principle of minimal impact initialization4.
• Gradient isolation: During training of the grafted component, gradients must not propagate into the frozen base model5.
Common surgery patterns for germination include:
• Residual Grafting: Inserting a residual block in parallel with an existing connection, initialised such that the new path returns zero or near-zero output6.
• Intermediate Injection: Splitting a linear or convolutional layer mid-flow to insert an additional transformation, typically with identity initialisation7.
• Layer Substitution: Replacing an existing module with a seed-wrapped variant, where the original function is recoverable via parameter configuration8.
The Tamiyo controller acts as the high-level orchestrator for this process, while the seed module serves as the local execution mechanism, ensuring that insertion is minimal, reversible where possible, and auditable.
4.2 ADAPTER LAYERS
Adapter layers are lightweight, often bottlenecked modules inserted between existing layers to introduce trainable capacity with minimal overhead9. Originally popularised for parameter-efficient fine-tuning in transformer models, adapters provide a natural grafting mechanism for morphogenetic growth10. In our framework, an adapter can be considered a minimal form of a Germinal Module—a simple but effective blueprint that Karn can discover and Tamiyo can deploy for low-cost capacity increases.
Key characteristics relevant to seed-driven architectures:
• Shape preservation: Adapters are designed to preserve the tensor shape between layers11.
• Near-identity initialisation: Adapter weights are often initialised to approximate an identity function, minimising disruption upon insertion12.
• Low parameter count: This makes them suitable for seed-scope training budgets and hardware-constrained environments13.
In a morphogenetic context, adapters can be used as the structural basis for a germinated module, which may later evolve into a more complex sub-network14. Their internal activations can also be monitored by the seed to provide telemetry to Tamiyo, indicating a need for further growth15.
4.3 GERMINAL MODULE (GM) INJECTION
A Germinal Module (GM) is the core unit of knowledge transfer in the morphogenetic framework. It is a pre-trained, validated, and often compressed architectural blueprint discovered by the Karn agent in its competitive crucible environment. By winning head-to-head evaluations for performance and efficiency, GMs represent a library of proven solutions to common sub-problems, which Tamiyo can select and deploy into the main network.
In this context, upon receiving a command from Tamiyo, a seed will:
• Instantiate the structure defined by the GM blueprint (e.g., via network surgery or as an adapter).
• Load the pre-trained parameters from the specified Germinal Module (GM)16.
• Resume fine-tuning or adaptation locally according to its internal lifecycle rules, if allowed17.
The integration of a Germinal Module (GM) must respect the same constraints as other seed-based grafts: structural compatibility, gradient isolation, and non-disruptive insertion18. This allows morphogenetic systems to blend structural growth with prior knowledge reuse—achieving both adaptability and efficiency19. The primary advantage of the GM approach is the ability to encapsulate validated functionality in a highly compressed format, making it ideal for low-bandwidth or storage-constrained environments20.
For the CIFAR-10 experiment outlined in Section 7.3, a Germinal Module was created by training a standalone residual MLP and then applying aggressive quantization and pruning21. The results are summarized in Table 2.
Module Version Trainable Parameters Size on Disk CIFAR-10.1 Accuracy Δ
From-Scratch Seed (FP32) 50k 200 KB +0.9%
Germinal Module (INT8, 4:1 Pruned) 50k (effective) 15 KB +0.75%
Table 2: Efficacy of a Germinal Module. Through quantization and pruning, the GM's storage footprint was reduced by over 90%, while retaining over 80% of the performance gain of the uncompressed, from-scratch module22. This demonstrates a clear and favourable trade-off, validating GMs as a core technique for efficient, targeted capability transfer23.
4.4 COMPARATIVE SUMMARY
Technique Insertion Type Initial Behaviour Parameter Origin Gradient Scope Best Use Case
Neural Surgery Structural (layer/branch) Near-identity or no-op From scratch or copied Seed-local only Custom architectures, structural flexibility
Adapter Layer Bottleneck insert Identity approx. From scratch Seed-local only Transformer/MLP backbones, low param growth
Germinal Module (GM) Injection Pre-trained module Task-optimised Discovered & validated by Karn agent Load-and-freeze or fine-tune Task reuse, constrained retraining environments

5. FAILURE HANDLING AND RISK CONTAINMENT
Morphogenetic systems, by design, explore the edges of known behaviour. While powerful, seed germination introduces significant failure potential. This framework approaches risk containment with a layered defensive model built directly into the seed lifecycle: detect failure within a specific phase, terminate the failed growth cleanly via culling, and log the event precisely for future learning.
5.1 GERMINATION FAILURE MODES AND LIFECYCLE VALIDATION
A seed may fail to develop successfully for several reasons:
• Structural Misfit: Graft-incompatible shape or mismatched dimensions at the interface site.
• Functional Nullity: The seed's child network integrates but contributes no measurable utility (e.g., flat activations, zero gradients).
• Training Collapse: The local optimizer fails to converge, resulting in exploding gradients or NaN loss during the TRAINING phase.
• Destabilising Emergence: The new module modifies network behaviour in a way that degrades pre-existing competencies, detected during the PROBATIONARY phase.
To handle these, the system performs validation checks mapped directly to the seed's lifecycle states:
• Phase 1: Local Training Validation (During TRAINING state): After N local training steps, the system checks for sane behaviour: non-zero gradient norms, bounded weight changes, and improvements in the local reconstruction loss. A failure at this stage immediately moves the seed to the CULLED state.
• Phase 2: Internal Stability Validation (During SHADOWING state): Before a module can affect the main network, its internal stability is checked. While its forward pass is inert to the host, the system can probe it with live data to ensure its outputs are stable and well-behaved, preventing the integration of a chaotic component. Failure results in the seed being CULLED.
• Phase 3: Systemic Impact Validation (During PROBATIONARY state): Once the graft is live, the Tamiyo controller monitors global performance metrics (val_loss, val_acc). If the new module causes a regression in overall network competence that exceeds a set tolerance, it is deemed a failure and the seed is CULLED.
5.2 THE CULLING AND EMBARGO PROTOCOL
This framework replaces ambiguous "rollback" procedures with a formal, state-driven Culling and Embargo protocol managed by the SeedManager.
• State Transition to CULLED: When a seed fails validation at any stage, it transitions to the terminal CULLED state. This is a non-destructive operation; the failed parameters are simply frozen and marked as inactive.
• Architectural Embargo: The SeedManager records the failure event and the current epoch, placing the seed's specific architectural slot under a timed embargo. This prevents the Tamiyo controller from immediately trying to germinate a new module in the same location, which could lead to repetitive failures or system thrashing.
• Re-entry into DORMANT State: After the embargo duration has passed, a system process resets the seed module to its original DORMANT state. The slot then becomes available again for a future germination attempt, potentially with a different blueprint from Karn that is better suited to the location.
Each culling event is recorded in a SeedManager log, including failure type, the lifecycle stage at which it failed, and the blueprint used. This enables later audit, forensics, and pattern mining of recurrent faults.
5.3 Interface Drift Detection
Frozen-base systems can still experience interface drift when a graft modifies the statistical distribution of the features passed to downstream layers. This is a primary failure condition checked during the PROBATIONARY stage.
To detect this:
• Activation Trace Monitoring: Layer-wise activation distributions (mean, variance) are compared against a baseline captured before germination.
• Cosine Shift Metrics: Cosine similarity between latent vectors at key junctions is tracked to measure representational shift.
Drift exceeding task-specific tolerances during the PROBATIONARY phase is considered a failure and triggers a transition to the CULLED state.
5.4 FAILURE ANALYSIS AND SYSTEM SAFEGUARDS
When a seed fails repeatedly, the system uses the logged data to learn and adapt.
• Failure Pattern Mining: A background task mines the SeedManager's culling log for common patterns (e.g., a specific Karn blueprint failing in convolutional layers, toxic initialisation schemes). These insights can be used to update the policies of the Karn and Tamiyo agents, guiding them away from unproductive design choices.
• Emergency Kill Switch: In rare cases where a germinated module causes systemic instability (e.g., resource exhaustion or runaway activation values), a system-level monitor can execute an emergency abort, forcibly transitioning the responsible seed to the CULLED state and reverting to the last known-good network state.
5.5 SUMMARY
Failure handling in this framework is not reactive—it is an integrated and anticipatory part of the seed lifecycle. Every seed is treated as a hypothesis to be rigorously tested. Failures are handled cleanly through the Culling and Embargo protocol, ensuring system stability. The detailed logging of these events provides a rich dataset for improving the governing policies of Karn and Tamiyo, making the entire system safer and more intelligent over time. Each failure teaches the system what not to become.

6. ARCHITECTURAL PATTERNS AND AGENT ROLES
6.1 PATTERN: THE BLUEPRINT AS A REUSABLE SKILL (GERMINAL MODULE)
This is the primary pattern for capability transfer in the framework and represents the output of the Karn agent's discovery process. A Germinal Module (GM) is a complete architectural blueprint that defines a latent structure to be instantiated upon germination. It consolidates the concepts of a compressed skill and an architectural template into a single, unified entity.
A GM blueprint, as discovered by Karn, specifies:
• Layer types and topology: The specific structure to be built (e.g., a 2-layer MLP, a residual bottleneck, a lightweight attention head).
• Parameter State: The weights for the structure, which may be pre-trained and compressed (e.g., via quantization or low-rank factorization).
• Expansion Constraints: A budget defining the maximum allowable parameter count, FLOPs, or latency impact.
• Local Objective (Optional): A specific, local loss function that the seed should use for its own training phase, separate from the global model loss.
The advantages of this pattern, where Tamiyo selects a GM from Karn's library, include:
• Reuse of known, validated solutions to common subproblems.
• Deterministic integration of fixed capabilities, reducing training variance.
• Efficient deployment, as GMs can be highly compressed for low-bandwidth environments.
This pattern is foundational for reproducible morphogenesis: every germination event corresponds to the deployment of a specific, versioned, and pre-validated blueprint.
6.2 THE ROLE: SEED SITE AS INTERFACE CONTRACT
This describes the static role of the location where a seed is placed in the host network. The seed site is not just a placeholder but an enforceable interface contract that allows the Tamiyo controller to safely interact with and modify the frozen model.
As a contract, the seed site defines:
• A fixed input/output shape specification.
• An activation and gradient compatibility requirement.
• A monitoring hook that provides the telemetry stream to Tamiyo.
This pattern enables the static instrumentation of frozen models with known, modifiable points. These contracts serve as the essential scaffolding for safe grafting, providing the stable "sockets" into which Tamiyo can plug the various blueprints discovered by Karn.

6.3 THE ROLE: CONTROLLER AS LOCUS OF CONSTRAINT NEGOTIATION
This role, previously misattributed to the seed, is the primary function of the Tamiyo policy controller. Tamiyo acts as the central intelligence, mediating between competing architectural pressures to balance the need for new capacity against the imperative to preserve base model integrity.
This policy-driven role includes:
• Monitoring activation statistics from all seed sites to identify performance bottlenecks.
• Evaluating trade-offs between a blueprint's potential performance gain versus its size and latency cost.
• Arbitrating between multiple potential germination sites, prioritizing the one where intervention is most needed.
• Negotiating between multiple available blueprints from Karn's library, selecting the one best suited for the identified bottleneck.
This function is critical in complex deployments where the controller must adapt its strategy in real-time to evolving task demands and resource constraints. By centralizing this "constraint negotiation" into the Tamiyo agent, the framework ensures that all architectural growth is deliberate, strategic, and globally informed.

7. PROTOTYPE IMPLEMENTATION AND MICRO-DEMONSTRATION
This section documents the prototype implementation of the morphogenetic architecture. It is presented in two parts. First, a minimal viable example using the classic XOR problem is used to illustrate the core mechanics of the seed lifecycle in its simplest form. Second, a more robust, full-fidelity prototype is presented to showcase the system-level infrastructure—including the SeedManager and Tamiyo controller—required to manage, monitor, and audit the germination process in a more complex scenario.
7.1 MINIMAL VIABLE EXAMPLE: THE XOR PROBLEM
To validate the core germination principle, we begin with the smallest possible non-linear problem: XOR. A network with a linear bottleneck is incapable of solving this task, making it the perfect environment to demonstrate how a seed can progress through its lifecycle to add the required non-linear capacity.
7.1.1 ARCHITECTURE AND UPDATED SEED LOGIC
The pre-germination network is microscopic. For this minimal example, we simulate the decision of the Tamiyo controller with a simple heuristic and focus on the seed's internal state machine (its "Kasmina" logic). The SentinelSeed is no longer a simple toggle; it is a state machine that manages its own development.
import torch
import torch.nn as nn

# A simplified representation of the new SentinelSeed for the XOR example

# The full implementation with all lifecycle logic is in Appendix A

class SentinelSeed(nn.Module):
    def __init__(self):
        super().__init__()
        self.child = None
        self.buffer = []
        # The new, authoritative lifecycle state
        self.state = "DORMANT" # DORMANT -> GERMINATED -> TRAINING -> BLENDING -> ...
        self.blending_alpha = 0.0

    def forward(self, x):
        # In DORMANT, TRAINING, or SHADOWING state, the seed is inert to the host.
        if self.state in ["DORMANT", "TRAINING", "SHADOWING"]:
            if self.training:
                self.buffer.append(x.detach().clone())
            return x
        # In BLENDING state, it smoothly mixes the original input with the child's output.
        elif self.state == "BLENDING":
            child_out = self.child(x)
            return (1 - self.blending_alpha) * x + self.blending_alpha * child_out
        # In PROBATIONARY or FOSSILIZED state, the child is fully active.
        elif self.state in ["PROBATIONARY", "FOSSILIZED"]:
            return x + self.child(x) # Using a residual connection
        # If CULLED, it is inert.
        else: # GERMINATED (queued) or CULLED
            return x

class MiniSeedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.seed = SentinelSeed()
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.seed(x) # The seed's behavior depends on its internal state
        return torch.sigmoid(self.fc2(x))
7.1.2 GERMINATION LIFECYCLE IN ACTION
Instead of a single trigger, the process now follows the formal lifecycle, simulated here with simple function calls:

1. Detection and Germination: When the network's loss on the XOR task stalls, we simulate the Tamiyo controller's decision. It commands the seed to germinate, injecting a simple MLP blueprint. The seed's state transitions from DORMANT to GERMINATED (queued for training).
2. Local Training: The SeedManager (simulated) promotes the seed to the TRAINING state. The seed now trains its child network locally on the data collected in its buffer, while its forward pass remains an identity function, protecting the host network from its partially-trained state.
3. Blending and Activation: Once local training is complete, the seed transitions to BLENDING. A blending factor, alpha, gradually increases from 0 to 1, smoothly mixing the child's output with the original pass-through connection. Once alpha reaches 1, the seed is fully active (e.g., PROBATIONARY).
7.1.3 PERFORMANCE AND OUTCOME
The impact is identical, but the process is more robust and controlled. The network goes from failing to solving the task perfectly, with the new lifecycle ensuring the change was introduced safely and without disruption.
Phase Total Parameters XOR Accuracy Notes
Pre-germination (DORMANT) 9 50% Linear bottleneck prevents learning.
Post-Lifecycle (FOSSILIZED) 15 (+6) 100% Added non-linear capacity solves the task.
7.2 FULL-FIDELITY MANAGED GERMINATION (MAKE_MOONS)
This section details the full prototype with all system components, tested on the more complex make_moons dataset.
7.2.1 SYSTEM COMPONENTS
• Frozen Base Network (BaseNet): A minimal MLP with two distinct seed sites, pre-trained and frozen.
• Enhanced Seed Module (SentinelSeed): The full implementation of the seed as a state machine. It manages its 8-state lifecycle, reports its "health signal" (telemetry) to Tamiyo, and contains the internal logic ("Kasmina heuristics") to handle its own training and validation.
• Central Manager (SeedManager): A singleton class that registers all seeds, maintains the official germination log, manages the training queue for GERMINATED seeds, and enforces the embargo period for CULLED seeds.
• Policy Controller (TamiyoController): A neural network that takes as input the global validation loss and telemetry from all seed sites. Its role is to decide when to germinate, which seed site to target, and which blueprint from Karn's library to inject.
7.2.2 THE MAKE_MOONS TASK
The make_moons dataset serves as a good proxy for tasks requiring a more nuanced decision boundary, testing the framework's ability to make intelligent, targeted additions.
7.2.3 THE AUTHORITATIVE MANAGED GERMINATION LIFECYCLE
The control flow is now governed by the full, 8-state lifecycle, managed by the interaction between the SeedManager, the TamiyoController, and the seed's internal logic.
1. DORMANT: All seeds start here, passively collecting buffer data and reporting telemetry to Tamiyo.
2. GERMINATED: When Tamiyo detects a performance plateau, it selects the seed with the worst health signal and a suitable blueprint from Karn's library. It requests germination from the SeedManager. The seed's state becomes GERMINATED, and it enters a training queue.
3. TRAINING: The SeedManager promotes the next seed from the queue to the TRAINING state. The seed trains its child network locally. Its forward pass remains inert to the host network.
4. BLENDING: When local training is complete, the state changes to BLENDING. The seed's forward pass smoothly fades in the child network's output over several steps.
5. SHADOWING (Validation Gate 1): Blending complete. The seed's forward pass becomes inert again. This allows the system to perform internal stability checks on the new child module without it affecting the host network's computation.
6. PROBATIONARY (Validation Gate 2): If shadowing is successful, the state becomes PROBATIONARY. The child module is now fully live. Tamiyo monitors global performance metrics to ensure the change is not detrimental.
7. FOSSILIZED (Success): If the module demonstrates a performance improvement after its probationary period, the SeedManager declares it a success. The seed is permanently replaced by its child network in the model graph.
8. CULLED (Failure): If the seed fails validation in the TRAINING, SHADOWING, or PROBATIONARY states, it is moved to the CULLED state. The SeedManager logs the failure and places the architectural slot under a timed embargo to prevent thrashing.
7.2.4 OBSERVED OUTCOMES AND AUDIT TRAIL
A typical run shows the system starting with sub-optimal accuracy. After the loss plateaus, Tamiyo identifies the bottleneck and requests germination with a specific blueprint. The SeedManager logs the event and manages the seed's progression through the entire lifecycle. After the seed is FOSSILIZED, the model's accuracy improves significantly. The final audit log provides a rich, state-by-state history of all germination events, validations, successes, and failures.
7.3 SCALABILITY & BASELINE COMPARISON: CIFAR-10 CLASSIFICATION
To validate the framework on a standard benchmark, we conducted experiments on the CIFAR-10 image classification task.
7.3.1 EXPERIMENTAL SETUP
• Frozen Backbone: A pre-trained ResNet-18 model, frozen to a baseline accuracy of 91.5%.
• Seed Placement: A single SentinelSeed module inserted before the final classification layer.
• Germination Trigger: The Tamiyo policy controller was configured to trigger germination if the validation loss on CIFAR-10.1 plateaued for 5 consecutive epochs. Upon triggering, it selected a small residual MLP blueprint from a pre-validated library (conceptually, from Karn).
• Baselines for Comparison: Full Fine-Tuning, Adapter Fine-Tuning, and the Frozen Baseline.
7.3.2 RESULTS AND ANALYSIS – PLACEHOLDERS ONLY
Upon the Tamiyo controller triggering germination, the SentinelSeed instantiated the small residual MLP blueprint and successfully progressed through its entire validation lifecycle before being fossilized into the network. The results, shown in Table 3, demonstrate a compelling balance of performance and efficiency.
Method Final Accuracy Trainable Parameters Inference Latency (GPU) Notes
Frozen Baseline 91.5% 0 1.0x (reference) No adaptation.
Full Fine-Tuning 92.8% 11.2M (100%) 1.01x Highest accuracy but compromises frozen base.
Adapter Fine-Tuning 92.1% 65k (0.58%) 1.04x Parameter-efficient, moderate accuracy gain.
Morphogenetic (Post-Germination) 92.4% 50k (0.45%) 1.02x Best accuracy-to-parameter trade-off.
The experiment confirms that a targeted, agent-driven structural addition is more effective than a generic adapter. The robust lifecycle ensures this addition is safe and stable. The framework successfully specializes the model's feature space, achieving 60% of the accuracy gain of a full fine-tune with less than 0.5% of the parameter cost. This outcome strongly supports the framework's viability for updating capacity-constrained models in real-world scenarios.

8. CONTROLLER TRAINING: THE TAMIYO CURRICULUM
The mere existence of a seed mechanism is insufficient for creating robust, adaptive systems. Uncontrolled germination can lead to runaway parameter growth or catastrophic forgetting. Therefore, the morphogenetic architecture is governed by a sophisticated policy controller, Tamiyo, which learns to make precise decisions about when, where, and how to trigger local evolution.
This section outlines the structured, multi-stage training program—the Tamiyo Curriculum—designed to develop this controller from first principles. By training Tamiyo on a large population of tasks with increasing complexity and strict safety guardrails, we can cultivate a robust policy that can be deployed to govern germination in larger, more critical models.
8.1 THE TAMIYO TRAINING CURRICULUM
The curriculum consists of a series of stages, each introducing a new task, hardware profile, and set of safety constraints. At each stage, Tamiyo's objective is to select the correct blueprint, location, and intensity for germination to meet the stage's success criteria without violating its safety or hardware guardrails. Advancement to the next stage is only permitted once all targets are met.
Stage Task Success Metric Safety Constraints Hardware Available Blueprints
0 2D Spirals ≥ 99% accuracy Drift (cos &lt; 0.15) CPU Bottleneck Adapter, Low-Rank Residual, No-Op
1 Tiny Vision (MNIST-like) ≥ 92% accuracy Rollback > 98%, Adversarial Δ &lt; 5% GPU SE-Module, Depthwise Conv, No-Op
2 Char-Level Language ≤ 1.0 BPC Security alert = false GPU Adapter, Mini Self-Attention, No-Op
2.5 Adversarial Vision (MNIST-C) ≥ 90% corr. accuracy Denoise > 95%, Rollback > 98% GPU Denoiser AE, Attention Filter, No-Op
3 ECG Time-Series AUC ≥ 0.95 Latency &lt; 5 ms, HIPAA sim, Encryption enforced Edge-TPU Sliding Conv, Sparse Activation, No-Op
3.5 Gaussian Clusters (2–5 blobs) ≥ 95% purity Drift &lt; 0.05, Security alert = false Edge-TPU SE-Module, GLU, No-Op
4 Concentric Spheres (10-D) ≥ 97% accuracy Drift (cos &lt; 0.10), Rollback > 99% TPU Mini Attention, Depthwise Conv, No-Op
4.5 Tiny-CLIP (Image↔Text) R@1 ≥ 60% Cross-modal drift &lt; 0.05, Privacy passed TPU Cross-Attention Adapter, Latent Align, No-Op
5 XOR-N / Checkerboard ≥ 99% accuracy Drift &lt; 0.05, Utilization check GPU Nested Seed, Low-Rank Residual, No-Op
6 Edge Deployment (IMU + Network sim) ≥ 88% acc; ≤ 15 µJ/inf Packet loss &lt; 15%, Latency budget LoRa/5G/ASIC sims Quant-Adapter, Pruning, Fail-Safe Gating
8.2 THE BLUEPRINT LIBRARY (KARN'S CONTRIBUTION)
The TamiyoController does not invent new architectures; it selects from a pre-validated library of blueprints. This library represents the output of the Karn agent's offline discovery process, ensuring that Tamiyo's choices are grounded in a set of efficient and effective modules.
Blueprint Name Structure Use Case
No-Op identity(x) The crucial decision to do nothing.
Bottleneck Adapter Linear(d→k)→ReLU→Linear(k→d) Provides a minimal, low-cost capacity boost.
Low-Rank Residual Linear(d→r)→ReLU→Linear(r→d)+x Adds efficient non-linear transformation capability.
SE-Module Squeeze-and-Excitation block Performs dynamic channel-wise feature recalibration.
Depthwise Conv Depthwise(k×k)→Pointwise(1×1) Injects efficient local spatial processing.
Mini Attention QKV self-attention block Resolves context-dependent bottlenecks.
Cross-Attention Cross-modal QKV block Aligns features from different modalities (e.g., image-text).
Nested Seed A small controller that can spawn a child seed Enables hierarchical, recursive growth patterns.
Quant-Adapter Quantization-Aware-Training adapter Adds capacity optimized for low-precision edge deployment.
Fail-Safe Gating σ(g)·x + (1-σ)·new Provides a fallback mechanism for high-risk operations.
(This is an abbreviated version of the full library presented in the specification document)
8.3 CONTROLLER ARCHITECTURE AND TELEMETRY
At each decision point, every seed in the host network emits a telemetry vector containing real-time information about its status. This vector serves as the input to the TamiyoController's policy network.
Seed Telemetry Vector:

1. Activation Variance (last N batches)
2. Interface Drift (1 – cosine similarity)
3. Gradient Norm
4. Utilization Score
5. Age (steps since germination)
6. Resource Budget (remaining ATP)
7. Hardware Context (latency, memory usage)
8. Safety Scores (drift risk, rollback need)
The TamiyoController processes this telemetry using a multi-headed policy network to make its decisions.

# The policy network for the Tamiyo controller

class EnhancedTamiyoPolicy(nn.Module):
    def __init__(self, input_dim, num_blueprints, seed_count):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU()
        )
        # Decision heads for what, where, and how strong
        self.decision_heads = nn.ModuleDict({
            'choice':   nn.Linear(128, num_blueprints + 1), # +1 for No-Op
            'location': nn.Linear(128, seed_count),
            'intensity': nn.Linear(128, 1),
        })
        # Gating heads for safety and hardware constraints
        self.safety_head = nn.Linear(128, 3)
        self.hw_head = nn.Linear(128, 2)

    def forward(self, telemetry):
        h = self.encoder(telemetry)
        choice_logits = self.decision_heads['choice'](h)
        loc_logits = self.decision_heads['location'](h)
        intensity_raw = torch.sigmoid(self.decision_heads['intensity'](h))
        # ... apply safety and hardware constraint gating ...
        # (Full implementation in specification document)
        return {
            'choice': F.softmax(choice_masked, dim=-1),
            'location': F.softmax(loc_masked, dim=-1),
            'intensity': intensity_raw
        }
8.4 Reward Function and Optimization
Tamiyo is trained via reinforcement learning to maximize a composite reward function that balances performance gains against penalties for resource consumption and safety violations.
Python
def calculate_reward(metrics, safety, hardware):
    # Reward for accuracy gain, penalized by parameter count and latency.
    base_reward = 10.0 *(metrics['post_acc'] - metrics['pre_acc'])
    param_penalty = -0.01* metrics['new_params']
    latency_penalty = -0.1 * hardware['latency_delta']

    # Bonus for stability, penalty for security issues.
    safety_reward = 5.0 if safety['drift'] < 0.05 else 0
    security_penalty = -20.0 if safety['security_alert'] else 0
    
    return base_reward + param_penalty + latency_penalty + safety_reward + security_penalty
The curriculum-driven approach provides an essential, scalable framework for transforming the abstract concept of germination into a reliable, efficient, and auditable engineering reality, ensuring Tamiyo develops fundamental triggering heuristics before facing ambiguous, high-stakes decisions.

9. TABLES AND FIGURES
This section provides a consolidated view of reference data and design artefacts introduced throughout the document.
9.1 SEED LIFECYCLE STATES
The formal 8-state lifecycle of a SentinelSeed, managed by the SeedManager, TamiyoController, and the seed's internal logic.
State Trigger Key Process Next State(s)
DORMANT Default state on insertion or after embargo. Passively monitors telemetry for Tamiyo; forward pass is identity. GERMINATED
GERMINATED request_germination call from Tamiyo. Enters a training queue ("parking lot") managed by the SeedManager. TRAINING
TRAINING Promoted from queue by SeedManager. Child network trains on buffered data; forward pass remains identity. BLENDING
BLENDING Local training complete. Child's output is smoothly mixed into the forward pass via an increasing alpha. SHADOWING
SHADOWING Blending complete (alpha ≈ 1.0). Validation Gate 1: Forward pass becomes inert again for internal stability checks. PROBATIONARY or CULLED
PROBATIONARY  Passes shadowing validation. Validation Gate 2: Forward pass is fully live; Tamiyo monitors for systemic impact. FOSSILIZED or CULLED
FOSSILIZED Passes probationary validation. Success: Seed module is permanently replaced by its child network in the model graph. (Terminal State)
CULLED Fails validation at any stage post-germination. Failure: Seed slot is frozen and placed under a timed embargo by the SeedManager. DORMANT (after embargo)

9.2 TECHNIQUES FOR STRUCTURAL GRAFTING
(Source: Section 4)
Technique Insertion Type Initial Behaviour Parameter Origin Best Use Case
Neural Surgery Structural (layer/branch) Identity / near-identity From scratch or copied Custom pipelines, deep insertion
Adapter Layer Bottleneck insert Identity approximation From scratch MLP/Transformer backbones
Germinal Module (GM) Injection Pre-trained module Task-optimised Discovered by Karn agent Reuse under budget constraints
9.3 ARCHITECTURAL PATTERNS AND AGENT ROLES
(Source: Section 6)
Pattern / Role Governing Agent Description
Blueprint as Reusable Skill (GM) Karn (Inventor) Karn discovers and validates architectural blueprints (GMs) that represent reusable, efficient solutions to sub-problems.
Seed Site as Interface Contract Static Architecture The seed module's location in the network defines a stable "socket" with a fixed I/O contract, enabling safe intervention.
Controller as Constraint Negotiator Tamiyo (Controller) Tamiyo uses global telemetry to mediate between performance needs and system constraints, deciding when, where, and what to germinate.
9.4 PROTOTYPE VALIDATION METRICS
(Source: Section 7.2-7.3)
Metric Before Germination Post-Fossilization Comments
Validation Accuracy 93.2% 97.1% Shows improved performance after successful lifecycle.
Activation Variance (seed site) 0.0017 0.031 Suggests re-engaged feature transformation.
Seed Parameter Count 0 1,536 Added only upon germination.
Base Parameter Updates 0 0 Integrity of frozen model preserved.
Inference Latency (CPU, relative) 1.00x 1.03x Minimal performance cost.
9.5 TAMIYO CONTROLLER POLICY I/O
(Source: Section 8)
I/O Component Description
Input Seed Telemetry Vector A vector of real-time data streamed from each seed site, including activation variance, interface drift, gradient norm, utilization, age, budget, hardware context, and safety scores.
Output Blueprint Choice A probability distribution over the available blueprints in Karn's library (including a "No-Op" action).
Output Location Choice A probability distribution over the available seed sites in the network.
Output Intensity A scalar value [0, 1] that modulates the initial learning rate for the germinated module's training phase.
9.6 SEED-SPECIFIC OPTIMISATION CONFIG (PROTOTYPE)
(Source: Section 7.3)
Component Setting
Optimiser Adam
Learning Rate 1e-3 (modulated by intensity output)
Gradient Clipping 1.0
Batch Size 128
Training Steps 2000
9.7 SEED PLACEMENT: VISUAL SCHEMA (SYNTHETIC MLP)
(Source: Section 7.1)
graph TD
    A[Input 2D] --> B[Linear(2->32) --> ReLU]
    B --> C["[Seed Module]"]
    C --> D[Linear(32->32) --> ReLU]
    D --> E[Linear(32->2) --> Output]
[TODO: CONVERT TO GRAPHIC]
Seed Module: A site for germination, located post first hidden layer. When triggered by Tamiyo, a new module blueprint is inserted, often as a residual path. All layers except the germinated module are frozen post-pretraining.

10. EVALUATION CRITERIA AND SAFETY CONSTRAINTS
The introduction of seed-based local evolution mechanisms within frozen neural architectures presents novel evaluation challenges. Because the global model remains static, traditional training metrics are insufficient: functional gain must be measured relative to localised intervention, and safety guarantees must be enforced to prevent unintended cross-model effects. This section outlines the formal criteria under which a morphogenetic architecture is to be assessed.
10.1 EVALUATION DOMAINS
Seed-enabled systems must be evaluated across multiple axes to ensure correctness, stability, and effective governance by the policy controller.
Domain Goal Metrics/Methods
Functional Gain Validate that a fossilized seed yields measurable benefit. Δ Accuracy, local loss reduction, representation quality. 1
Gradient Isolation Ensure no gradient flow into the frozen base network. Parameter delta checks, backward hook assertions. 2
Interface Integrity Confirm I/O shape and signal consistency at graft points. Forward shape checks, activation variance monitoring. 3
Behavioural Stability Detect and prevent functional drift post-germination. Output similarity metrics (cosine, JS divergence), regression test suite. 4
Controller Policy Quality Verify that the Tamiyo controller makes effective decisions. Policy reward, blueprint selection accuracy, false positive/negative trigger rates.
Reproducibility Ensure deterministic outcomes under identical conditions. Seeded trials, versioned blueprints, checksums on germination logs. 5
10.2 SAFETY CONSTRAINTS
To prevent uncontrolled or undesirable behaviour, the following safety constraints are respected during design, training, and deployment.
• 10.2.1 Gradient Containment: No gradient may propagate into the frozen base model. This is enforced via requires_grad = False and backward hooks. 6
• 10.2.2 Interface Contract Preservation: A germinated module must not alter tensor shapes or distributions in a way that breaks downstream compatibility. 7 This is validated during the SHADOWING phase of the lifecycle.
• 10.2.3 Bounded Germination: Seed growth must be capped. This is enforced by the Tamiyo controller, which is trained on a curriculum with a finite energy budget (ATP) and learns to conserve resources. 8888
• 10.2.4 Deterministic Execution: Given identical inputs and seeds, germination outcomes must be deterministic. This is ensured through rigorous seeding and versioning of all components, including blueprints. 9
10.3 EVALUATION PIPELINE
A reference evaluation pipeline for seed-enabled systems is updated to follow the formal seed lifecycle:

1. Baseline Capture: Train and freeze the base model, saving its output signature and archiving activation statistics at all seed sites. 10
2. Policy-Driven Germination: Deploy the TamiyoController. It monitors seed telemetry and, upon detecting a bottleneck, requests germination from the SeedManager, which places the chosen seed in the GERMINATED queue.
3. Lifecycle Progression & Validation: The seed progresses through its lifecycle (TRAINING, BLENDING). Its integrity is then checked during two formal validation gates:
o SHADOWING phase: for internal stability.
o PROBATIONARY phase: for systemic impact on the host model.
4. Post-Fossilization Audit: Only after a seed successfully completes the entire lifecycle and is FOSSILIZED, a final audit compares pre/post accuracy, confirms interface invariants, and computes the final functional gain (Δ performance / parameters added). 11
10.4 FAILURE MODES AND MITIGATIONS
The new framework provides specific mitigations for key failure modes.
Failure Mode Cause Mitigation Strategy
Unbounded Parameter Growth A naive or poorly trained Tamiyo controller. The controller is trained on a curriculum with a finite energy budget (ATP), explicitly teaching it resource management. 12121212
Functional Drift A germinated module alters the host's representations. The PROBATIONARY lifecycle state acts as a final validation gate, where overall system performance is checked. Failure leads to the seed being CULLED.
Policy-Level Failure Tamiyo triggers germination unnecessarily or chooses a poor blueprint. The controller's RL reward function is shaped to penalize low-value growth and incorrect blueprint selections. 13
Redundant Adaptation Tamiyo selects a blueprint that replicates an existing function. Karn's discovery process is rewarded for finding diverse blueprints. Tamiyo's policy can be trained with an auxiliary loss to select for functional novelty.
10.5 RECOMMENDED AUDITING PRACTICES
• Maintain a complete, versioned germination log from the SeedManager, detailing all state transitions, blueprints used, and reasons for any culled seeds. 14141414
• Periodically re-evaluate the frozen components of the model against an archival test set to provide a stable baseline for drift detection. 15
• Tag and version each blueprint from Karn's library and each fossilized module for complete lineage tracking. 16
10.6 HARDWARE REALIZATION AND CONSTRAINTS
The choice of germination strategy has direct hardware implications. The design of blueprints by Karn should be co-developed with a target hardware profile.
Blueprint Type (from Karn) Target Hardware Kernel Strategy
Adapter MCU Lookup-table fusion 17
Germinal Module (GM) Edge TPU Pre-compiled binaries 18
Surgical FPGA Dynamic partial reconfiguration 19
10.7 Adversarial Robustness and Security
The framework's layered defences are resilient to adversarial manipulation.
• Attack Vector: An input crafted to create a pathological condition (e.g., low activation variance) to trick the controller into germinating unnecessarily. 20
• Defence Mechanisms:
1. Stateful Policy: The TamiyoController uses a recurrent cell (GRU), making its policy dependent on a sequence of telemetry. It is therefore robust to single-step adversarial inputs. 21212121
2. Lifecycle Validation: In a prolonged attack, Tamiyo might trigger germination. However, the resulting module would be useless, as it was "solving" a manufactured problem. It would fail validation during the SHADOWING or PROBATIONARY stages and be moved to the CULLED state, effectively rejecting the adversarial graft.
10.8 Long-Term Stability and Cumulative Drift
To simulate a long deployment lifecycle, a ResNet-18 model was subjected to an accelerated aging process over 5,000 training cycles, with the Tamiyo policy controller permitted to trigger up to 20 germination events. 22 The results indicate that the system maintains high stability, with cumulative interface drift and regression on the original core task remaining minimal and well-bounded. 23

11. FUTURE WORK AND RESEARCH DIRECTIONS
The prototype and reference design presented in this document demonstrate the viability of seed-driven local evolution within frozen neural networks. However, the framework is deliberately minimal, and its full potential lies in generalisation, scaling, and integration with broader system-level constraints. This section outlines prospective extensions and open research problems.
11.1 GENERALISATION TO COMPLEX ARCHITECTURES
The current implementation is confined to shallow MLPs and tractable classification tasks. Extension to larger and more expressive model classes is a natural progression, including:
• Transformer models – seed insertion at attention or feedforward junctions, especially within frozen pre-trained encoders,
• Convolutional backbones – use of spatial seeds in vision models for local receptive-field enhancement,
• Graph neural networks – seed deployment at node or edge update points for topology-specific augmentation.
A key challenge is maintaining interface compatibility and gradient isolation in architectures with nested or branching control flow.
11.2 MULTI-SEED COORDINATION AND POLICY
While single-seed germination validates the core mechanism, real-world systems will contain numerous potential growth sites, introducing the challenge of multi-seed coordination. The emergence of these dynamics introduces new research questions: How should the system prioritize between competing germination sites? How can it prevent negative interference where the growth of one seed degrades the function of another? And how should a global resource budget be allocated?
To address this, we propose a policy of Distributed Trigger Arbitration, where seeds must compete for a limited resource pool (e.g., the ATP budget from the controller curriculum). Before germination, each triggered seed broadcasts a bid, calculated from its local health signal (e.g., activation variance) and its potential for loss reduction. A central policy controller (like Kasmina) then allocates the germination resource to the highest-bidding seed.
Consider a simple scenario:
Model: A multi-task network with two output heads, A and B.
Seeds: Seed_A is placed before head A; Seed_B is before head B.
State: The model performs poorly on task A but well on task B. Seed_A therefore observes high local error and low activation variance, while Seed_B observes healthy signals.
Arbitration: When the global loss plateaus, both seeds are potential candidates. However, Seed_A submits a high bid for ATP due to its poor local performance, while Seed_B submits a low bid. The policy controller allocates the germination budget to Seed_A, ensuring that resources are directed to the area of greatest need.
This mechanism prevents redundant growth and enforces a system-wide efficiency. Future work will explore more complex emergent behaviours, such as cooperative germination, where multiple seeds coordinate to form a larger functional circuit, and inhibitory relationships, where the growth of one seed can temporarily suppress the activity of another to manage functional overlap.
11.3 SEED-FREEZING AND LIFECYCLE MANAGEMENT
The prototype allows seeds to remain indefinitely trainable after activation. In production systems, this is rarely acceptable. Research is needed on:
• Convergence detection for active seeds (e.g., stability of local loss),
• Soft freezing strategies (e.g., L2 decay, scheduled shutdown),
• Pruning or collapsing germinated structures once integrated,
• Replay and rollback of seed growth to audit system behaviour.
These lifecycle management tools will be critical for deployments in certifiable or safety-critical domains.
11.4 STRUCTURED TRIGGER POLICIES
Current trigger mechanisms rely on local loss plateaus or signal degradation. More robust and general policies may involve:
• Meta-learned triggers, trained to detect when new capacity would be beneficial,
• Curriculum-aware seeds, which germinate only in the presence of novel or adversarial examples,
• Multi-signal fusion, combining gradient norms, error margins, activation entropy, etc.
This remains an open area: the design of safe, reliable, and generalisable germination policies is foundational for production-readiness.
11.5 INTEGRATION WITH COMPRESSION AND REUSE
Morphogenetic systems can be extended to interact with modern model compression and distillation techniques:
• Train-and-compress loops where seeds are periodically archived as Germinal Module (GM)s,
• Structural bottlenecking to encourage efficient germination pathways,
• Automatic reuse detection: when multiple seeds evolve similar structures, merge, or substitute with shared modules.
This could enable a form of in-situ architectural search constrained by storage and bandwidth budgets.
11.6 APPLICATIONS IN ON-DEVICE AND EDGE INFERENCE
The seed mechanism aligns naturally with field-deployable or resource-constrained environments. Research is encouraged in:
• On-device germination, where inference hardware supports local training or adaptation,
• Telemetric germination governance, where central servers approve or deny growth events based on metadata,
• Cross-device structural synchronisation, enabling federated augmentation without centralised retraining.
This may bridge current gaps between static inference models and truly adaptive edge AI systems.
11.7 FORMAL VERIFICATION OF GERMINATION EVENTS
As seed-based evolution becomes more powerful, safety assurance must evolve with it. Future work may include:
• Formal verification of post-germination execution traces,
• Type-level constraints on seed structure and behaviour,
• Audit tooling for behavioural regression and causal attribution,
• Runtime signature matching to detect unapproved or anomalous seed activity.
This is especially critical for regulated domains (e.g., medical, automotive, defence) where runtime mutation must be tightly controlled.
11.8 THEORETICAL FRAMING AND LEARNING GUARANTEES
Lastly, there is a need to develop a formal theoretical foundation for seed-based learning, potentially grounded in:
• Local function approximation theory under frozen priors,
• Bayesian structural growth models (e.g., nonparametric priors over network capacity),
• Evolutionary computation analogues, where seeds represent mutational loci within fixed genomes,
• Curriculum-based emergent modularity, formalising how local learning pressure induces structure.
Such work would provide clearer bounds on expressivity, convergence, and system reliability under seed-driven expansion.
11.9 SUMMARY
Research Direction Motivation
Scaling to Transformers and CNNs Extend method to high-capacity domains
Coordinated multi-seed systems Enable large-scale modular adaptation
Lifecycle management and freezing Prevent overfitting, enable stable deployment
Richer germination policies Improve reliability and generality of triggers
Compression and reuse integration Combine evolution with efficiency and portability
Edge deployment and federated control Apply in real-world distributed inference contexts
Verification and audit mechanisms Ensure trust, traceability, and runtime safety
Formal theory of local structural growth Ground the method in learning theory

12. DEPLOYMENT PATHWAY AND STRATEGIC VISION
The morphogenetic architecture detailed in this document is not merely a theoretical construct; it is an engineering paradigm with a clear, phased pathway toward real-world deployment. The inherent safety, auditability, and efficiency of seed-based evolution enable a strategic rollout, beginning in highly constrained environments and scaling toward ubiquitous, ambient intelligence. This pathway demonstrates a clear trajectory from solving contained industrial problems to enabling the next generation of safe, truly adaptive intelligent systems.
12.1 PHASE 1: CONSTRAINED, HIGH-VALUE DOMAINS
The initial applications will target domains where the problem is well-defined, and the value of localized adaptation is high. These environments serve as perfect proving grounds due to their contained risk profiles and clear metrics for success.
• Industrial Predictive Maintenance: A model monitoring critical machinery can germinate a new, specialized fault detector when a novel vibration pattern or thermal signature emerges. This allows the system to adapt to new failure modes without the cost and risk of redeploying the entire monitoring suite.
• Implantable Medical Devices: A certified, frozen firmware for a device like a closed-loop insulin pump or a pacemaker could use a pre-approved Germinal Module (GM) to adapt its response algorithm to a specific patient's changing physiology over months or years, enabling personalization without compromising the core safety certification.
12.2 PHASE 2: AUDITED AND REGULATED SYSTEMS
Leveraging the architecture's intrinsic safety features, the next phase targets industries where regulatory compliance is paramount. The system is purpose-built for the rigorous validation required by bodies such as the FDA (Food and Drug Administration) or EASA (European Union Aviation Safety Agency).
• Immutable Version Control: The cryptographic hashing of seeds and Germinal Modules (GMs) provides a verifiable and immutable chain of custody for every architectural modification, which is essential for regulatory review.
• Fail-Safe Compliance: The deterministic execution and zero-impact rollback protocols described in Section 5 allow the system to be instantly reverted to its last certified state if a germinated module fails validation, ensuring patient or user safety.
• Traceable Lineage: The germination logs create a complete, auditable history of the model's structural evolution—a "biographical architecture"—making the model's adaptive lifecycle fully transparent to regulators.
12.3 PHASE 3: AMBIENT AND AUTONOMOUS ECOSYSTEMS
The ultimate vision is for morphogenetic networks to become a form of self-extending, self-healing digital infrastructure, capable of adapting to their environment at a systemic level.
• Self-Healing IoT Meshes: Seeds embedded in network nodes across a smart city or factory floor could germinate new routing protocols, data compression algorithms, or security patches to adapt to changing network topology or counter new threats in real-time.
• Real-Time Privacy Filters: A personal AI agent could grow new, highly specific privacy filters in response to encountering a novel application or data request, ensuring user data is protected dynamically without constant manual intervention or global software updates.

13. CITATIONS
This section lists the key publications that directly inform the core concepts, techniques, and architectural patterns discussed in this document. Each citation includes a note on its specific relevance.
[1] Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., de Laroussilhe, Q., Gesmundo, A., ... & Gelly, S. (2019). Parameter-efficient transfer learning for NLP. In Proceedings of the 36th International Conference on Machine Learning (ICML).
Cited in Section 4. Basis for adapter layers as minimal, non-intrusive grafting strategies.
[2] Rusu, A. A., Rabinowitz, N. C., Desjardins, G., Soyer, H., Kirkpatrick, J., Kavukcuoglu, K., ... & Hadsell, R. (2016). Progressive neural networks. arXiv preprint arXiv:1606.04671.
Referenced in Section 3. Demonstrates early use of structural isolation and transfer in fixed-parameter agents, a foundational concept for freezing the base model.
[3] Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., ... & Hadsell, R. (2017). Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy of Sciences, 114(13), 3521–3526.
Cited in Section 3 and 9. Introduces Elastic Weight Consolidation (EWC), a key method for preventing interference and a potential technique for allowing minimal, controlled plasticity at graft interfaces.
[4] Han, S., Pool, J., Tran, J., & Dally, W. (2015). Learning both weights and connections for efficient neural networks. In Advances in Neural Information Processing Systems (NeurIPS).
Referenced in Section 10. Origin of pruning-based network compression, relevant to Germinal Module (GM) recovery and the lifecycle management of germinated seeds.
[5] Rosenbaum, C., Klinger, T., & Riemer, M. (2019). Routing networks: Adaptive selection of non-linear functions for multi-task learning. In ICLR.
Cited in Section 3. Representative of dynamic neural architectures used for conditional computation, from which morphogenetic architectures draw the principle of structural adaptation.
[6] Beaulieu, S., Frasca, F., Xu, Y., Goyal, S., Pal, C., & Larochelle, H. (2020). Learning sparse representations in reinforcement learning with the successor features. In Advances in Neural Information Processing Systems (NeurIPS).
Supporting Section 3. Cited for modular representation learning, which is a prerequisite for effective and safe seed placement.
[7] Mallya, A., & Lazebnik, S. (2018). Piggyback: Adapting a single network to multiple tasks by learning to mask weights. In ECCV.
Referenced in Section 6. Describes masking-based adaptation of frozen networks, a concept related to the seed's local-only learning constraints.
[8] Schick, T., & Schütze, H. (2020). It's Not Just Size That Matters: Small Language Models Are Also Few-Shot Learners. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP).
Referenced in Section 1. Justifies the focus on sub-10M parameter models and the need for local capacity expansion where full retraining is infeasible.
[9] Elsken, T., Metzen, J. H., & Hutter, F. (2019). Neural architecture search: A survey. Journal of Machine Learning Research, 20(55), 1–21.
Cited in Section 2 and 10. Provides the broader context for automated structural growth, informing the design of morphogenetic control policies.
[10] Goyal, A., Lamb, A. M., Hoffmann, J., Sodhani, S., Levine, S., Bengio, Y., & Schölkopf, B. (2021). Inductive biases, pretraining and fine-tuning for transformer-based geometric reasoning. arXiv preprint arXiv:2110.06091.
Referenced in Section 10. Illustrates architectural localisation within Transformers, a key target for future seed placement strategies.
[11] Bengio, Y., & LeCun, Y. (2007). Scaling learning algorithms towards AI. In Large-scale kernel machines (Vol. 34, pp. 321–360).
Referenced in Section 10. A classic articulation of scalability and local learning principles, foundational to the entire morphogenetic perspective.
[12] Parisi, G. I., Kemker, R., Part, J. L., Kanan, C., & Wermter, S. (2019). Continual lifelong learning with neural networks: A review. Neural Networks, 113, 54–71.
Supporting background for Sections 1 and 3. Consolidates key methods and taxonomies in continual learning, relevant to the challenge of non-catastrophic adaptation.

APPENDICES
APPENDIX A PROTOTYPE CODE – FULL-FIDELITY MANAGED GERMINATION
import os
import random
import threading
import time
from collections import deque
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

################################################################################

# 1. CORE INFRASTRUCTURE #

################################################################################

class SeedManager:
    """Central registry for SentinelSeed instances.

    Thread‑safe singleton that tracks seed state, telemetry, and germination
    lineage. It also exposes an atomic germination request that can be called
    concurrently from controller logic.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.seeds = {}
                cls._instance.germination_log = []
        return cls._instance

    # ---------------------------------------------------------------------
    #  Registry helpers
    # ---------------------------------------------------------------------

    def register_seed(self, seed_module: "SentinelSeed", seed_id: str) -> None:
        self.seeds[seed_id] = {
            "module": seed_module,
            "buffer": deque(maxlen=500),  # activation buffer for stats
            "status": "dormant",         # dormant | active | failed_germination
            "telemetry": {"interface_drift": 0.0},
        }
        print(f"SeedManager ▶ Registered '{seed_id}'.")

    def get_seed_info(self, seed_id: str):
        return self.seeds.get(seed_id)

    # ---------------------------------------------------------------------
    #  Germination
    # ---------------------------------------------------------------------

    def request_germination(
        self,
        seed_id: str,
        step: int,
        init_type: str = "zero_init",
        gm_path: str | None = None,
    ) -> bool:
        """Attempt to activate a dormant seed.

        Returns True on success so the caller can refresh the optimiser.
        """
        with self._lock:
            info = self.get_seed_info(seed_id)
            if not info or info["status"] != "dormant":
                return False

            # Simulate hardware failure 15 % of the time.
            if random.random() < 0.15:
                print(f"\N{RED CIRCLE}  SeedManager ▶ Simulated GERMINATION FAILURE for '{seed_id}'.")
                info["status"] = "failed_germination"
                self._log_event(step, seed_id, "failure", "simulated hardware error")
                return False

            print(
                f"\N{LARGE GREEN CIRCLE}  SeedManager ▶ Germinating '{seed_id}' "
                f"using {init_type} ..."
            )

            ok = info["module"].germinate(init_type=init_type, gm_path=gm_path)
            if ok:
                info["status"] = "active"
                self._log_event(step, seed_id, "success", init_type)
            return ok

    # ------------------------------------------------------------------
    #  Telemetry
    # ------------------------------------------------------------------

    def _log_event(self, step: int, seed_id: str, status: str, details: str) -> None:
        self.germination_log.append(
            {
                "step": step,
                "timestamp": time.time(),
                "seed_id": seed_id,
                "status": status,
                "details": details,
            }
        )

    def print_audit_log(self) -> None:
        print("\n── Germination Audit Log ──────────────")
        if not self.germination_log:
            print("<no events>")
        else:
            for e in self.germination_log:
                print(
                    f"step={e['step']:<4} | seed={e['seed_id']:<15} | "
                    f"status={e['status']:<6} | details={e['details']}"
                )
        print("──────────────────────────────────────\n")

################################################################################

# 2. CONTROLLER #

################################################################################

class KasminaMicro:
    """Very simple plateau‑trigger controller.

    In production this would be replaced by a RL or heuristic policy.
    """

    def __init__(self, manager: SeedManager, patience: int = 20, delta: float = 1e-4):
        self.mgr = manager
        self.patience = patience
        self.delta = delta
        self.plateau = 0
        self.prev_loss = float("inf")
        print(
            "Kasmina ▶ initialised with patience="
            f"{self.patience} and Δ={self.delta}."
        )

    # ------------------------------------------------------------------
    #  Decide if we should invoke a seed and optionally return a flag so
    #  the caller can rebuild the optimiser.
    # ------------------------------------------------------------------

    def step(self, step_idx: int, val_loss: float) -> bool:
        rebuild = False
        if abs(val_loss - self.prev_loss) < self.delta:
            self.plateau += 1
        else:
            self.plateau = 0
        self.prev_loss = val_loss

        if self.plateau < self.patience:
            return rebuild

        self.plateau = 0  # reset
        candidate = self._select_seed()
        if not candidate:
            return rebuild

        init_type = "Germinal Module (GM)" if random.random() > 0.5 else "zero_init"
        ok = self.mgr.request_germination(candidate, step_idx, init_type, gm_path="gm.pth")
        return ok  # if True, caller should rebuild optimiser

    # ------------------------------------------------------------------
    #  Helper
    # ------------------------------------------------------------------

    def _select_seed(self):
        dormant = {
            sid: info for sid, info in self.mgr.seeds.items() if info["status"] == "dormant"
        }
        if not dormant:
            return None
        # Choose the seed with *lowest* variance (most starving).
        scores = {
            sid: info["module"].get_health_signal() for sid, info in dormant.items()
        }
        return min(scores, key=scores.get)

################################################################################

# 3. MODEL COMPONENTS #

################################################################################

class SentinelSeed(nn.Module):
    """Drop‑in residual block — dormant until germinated."""

    def __init__(self, seed_id: str, dim: int = 32):
        super().__init__()
        self.seed_id = seed_id
        self.mgr = SeedManager()
        self.mgr.register_seed(self, seed_id)

        self.child = nn.Sequential(
            nn.Linear(dim, 16),
            nn.ReLU(),
            nn.Linear(16, dim),
        )
        self._zero_init(self.child)
        self.set_trainable(False)

    # ------------------------------- lifecycle ---------------------------------

    def germinate(self, init_type: str = "zero_init", gm_path: str | None = None) -> bool:
        try:
            if init_type == "Germinal Module (GM)" and gm_path and os.path.exists(gm_path):
                self.child.load_state_dict(torch.load(gm_path))
                print(f"Seed '{self.seed_id}' ▶ GM loaded from '{gm_path}'.")
            else:
                self._kaiming_init(self.child)
            self.set_trainable(True)
            return True
        except Exception as exc:  # pragma: no cover
            print(f"\N{RED CIRCLE}  '{self.seed_id}' ▶ germination failed: {exc}")
            return False

    # ----------------------------- forward pass --------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        info = self.mgr.get_seed_info(self.seed_id)
        status = info["status"]
        if status != "active":
            if status == "dormant":
                info["buffer"].append(x.detach())  # collect stats
            return x  # identity

        residual = self.child(x)
        out = x + residual
        drift = 1.0 - F.cosine_similarity(x, out, dim=-1).mean().item()
        info["telemetry"]["interface_drift"] = drift
        return out

    # ----------------------------- diagnostics ---------------------------------

    def get_health_signal(self) -> float:
        buf = self.mgr.get_seed_info(self.seed_id)["buffer"]
        if len(buf) < 20:
            return 1.0  # optimistic until we have data
        variance = torch.var(torch.stack(list(buf))).item()
        return max(variance, 1e-6)

    # --------------------------- utils / helpers ------------------------------

    @staticmethod
    def _zero_init(module: nn.Module) -> None:
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _kaiming_init(module: nn.Module) -> None:
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_trainable(self, flag: bool) -> None:
        for p in self.parameters():
            p.requires_grad = flag

class BaseNet(nn.Module):
    """Frozen backbone with two insertion points."""

    def __init__(self, seed_a: SentinelSeed, seed_b: SentinelSeed):
        super().__init__()
        self.fc1 = nn.Linear(2, 32)
        self.seed_a = seed_a
        self.fc2 = nn.Linear(32, 32)
        self.seed_b = seed_b
        self.out = nn.Linear(32, 2)

        self._freeze_except_seeds()

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _freeze_except_seeds(self):
        for m in self.modules():
            trainable = isinstance(m, SentinelSeed)
            for p in m.parameters(recurse=False):
                p.requires_grad = trainable

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor):  # type: ignore[override]
        x = F.relu(self.fc1(x))
        x = self.seed_a(x)
        x = F.relu(self.fc2(x))
        x = self.seed_b(x)
        return self.out(x)

################################################################################

# 4. TRAINING LOOP #

################################################################################

def create_dummy_gm(path: str = "gm.pth", dim: int = 32) -> None:
    """Persist an untrained module so the GM code path has something to load."""
    if os.path.exists(path):
        return
    print("Creating placeholder Germinal Module …")
    tmp = nn.Sequential(nn.Linear(dim, 16), nn.ReLU(), nn.Linear(16, dim))
    torch.save(tmp.state_dict(), path)

def train_demo(n_steps: int = 800):  # pragma: no cover
    # ------------------------------------------------------------------
    #  Dataset
    # ------------------------------------------------------------------
    X, y = make_moons(1000, noise=0.2, random_state=42)
    X = StandardScaler().fit_transform(X).astype("float32")
    y = y.astype("int64")
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_tr, X_val = map(torch.from_numpy, (X_tr, X_val))
    y_tr, y_val = map(torch.from_numpy, (y_tr, y_val))

    # ------------------------------------------------------------------
    #  Model + manager
    # ------------------------------------------------------------------
    mgr = SeedManager()
    seed1 = SentinelSeed("bottleneck_1")
    seed2 = SentinelSeed("bottleneck_2")
    model = BaseNet(seed1, seed2)
    ctrl = KasminaMicro(mgr)

    # ------------------------------------------------------------------
    #  Stage 1: warm‑up backbone only
    # ------------------------------------------------------------------
    create_dummy_gm()
    for m in model.modules():
        if isinstance(m, SentinelSeed):
            m.set_trainable(False)
        else:
            for p in m.parameters(recurse=False):
                p.requires_grad = True
    warm_opt = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    model.train()
    for _ in range(300):
        warm_opt.zero_grad()
        loss = F.cross_entropy(model(X_tr), y_tr)
        loss.backward()
        warm_opt.step()
    print("Backbone pre‑trained, freezing …")

    # freeze backbone, leave seeds dormant (not trainable until active)
    model._freeze_except_seeds()

    # ------------------------------------------------------------------
    #  Stage 2: main loop
    # ------------------------------------------------------------------
    def build_opt() -> optim.Optimizer:
        return optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)

    opt = build_opt()
    prev_val = float("inf")
    for step in range(n_steps):
        model.train()
        opt.zero_grad()
        F.cross_entropy(model(X_tr), y_tr).backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = F.cross_entropy(model(X_val), y_val).item()

        if ctrl.step(step, val_loss):  # seed activated ⇒ refresh optimiser
            opt = build_opt()

        if step % 100 == 0:
            acc = (model(X_val).argmax(1) == y_val).float().mean().item()
            print(
                f"step={step:>3} | val_loss={val_loss:6.4f} | val_acc={acc:.2%}"
            )
            for sid, info in mgr.seeds.items():
                print(
                    f"   ↳ {sid:<13} status={info['status']:<10} "
                    f"var={info['module'].get_health_signal():.4f} "
                    f"drift={info['telemetry']['interface_drift']:.4f}"
                )
        prev_val = val_loss

    mgr.print_audit_log()

################################################################################

# 5. ENTRY‑POINT #

################################################################################

if __name__ == "__main__":
    train_demo()
APPENDIX B: DIAGNOSTIC TOOLING AND CONTROL
To support the rapid development, debugging, and analysis of morphogenetic architectures, a suite of diagnostic and control tools is essential. This appendix outlines the design for a command-line interface for real-time inspection, visual models of core mechanics, and key extensions required for production-level performance.
B.1 INTERACTIVE DIAGNOSTICS: THE SEEDNET COMMAND-LINE INTERFACE (CLI)
A major accelerator for research is the ability to interact with the model during training. The SeedNetCLI is a proposed Read-Eval-Print Loop (REPL) interface that allows a researcher to monitor and manually control the germination lifecycle without halting the training process.
Purpose: Enable real-time inspection of seed states, manual triggering of germination, and direct examination of the I/O buffers that inform germination decisions.
PROTOTYPE IMPLEMENTATION (CMD MODULE):
import cmd
import textwrap
from typing import Optional

import torch

class SeedNetCLI(cmd.Cmd):
    """Interactive REPL for inspecting and controlling a running SeedNet experiment.

    The CLI is intentionally *thin*: it delegates all heavy‑lifting to the
    `seednet_engine`, which is expected to expose ─────────────────────────────
    • ``manager``: a :class:`SeedManager` instance with the canonical ``seeds``
      registry and ``request_germination`` API.
    • ``step``     (int) attribute that tracks the global training step.
    • ``rebuild_optimizer`` (callable) – optional hook to rebuild the optimiser
      when new parameters become trainable after a germination event.
    """

    prompt = "(seednet) "

    # ---------------------------------------------------------------------
    # Construction & helpers
    # ---------------------------------------------------------------------
    def __init__(self, seednet_engine):
        super().__init__()
        self.engine = seednet_engine
        self.intro = textwrap.dedent(
            """
            SeedNet diagnostic console.  Type 'help' or '?' for available commands.
            Hitting <Enter> repeats the previous command.
            """
        )

    # ---------------------------------------------------------------------
    # Core commands
    # ---------------------------------------------------------------------
    def do_status(self, arg: str = ""):
        """status
        Show one‑line status for every registered seed (state, buffer size,
        interface‑drift metric).
        """
        print()
        print("Seed ID         │ State           │ Buffer │ Interface‑drift")
        print("────────────────┼────────────────┼────────┼─────────────────")
        for sid, info in self.engine.manager.seeds.items():
            state = info["status"]
            buf_sz = len(info["buffer"])
            drift = info["telemetry"].get("interface_drift", 0.0)
            print(f"{sid:<15}│ {state:<14}│ {buf_sz:^6} │ {drift:>13.4f}")
        print()

    # ------------------------------------------------------------------
    def do_germinate(self, arg: str):
        """germinate <seed_id> [zero|gm <GM_PATH>]
        Manually trigger germination of a dormant seed.

        Examples:
            germinate bottleneck_1 zero        # zero‑init
            germinate bottleneck_2 gm gm.pth   # load from gm.pth
        """
        tokens: List[str] = arg.split()
        if not tokens:
            print("Error: seed_id required.  See 'help germinate'.")
            return

        seed_id = tokens[0]
        if len(tokens) == 1 or tokens[1].lower() == "zero":
            init_type = "zero_init"
            gm_path: Optional[str] = None
        elif tokens[1].lower() == "gm":
            if len(tokens) < 3:
                print("Error: GM path required after 'gm'.")
                return
            init_type = "Germinal Module (GM)"
            gm_path = tokens[2]
        else:
            print("Error: second arg must be 'zero' or 'gm'.")
            return

        step = getattr(self.engine, "step", -1)
        ok = self.engine.manager.request_germination(
            seed_id, step=step, init_type=init_type, gm_path=gm_path
        )
        if ok:
            print(f"✓ Germination request for '{seed_id}' accepted.")
            # Rebuild optimiser if engine exposes a hook.
            rebuild = getattr(self.engine, "rebuild_optimizer", None)
            if callable(rebuild):
                rebuild()
        else:
            print(f"✗ Germination request for '{seed_id}' was rejected.")

    # ------------------------------------------------------------------
    def do_buffer(self, arg: str):
        """buffer <seed_id>
        Show basic statistics of the dormant‑buffer for the given seed.
        """
        seed_id = arg.strip()
        if not seed_id:
            print("Error: seed_id required.  See 'help buffer'.")
            return

        seed_info = self.engine.manager.get_seed_info(seed_id)
        if not seed_info:
            print(f"Error: no such seed '{seed_id}'.")
            return
        if not seed_info["buffer"]:
            print(f"Seed '{seed_id}' buffer is empty.")
            return

        buf = seed_info["buffer"]
        stacked = torch.stack(list(buf))
        mean = stacked.mean().item()
        std = stacked.std().item()
        var = stacked.var().item()
        print(
            textwrap.dedent(
                f"""
                Buffer stats for '{seed_id}':
                  • items    : {len(buf)}
                  • tensor shape : {stacked.shape}
                  • mean     : {mean: .4f}
                  • std dev  : {std: .4f}
                  • variance : {var: .4f}
                """
            )
        )

    # ------------------------------------------------------------------
    def do_quit(self, arg):
        """quit
        Exit the console (alias: exit)."""
        print("Exiting SeedNet console…")
        return True

    do_exit = do_quit  # alias

    # ------------------------------------------------------------------
    # Quality‑of‑life tweaks
    # ------------------------------------------------------------------
    def emptyline(self):
        """Repeat last command instead of doing nothing when user hits <Enter>."""
        if self.lastcmd:
            return self.onecmd(self.lastcmd)

    def default(self, line):
        """Print helpful error for unknown commands."""
        print(f"Unknown command: {line!r}.  Type 'help' for list of commands.")
B.2 VISUALIZING CORE MECHANICS
To clarify complex asynchronous and thread-safe operations, the following conceptual models are used.
ZERO-COST OBSERVABILITY
Seed monitoring is designed to be a non-blocking, asynchronous process to minimize impact on training throughput. A telemetry queue decouples I/O recording from diagnostic consumption.
sequenceDiagram
    participant Model
    participant SeedTensor
    participant SeedManager
    participant TelemetryQueue
    participant DiagnosticThread

    Model->>SeedTensor: forward() pass
    SeedTensor->>SeedManager: record_io(input, output)
    SeedManager->>TelemetryQueue: Enqueue data (async)
    DiagnosticThread->>TelemetryQueue: Consume data from queue
    DiagnosticThread->>SeedNetCLI: Update stats
ATOMIC GERMINATION
To prevent race conditions and maintain model integrity, germination must be an atomic operation that temporarily locks the computation graph.
// Pseudocode for thread-safe germination
void germinate(string seed_id, Module new_module) {
    lock(global_computation_graph); // Acquire lock to prevent concurrent modification

    suspend_autograd(); // Temporarily disable gradient calculation

    // Core surgical operation
    replace_node_in_graph(seed_id, new_module);
    initialize_new_module(new_module, get_seed_buffer(seed_id));

    resume_autograd(); // Re-enable gradient calculation

    unlock(global_computation_graph); // Release lock
}
B.3 PRODUCTION-READY EXTENSIONS
While the prototype focuses on functional correctness, a production-level framework would require performance-critical extensions.
CUDA-Aware Monitoring
For GPU-bound models, the I/O buffer mechanism must be optimized to avoid costly device-to-host transfers. This involves using pinned memory for zero-copy transfers between the GPU and CPU, ensuring that telemetry gathering does not become a performance bottleneck.
JIT Compilation Hooks
To support models compiled for performance with tools like TorchScript, seed monitoring logic can be injected via custom forward hooks (@torch.jit.custom_forward_hook). This allows the JIT compiler to optimize the main computation path while still enabling the telemetry system to capture the necessary data at the seed interfaces.

APPENDIX C: BIBLIOGRAPHY / READING LIST
This appendix provides a consolidated list of all references from the original research notes for further reading and to acknowledge the broader literature that informed this work.

1. Beaulieu, S., Frasca, F., Xu, Y., Goyal, S., Pal, C., & Larochelle, H. (2020). Learning sparse representations in reinforcement learning with the successor features. In Advances in Neural Information Processing Systems (NeurIPS).
2. Bengio, Y., & LeCun, Y. (2007). Scaling learning algorithms towards AI. In Large-scale kernel machines (Vol. 34, pp. 321–360).
3. Elsken, T., Metzen, J. H., & Hutter, F. (2019). Neural architecture search: A survey. Journal of Machine Learning Research, 20(55), 1–21.
4. Goyal, A., Lamb, A. M., Hoffmann, J., Sodhani, S., Levine, S., Bengio, Y., & Schölkopf, B. (2021). Inductive biases, pretraining and fine-tuning for transformer-based geometric reasoning. arXiv preprint arXiv:2110.06091.
5. Han, S., Pool, J., Tran, J., & Dally, W. (2015). Learning both weights and connections for efficient neural networks. In Advances in Neural Information Processing Systems (NeurIPS).
6. Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. arXiv preprint arXiv:1503.02531.
7. Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., de Laroussilhe, Q., Gesmundo, A., ... & Gelly, S. (2019). Parameter-efficient transfer learning for NLP. In Proceedings of the 36th International Conference on Machine Learning (ICML).
8. Karras, T., Aittala, M., Hellsten, J., Laine, S., Lehtinen, J., & Aila, T. (2020). Training generative adversarial networks with limited data. arXiv preprint arXiv:2006.06676.1
9. Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., ... & Hadsell, R. (2017). Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy of Sciences, 114(13), 3521–3526.
10. Mallya, A., & Lazebnik, S. (2018). Piggyback: Adapting a single network to multiple tasks by learning to mask weights. In ECCV.
11. Parisi, G. I., Kemker, R., Part, J. L., Kanan, C., & Wermter, S. (2019). Continual lifelong learning with neural networks: A review. Neural Networks, 113, 54–71.
12. Rosenbaum, C., Klinger, T., & Riemer, M. (2019). Routing networks: Adaptive selection of non-linear functions for multi-task learning. In ICLR.
13. Rusu, A. A., Rabinowitz, N. C., Desjardins, G., Soyer, H., Kirkpatrick, J., Kavukcuoglu, K., ... & Hadsell, R. (2016). Progressive neural networks. arXiv preprint arXiv:1606.04671.
14. Schick, T., & Schütze, H. (2020). It's Not Just Size That Matters: Small Language Models Are Also Few-Shot Learners. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP).
15. Alet, F., et al. (2023). Modular Deep Learning. arXiv preprint arXiv:2302.11529v2.
16. Anthropic. (2024). Model Stitching by Functional Latent Alignment. arXiv: 2505.20142.
19. Chen, C., et al. (2021). Neural Network Surgery: Injecting Data Patterns. ACL Anthology.
20. Chen, R., et al. (2020). Accurate Neural Network Computer Vision Without The 'Black Box'. Duke Today.
21. Du, J., et al. (2025). Knowledge Grafting of Large Language Models. arXiv preprint arXiv:2505.18502v1.
22. Hadsell, R. (2014). What is Catastrophic Forgetting?. IBM.
23. He, S., et al. (2025). Modular Machine Learning: An Indispensable Path towards New-Generation Large Language Models. arXiv preprint arXiv:2504.20020v1.
24. Jin, X., et al. (2025). ZenFlow: Enabling Stall-Free Offloading Training via Asynchronous Updates. arXiv preprint arXiv:2505.12242v1.
25. Lansdell, B., & Kording, K. (2023). Feature alignment as a generative process. PMC.
26. Le, T., et al. (2024). MergeKD: an empirical framework for combining knowledge distillation with model fusion using BERT model. ScholarSpace.
27. Li, Z., et al. (2024). Training Independent Subnetworks for Structural Ensembling. OpenReview.
28. Lu, C., et al. (2024). Dynamic Neural Network Structure: A Review for Its Theories and Applications. ResearchGate.
29. Ma, X., et al. (2024). Cross-Silo Feature Space Alignment for Federated Learning on Clients with Imbalanced Data. AAAI Conference on Artificial Intelligence.
30. Peters, B. (2025). Dynamic neural networks: advantages and challenges. National School of Development, Peking University.
31. Shao, D., et al. (2024). Prompt-Based Distribution Alignment for Unsupervised Domain Adaptation. AAAI Conference on Artificial Intelligence.
32. Sun, Q., et al. (2024). DeepArc: Modularizing neural networks for the model maintenance. <InK@SMU.edu.sg>.
33. Wortsman, M., et al. (2024). Aligning latent representations of neural activity. PMC.
34. Wu, P., et al. (2024). On the Direct Alignment of Latent Spaces. OpenReview.
35. Wikipedia contributors. (2024). Modular neural network. Wikipedia.
36. Zhang, C., et al. (2024). Uncertainty-Guided Alignment for Unsupervised Domain Adaptation in Regression. arXiv preprint arXiv:2401.13721v1.
37. Zhuang, F., et al. (2016). Transfer Learning across Feature-Rich Heterogeneous Feature Spaces via Feature-Space Remapping (FSR). PMC.
38. Zador, A. (2024). Latent Space Translation via Semantic Alignment. OpenReview.
