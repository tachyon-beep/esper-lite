# Esper: Architectural Ecology in Neural Networks

**Draft Discussion Paper on Morphogenetic AI (Revised v2.2)**

## Abstract

Contemporary deep learning largely follows a paradigm of **architectural engineering**: models are statically designed, initialised, and trained as monolithic blocks. Neural Architecture Search (NAS) automates parts of this process, but typically remains a discrete, offline optimisation loop.

**Esper** proposes a shift to **architectural ecology**: an online, continuous process where neural modules (seeds) germinate, compete, and stabilise inside a living host network during training. Esper separates concerns across a **substrate** that executes and prices decisions, and an **organism layer** that grows and prunes structure online.

**Organism (agents and organs)**

1. **Kasmina (The Body):** A differentiable host that supports controlled growth via SeedSlots and blending.
2. **Tamiyo (The Brain, Growth Policy):** A policy that germinates and develops *Kasmina seeds* during training.
3. **Emrakul (The Immune System, Decay Policy):** Custodian of committed structure, enforcing long-term ROI via probe-and-lysis protocols.

**Substrate (infrastructure and physics)**
4. **Tolaria (The Metabolism):** The training engine for Model Alpha: high-throughput execution, deterministic replay, and flight-recorded data capture. Includes an internal safety governor for NaNs and divergence rollback.
5. **Simic (The Selective Pressure):** The economy and accounting layer that prices complexity (Rent and Churn) and provides credit signals (including Shapley audits in Phase 1).

**Kasmina seeds** and **phages** are peer primitives. Seeds are creative tissue (can become anything, Tamiyo-managed), while phages are infrastructure wrappers (measured, priced, lysable, Emrakul-managed). Fossilisation rewraps a seed as a phage and transfers custodianship.

A key clarification is that Esper is intended to be **two-timescale**:

* In **Phase 1 ("train the trainer")**, we deliberately spend compute on audited credit assignment (including exact, full-retrain Shapley values on small candidate sets) to teach Tamiyo what “good growth” looks like under a fixed protocol.
* In **deployment**, a trained Tamiyo is intended to grow many new models **without** the Shapley harness, using only cheap online signals and learned critics. This amortises Phase 1 scaffolding across downstream runs.

**Key point:** Shapley values are used as *training-only teacher labels* to shape learning updates. They are **not** included in Tamiyo’s observation space, which remains identical to deployment telemetry. This prevents privileged-information leakage between training and deployment.

---

## 1. Introduction: From Fossils to Flora

A standard ResNet or Transformer is a **fossil**: its skeletal structure is fixed before the first gradient is calculated. If the task proves more complex than anticipated, the model cannot grow. If the task is simpler, the model wastes compute on redundant parameters.

We argue that network topology should be a function of training history, not solely its creator’s intuition. Biology does not build a brain and then switch it on. The brain grows in response to stimuli while paying ongoing metabolic costs.

Esper aims to build an environment where networks can grow safely, and where growth is not free. Seeds are introduced, evaluated under selective pressure, and either integrated (**Committed**) or removed (**Pruned**). The focus is not “the perfect architecture”, but the **physics of a system** in which useful architectures can emerge under cost and stability constraints.

### 1.1 Two-timescale learning: training models vs training Tamiyo

A common failure mode in describing morphogenetic systems is accidentally selling them as “we trained many expensive models”. Esper is not primarily about producing thousands of partially trained CIFAR-10 models.

Esper is about training a **reusable architect policy**:

* **Inner loop (model training):** Kasmina’s weights learn the task while the architecture changes online.
* **Outer loop (policy training):** Tamiyo learns *how to cause beneficial architectural change* across runs, checkpoints, and hosts.

In Phase 1, expensive audits (for example Shapley on small candidate sets) act as teacher labels. The intended outcome is a Tamiyo that can later be deployed to grow new models **without** repeating full audit machinery.

---

## 2. System Overview and Terminology

### 2.1 The loop (online morphogenesis)

At a high level, Esper runs a closed feedback loop across substrate and organism.

**Substrate (executes and prices):**

1. **Tolaria** executes training and evaluation, enforces determinism, records events, and rolls back on divergence.
2. **Simic** defines rewards and costs, computes rent and churn, and produces credit labels (Phase 1 audits only).

**Organism (decides and acts):**
3.  **Kasmina** performs forward and backward passes while supporting dormant and active SeedSlots.
4.  **Tamiyo** observes training dynamics and slot states, then chooses **growth** actions (germinate, blend, commit).
5.  **Emrakul** observes phage ledgers and global stress, then chooses **decay** actions (probe, sedate, lyse).

**Schematic:**

```text
Data -> Tolaria (training engine incl. governor) -> Kasmina (Host + Seeds/Phages) -> Loss / Metrics
                     |                                          ^
                     v                                          |
              Slot and wrapper state                  Tamiyo (growth) + Emrakul (decay)
                     |                                          ^
                     v                                          |
                        Simic (Reward + Accounting + Phase 1 Audits)

```

### 2.2 Train-the-trainer vs deploy-the-trainer

Esper has two operational modes.

**Train-the-trainer (Phase 1):**

* Instrumented runs generate high-quality credit signals, including expensive audits on small candidate sets.
* Tamiyo and its critics are trained to predict value and credit, and to choose actions reliably.
* Emrakul’s pruning components are calibrated using probe outcomes and optional short-horizon ablation sweeps (Section 9.4), not full retrain Shapley.

**Deployment (Phase 2):**

* Tamiyo grows new models with **cheap online signals** (loss trends, gradient statistics, gate stability, cost signals, critic predictions), without full retrain Shapley.
* Emrakul maintains efficiency using **probe-and-lysis protocols**, primarily via reversible sedation and ledger-based safety.

A useful cost framing is:

Where  is the one-time Phase 1 meta-training and auditing cost,  is the cost per downstream run, and  is the number of downstream runs.

### 2.3 Glossary (working definitions)

**Substrate terms**

* **Host:** The base backbone (for example ResNet, Transformer) plus insertion points for seeds and phage-wrapped infrastructure.
* **Tolaria:** The training engine for Model Alpha, responsible for high-throughput execution, determinism and replay, and event-sourced telemetry capture.
* **TolariaGovernor:** A safety subsystem inside Tolaria that monitors training health (NaNs, divergence, loss explosions) and can trigger rollback or emergency interventions.
* **Rent:** The ongoing cost of keeping a structure active (compute, memory, latency proxy). Defined by Simic.
* **Churn:** A penalty for structural volatility (rapid add and remove, oscillating gates, frequent state flips). Defined by Simic.
* **Shapley value (φ):** A principled attribution of marginal contribution that shares interaction gains across participants.

**Organism terms**

* **SeedSlot:** A pre-allocated insertion point. When dormant, it behaves as an identity (or no-op) under a defined contract. The heavy wrapper for growth.
* **Kasmina seed:** A growable module instantiated in a SeedSlot during exploration and development. Creative tissue that can become anything, Tamiyo-managed.
* **Phage (wrapper):** A standard infrastructure wrapper that exposes telemetry and a lysis interface. Committed seeds are rewrapped as phage-wrapped infrastructure.
* **Blueprint:** The family or type of module a seed can become (the genome library).
* **Alpha (α):** A continuous gate controlling a seed’s contribution to the host forward path (SeedSlot only). Phages run at effective .

**Lifecycle terms**

* **Committed (Tamiyo-locked):** A seed state where Tamiyo will not modify the slot further. Custodianship transfers to Emrakul.
* **Compacted:** A physical rewrite step (optional) that removes edit machinery and fuses committed structure at a safe boundary (end of training or checkpoint compaction).
* **Lysis:** Controlled dissolution via a **sedation mask** (distinct from ). Phages expose `apply_sedation_mask()` for soft gating and `physical_lysis()` for reclamation at safe boundaries.
* **ScarSlot:** An identity SeedSlot left after lysis, with metadata (age, last blueprint) and quarantine, used as a regrowth surface.
* **Emrakul:** The immune system policy that controls phages and decides when to trigger lysis.

**Value terms**

* **Online return:** Discounted reward used by PPO (typically , GAE ) for short-horizon control and stability.
* **Audit value :** Used for Phase 1 Shapley labels: a fixed training budget under a specified protocol with a final evaluation metric (for example validation loss).

### 2.4 Seed lifecycle (state machine)

Esper treats growth as a controlled state machine rather than ad hoc graph surgery:

1. **Dormant:** Slot is identity, no effect ().
2. **Germinated:** Module instantiated, sanity checks complete.
3. **Training (Incubator):** Seed receives surrogate gradients via STE while host forward-path is unchanged.
4. **Blending:** Seed is gradually introduced via .
5. **Holding:** Final validation period before commitment.
6. **Committed (Tamiyo-locked):** Tamiyo will not modify this slot thereafter. The module is rewrapped as a phage and becomes subject to Emrakul. Not yet compacted.
7. **Pruned/Lysed → ScarSlot → Resetting:** Upon Lysis, the module collapses into a ScarSlot (identity with metadata). Tamiyo may perform "Trauma Surgery" (regrowth) or allow it to fade into a Dormant state.

```text
DORMANT → GERMINATED → TRAINING → BLENDING → HOLDING → COMMITTED (Tamiyo-locked)
   ^                      │          │          │               │
   │                      │          │          │               v
   │                      │          │          │        (rewrap) PHAGE-WRAPPED
   │                      │          │          │               │
   │                      │          │          │        (Emrakul) LYSIS
   │                      │          │          │               │
   │                      └──────────┴──────────┴─────────────→ SCARSLOT → DORMANT ─┘

```

**Authority over lifecycle transitions**

| Actor | Can modify ? | Can change lifecycle? | Scope |
| --- | --- | --- | --- |
| **Tamiyo** | Yes, pre-commit | Yes, up to Commit (including Prune) | Seeds in Dormant through Holding |
| **Emrakul** | No (uses sedation mask) | Yes (lysis only) | Committed seeds (phage-wrapped) |
| **Tolaria** | Yes (emergency) | Yes (rollback, emergency lysis) | Anything, catastrophic triggers only |

---

## 3. Physiology: Kasmina as a Morphogenetic Host

### 3.1 The morphogenetic plane (SeedSlots)

Instead of cutting and pasting network graphs mid-training, Esper uses pre-allocated **SeedSlots** placed at structurally meaningful points (for example residual branches, MLP blocks, attention subpaths). Each SeedSlot is an identity when dormant, ensuring the host’s baseline function is intact.

A typical residual-style integration is:

Where:

* is the host path
* is the seed module
* gates how “real” the seed is

This turns “adding a module” into “activating a pre-existing dormant organ”.

### 3.2 The incubator: gradient isolation via Straight-Through Estimator

A recurring failure mode in naive morphogenesis is destabilisation: a newly initialised seed produces noisy outputs that disrupt the host.

Esper mitigates this via **Gradient Isolation**. During the Training stage, the seed receives the same inputs as it would in production, but its output does not affect the host’s forward computation.

Implementation uses a **Straight-Through Estimator (STE)**:

```python
# Forward: returns host_features (seed contribution cancels out)
# Backward: gradients flow to both host AND seed parameters
return host_features + (seed_features - seed_features.detach())

```

This provides a surrogate gradient so the seed receives task-aligned gradient signal while its forward contribution is cancelled. Training behaves like  in the forward pass and approximately  in the backward pass.

**Additional isolation:** Host input into the seed path is detached during Training, ensuring host gradients remain identical to a host-only model.

**Maturity gate (G2):** A `GradientHealthMonitor` tracks the seed’s gradient-to-host ratio. Maturity gating uses gradient health plus a bounded activation-stat probe under a small test blend to reduce blend-time distribution shock. Once the seed passes stability criteria (bounded norm, healthy gradients, acceptable blend probe), it transitions to **Blending**.

### 3.3 Blending and gate schedules

Blending is the controlled ramp of  from 0 towards an operational value. This is the primary safety valve against destabilisation.
**Practical notes:**

* can be scheduled (handcrafted ramps) or policy-controlled (Tamiyo selects step sizes).
* Rent can be made -dependent to prevent “gate gaming” (paying almost nothing at tiny  while still extracting benefit).

---

## 4. Metabolism: Tolaria and High Throughput

### 4.1 Inverted control flow

Morphogenesis multiplies evaluation. Python stepping and the GIL become bottlenecks.

Tolaria implements **inverted control flow**: the high-performance execution engine drives training and evaluation rather than a Python agent stepping the environment. Multiple environments, candidate sets, and evaluation branches can be executed in parallel, treating the GPU as a batch-processing metabolic substrate.

### 4.2 Vectorised determinism (the determinism contract)

A core engineering goal is replayability. If a specific architecture emerges, we should be able to reproduce the history that produced it.

**Determinism contract (intended):** Given fixed seeds, fixed data order and augmentations, and deterministic kernel settings, the morphogenetic stack (including seed germination decisions and policy updates) replays identically.

**Boundary conditions:** In practice, determinism depends on framework and kernel settings, reduction order, and careful RNG handling. Esper treats determinism as a first-class system feature: RNG states, action events, and relevant execution settings are logged so that replay drift becomes diagnosable rather than mysterious.

---

## 5. Selection Pressure: Simic’s Economy

### 5.1 The rent and churn economy

Tamiyo is not rewarded solely for task accuracy. It plays an ROI game: improve task performance while paying ongoing costs.

A generic shaping form:

To make this auditable, rent can be expressed in parameter-proxy units with logarithmic scaling:

Where:

* `BaseSlotRent` : a fixed overhead per occupied slot
* `P_seed`: seed parameter count
* `P_host`: host parameter count
* Log scaling prevents runaway penalties while preserving sensitivity to growth

Rent penalty may be capped (`max_rent`) to avoid overwhelming the task signal.

**Churn** penalises structural volatility (edits per window, oscillations, frequent state flips). Its purpose is to discourage tax-loophole behaviours such as rapid add/remove cycles or  oscillation around thresholds.

### 5.2 Value functions and horizons

Esper uses two value definitions on different timescales:

1. **Online return** is the discounted reward used by PPO (for example ) for short-horizon control and stability.
2. **Audit value** is used for Phase 1 Shapley labels: a fixed training budget under a specified protocol with a final evaluation metric.

Phase 1 learns critics that predict audit credit from cheap online telemetry, enabling deployment without repeated Shapley runs.

### 5.3 Credit assignment: exact Shapley as a calibration oracle (Phase 1)

Credit assignment is hard because contributions are contextual. A seed can be useful only in combination with others, or only at a particular stage of training.

For small candidate sets, Esper uses exact Shapley values as an **oracle-style calibration signal**. Given a candidate set  of size  and a value function  over subsets , the Shapley value for seed  is:

**Phase 1 protocol (example, CIFAR-10, small n):**

* For each subset , train the model under a fixed budget with only the seeds in  enabled.
* Evaluate  via a fixed metric (validation accuracy or loss).
* Compute  across all subsets.

This is expensive, but provides an audit trail.

**Shapley-as-teacher (no observation leakage):** Shapley values  are treated as privileged **labels**, not state. Tamiyo receives only deployment telemetry in its observations. Shapley is computed offline and joined post hoc to recorded events to shape learning targets (reward relabelling, advantage correction, critic supervision). Acting never conditions on Shapley.

### 5.4 From oracle to deployment: distillation and critics

The intended path is to convert expensive oracle labels into cheap deployment behaviour:

1. Shapley-labelled runs become a dataset.
2. Critics learn to predict credit or cost-adjusted advantage from telemetry.
3. Tamiyo’s policy is trained to act using critic outputs and cheap online features.
4. Deployment uses the trained policy and critics, not the Shapley harness.

---

## 6. Tamiyo: Policies and Scaling

Tamiyo uses a **factored action space** with 8 independent heads, enabling compositional control:

| Head | Options | Purpose |
| --- | --- | --- |
| `slot` | N slots | Which slot to target |
| `blueprint` | 13 types | Module type to germinate (NOOP, CONV_LIGHT, ATTENTION, NORM, etc.) |
| `style` | 4 styles | Germination style combining blend algorithm and alpha algorithm |
| `tempo` | 3 speeds | Blending tempo (FAST 3 epochs, STANDARD 5, SLOW 8) |
| `alpha_target` | 3 values | Target alpha amplitude (0.5, 0.7, 1.0) |
| `alpha_speed` | 4 speeds | Schedule speed in controller ticks |
| `alpha_curve` | 5 curves | Schedule curve (LINEAR, COSINE, SIGMOID variants) |
| `op` | 6 operations | Lifecycle operation |

**Lifecycle operations:**

* `WAIT`
* `GERMINATE`
* `SET_ALPHA_TARGET`
* `ADVANCE`
* `PRUNE`
* `COMMIT`

Scaling to large networks is treated as a curriculum problem. A policy that does “2 seeds, 3 slots on CIFAR-10” well is not automatically a policy that can manage 1000 seeds across diverse hosts.

**A plausible scaling path:**

1. Oracle bootcamp (small , Shapley audits)
2. Distillation (critics to predict  or cost-adjusted advantage)
3. Candidate selection (Scouts propose  candidates for expensive eval)
4. Approximate credit (Monte Carlo Shapley, leave-one-out, short proxies, learned critics), validated by periodic audits
5. Full scale (many slots and blueprints, bounded by shortlists and occasional audits)

---

## 7. Case Study: The Rise of the Norm (Phase 1, CIFAR-10)

In early CIFAR-10 runs we observed **Norm Dominance**. Given a blueprint library (Conv, Attention, Norm), Tamiyo strongly preferred normalisation layers.

* **Low rent**
* **High stability**
* **Interpretable:** policy exploits optimisation physics, buying stability before buying capacity

This validates that the economy is real (Tamiyo optimises ROI), and warns that local optima and reward hacking are default behaviours.
We propose reporting not only best-run accuracy, but **reliability under fixed budget**, for example median and lower-quantile accuracy across seeds at matched compute.

| Setting | Task metric (median / q25) | Added params | FLOPs delta | Blueprint distribution | Notes |
| --- | --- | --- | --- | --- | --- |
| Baseline host | TBD | 0 | 0 | none |  |
| Esper (low rent) | TBD | TBD | TBD | TBD | expect more growth |
| Esper (high rent) | TBD | TBD | TBD | TBD | expect norm-heavy |
| No churn penalty | TBD | TBD | TBD | TBD | expect oscillation |

---

## 8. Engineering for Learning: The Flight Recorder

Because Tolaria is designed for replayability, Esper treats runs as dataset generation, not just online training.

### 8.1 Event-sourced logging

Every Tamiyo decision is logged as an immutable event. Currently implemented:

* explicit action (seed, slot, blueprint, gate changes)
* slot state at decision time
* policy outputs (confidence, per-head entropies)
* expected value  and TD advantage (computed later)
* delayed teacher labels joined post hoc (Shapley , oracle rewards, long-horizon outcomes)

Teacher labels are joined post hoc and used for learning targets. They are not emitted as online observations.

**Planned:**

* observation vectors (pre-action and post-action)
* action masks and full logit vectors
* identifiers for specs and versions (observation spec, reward spec, seed library, host manifest)

### 8.2 Forensic replay and counterfactuals

Combined with determinism, the Flight Recorder enables forensic replay:

* reload exact universe state at decision step
* inject a counterfactual action
* replay forward to observe divergence

**Current:** determinism infrastructure exists; episode-end counterfactual analysis exists. Full step-level forensic replay is planned.

---

## 9. Emrakul: The Immune System (Probe-and-Lysis)

Esper’s growth story is incomplete without decay. **Emrakul** is the peer policy to Tamiyo. Tamiyo controls growth (seeds) via prediction; Emrakul controls decay (phages) via empirical evidence.

Pruning is asymmetric. Growth is reversible and stabilised by blending. Deletion can silently remove a capability that only becomes obvious long after the event. Emrakul therefore does not rely on guesses from telemetry alone. She **probes, validates, and lyses**.

### 9.1 The Magistrate pattern: phage wrappers as local controllers

Rather than treating Emrakul as a single agent that micromanages every parameter, Esper uses a **Magistrate pattern**:

1. **Emrakul (global):** emits a scalar metabolic stress signal () that expresses how strongly efficiency is valued relative to redundancy.
2. **Phage wrappers (local):** run a lightweight, shared policy per guild (the **Magistrate**) that interprets local telemetry and outputs **control laws** (thresholds and severities), not individual per-weight decisions.

This design keeps the global action space small and pushes local detail into standardised wrappers.

### 9.2 Saliency estimation: first-order Taylor proxy (deployment-safe)

To prioritise what to probe and sedate, phages use a cheap saliency proxy derived from local gradients.
For a weight  with gradient , a first-order estimate of loss change under deletion (setting ) is:

For ranking, use magnitude:

For a subtarget group  (for example a channel, head, neuron-group), use:

**Important:** Most deployments should default to **grouped subtargets** (channels, heads, neuron-groups), not raw parameter-level pruning. Parameter-level masking can be enabled as a research mode, but it is not assumed to be the default.

The Magistrate observes summary statistics of saliency (mean, variance, sparsity, top-k mass) and stress (), then outputs:

* : saliency threshold for candidate selection
* : severity or steepness for soft-gating (how aggressively to sedate)

### 9.3 Defining “contribution” without expensive counterfactuals

Emrakul’s ledger requires a definition of contribution that does not rely on expensive counterfactual retrains. Esper supports two deployment-safe estimators:

1. **Probe delta (primary):** During sedation probes (Section 9.4), measure the short-horizon change in loss trend and stability metrics. This is treated as empirical evidence of marginal utility or redundancy.
2. **Critic prediction (optional):** A learned utility critic trained during Phase 1 predicts expected probe harm or benefit from local telemetry.

This keeps the deployment contract aligned: no full Shapley, no privileged labels in observations.

### 9.4 The probe-and-lysis protocol

Phages do not delete immediately. They test hypotheses.

**Stage 1: Sedation probe (reversible)**

* **Trigger:** high metabolic debt (defined below) or low saliency mass under stress ().
* **Action:** apply a sedation mask to selected subtargets below threshold ().
* **Probation:** hold sedation for a window of  steps while monitoring:
* loss trend delta (EWMA)
* gradient norm and gradient noise changes at the phage boundary
* activation distribution stability
* proxy cost reduction (rent proxy, step-time proxy if available)

**Stage 2: Immune ledger (persistent memory)**
Each phage maintains a non-differentiable ledger:

| Ledger item | Meaning | Update behaviour |
| --- | --- | --- |
| **Metabolic debt** | accumulated wasted rent | increases when contribution estimate < rent, decays slowly |
| **Antibodies** | confidence that subtargets are redundant | increases when probes cause negligible harm |
| **Criticality lock** | do-not-touch safety flag | set if probe causes a spike (loss explosion, instability), decays or cools down |

This prevents oscillation (sedate, regret, unsedate, repeat) and turns pruning into evidence accumulation.

**Stage 3: Physical lysis (irreversible, safe boundary only)**
Only when antibodies exceed a safety threshold and no criticality lock is active does the system schedule **physical lysis**:

* weights are permanently removed, or tensors are compacted or reshaped
* performed only at safe boundaries (checkpoint compaction, end of training, or a governed maintenance step)
* **Creation of ScarSlot:** Upon lysis, the structure is replaced by a ScarSlot (identity + metadata) which serves as a potential regrowth surface for Tamiyo.

### 9.5 Tolaria as an optional ablation lab (not Phase 1 Shapley)

To accelerate immune calibration without destabilising the main run, Tolaria can optionally perform **batched ablation sweeps** at checkpoints:

1. Fork  short clones of the current state.
2. Apply different sedation masks to each clone (different phages or subtargets).
3. Run a short safety horizon (for example 50 steps).
4. Use outcomes to update antibody and criticality signals.
5. Discard clones and return to the main line.

These sweeps are **optional diagnostics** and can be disabled in deployment. They are short-horizon safety probes, not the Phase 1 Shapley harness.

### 9.6 The granularity ladder (default: micro-surgery)

Emrakul follows a hierarchy of intervention by blast radius and safety:

1. **Micro-surgery (default):** heads, channels, neuron-groups (low blast radius, hardware-friendly)
2. **Compression maintenance:** low-rank factorisation, merge near-identical branches, quantisation of stable phages
3. **Module lysis:** remove entire module (highest blast radius, requires strongest evidence)

Parameter-level unstructured pruning is considered research mode, not the default.

---

## 11. Esper Master Roadmap: Tamiyo (growth) and Emrakul (decay)

This is a capability roadmap, not a calendar. Each horizon describes what the system can reliably do, plus the infrastructure needed before moving on. The demo stages turn that capability roadmap into a sequence of storyboardable, auditable milestones.

### 11.1 System premise and contracts

**Operating premise:**
You set the physics (topology boundaries, determinism, observability). The agents discover tactics. You validate by Pareto frontier movement and robustness, not by vibes or cherry-picked hero runs.

**Core safety and determinism contract:**

* **Topology changes happen only at predefined boundaries:** SeedSlots, phage ports, safe compaction boundaries.
* **“Micro-surgery beats amputation”:** smaller blast radius than module deletion whenever possible.
* **Hardware-aware sparsity or it is theatre:** if it cannot run faster in real step-time, it does not count.

### 11.2 Summary table (capability horizons)

| Horizon | Tamiyo (growth) | Emrakul (decay) | Core contract |
| --- | --- | --- | --- |
| **Near term** | Module-level seeds in SeedSlots, learns development programmes (incubate, blend, scaffold, commit) | Module-level probe-and-lysis on phages, leaves a **ScarSlot** (SeedSlot-like regrowth surface) | Topology changes only at predefined boundaries (SeedSlots, phage ports, safe compaction) |
| **Mid term** | Submodule morphogenesis (heads, channels, neuron groups, width multipliers), zone-level control | Submodule surgery (head/channel pruning, low-rank swaps, merges, quantisation), evidence-led | “Micro-surgery beats amputation”: smaller blast radius than module deletion |
| **Far term** | Parameter-level placement and rewiring (dynamic sparse growth), mostly research mode | Parameter-level sedation and deletion (saliency ranked), plus compaction pipelines | Hardware-aware sparsity or it is theatre |

### 11.3 Near term: make the ecology stable and legible

**Tamiyo v1: module-based morphogenesis (what you largely have now)**

* **Capabilities:** Choose slot, blueprint, and a development protocol (incubator → blending → commit). Use scaffolding motifs: train-behind, blend-in then blend-out, staged replacement. Learn placement and ordering under Simic pricing (rent/churn), with Phase 1 teacher labels.
* **Deliverables:** Stable behavioural competence (improves Pareto frontier). Developmental motifs become repeatable.
* **Telemetry (non-negotiable):** Sampled action → resolved config → state transition. Per-head logits/entropy. Slot-local stats (activation moments, gradient alignment).

**Emrakul v0: module-level probe-and-lysis plus scar**

* **Capabilities:** Operate only on committed phages. Do reversible sedation probes, maintain immune ledger. Schedule physical lysis only at safe boundaries. On lysis, collapse into a **ScarSlot**.
* **Coordination:** Enrich slot state with `slot_kind = SCAR`, `scar_age`, `last_removed_blueprint`, `quarantine_remaining`.
* **Success criteria:** Delete modules without quality loss. ScarSlots do not proliferate uncontrollably.

### 11.4 Mid term: submodule control and evidence-led efficiency

**Tamiyo v2: submodule morphogenesis**

* **Capabilities:** Attention (head count, KV dims), Conv (channel groups), MLP (width multipliers). Seeds become containers spawning internal substructure.
* **Policy shape:** Hierarchical or token-based policy. Curriculum training.
* **Telemetry:** Subtarget inventory per phage/seed. Boundary gradients for marginal utility estimates.

**Emrakul v1: submodule surgery and maintenance**

* **Capabilities:** Micro-surgery (head/channel sedation). Maintenance (low-rank swaps, branch merges). Evidence engine via sedation probes.
* **Why this matters:** Smaller blast radius allows faster learning. Hardware-aligned efficiency.
* **Success criteria:** Shift from "delete whole module" to "shave cost while preserving function."

### 11.5 Far term: parameter-level morphogenesis and surgery (research frontier)

**Tamiyo v3: parameter-level placement and rewiring**

* **Practice:** Dynamic sparse training, regrowth rules (RigL-style), per-parameter gating inside a guild.
* **Guardrails:** Restrict to hardware-friendly sparsity (block, N:M) or it will be slow.

**Emrakul v2: parameter-level sedation and deletion**

* **Practice:** Rank parameters by saliency proxies. Apply reversible sedation masks, compact at safe boundaries.
* **Prerequisites:** Real compaction pipeline (tensor resize, weight packing). Per-parameter work must be measurable in step-time.

### 11.6 Cross-cutting milestones (unlock each step)

1. **Telemetry integrity:** If you cannot prove “sampled action → resolved config → executed behaviour”, you will chase ghosts.
2. **Stable identifiers:** Slots, phages, and subtargets need persistent IDs.
3. **Compaction discipline:** Physical rewrites happen only at controlled boundaries.
4. **Pareto evaluation:** Track frontier movement and reliability, not just champions.
5. **Anti-oscillation primitives:** Quarantine, cooldown, criticality locks.

### 11.7 Demo-stage roadmap (storyboarded milestones)

**Stage 0: Demo foundation**

* **Goal:** You can trust what you are seeing.
* **Exit:** Deterministic replay works. Telemetry is end-to-end correct. Flight recorder reconstructs "why this happened."

**Stage 1: Tamiyo demo (module-level growth)**

* **Goal:** Tamiyo grows useful structure safely and repeatably.
* **Exit:** Median performance improves vs baseline under matched cost. No endless oscillation or NOOP collapse.

**Stage 2: Emrakul demo (module-level decay plus ScarSlot)**

* **Goal:** Emrakul removes redundant modules auditably.
* **Exit:** Emrakul performs N successful prunes where loss does not spike, criticality locks work, and system ends up cheaper without losing accuracy.

**Stage 3: Handoff demo (Emrakul cuts, Tamiyo heals)**

* **Goal:** Demonstrate the "trauma surgeon" loop.
* **Exit:** Module removed -> Scar remains -> Tamiyo regrows capacity at that site and recovers lost metric.

**Stage 4: Submodule work**

* **Goal:** Finer granularity without chaos.
* **Exit:** Cost reduction at same performance via submodule surgery.

**Meta-rule:** Each stage needs a storyboarded scenario, a single page of metrics, and a forensic trace.

---

## 12. Speculative Extensions

These extensions are speculative and not required for the core ecology, representing potential scaling paths beyond the current roadmap.

### 12.1 Narset: meta-coordination

A slow-timescale coordinator that allocates per-zone budgets using telemetry only (performance trends, cost, churn, health). Narset does not observe architecture or seed inventory and cannot directly select modules.

### 12.2 Fractal growth

A seed can become a container for another morphogenetic model, enabling zoom-in growth at bottlenecks. This depends on stabilising the non-recursive ecology first.

### 12.3 Blueprint meta-loop

Another policy observes telemetry and generates new blueprints (offline evolutionary search), expanding the genome library.

### 12.4 Phage healing

Beyond lysis, Emrakul could authorise healing operations: merging similar branches via internal distillation, or quantising stable phages to reduce cost without removing function.

### 12.5 Weaning from counterfactual supervision (Phase 1 to deployment)

A critical practical question is whether Tamiyo can learn comparable behaviour under non-counterfactual reward signals. Phase 1 uses Shapley-labelled supervision and counterfactual evaluation to reduce variance. Deployment must rely on cheap telemetry and learned critics. A principled ablation is to progressively remove counterfactual teacher signals while preserving observation parity, then measure whether behavioural fingerprints (module mixtures, ordering statistics, Pareto efficiency) remain stable.

---

## 13. Conclusion

Esper proposes a move away from the intelligent-designer model of architecture. The goal is not to build the perfect network by hand. The goal is to build a **world with rules, then let ecology happen inside it**.

The system has two layers.
**Substrate (infrastructure and physics):**

* **Tolaria:** execution engine for Model Alpha, high throughput, determinism, replay, safety
* **Simic:** economy and accounting, rent, churn, Phase 1 Shapley-based credit

**Organism (agents and organs):**

* **Kasmina:** body, SeedSlots and phages as peer primitives for growth and decay
* **Tamiyo:** growth policy, controls seeds via factored RL
* **Emrakul:** decay policy, controls phages via probe-and-lysis protocols

Fossilisation is custody transfer: a seed becomes **committed** (Tamiyo-locked), rewrapped as a phage, and handed to Emrakul. Compaction occurs later at safe boundaries.

In Phase 1, we deliberately spend compute on audited attribution to ensure the feedback signal is trustworthy while Tamiyo learns. The longer-term plan is to convert expensive truth into scalable behaviour via distillation, approximation, candidate selection, and durable flight-recorded experience, so a trained Tamiyo can reliably grow many future models without the Phase 1 harness.

---

## Appendix A: Pending Validation (Do Not Publish)

> **Status:** This appendix contains draft observations and claims that require validation before inclusion in any public-facing document.

### A.1 Emergent restoration: the “Lobotomy” stress test

To test Tamiyo’s capacity for structural independence, we subjected the system to a stress configuration (“Lobotomy”) in which the host model was intentionally under-structured and performed poorly on CIFAR-10.

**Observed results (requires validation):**

1. **Panic search (early epochs):** high-entropy exploration with rapid germination and pruning.
2. **Identification:** Tamiyo selects a conv-light blueprint capable of restoring spatial processing.
3. **Bypass grafting:** the seed carries most of the effective signal, with strong ablation separation versus the host.

**Tentative interpretation:** If confirmed, this suggests Esper can grow load-bearing modules when the host lacks necessary inductive bias or capacity, and can do so without destabilising optimisation due to gradient isolation and blending.

### A.2 Draft figure caption (for TUI screenshot)

**Figure X: Forensic analysis of a rescue operation.**
Telemetry from the “Lobotomy” stress test on CIFAR-10. Best run reaches 46.1% accuracy under an intentionally weak host baseline. The **Seed Graveyard** shows rejection of multiple candidates before committing a successful conv-light module. Counterfactual analysis indicates the committed seed provides the majority of the effective signal.

### A.3 Engineering sanity checks

Before any publication, validate:

1. **Rent calculation under weak hosts:** Ensure `P_host` is physical parameter count, not tied to performance. Avoid division-by-zero behaviour.
2. **Reproducibility:** Confirm rescue behaviour across multiple RNG seeds. Characterise variance of the “panic search” phase.
3. **Counterfactual validity:** Confirm seed-versus-host split is from correct ablations (seed zeroed vs host zeroed), not a logging artefact.
4. **Gradient isolation verification:** Confirm gradients flow to seed in isolation. Confirm no unintended signal path leaks to or from the host.
