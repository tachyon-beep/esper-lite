## Title page

**Metamorphic Morphogenesis in Seed-Grafted Neural Networks**
**Parasitism, Symbiosis, Succession, and Model Hygiene via Emrakul**

**Authors**

* [Author 1] (corresponding)
* [Author 2]
* [Author 3]

**Affiliations**

* [1] [Org / Lab], [City], [Country]
* [2] [Org / Lab], [City], [Country]

**Correspondence**

* [Name], [email]
* [Postal address (optional)]

**Artefacts**

* Code: [repository link]
* Trained checkpoints: [link]
* Experiment logs / telemetry: [link]
* Repro configs / task specs: [link]

**Version / date**

* Version: [v0.x]
* Date: [YYYY-MM-DD]

**Suggested citation**
[Authors]. *Metamorphic Morphogenesis in Seed-Grafted Neural Networks: Parasitism, Symbiosis, Succession, and Model Hygiene via Emrakul.* [Venue / arXiv], [Year].

---

## Abstract

Seed-grafted morphogenetic training systems dynamically grow and prune neural topology during training while managing interference. A central ambiguity is that a grafted module (“seed”) can become highly counterfactually load-bearing either by adding genuine capability (**symbiosis**) or by displacing and weakening host pathways (**parasitism**). We extend this framing with an orthogonal regime, **metamorphic succession**, where host displacement is not inherently harmful: early seeds may capture representation, later seeds may consume and compose upstream seed outputs, and the final organism can outperform an equivalent “full-fat” model after redundant structure is removed.

We describe Esper, a lifecycle-managed seed-grafting system, and formalise the controller (**Tamiyo**) as a reinforcement learning problem with factored actions, action masking, and a reward design that deliberately stays minimal: counterfactual-grounded contribution, efficiency rent, terminal outcome, and germination-only budget pressure. We present a practical counterfactual measurement scheme that scales as (O(n)) forward passes for (n) concurrent seeds during training, and a capped diagnostic regime ((O(2^n)), (n \le 3)) that decomposes interaction terms to distinguish synergy, interference, and independence.

Because parasitism and early metamorphosis can look identical in snapshot metrics, we frame regime identification as an in-flight detection problem that is weakly identifiable from instantaneous state but separable in expectation from trajectory statistics. We propose leading indicators of “recovering turbulence”, highlighting the interaction-term derivative over time as a tractable signal, and show how Tamiyo can act under uncertainty via a turbulence budget and risk management through adaptive checkpointing and rollback.

Finally, we introduce **Emrakul**, a model-hygiene subsystem that prunes detritus (unreachable or low-traffic structure), stabilises attribution, and consolidates metamorphic transitions into coherent adult topologies. We also sketch a shared stability-coupled “energy budget” that dynamically allocates resources between growth (Tamiyo) and pruning (Emrakul) based on measured training stability.

**Results and evidence** are provided in Section [14], including: (i) regime frequencies across tasks, (ii) Pareto comparisons against matched baselines, (iii) interaction-telemetry trajectories, (iv) reward-ablation learning curves, and (v) hygiene and efficiency deltas.
**Placeholders:** [Insert numeric results, confidence intervals, and figures.]

---

## Keywords

Morphogenetic training; dynamic neural architectures; seed grafting; reinforcement learning controller; counterfactual attribution; interaction decomposition; pruning; continual learning; modular networks; training stability.

---

## Plain-language summary (optional)

We build neural nets that can grow new bits while training. Sometimes a new bit becomes “important” because it genuinely helps; sometimes it becomes “important” because it broke the old path and made itself mandatory. Annoyingly, those two cases can look the same mid-training. We treat growth control as an RL problem (Tamiyo), measure seed value with counterfactual ablations, and use interaction telemetry to see whether multiple seeds are cooperating or fighting. We also run a clean-up monster (Emrakul) that eats dead, unused parts of the network so the final model is leaner and easier to reason about.
**Placeholders:** [Insert the main empirical takeaway once results are final.]

---

## Paper outline

Section 2 summarises the Esper system and seed lifecycle. Sections 3–4 formalise Tamiyo as an RL controller and document reward-engineering lessons that led to a minimal, learnable signal. Sections 5–6 define scalable counterfactual attribution and capped interaction decomposition. Sections 7–9 introduce the symbiosis/parasitism/metamorphosis taxonomy and the in-flight detection problem, with trajectory-based indicators and probabilistic control mechanisms. Sections 10–12 cover depth-ordering/traffic dynamics, Emrakul model hygiene, and a stability-coupled shared energy budget. Sections 13–16 present experiments, results, discussion, and limitations.

---

## Contributions

This paper makes the following contributions (with supporting evidence in the Results section):

1. **A three-regime taxonomy** for seed integration: symbiosis, parasitism, and metamorphosis (succession).
2. **Tamiyo as an RL controller** with factored actions and masking, aimed at learning integration strategy rather than hard-coding it.
3. **Scalable counterfactual attribution** for training ((O(n))) and **interaction decomposition** diagnostics capped to (n \le 3) ((O(2^n))).
4. **Trajectory-based in-flight detection framing**, proposing “recovering turbulence” indicators (notably the interaction-term derivative) to separate early metamorphosis from parasitism.
5. **Emrakul model hygiene** as a first-class subsystem: detritus pruning to stabilise attribution and consolidate adult topologies.
6. **A shared stability-coupled energy budget** concept allocating resources between growth and pruning based on measured stability.

**Placeholders:** [Insert citations to figures/tables once numbered.]

---

## Definitions and notation

* **Host:** the current main network excluding newly grafted seeds.
* **Seed:** a grafted module trained to address residual error or add capacity.
* **Slot:** a graft location in the host (e.g., early/mid/late).
* **Blending:** staged integration via an interpolation schedule (e.g., (\alpha)).
* **Fossilise / cull:** keep permanently / remove the seed.
* **Counterfactual marginal:** performance difference between the full model and the model with a seed ablated under a specified protocol.
* **Interaction term:** deviation from additivity when multiple seeds are present (synergy vs interference).
* **Detritus:** unreachable or low-traffic structure not carrying meaningful computation.
* **Model hygiene:** systematic removal of detritus to improve efficiency and measurement clarity.

---

## Reproducibility and reporting commitments

* All experiments report **random seeds**, **run counts** [N=]( ), and **mean ± uncertainty** ([CI method]).
* Counterfactual contributions are reported under at least **two ablation protocols**: gate-to-zero and bypass/identity, with optional recalibration where relevant.
* Interaction decomposition is enabled for diagnostic runs with (n \le 3) seeds; beyond that, only marginal contributions are used for training-time control.
* Telemetry schemas for host metrics, per-seed state, traffic estimates, and controller actions are versioned and published.
* Definitions of **collapse**, **stability**, and **traffic** (including thresholds and smoothing) are declared once and used consistently across the paper.
  **Placeholders:** [Insert links and exact reporting conventions.]

---

## Declarations

**Funding:** [None / list funding sources]
**Competing interests:** [None / disclose]
**Ethics:** [Not applicable / approvals]
**Data and materials availability:** [links / access conditions]
**Author contributions:** [CRediT roles]
**Acknowledgements:** [people / infra]

---

## 1. Introduction

Modern neural networks are usually trained as if the architecture is a settled matter: pick a topology, optimise weights, ship it. That works—until it doesn’t. As soon as the task distribution shifts, compute budgets fluctuate, or the model needs to expand capacity in a targeted way, “architecture as a constant” becomes a brittle assumption. Morphogenetic training systems tackle this by letting the model **change its topology during training**: growing new modules when needed, integrating them safely, and pruning away what no longer matters.

**Esper** (and its lighter implementation, *esper-lite*) explores a specific morphogenetic pattern: **seed grafting**. The model periodically germinates small candidate modules (“seeds”), trains them on residual error without destabilising the host, blends them in gradually, and then either fossilises them permanently or culls them. This sounds straightforward, but in practice it creates two coupled problems.

1. **Attribution is ambiguous.** A seed can become counterfactually “load-bearing” either because it genuinely adds capability or because it has displaced the host and made itself indispensable. Both cases look like “high contribution” if you only look at a single ablation number.

2. **Control is a learning problem.** When to germinate, where to graft, what to graft, how long to blend, when to cull—these are sequential decisions under uncertainty, under budget constraints, with delayed outcomes. Hard-coded policies tend to either overfit a single failure mode or become so conservative that growth never happens.

This paper treats those issues as first-class. We separate **what** we measure (counterfactuals and interactions), **how** we decide (Tamiyo as an RL controller), and **how** we keep the evolving model sane (Emrakul as hygiene and pruning).

### 1.1 The core ambiguity: indispensability is not value

A common instinct in modular systems is to measure importance via counterfactual removal: “if I ablate this module, how much does performance drop?” This metric is useful—but it is not sufficient.

In seed-grafted morphogenesis, a seed can become counterfactually important through at least two qualitatively different mechanisms:

* **Symbiosis:** the seed adds representational power the host did not have; the host remains competent; the seed is valuable.
* **Parasitism:** the seed disrupts or suppresses host pathways during integration; the host atrophies; the seed becomes necessary because it created a dependency.

There is a third regime that complicates the story further:

* **Metamorphosis (succession):** early seeds may “steal” functionality from the host, later seeds may consume and compose upstream seed outputs, and the final network can be superior *after* redundant larval pathways are pruned away. In this regime, **host-only competence is not sacred**, and early host degradation can be part of a productive transition.

So the question is not “did the seed become important?” but:

> **Was importance achieved by adding capability, by creating dependency, or by enabling a successor topology that ultimately replaces the host?**

Answering that requires looking beyond snapshot metrics and into **trajectory**.

### 1.2 The in-flight detection problem

Parasitism and early-stage metamorphosis can look identical mid-run: the host degrades while a seed becomes load-bearing, and interactions between seeds can be negative during turbulent integration. You only learn which regime you were in once you see whether the system recovers into a coherent successor circuit or collapses into brittle dependency.

This motivates a practical framing: regime identification is **weakly identifiable from instantaneous observations**, but becomes separable in expectation from **trajectory statistics**. In particular, when interactions between a small number of concurrent seeds can be measured, the **interaction term over time** becomes a high-leverage signal. Parasitism tends to produce “sticky negative” interaction; metamorphosis tends to produce “negative-but-recovering” interaction, where the derivative trends upward as succession completes.

Crucially, the controller does not need to perfectly label regimes online. It needs to learn which bets pay off **on average**, and to manage risk when turbulence does not resolve.

### 1.3 Control as reinforcement learning: Tamiyo

Esper uses **Tamiyo**, a learned controller, to decide seed lifecycle operations: germinate, blend, wait, cull, and fossilise across multiple slots. This is naturally posed as a reinforcement learning problem with:

* a rich observation space (host dynamics; per-slot stage/age/trajectory; resource usage; multi-slot occupancy),
* a factored action space (slot × blueprint × blend algorithm × lifecycle operation),
* action masking for invalid moves,
* and a reward signal that must incentivise **net system improvement** without hard-coding *how* to integrate.

A key empirical lesson (documented later via ablations) is that “clever” reward shaping that anticipates every failure mode can make learning impossible. The working approach is deliberately minimal: counterfactual-grounded contribution, an efficiency rent, a terminal outcome bonus, and a germination-only budget pressure. This keeps the landscape learnable and allows Tamiyo to discover integration strategies that were not explicitly programmed.

### 1.4 Measurement: scalable counterfactuals and capped interaction decomposition

To drive both control and diagnosis, Esper uses a two-tier measurement scheme.

* **Training regime (scales as (O(n))):** with (n) concurrent seeds, estimate per-seed marginal contributions via (n+1) forward passes (full model plus each seed ablated). This is cheap enough to use as a reward signal and for lifecycle decisions.

* **Diagnostic regime (scales as (O(2^n)), capped):** for (n \le 3), evaluate all seed combinations and decompose interaction terms (synergy vs interference vs independence). This adds the missing context needed to interpret “importance”, and it enables trajectory features (notably the interaction derivative) for in-flight discrimination.

Counterfactual estimates are also **protocol-dependent** (e.g., gate-to-zero vs bypass/identity), so reporting multiple ablation protocols is not optional theatre—it is necessary for trustworthy attribution.

### 1.5 Hygiene: Emrakul and the detritus problem

Morphogenesis is messy by design. During growth and blending, the network temporarily contains redundant pathways and components that receive little or no traffic. If left unaddressed, this **detritus** causes three practical failures: wasted compute, distorted attribution baselines, and training instability from stale or noisy structure.

Esper therefore includes **Emrakul**, a hygiene subsystem responsible for:

* pruning unreachable structure,
* pruning persistently low-traffic components (with safeguards),
* and assisting in removing non-performant seeds and cleaning up after topology changes.

In the metamorphosis framing, Emrakul is what turns “messy takeover” into “clean transition”: it consolidates the adult topology by eating dissolved larval tissue.

We also describe a shared **stability-coupled energy budget** mechanism that dynamically allocates resources between Tamiyo (growth) and Emrakul (pruning) based on measured stability: when the system is unstable, invest in growth and risk control; when stable, invest in compaction and cleanup.

### 1.6 Contributions and what to expect

This paper contributes a regime-aware, controller-aware, measurement-aware framing for seed-grafted morphogenesis:

* a three-regime taxonomy (symbiosis, parasitism, metamorphosis),
* Tamiyo as an RL controller with factored actions, masking, and a minimal learnable reward,
* scalable counterfactual attribution for control plus capped interaction decomposition for diagnostics,
* an explicit treatment of in-flight ambiguity using trajectory statistics (notably interaction derivatives),
* Emrakul as model hygiene, and a stability-coupled budget that ties growth and pruning into one system.

**Evidence and conclusions are reported later** (Sections 13–16), including regime frequencies, interaction-telemetry trajectories, reward-ablation learning curves, hygiene/efficiency deltas, and Pareto comparisons against matched baselines.
**Placeholders:** [Insert figure/table references once numbered.]

### 1.7 Paper organisation

* **Section 2** introduces Esper’s lifecycle-managed seed-grafting system and telemetry.
* **Sections 3–4** formalise Tamiyo as an RL controller and document reward-design lessons.
* **Sections 5–6** define counterfactual attribution and interaction decomposition.
* **Sections 7–9** present the taxonomy, the in-flight detection problem, and probabilistic control mechanisms (turbulence budget, checkpointing/rollback).
* **Sections 10–12** cover depth ordering/traffic flow, Emrakul hygiene, and the shared energy budget concept.
* **Sections 13–16** describe experiments, results, discussion, and limitations.

---

## 2. System overview: Esper as a morphogenetic stack

Esper (and *esper-lite*) is a lifecycle-managed framework for **seed-grafted morphogenesis**: it grows new modules during training, integrates them gradually, and prunes away structure that is no longer carrying useful computation. This section defines the core objects (host, slots, seeds), the seed lifecycle, and the telemetry/control surfaces used by the RL controller (**Tamiyo**) and the hygiene subsystem (**Emrakul**).

### 2.1 Architectural decomposition (subsystems)

Esper is organised into domain-oriented subsystems. The system behaviour is not “just the model”; it emerges from the coupling of lifecycle machinery, measurement/telemetry, a learned controller, and hygiene/pruning.

* **Kasmina (Body/Model):** seed lifecycle implementation, quality gates, host networks, blueprint instantiation, slot wiring.
* **Leyline (Nervous System):** shared contracts (enums, dataclasses), signals, telemetry schemas, and cross-module invariants.
* **Tolaria (Hands):** training loops, optimisation, blending schedules, watchdog/governor logic.
* **Simic (Gym):** RL infrastructure (e.g., PPO-style training), feature engineering, reward computation, rollout plumbing.
* **Tamiyo (Brain):** decision-making policy over lifecycle operations across slots.
* **Nissa (Senses):** telemetry hub and output backends (console, files, dashboards).
* **Runtime / Scripts / Utils:** task registry, presets, entry points, data loading and helpers.

**Placeholder:** [Insert a dependency diagram / subsystem interaction figure.]

---

### 2.2 Core abstractions: host, slots, and seeds

Esper treats the trained model as a **host network** plus a set of **seed slots** at predetermined attachment points.

* **Host network** (H): the baseline architecture that provides end-to-end capability absent any seeds.
* **Slots** ({L_1, \dots, L_k}): attachment points (e.g., “early/mid/late” positions such as post-block1, post-block2, post-block3).
* **Seeds** ({s_j}): candidate modules attached to slots. Seeds are instantiated from **blueprints** (small architecture templates).

Each occupied slot modifies an intermediate activation via a blend operator. A canonical mental model is alpha mixing:

[
y = (1-\alpha),f_H(x) + \alpha,f_{s}(x),
]

where (f_H) is the host pathway at that slot, (f_s) is the seed pathway, and (\alpha \in [0,1]) is a scheduled blending parameter. Esper supports multiple blend algorithms; alpha mixing is the simplest reference form.

The operational goal is not “add seeds forever”, but **selective growth and consolidation**:

* seeds that add value are **fossilised** (retained),
* seeds that don’t are **culled** (removed),
* unused structure is pruned by **Emrakul** to maintain clarity and efficiency.

---

### 2.3 Seed lifecycle and stage semantics

Seeds progress through discrete lifecycle stages governed by invariants, constraints, and evaluation gates. Exact stage naming is implementation-defined; conceptually Esper uses a flow like:

1. **EMPTY:** slot unoccupied.
2. **GERMINATING:** seed instantiated; wiring validated; initial baselines captured.
3. **ISOLATED TRAINING:** seed trained primarily against host residuals with mechanisms to limit destabilisation of host pathways (e.g., controlled routing / gradient isolation).
4. **BLENDING:** seed integrated gradually via an (\alpha) schedule (or alternative blend algorithm).
5. **PROBATION:** joint training continues; stability and net benefit assessed.
6. **FOSSILISE:** seed becomes permanent; lifecycle ends.
7. **CULL:** seed removed; slot returns to EMPTY (optionally with cooldown).

Stage-specific constraints (examples):

* **Fossilisation gates:** minimum evidence of net benefit under the declared evaluation suite and protocol(s).
* **Culling permission:** always available, but may have costs (lost compute investment, topology churn).
* **Blending safety:** incremental integration is the default; hard switches are avoided unless explicitly configured.

**Placeholder:** [Insert a lifecycle state diagram / transition table, including stage gate conditions.]

---

### 2.4 Blueprints: seed classes as plugins

Seeds are instantiated from **blueprints**, which define the seed architecture and its parameter/compute profile (e.g., “depthwise”, “attention-lite”, “conv-heavy”). Esper uses a registry pattern so new blueprints can be added without modifying the controller or lifecycle machinery.

Blueprints vary in:

* **capacity** (parameter count, receptive field, attention span),
* **inductive bias** (convolutional, depthwise separable, attention-style),
* **integration risk** (e.g., high-capacity seeds can dominate local gradient flow and displace host pathways).

This matters for the regime taxonomy: blueprint choice changes the likelihood of symbiosis, parasitism, or metamorphosis.

**Placeholder:** [Insert a blueprint table with typical parameter counts and intended use-cases.]

---

### 2.5 Training orchestration (Tolaria) and stability guardrails

Tolaria orchestrates training and integration. Core responsibilities include:

* per-stage training modes (isolated vs joint),
* blend scheduling (e.g., (\alpha) progression),
* evaluation cadence for lifecycle decisions,
* watchdog/governor logic for safety (e.g., divergence detection, NaNs, runaway growth).

The orchestration is designed to make topology changes *safe enough to attempt frequently*:

* topology changes are incremental,
* evaluation is continuous rather than “train then judge once”,
* rollback and checkpointing hooks exist for turbulence management (discussed later in Section 9).

**Placeholder:** [Insert high-level pseudocode for the training + lifecycle loop.]

---

### 2.6 Telemetry: the shared language between Tamiyo and Emrakul

Esper is telemetry-first: telemetry is not merely for dashboards; it is the **state** observed by Tamiyo and the **evidence** used by Emrakul.

Telemetry is emitted at multiple levels.

**Host-level signals**

* training/eval loss and task metrics,
* gradient norm summaries and instability flags,
* plateau and regression indicators (e.g., trend slopes, windowed variance).

**Per-slot / per-seed signals**

* lifecycle stage, age, epochs-in-stage,
* blend state (e.g., current (\alpha)),
* improvement trajectory summaries (short- and long-horizon deltas),
* traffic/utilisation estimates (activation flow, routing share, gradient share),
* counterfactual marginal contribution estimates (when computed).

**Resource signals**

* seed budget consumed/remaining,
* parameter growth ratio vs host,
* compute overhead and wall-clock signals (if available),
* energy-pool allocation (if the shared budget is enabled).

**Placeholder:** [Insert the telemetry schema summary or key field list.]

---

### 2.7 Control surfaces: who is allowed to do what

Esper separates responsibilities between the growth controller and the hygiene subsystem.

**Tamiyo (controller)**
Chooses lifecycle operations and integration strategy, such as:

* **GERMINATE** (choose slot, blueprint, blend algorithm),
* **WAIT** (continue training within current stage),
* **ADVANCE_STAGE** (when stage gates permit),
* **CULL**,
* **FOSSILISE**.

**Emrakul (hygiene)**
Maintains structural cleanliness and measurement clarity:

* prune unreachable components,
* prune persistently low-traffic components (with safeguards such as minimum age),
* remove detritus created by topology change,
* optionally trigger normalisation recalibration after pruning.

This separation is intentional. Tamiyo is rewarded on net outcomes and may take bounded risks; Emrakul is conservative and aims to keep the organism coherent, efficient, and measurable. Where they intersect, coordination occurs via shared telemetry, constraints, and (optionally) the shared energy budget described in Section 12.

---

### 2.8 Counterfactual evaluation surfaces (what Esper can measure)

Esper’s primary “grounded” value signal is counterfactual contribution computed by ablating seeds under a declared protocol. Two measurement regimes are supported:

* **Training-time marginals:** (\mathcal{O}(n)) evaluations (specifically (n+1)) to estimate per-seed marginal contributions when (n) seeds are active.
* **Diagnostic decomposition:** (\mathcal{O}(2^n)) evaluations capped to (n \le 3) to estimate interaction terms (synergy / interference / independence).

These measurements feed both:

* controller reward signals (Simic/Tamiyo), and
* analysis and auditing (Nissa logs, offline regime classification).

**Placeholder:** [Insert the precise ablation protocols used (gate-to-zero, bypass/identity, recalibration).]

---

### 2.9 Summary

Esper is best understood as a coupled system:

* a **host network** with predefined **seed slots**,
* a staged **seed lifecycle** designed for safe incremental integration,
* a telemetry substrate that makes dynamics observable,
* a learned RL controller (**Tamiyo**) that allocates growth under constraints,
* a hygiene subsystem (**Emrakul**) that prunes detritus and consolidates transitions.

This architecture is explicitly built to support both **host-preserving growth** and **host-replacing succession**, and to make those regimes measurable via counterfactual attribution and (when feasible) interaction telemetry.

---

## 3. Tamiyo: Reinforcement Learning Control for Morphogenetic Decisions

Esper’s core premise is that *when and how to grow* should not be hard-coded as a brittle set of heuristics. Instead, the system uses **Tamiyo**, a learned controller, to make lifecycle decisions over a seed bank embedded in a host network. This controller operates under delayed rewards, partial observability, and hard structural constraints, making the control problem naturally well-modelled as reinforcement learning.

### 3.1 Problem formulation

We model morphogenetic control as an episodic (or continuing) Markov decision process (MDP):

* **Environment:** training and evaluation of a host network with a set of seed slots; environment transitions are driven by both optimiser updates and lifecycle operations (germination, blending, culling, fossilisation, hygiene events).
* **Agent:** Tamiyo, which selects lifecycle actions that modify architecture and slot state.
* **Objective:** maximise expected task performance subject to parameter/compute budgets, while minimising collapse risk and encouraging stable learning dynamics.

A key complication is that environment dynamics are dominated by non-stationary learning processes: gradient updates change the mapping from architecture to performance over time. Tamiyo therefore acts as a controller over a drifting, partially observed system rather than a stationary “game”.

### 3.2 Observations (state representation)

Tamiyo’s observation vector is structured to preserve the information that makes regime dynamics identifiable from trajectories rather than snapshots. We group features into four blocks.

1. **Host training dynamics**

   * current loss and task score (train/val where available)
   * plateau/regression indicators (e.g., EMA slope of validation score)
   * stability proxies (e.g., windowed variance of loss/score)
   * optional optimiser/gradient summaries (e.g., gradient norm, update magnitude)

2. **Per-slot seed state (for each slot)**

   * lifecycle stage (e.g., GERMINATING / ISOLATED / BLENDING / PROBATION / FOSSILISED)
   * age (epochs-in-stage, wall-clock age, or update count)
   * current blend parameter (\alpha) and schedule position (where applicable)
   * improvement trajectory summaries (short/medium horizon deltas)
   * traffic/utilisation estimates (activation magnitude, routing share, gradient-flow proxy)
   * counterfactual marginal contribution estimates when computed (Section 5)

3. **Resource utilisation**

   * seed budget consumed and remaining
   * parameter growth ratio relative to host
   * compute / latency estimates (optional; platform-dependent)
   * energy allocation between Tamiyo and Emrakul when using a shared pool (Section 12)

4. **Multi-slot context**

   * occupancy bitmap (which slots are active)
   * relative per-slot contributions and (when diagnostics are enabled) interaction summaries
   * slot depth identifiers (early/mid/late) and adjacency indicators

This structure is designed for the paper’s central control challenge: parasitism and early metamorphosis may share the same instantaneous phenotype, so the controller must condition on *trend features* (e.g., improving vs worsening turbulence) rather than a single score.

### 3.3 Action space: factored decisions with masking

A naïve formulation treats each “slot × blueprint × blend variant × lifecycle op” tuple as a single flat discrete action. This is easy to implement but scales poorly: the action count grows multiplicatively with the number of options, producing a combinatorial explosion as the system grows.

Esper instead uses a **factored action space**, conceptually:

[
a = (a_{\text{slot}},\ a_{\text{blueprint}},\ a_{\text{blend}},\ a_{\text{op}})
]

where:

* (a_{\text{slot}}): which slot is targeted
* (a_{\text{blueprint}}): which seed class/blueprint to germinate (only meaningful when germinating)
* (a_{\text{blend}}): which blend schedule/algorithm variant to use (if configurable)
* (a_{\text{op}}): lifecycle operation (e.g., WAIT, GERMINATE, ADVANCE_STAGE, CULL, FOSSILISE)

Factorisation allows policy heads to scale approximately linearly with the number of choices rather than multiplicatively.

**Action masking** is required because many factor combinations are invalid in a given state. Examples include:

* cannot **FOSSILISE** an empty slot
* cannot **GERMINATE** when the seed budget is exhausted
* cannot **ADVANCE_STAGE** from a non-advanceable stage
* blueprint-slot incompatibilities (if enforced)
* optional hygiene-only operations disallowed when Emrakul budget is zero (in shared-pool runs)

Masking is applied both to reduce wasted probability mass and to enforce structural safety constraints.

### 3.4 Decision cadence and temporal abstraction

Tamiyo does not need to act at every optimiser step. In practice, it is invoked on a **controller cadence** (e.g., every (k) steps or at epoch boundaries), while telemetry is computed continuously and aggregated into trajectory summaries. This reduces action noise and improves credit assignment by aligning decisions with the natural timescales of:

* blending schedules (where (\alpha) changes over a stage window),
* evaluation ticks used for counterfactual measurement,
* checkpointing intervals (for rollback safety),
* and hygiene passes (which may be periodic or stability-triggered).

Cadence is treated as a configuration choice and is reported in TaskSpecs for reproducibility.

### 3.5 Why reinforcement learning rather than heuristics

Two properties make this domain unfriendly to purely hand-built policies:

1. **Multi-objective optimisation under uncertainty.** The controller must trade off short-term turbulence versus long-term gains, and performance versus parameter/compute budgets, while operating with noisy measurements.

2. **Non-stationary dynamics and regime ambiguity.** Parasitism and early metamorphosis can look identical in snapshot metrics; good decisions depend on the probability of recovery, which is learned from experience rather than deduced from a single threshold.

Tamiyo’s goal is not to output symbolic regime labels online. Instead, it learns a policy that maximises expected downstream return given the observed trajectory and resource context—supported by minimal, counterfactual-grounded rewards (Section 4) and risk-control mechanisms such as turbulence budgeting and adaptive checkpointing (Section 9).

### 3.6 Practical notes on implementation boundaries (reporting requirements)

To make controller behaviour interpretable and reproducible, experiments report:

* the exact observation feature set and normalisation scheme,
* the factored action heads and masking rules,
* the controller cadence,
* the RL algorithm and hyperparameters (e.g., PPO variant, rollout length, entropy regularisation),
* and the mapping from controller actions to lifecycle transitions (including any safety overrides).

**Placeholders:** [Insert the concrete Tamiyo architecture and training hyperparameters once finalised.]

---

## 4. Reward Design That Learns (and Reward Design That Doesn’t)

Tamiyo’s ability to learn useful morphogenetic policies depends heavily on reward design. Seed-grafted morphogenesis is a magnet for over-engineering: there are many plausible failure modes (parasitism, “ransomware” dependency, blending regressions, WAIT-farming, budget abuse), and it is easy to add a bespoke penalty for each. Empirically, this instinct can backfire. In our setting, heavily-shaped rewards created an unlearnable landscape and drove the controller toward conservative “do nothing” policies that degraded end-task performance.

This section documents a key operational lesson: **minimal rewards grounded in counterfactual contribution are learnable**, while complex “protective” shaping often collapses learning—sometimes before the very failure modes it tries to prevent can even appear.

### 4.1 Design requirements for morphogenetic rewards

A reward function for morphogenetic control must satisfy several constraints simultaneously:

1. **Seed-specific credit assignment.** Tamiyo needs to distinguish “this seed helped” from “the host improved anyway”, especially when multiple seeds are active and stages differ.

2. **Budget and efficiency pressure.** The controller must face a clear incentive not to grow without bound; otherwise the easiest strategy is to add capacity everywhere.

3. **Stage robustness.** Rewards should not require fragile stage-by-stage special cases that effectively hard-code a policy into the reward (and then break when stage dynamics change).

4. **Learnability under partial observability.** The environment is non-stationary (training changes the mapping from actions to outcomes), and the controller observes only telemetry proxies. Rewards must remain smooth enough that exploration is not uniformly punished.

5. **Outcome alignment.** Ultimately, the system must optimise the real task objective (accuracy/reward), not a proxy that can be gamed.

Counterfactual marginal contribution (Section 5) provides a natural anchor: it is seed-specific, directly tied to performance, and can be computed frequently enough to serve as the primary learning signal.

### 4.2 Reward evolution: three phases

We summarise the reward’s evolution because the “failed” versions are as instructive as the working one. The intent here is not to dunk on earlier designs; it is to explain why apparently sensible shaping can create brittle, adversarial optimisation landscapes.

#### Phase 1: loss-delta rewards (failed)

Early versions used reward signals derived from loss improvement (e.g., (\Delta)loss over a window). In practice, this signal was noisy and poorly aligned with seed-specific causality. Loss fluctuations were dominated by optimiser noise, batch variance, and general host dynamics rather than by the controller’s discrete lifecycle actions. The controller could not reliably infer “my action caused that loss change”, so policies either failed to improve or became unstable.

**Placeholder:** [Insert learning curves showing reward variance and weak policy improvement.]

#### Phase 2: contribution-based with extensive protections (failed)

A later design anchored reward on contribution but attempted to anticipate multiple theoretical failure modes with extensive shaping (historically described as ~974 lines). Mechanisms included, for example:

* attribution discounting based on total improvement trajectories
* ratio penalties for “high contribution / low improvement” patterns
* escalating blending warnings for negative trajectories
* anti-farming penalties for WAIT-heavy behaviour
* stage bonuses with anti-exploitation logic (PBRS-like)
* special-case handling for CULL and FOSSILISE
* explicit “ransomware signature” detection heuristics
* legitimacy discounts and stacked guardrail penalties

Each mechanism addressed a real concern. However, the combined system frequently became **unlearnable**:

* **Penalty stacking drowned the signal.** Multiple penalties often fired on the same seed at the same time, overwhelming any positive contribution term.
* **Interacting terms produced unstable gradients.** Seemingly independent shaping terms interacted through shared telemetry features and stage transitions, creating discontinuous reward cliffs.
* **Exploration became uniformly unsafe.** The policy learned that taking actions was more likely to trigger penalties than to earn reward, converging toward a conservative WAIT/inaction equilibrium.
* **End-task performance regressed.** In some runs, the controller’s safest policy actively reduced baseline accuracy by suppressing useful growth or locking the system into unproductive dynamics.

The headline lesson is blunt: reward shaping that is “smart” in code can be hostile in optimisation space.

**Placeholder:** [Insert ablation results: shaped reward → do-nothing policy; baseline drop; action histograms.]

#### Phase 3: minimal counterfactual-grounded reward (worked)

The reward design that consistently enabled learning was deliberately simple and grounded in counterfactual marginals:

[
r_t ;=; \sum_{i \in \text{active seeds}} m_{i,t}
;-; \lambda_{\text{rent}} ,\log!\bigl(1+\text{params}*t\bigr)
;+; \lambda*{\text{term}} ,\mathcal{E}*{\text{final}}
;-; \lambda*{\text{bud}} ,u_t^{2},\mathbf{1}{\text{GERMINATE}}
]

Where:

* (m_{i,t} = \mathcal{E}(M_t) - \mathcal{E}(M_t \setminus s_i)) is the marginal counterfactual contribution (Section 5).
* (\text{params}_t) is current parameter count (or growth ratio vs host), used as a soft efficiency rent.
* (\mathcal{E}_{\text{final}}) is the terminal task score for the episode (or evaluation window).
* (u_t) is budget utilisation (e.g., fraction of seed budget consumed).
* (\mathbf{1}{\text{GERMINATE}}) applies budget pressure primarily when the controller chooses to spawn a new seed.

This reward does *not* attempt to explicitly encode parasitism/metamorphosis rules. Instead, it pushes Tamiyo toward actions that produce measurable improvements, while discouraging bloat and uncontrolled spawning.

**Placeholder:** [Insert evidence: learning curves and performance recovery under minimal reward.]

### 4.3 Why minimal reward works (and what it assumes)

The minimal reward works in this domain for two pragmatic reasons.

**First, it preserves a usable signal-to-noise ratio.** Counterfactual marginals provide seed-specific credit. The rent term provides global pressure without relying on fragile stage heuristics. Together they produce a reward signal with stable sign and meaningful gradients for policy optimisation.

**Second, it avoids pre-emptive moralising.** Many failure modes are real but rare, or emerge only once the controller learns to act. If shaping terms punish the exploration required to reach those states, the agent never learns anything except “don’t touch it”.

This is especially important given the paper’s central ambiguity: parasitism and early metamorphosis can look the same mid-run. If the reward harshly penalises early host degradation or negative interaction, it risks systematically suppressing productive metamorphosis. A minimal reward allows the agent to learn, statistically, when tolerating turbulence pays off—while risk is handled through **separate mechanisms** (turbulence budgets and checkpointing/rollback, Section 9) rather than by cramming those mechanisms into reward.

### 4.4 What the reward does *not* solve

Minimal reward is not a magic spell. It deliberately leaves several issues to other parts of the system:

* **Interaction blindness:** (\sum m_i) is not a Shapley value and does not fully account for seed interactions (addressed diagnostically in Section 6).
* **Protocol sensitivity:** marginals depend on ablation protocol and evaluation batch choice (Section 5).
* **Risk management:** the reward alone does not prevent occasional catastrophic bets; checkpointing/rollback exists precisely because the controller will sometimes be wrong (Section 9).
* **Hygiene:** reward does not remove detritus; Emrakul exists to keep the grown model legible and efficient (Section 11).

This division of labour is intentional. In our experience, making the reward “smarter” by absorbing these concerns tended to make it less learnable. The strategy that worked was to keep the reward minimal and handle robustness through observability, constraints, and hygiene.

### 4.5 Practical guidance: how to iterate reward design safely

Based on the observed reward evolution, we recommend an iterative approach:

1. Start with the minimal reward anchored on counterfactual contribution + rent + terminal outcome.
2. Instrument heavily (action distributions, collapse triggers, interaction diagnostics where feasible).
3. Add new shaping terms only when a failure mode appears repeatedly *and* can be targeted with a single, well-isolated penalty.
4. Re-run ablations to confirm that the new term improves outcomes without collapsing exploration.

**Placeholder:** [Insert a table of reward variants and their observed behavioural failure modes.]

---

## 5. Counterfactual Attribution at Scale

Morphogenetic systems need a value signal that is (i) **seed-specific**, (ii) **comparable across time**, and (iii) **cheap enough** to compute frequently. Esper uses counterfactual evaluation as its primary grounding mechanism: a seed’s contribution is estimated by measuring performance with the seed present versus absent under a specified ablation protocol.

This section defines the measurement problem precisely, describes practical ablation protocols (including reporting requirements), and presents a scalable scheme for attributing value to (n) concurrent seeds during training.

### 5.1 What “counterfactual contribution” means (and what it does not)

Let (M) be the full model containing a host network (H) and a set of active seeds (S = {s_1,\dots,s_n}). Let (\mathcal{E}(\cdot)) denote an evaluation function appropriate to the task (e.g., accuracy on a fixed validation batch, episodic return, or another scalar score computed on a held-out evaluation stream).

We define the (training-time) **marginal counterfactual contribution** of seed (s_i) at time (t) as:

[
m_i(t) ;=; \mathcal{E}!\left(M_t\right) ;-; \mathcal{E}!\left(M_t \setminus s_i\right)
]

where (M_t \setminus s_i) denotes the model obtained by **ablating** (disabling) seed (s_i) at time (t) using a declared protocol. The key intent is pragmatic: “if I remove this seed right now, how much worse do I get?”

This quantity is deliberately *not* a claim of:

* Shapley-optimal credit assignment,
* causal identification in the presence of confounding optimisation dynamics,
* or a global decomposition of responsibility across all components.

It is a local, time-indexed signal that is useful for control because it gives **seed-specific credit assignment** without requiring exponential computation.

Two limitations matter immediately:

1. **Protocol dependence.** The value of (\mathcal{E}(M_t \setminus s_i)) depends on *how* ablation is performed (Section 5.2).
2. **Interaction blindness.** (m_i(t)) can be inflated or suppressed by interactions among seeds; it is therefore insufficient on its own to distinguish regimes like parasitism and early metamorphosis (Section 6).

### 5.2 Ablation protocols (reporting requirement)

Counterfactual estimates can be artefactually inflated by graph-shape changes, normalisation drift, or bypass behaviour. For this reason, the ablation method is part of the measurement definition, not an implementation detail. Esper reports at least two ablation protocols; results should be shown under both unless stated otherwise.

**Protocol A — Gate-to-zero (“soft excision”)**
The seed’s output is forced to zero while preserving tensor shapes and graph structure. In alpha-blended wiring, this is often equivalent to forcing the seed path’s effective contribution to zero (e.g., setting the seed gate to 0 or (\alpha = 0) for that path).

* Pros: preserves shapes; minimal graph changes; cheap.
* Cons: can interact with normalisation statistics; may not reflect “remove the module” semantics if the seed path influences stateful layers.

**Protocol B — Bypass / identity (“structural bypass”)**
The seed path is replaced by a defined bypass (e.g., identity, skip connection, or routing around the seed) that preserves end-to-end functionality without invoking the seed.

* Pros: closer to “module removed” semantics; can reduce some normalisation artefacts.
* Cons: only valid when a meaningful bypass exists; may change effective depth or routing.

**Optional Protocol C — Recalibrated ablation (recommended when normalisation is present)**
After ablation, run a short calibration pass to refresh normalisation statistics (e.g., BatchNorm running means/variances or EMA stats) *without* weight updates. Calibration data and procedure must be reported.

* Pros: reduces artefacts introduced by stateful normalisation.
* Cons: adds compute; introduces an extra methodological choice (calibration set and length).

**Reporting commitment:** in the Results section, counterfactual marginals and any claims that rely on them should be supported by Protocols A and B. Protocol C should be used (and reported) when protocol sensitivity is observed or when stateful normalisation is known to be a confound.

**Placeholder:** [Insert protocol-sensitivity summary table and the exact implementation details for each protocol.]

### 5.3 Training-time attribution: (O(n+1)) evaluations for (n) concurrent seeds

With (n) concurrent seeds, full interaction attribution is exponential in (n) (Section 6). For training-time control, Esper uses a linear-cost approximation that remains informative and scalable.

At a given evaluation tick:

1. Evaluate the **full model** once:
   [
   \mathcal{E}(M_t)
   ]

2. For each seed (s_i), evaluate the model with that seed ablated:
   [
   \mathcal{E}(M_t \setminus s_i)\quad \text{for } i \in {1,\dots,n}
   ]

This yields (n) marginals:
[
m_i(t) = \mathcal{E}(M_t) - \mathcal{E}(M_t \setminus s_i)
]

Total cost: **(n+1) evaluations**, i.e. (O(n)) in the number of active seeds.

This is sufficient to drive the controller’s core operations:

* **Reward:** use (\sum_i m_i(t)) (with efficiency rent and terminal terms) as the primary signal for Tamiyo (Section 4).
* **Lifecycle gating:** detect non-performing seeds (persistently low/negative marginals) and candidates for fossilisation (stable positive marginals with acceptable cost).
* **Budget management:** trade off contribution against parameter growth and compute rent.
* **Auditability:** log marginals for offline analysis and regime labelling.

Because counterfactual evaluation itself can be expensive, the system can amortise cost by controlling cadence:

* perform marginals at stage boundaries (e.g., during BLENDING and PROBATION),
* reduce frequency for stable fossilised seeds,
* and increase frequency during turbulence (e.g., when instability triggers checkpointing or the turbulence budget is active).

**Placeholder:** [Insert the exact evaluation cadence policy used in experiments.]

### 5.4 Comparability across time: evaluation control and batching

Counterfactuals are only useful if they are comparable across time. Esper therefore treats the evaluation surface as part of the experimental contract:

* Use a fixed evaluation batch or fixed evaluation stream per run (or a deterministic schedule).
* Keep ablation protocol constant when analysing trends.
* Align evaluation ticks to lifecycle stages to avoid mixing “pre-blend” and “post-blend” semantics.

For sequence or RL tasks where evaluation is inherently stochastic, we recommend:

* reporting marginals as rolling estimates over multiple episodes,
* logging the variance or confidence interval of (\mathcal{E}) estimates,
* and ensuring that full and ablated evaluations share the same episode seeds where practical.

**Placeholder:** [Insert evaluation determinism strategy for each task family.]

### 5.5 Interpreting marginals in a dynamic morphogenetic system

Marginal counterfactual contributions are **local** signals. They should be read as “current dependency” rather than “intrinsic worth.”

In particular:

* A large (m_i(t)) means the system is currently relying on seed (s_i), but it does not distinguish:

  * symbiosis (healthy additive value),
  * parasitism (harmful displacement),
  * or metamorphosis (successor circuit under formation).
* A small or negative (m_i(t)) suggests the seed is not helping *at this moment*, but can occur transiently during early germination and blending, especially when the seed is not yet carrying traffic.
* Comparisons across time are most meaningful when evaluation batches/streams, ablation protocols, and stage context are held constant.

These limits motivate the interaction decomposition introduced in Section 6, which provides the missing information about synergy and interference between seeds.

## 6. Interaction Decomposition and Regime Signals

Marginal counterfactuals (Section 5) are sufficient for scalable control, but they are blind to synergy and interference. In multi-seed morphogenesis, seeds can cooperate (the whole is greater than the sum), conflict (the whole is less than the sum), or remain largely independent. Capturing that structure is essential for understanding regime dynamics (symbiosis vs parasitism vs metamorphosis) and for constructing tractable in-flight signals.

This section introduces interaction decomposition as a *diagnostic lens*: we do not claim Shapley-optimal credit assignment or complete causal identification. Instead, we aim for a practical, low-(n) decomposition that exposes when seeds are composing versus fighting, and that yields trajectory features (notably an interaction derivative) that are useful for control.

### 6.1 From marginals to interactions

Let (M) be the full model containing host (H) and a set of active seeds (S={s_1,\dots,s_n}). Let (\mathcal{E}(\cdot)) denote the evaluation function used for counterfactual measurement (e.g., accuracy on a fixed validation microbatch, averaged over a small stream). For a subset (A \subseteq S), let (M_A) denote the model with exactly the seeds in (A) enabled (and all other seeds ablated under a specified protocol). Let (M_{\varnothing}) denote the host-only model (no seeds enabled), evaluated under the same training history.

For two seeds (s_1, s_2), we define:

* Host-only performance:
  [
  e_{\varnothing} = \mathcal{E}(M_{\varnothing})
  ]
* Standalone performances:
  [
  e_1 = \mathcal{E}(M_{{s_1}}), \quad e_2 = \mathcal{E}(M_{{s_2}})
  ]
* Full (both enabled):
  [
  e_{12} = \mathcal{E}(M_{{s_1,s_2}})
  ]

Define the standalone gains over host-only:

[
g_1 = e_1 - e_{\varnothing}, \quad g_2 = e_2 - e_{\varnothing}
]

Then the pairwise interaction term is:

[
I_{12} = e_{12} - e_{\varnothing} - g_1 - g_2
]

Equivalently:

[
I_{12} = e_{12} - e_1 - e_2 + e_{\varnothing}
]

Interpretation:

* (I_{12} > 0): **synergy** — the combination outperforms additivity.
* (I_{12} < 0): **interference** — the combination underperforms additivity.
* (I_{12} \approx 0): near-additive / weak interaction.

This decomposition is intentionally simple. It is sufficient to answer the operational question: *are these seeds helping each other, harming each other, or mostly ignoring each other?*

### 6.2 Extension to three seeds and higher-order effects

For three seeds (s_1,s_2,s_3), we can compute all subset evaluations ((2^3 = 8)):

[
e_A = \mathcal{E}(M_A) \quad \text{for all } A \subseteq {s_1,s_2,s_3}
]

Pairwise interaction terms ((I_{12}, I_{13}, I_{23})) can be computed as above using the relevant subsets. A third-order interaction term can also be defined:

[
I_{123} = e_{123} - e_{12} - e_{13} - e_{23} + e_1 + e_2 + e_3 - e_{\varnothing}
]

In practice, we treat (I_{123}) as optional: it is informative when composition is strongly non-additive, but it is also noisier and less directly actionable for control than pairwise interactions. When reported, we use it to support qualitative claims about multi-seed succession rather than as a primary training signal.

### 6.3 Diagnostic scaling: (O(2^n)) capped at (n \le 3)

Full interaction decomposition requires evaluating all subsets of seeds, i.e. (2^n) forward passes for (n) seeds, which does not scale.

Esper therefore uses a dual measurement regime:

* **Training regime (scalable):** marginals computed with (O(n+1)) evaluations for any (n) (Section 5.3).
* **Diagnostic regime (decomposed):** full subset evaluation only when (n \le 3) (at most 8 evaluations), or during occasional sampled windows designed specifically for analysis.

This cap is a deliberate trade-off: it provides enough visibility to validate the interaction hypothesis and to develop in-flight signals, without imposing an exponential tax on routine training. For high-concurrency settings ((n \gg 3)), we rely on marginals for control and use sampled/rotating diagnostics to spot-check interaction structure.

**Reporting requirement:** when interaction diagnostics are used, the paper reports the diagnostic cadence, subset-evaluation batch/stream definition, and the ablation protocol used to enable/disable seeds.

### 6.4 Interaction terms as regime signals

Interaction terms provide structure that marginals alone cannot, and they align naturally with the three-regime taxonomy (Section 7).

**Symbiosis (host-preserving additive growth):**

* Standalone gains (g_i) are generally positive.
* Pairwise interactions (I_{ij}) are near zero or mildly positive.
* Seeds behave like mostly independent contributors; the host remains competent.

**Parasitism (harmful displacement / cannibalisation):**

* One seed’s standalone gain may be positive while another’s is near-zero or negative.
* Interactions tend to be persistently negative, reflecting interference.
* Large marginals can appear because the system is dependent, but dependency does not translate to net improvement.

**Metamorphosis (succession / productive replacement):**

* Interactions can be negative early (turbulence) as traffic and gradients reroute.
* Interactions trend toward zero or positive as later seeds learn to consume upstream outputs and the successor circuit stabilises.
* Conditional dependence can be healthy: a late seed can benefit specifically from upstream seed representations without that being parasitic, provided final Pareto outcomes improve and the system remains robust post-hygiene (Section 11).

A key point is that the sign of (I_{ij}) at a single time step is not decisive. The regime distinction is often **trajectory-defined**.

### 6.5 Interaction derivatives and “recovering turbulence”

Because parasitism and early metamorphosis can look identical in snapshot metrics (Section 8), we propose a trajectory feature that is both tractable and strongly tied to regime behaviour: the **interaction derivative over time**.

Let (I_t) denote a chosen interaction statistic at time (t) (typically a pairwise (I_{ij}), or a summary such as the minimum pairwise interaction when multiple are logged). Because (I_t) is noisy, we use an exponential moving average (EMA) to obtain (\tilde{I}_t). Over a window of (k) diagnostic ticks, define:

[
\Delta \tilde{I}_t = \tilde{I}*t - \tilde{I}*{t-k}
]

We hypothesise:

* **Parasitism:** (I_t < 0) with (\Delta \tilde{I}_t \le 0) (sticky negative interference).
* **Metamorphosis:** (I_t < 0) with (\Delta \tilde{I}_t > 0) (negative-but-recovering turbulence).

This is not intended as a deterministic classifier. It is an informative feature: it tells the controller and the analyst whether negative interaction is *resolving* or *entrenching*. In later sections, this derivative is used in two ways:

1. as part of the evidence for post-hoc regime classification; and
2. as a trajectory signal that can support turbulence budgets and risk controls (Section 9).

### 6.6 Telemetry requirements for interaction diagnostics

When diagnostics are enabled (typically (n \le 3)), Esper logs per diagnostic tick:

* subset performances (e_A) for all (A \subseteq S) in the diagnostic set
* derived standalone gains (g_i)
* pairwise interactions (I_{ij})
* optional higher-order interaction (I_{123}) (when (n=3))
* smoothed interactions (\tilde{I}_t) and window deltas (\Delta \tilde{I}_t)
* the evaluation batch/stream ID and ablation protocol identifier

This telemetry supports three downstream needs:

* validating that measured interactions are stable across ablation protocols (Section 14.3);
* stratifying interaction behaviour by slot depth ordering (Section 10); and
* distinguishing parasitism from metamorphosis via trajectory statistics (Sections 8–9).

---

## 7. Three Regimes of Seed Integration: Symbiosis, Parasitism, Metamorphosis

Seed grafting introduces a structural ambiguity into training dynamics: a module can become central to performance for reasons that are either desirable (genuine capability gain) or pathological (dependency through displacement). This section defines three regimes of integration—**symbiosis**, **parasitism**, and **metamorphosis**—and gives operational criteria for distinguishing them. The goal is not to “name vibes”, but to provide decision-relevant categories for Tamiyo and analysis-relevant categories for telemetry.

### 7.1 Why a taxonomy is necessary

Counterfactual contribution (e.g., the drop in evaluation score when a seed is ablated) is a necessary ingredient for morphogenetic control, but it is not sufficient. The same magnitude of counterfactual drop can arise when:

* the seed adds useful capability,
* the seed suppresses the host and makes itself mandatory,
* the seed participates in a successor circuit that replaces the host.

In other words, **indispensability is underdetermined** without context. A taxonomy makes that context explicit and prevents the control system from optimising the wrong objective (for example, prematurely culling productive successors because the host baseline fell).

### 7.2 Definitions and measurement primitives

Let:

* (H) denote the host network (with all seed slots disabled).
* (S = {s_1, \dots, s_n}) denote the set of active seeds.
* (M(H, S)) denote the full model obtained by inserting seeds (S) into the host (H).
* (\mathcal{E}(\cdot)) denote the evaluation function (e.g., accuracy on a fixed validation batch/stream, task reward, or another scalar metric).

For a given ablation protocol (\pi) (gate-to-zero, bypass/identity, optionally recalibrated), define the ablated model (M \setminus_{\pi} s_i) as the model with seed (s_i) disabled under protocol (\pi).

**Training-time marginal contribution** (Section 5) is defined as:

[
m_i^{(\pi)} ;=; \mathcal{E}\big(M(H,S)\big) ;-; \mathcal{E}\big(M(H,S)\setminus_{\pi} s_i\big)
]

This is a local, time-indexed credit signal: “how much worse do we get if we remove this seed right now?” It is not a global causal decomposition and it is blind to interactions among seeds.

When (n \le 3), Esper enables a diagnostic regime (Section 6) that evaluates all seed subsets and computes **interaction terms**. For two seeds (s_1) and (s_2), define:

* (acc_{\varnothing} = \mathcal{E}(M(H, \varnothing))) (host-only, same training history)
* (acc_{1} = \mathcal{E}(M(H, {s_1})))
* (acc_{2} = \mathcal{E}(M(H, {s_2})))
* (acc_{12} = \mathcal{E}(M(H, {s_1,s_2})))

Standalone gains:
[
g_1 = acc_1 - acc_{\varnothing}, \qquad g_2 = acc_2 - acc_{\varnothing}
]

Pairwise interaction:
[
I_{12} = acc_{12} - acc_{\varnothing} - g_1 - g_2
]

Interpretation:

* (I_{12} > 0): synergy (whole > sum of parts)
* (I_{12} < 0): interference (whole < sum of parts)
* (I_{12} \approx 0): independence / additivity

These primitives—marginals and interactions—are used both for controller learning (Tamiyo reward) and for post-hoc regime classification.

### 7.3 Symbiosis: host-preserving additive growth

**Concept.** A seed integrates by adding representational capacity the host did not have, while the host retains its baseline competence. The seed is useful, but not destructive.

**Operational signature (typical):**

* **Net improvement:** final performance improves relative to pre-germination baseline, (\Delta \mathcal{E}_{\text{final}} > 0).
* **Host retention:** host-only performance remains close to baseline (within a declared tolerance), (\mathcal{E}(M(H,\varnothing)) \approx \mathcal{E}(H_{\text{pre}})).
* **Positive marginals:** (m_i^{(\pi)} > 0) across ablation protocols, with stable ranking.
* **Low interaction:** interactions are near-zero or mildly positive; standalone gains are positive.

**Interpretation.** The seed behaves like an additive feature: removing it hurts, but the host path still functions. This is the regime where “importance” usually means what you want it to mean.

**Decision implications.** Fossilisation is appropriate when improvements persist across evaluations and the efficiency rent does not outweigh contribution. Hygiene is mostly about removing transient detritus rather than consolidating a replacement.

### 7.4 Parasitism: harmful displacement and brittle dependency

**Concept.** A seed becomes load-bearing by suppressing or cannibalising the host (and often other seeds), creating dependency without net value. The seed is “important” because it made itself indispensable.

**Operational signature (typical):**

* **Net regression:** final performance fails to improve or regresses relative to pre-germination baseline, often with degradation concentrated during blending.
* **Host collapse:** host-only performance drops sharply, (\mathcal{E}(M(H,\varnothing)) \ll \mathcal{E}(H_{\text{pre}})).
* **High dependency:** marginals (m_i^{(\pi)}) can be large, but reflect dependency rather than additive gain.
* **Sticky negative interaction:** pairwise interactions tend to be negative and do not recover, i.e., (I_t < 0) with (\Delta \tilde{I}_t \le 0) over relevant windows (Section 8).
* **Protocol fragility (often):** counterfactual estimates may vary substantially across ablation protocols, indicating measurement sensitivity or artefacts.

**Interpretation.** The seed acts as an optimisation sink: gradients route through it, the host receives diminished learning signal, and the system becomes brittle. The defining trait is not “host got worse” alone; it is “host got worse *and nothing coherent replaced it*”.

**Decision implications.** Culling is typically correct once negative net trajectory persists and recovery indicators (interaction derivative, gradient health, downstream consumption) fail to materialise. Importantly, parasitism is a trajectory diagnosis, not a snapshot diagnosis.

### 7.5 Metamorphosis: productive replacement through succession

**Concept.** Host displacement is allowed—sometimes required—because a successor circuit emerges that ultimately yields a better, leaner model after consolidation and hygiene. In this regime, the host is larval tissue: useful during early development, but not a required invariant.

Metamorphosis differs from symbiosis primarily in *what is preserved*: rather than preserving host-only competence, the system preserves (and improves) end-to-end competence while allowing internal reconfiguration.

**Operational signature (typical):**

* **Host-only can degrade:** (\mathcal{E}(M(H,\varnothing))) may fall during transition; this is not a failure condition.
* **Eventual Pareto improvement:** after succession and consolidation, the final model improves on a declared Pareto frontier (e.g., performance vs params/latency/FLOPs), relative to matched baselines.
* **Recovering turbulence:** interactions can be negative early (turbulence), but trend upward toward zero or positive, i.e., (I_t < 0) with (\Delta \tilde{I}_t > 0) over time.
* **Downstream consumption:** later seeds measurably consume upstream seed outputs in a meaningful way (visible as conditional dependence and/or positive interaction emerging late).
* **Larval dispensability post-hygiene:** after Emrakul’s hygiene passes, removing dissolved host pathways has minimal impact, while removing the successor circuit causes a large drop.

**Interpretation.** What looks like “the seed stole the model” can be a valid transition if later modules consolidate that representation into a coherent successor. Metamorphosis is not “parasitism that worked”; it is a different success criterion with different invariants.

**Decision implications.** The controller must tolerate bounded turbulence—often via a turbulence budget and risk controls (Section 9). Success should be judged on final Pareto outcomes and robustness, not on preserving host-only competence.

### 7.6 Regime-aware scorecards

Because metamorphosis relaxes host retention, a single scalar “health” metric is frequently misleading. We therefore recommend reporting regime-aware scorecards.

**Symbiosis scorecard**

* Final net improvement (\Delta \mathcal{E}_{\text{final}}) vs pre-germination baseline
* Host retention: (\mathcal{E}(M(H,\varnothing))) relative to baseline
* Marginal stability across ablation protocols (rank correlation, variance)
* Efficiency rent vs contribution (params/FLOPs/latency deltas)

**Metamorphosis scorecard**

* Final Pareto position vs matched baselines (fixed-host and full-fat)
* Seed economy (gain per seed; diminishing returns)
* Successor robustness (variance across runs; sensitivity to init/order)
* Hygiene delta (detritus ratio decreases; efficiency improves)
* Interaction recovery statistics (time-to-recovery, (\Delta \tilde{I}) distributions)

**Parasitism red flags**

* Net regression + host collapse
* Sticky negative interaction (no recovery)
* No downstream consumption signal
* High protocol sensitivity of counterfactuals
* Persistent instability / elevated collapse risk

**Placeholders:** [Map each scorecard field to a logged telemetry field and cite the corresponding figures/tables in Section 14.]

### 7.7 Summary: why this matters for control

This taxonomy exists because **Tamiyo’s incentives and safety mechanisms depend on which regime is unfolding**. Symbiosis can be rewarded and fossilised with relatively simple gates. Parasitism must be culled early to avoid entrenched dependency. Metamorphosis must be allowed to proceed through bounded turbulence, after which Emrakul can consolidate the adult topology.

The rest of the paper develops the practical machinery needed to make these regimes tractable:

* scalable counterfactual marginals for control (Section 5),
* capped interaction decomposition for diagnostics (Section 6),
* trajectory-based in-flight detection features (Section 8),
* probabilistic control and risk management (Section 9),
* hygiene and budget coupling that make metamorphosis safe and efficient (Sections 11–12).

---

## 8. The In-Flight Detection Problem: Early Metamorphosis vs Parasitism

The central operational difficulty in seed-grafted morphogenesis is that **parasitism** and **early-stage metamorphosis** can be indistinguishable mid-run. In snapshot metrics, both regimes can present the same early phenotype:

* the host pathway degrades (sometimes sharply),
* one or more seeds become counterfactually load-bearing,
* multi-seed interaction terms can be negative during blending and early probation.

Yet the correct response differs. Parasitism should usually be culled early to prevent entrenched dependency. Metamorphosis often needs *time* to complete a successor circuit and may look “bad” before it looks good.

This section frames regime identification as a tractable **trajectory inference** problem: regimes are weakly identifiable from instantaneous state, but become separable in expectation from **time-series statistics**.

### 8.1 Why snapshot rules fail

Any deterministic, snapshot-based gating rule—e.g., “cull if host-only accuracy drops below (X)” or “cull if interaction (I_t < 0)”—will reliably make the wrong call in a subset of runs. The reason is structural: the early phenotype is shared.

A seed can become indispensable because it adds capability (healthy), because it displaces the host (harmful), or because it is part of a transitional successor circuit (potentially healthy). Snapshot contributions and host-only performance can confirm that dependency exists, but they cannot explain **why** it exists or whether it will resolve into a stable adult topology.

In other words, in-flight regime detection is a partial observability problem: the latent regime is not directly observed; Tamiyo receives noisy proxies through telemetry. The best we can do mid-run is reduce uncertainty and manage risk.

### 8.2 Leading indicators of “recovering turbulence”

Although deterministic classification is unrealistic, the system can still look for **leading indicators** that negative integration dynamics are likely to resolve. We consider five families of indicators, chosen because they are measurable online and plausibly related to whether a successor circuit is forming:

| Signal family                         | Parasitism (harmful)                                                                | Early metamorphosis (productive)                                                       |
| ------------------------------------- | ----------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| **Seed gradient health**              | stagnant, noisy, or chaotic gradients; seed-path loss does not reliably decrease    | stable gradients; seed-path loss decreases in a consistent direction                   |
| **Seed representation structure**     | collapsed / degenerate features; low effective rank without task-specific structure | structured feature development; increasing or stabilising rank with meaningful spectra |
| **Host degradation shape**            | catastrophic or accelerating decline; no stabilisation                              | gradual decline that stabilises as successor forms                                     |
| **Later-seed behaviour** (if present) | ignores, competes with, or cancels earlier seed outputs                             | consumes upstream seed outputs; composition becomes detectable                         |
| **Interaction trajectory**            | persistently negative or worsening                                                  | negative but trending toward zero (and sometimes positive)                             |

Two notes matter here:

1. **No single indicator is decisive.** Each can be noisy, task-dependent, or misled by non-stationary training dynamics.
2. **Indicators are most useful as trends.** The absolute value of a proxy (e.g., a rank estimate) is less informative than whether it is stabilising, collapsing, or improving under continued training.

In practice, the most operationally convenient indicator is the **interaction trajectory**, because it directly captures whether seeds are cooperating or interfering, and because it aligns with the metamorphosis hypothesis: turbulence is acceptable if it is resolving.

### 8.3 Interaction derivatives as a practical discriminator

When diagnostic interaction decomposition is enabled (typically when (n \le 3)), we can compute interaction terms as in Section 6. For two seeds (S_1, S_2), define:

[
I_t = acc_{12,t} - acc_{\varnothing,t} - (acc_{1,t} - acc_{\varnothing,t}) - (acc_{2,t} - acc_{\varnothing,t})
]

where all accuracies are measured under a declared ablation protocol on a fixed evaluation stream.

Because (I_t) is noisy, we use a smoothed estimate (\tilde{I}_t) (e.g., EMA), and then compute a windowed change:

[
\Delta \tilde{I}_t = \tilde{I}*t - \tilde{I}*{t-k}
]

for a window size (k) chosen relative to the controller cadence.

**Hypothesis (trajectory separability, testable):**

* **Parasitism:** (I_t < 0) and (\Delta \tilde{I}_t \le 0) (sticky interference).
* **Metamorphosis:** (I_t < 0) but (\Delta \tilde{I}_t > 0) (negative-but-recovering turbulence).

This does not claim that the sign of (\Delta \tilde{I}_t) is a perfect classifier. Rather, it provides a *useful directional feature* that distinguishes “conflict that is resolving” from “conflict that is persisting”.

A practical advantage is that (\Delta \tilde{I}_t) is meaningful even when absolute interaction magnitudes differ across tasks or evaluation protocols: the derivative is scale-tolerant in a way that raw thresholds are not.

**Placeholders (to be populated in Results):**

* [Figure: (I_t) and (\tilde{I}_t) trajectories for parasitic vs metamorphic runs]
* [Figure: distribution of (\Delta \tilde{I}_t) early in training, coloured by final regime]
* [Table: AUC / PR for early-window prediction using (\Delta \tilde{I}_t), seed gradient health, and combinations]

### 8.4 “Consume upstream outputs” as a signature of successor formation

Metamorphosis is characterised by **succession**: later modules progressively become end-to-end meaningful by consuming upstream representations (which may be produced by earlier seeds that “stole” the host pathway).

We therefore treat downstream consumption as a structural signal: late-stage seeds should show measurable dependence on upstream seed outputs in a way that is consistent with **composition**, not just cancellation. In the diagnostic regime, this can be assessed by conditional effects:

* does the downstream seed’s standalone gain increase when upstream seeds are present?
* does interaction move from negative toward zero/positive as later seeds learn?

In high-concurrency regimes where full decomposition is infeasible, this can be approximated via cheaper proxies such as traffic flow and conditional marginals on sampled subsets, but the key point remains: productive metamorphosis tends to produce increasing coordination over time.

### 8.5 Practical implication for Tamiyo: act under uncertainty, manage downside

Because regimes are not deterministically identifiable mid-run, the controller should behave like a probabilistic decision-maker. The aim is not to “label” metamorphosis, but to learn which actions are good bets on average and to bound the cost of bad bets.

This motivates three practical controller behaviours (expanded in Section 9):

1. **Turbulence budgets:** tolerate negative interaction for a bounded horizon when seed health indicators remain good.
2. **Learning the odds:** train across many runs so the value function learns statistical associations between trajectories and eventual recovery.
3. **Adaptive checkpointing:** when turbulence rises, checkpoint more often so the system can roll back if recovery fails to materialise.

Under this framing, in-flight detection is not a binary classifier bolted onto the system; it is a set of trajectory features that make the control problem learnable and safer to explore.

### 8.6 Summary

Parasitism and early metamorphosis share an early phenotype, so snapshot gating rules are brittle. Trajectory features—especially interaction-term derivatives—provide a tractable way to distinguish “sticky interference” from “recovering turbulence” in expectation. This does not eliminate uncertainty, but it enables Tamiyo to make statistically grounded bets and to manage risk through bounded tolerance and rollback mechanisms.

---

## 9. Tamiyo as Probabilistic Control Under Uncertainty

Tamiyo operates in a setting where the “true regime” of integration (symbiosis, parasitism, or metamorphosis) is not reliably observable mid-run. In particular, parasitism and early metamorphosis can share the same snapshot phenotype: host degradation, a seed becoming load-bearing, and negative interaction terms during blending. As a result, Tamiyo is best modelled not as a regime classifier, but as a policy that maximises expected downstream return under uncertainty.

This section describes three mechanisms that make probabilistic control practical in Esper: (i) a **turbulence budget** that bounds how long the controller will tolerate recoverable turbulence, (ii) learning statistical associations across many runs rather than requiring deterministic mid-run labels, and (iii) **adaptive checkpointing and rollback** to make high-upside bets safe to explore.

### 9.1 Turbulence budget: bounded tolerance for recoverable interference

Metamorphic succession may require passing through an unstable integration window in which interaction terms are negative and host-only competence degrades. Parasitism can produce the same symptoms but does not resolve. A controller that culls aggressively on first sign of turbulence will suppress metamorphosis; a controller that never culls will accumulate brittle dependency. Esper therefore introduces a **turbulence budget**, which makes “hold your nerve” a bounded, auditable decision rather than an implicit hope.

Let (I_t) be an interaction estimate at time (t) (defined in Section 6), and let (\tilde{I}_t) be a smoothed version (e.g., EMA). Define a windowed interaction derivative
[
\Delta \tilde{I}_t = \tilde{I}*t - \tilde{I}*{t-k}.
]
We also assume a set of auxiliary “health” indicators (\mathcal{H}_t) such as seed-path loss trend, seed gradient norms, and host degradation slope (Section 8.2).

A turbulence budget can be implemented as a per-slot or global scalar (B_t \ge 0) that is consumed when the system remains in a turbulent state. Operationally:

* Turbulence is “active” when (\tilde{I}_t < 0) (or when stage-specific blending deltas are negative, if interaction diagnostics are unavailable).
* Budget is consumed more slowly when recovery indicators are positive (e.g., (\Delta \tilde{I}_t > 0) and seed gradient health is within bounds).
* Budget is consumed rapidly when turbulence is sticky (e.g., (\Delta \tilde{I}_t \le 0)) or health indicators indicate collapse risk.

One concrete sketch is:
[
B_{t+1} =
\begin{cases}
\min(B_{\max},, B_t + r) & \text{if recovery is strong} \
B_t - c_{\text{mild}} & \text{if turbulent but improving} \
B_t - c_{\text{hard}} & \text{if turbulent and not improving,}
\end{cases}
]
with (c_{\text{hard}} > c_{\text{mild}} > 0), and (r \ge 0) a small replenishment rate.

When (B_t) reaches zero, Tamiyo becomes biased toward actions that reduce downside—typically culling the offending seed, pausing germination, adjusting blending, or rolling back (Section 9.3). The important point is not the exact update rule, but the behavioural guarantee:

> Tamiyo may tolerate turbulence, but only within a bounded horizon and only while leading indicators suggest recovery remains plausible.

This framing also avoids turning metamorphosis into “free chaos”: negative interaction is permitted, but it has to *earn its keep* by trending toward recovery.

**Reporting requirements (placeholders):**

* (B_{\max}), (k), smoothing method for (\tilde{I}_t), and which indicators comprise (\mathcal{H}_t): [ ]
* Ablation comparing turbulence budget on/off: premature cull rate, collapse rate, Pareto wins: [ ]

### 9.2 Learning the odds: policies trained across many runs

Because regimes are latent, deterministic mid-run rules are brittle. Tamiyo instead learns a mapping from observation trajectories to actions that maximise expected return. In practice, this means learning empirical regularities of the form:

> “In states like this, holding for (N) steps tends to pay off with probability (p); in states like that, it rarely does.”

This shifts regime discrimination into value estimation: rather than labelling “metamorphosis”, the policy learns that certain trajectory patterns are high-upside.

Two aspects of the system make this feasible:

1. **Factored actions with masking** (Section 3) reduce combinatorial explosion and keep exploration sane.
2. **Minimal reward** (Section 4) preserves a learnable signal-to-noise ratio. Over-shaped rewards that anticipate every failure mode tend to create “everything is punished” landscapes, encouraging a do-nothing equilibrium. Minimal counterfactual-grounded reward, by contrast, lets the policy see the actual payoff structure.

This approach also matches the practical objective. The controller does not need interpretability in the form of explicit regime labels. It needs a policy that, averaged across runs, makes better bets: tolerate turbulence when it tends to resolve, and cut losses when it tends not to.

**Reporting requirements (placeholders):**

* Action distributions conditioned on binned indicators ((\tilde{I}_t), (\Delta \tilde{I}_t), seed gradient health): [ ]
* Recovery probability estimates by state cluster (or indicator bins): [ ]

### 9.3 Adaptive checkpointing and rollback: making turbulence safe to explore

Even a well-trained policy will sometimes bet wrong. Metamorphosis is inherently a controlled gamble: accept short-term regressions to pursue a superior successor topology. Esper therefore treats checkpointing not as a purely operational concern, but as part of the control strategy.

The idea is simple: when early-warning signals indicate rising risk, increase checkpoint frequency so the system can roll back if recovery fails to materialise.

Let (R_t) be a risk score computed from telemetry. A minimal implementation might combine:

* turbulence indicators (e.g., (\tilde{I}_t < 0), negative blending deltas),
* non-recovery indicators (e.g., (\Delta \tilde{I}_t \le 0)),
* and stability signals (loss variance spikes, gradient explosions, rapid host degradation).

Checkpoint cadence can then be adapted via a rule such as:
[
\text{checkpoint_interval}_t = f(R_t),
]
where higher risk implies shorter intervals. Rollback triggers can be defined in terms of:

* budget exhaustion ((B_t = 0)),
* crossing a hard stability threshold,
* or failing to show recovery within (N) steps of sustained turbulence.

Operationally:

1. Enter turbulence → increase checkpoint frequency.
2. If recovery indicators improve → resume normal cadence.
3. If recovery fails within the turbulence budget → roll back to last stable checkpoint and take an alternative branch (e.g., cull, switch blueprint, change slot, alter blending).

This converts topology changes from irreversible commitments into reversible exploration steps. It also allows Tamiyo to be less conservative: high-upside transitions can be attempted because the downside is bounded.

**Reporting requirements (placeholders):**

* Definition of (R_t) and checkpoint policy (f(\cdot)): [ ]
* Compute/storage overhead of adaptive checkpointing: [ ]
* Collapse rate and final performance with/without rollback: [ ]

### 9.4 Coupling probabilistic control to shared energy allocation

The turbulence budget and checkpointing mechanisms integrate naturally with the shared energy pool described in Section 12. During unstable periods, energy is preferentially allocated to Tamiyo (growth/control) and to richer diagnostics or risk controls (e.g., higher checkpoint cadence). During stable periods, energy shifts toward Emrakul (hygiene) to prune detritus and consolidate the adult topology.

In metamorphic runs, this coupling yields an intuitive sequence:

1. **Instability rises** during early succession → Tamiyo receives budget to act and risk-manage.
2. **Recovery indicators improve** (e.g., (\Delta \tilde{I}_t > 0)) → turbulence budget stops draining; checkpoint cadence relaxes.
3. **Stability increases** → energy shifts toward Emrakul to prune dissolved host tissue, improving efficiency and attribution clarity.

This makes “growth vs hygiene” an adaptive systems variable rather than a fixed engineering schedule, and provides a coherent control-theoretic interpretation: instability funds corrective action; stability funds consolidation.

**Reporting requirements (placeholders):**

* Energy allocation traces alongside stability (S_t) and key lifecycle events: [ ]
* Comparison of fixed vs sloshing budgets under identical training budgets: [ ]

---

## 10. Depth Ordering and Traffic Flow in Multi-Slot Morphogenesis

Esper’s seed bank is typically instantiated across multiple **slots** at different depths in the host network (for example: early/post-block1, mid/post-block2, late/post-block3). This introduces depth-dependent dynamics that do not appear in single-slot morphogenesis: where a seed is grafted changes what it can “see”, what gradients it receives, and how it competes or composes with other seeds.

Throughout this section we treat slot depth as a structural prior, not a deterministic rule. The same slot pairing can produce different outcomes depending on blueprint capacity, blending schedule, and the controller’s risk posture.

### 10.1 Slot depth as an implicit coordination problem

With multiple concurrent seeds, the system is no longer “host + independent add-ons”. It becomes a coordination problem in which seeds can either:

* **compete** for the same representational niche (interference),
* **compose** (downstream seeds consume upstream representations), or
* remain mostly **independent** (additive improvements).

Depth matters because it changes the causal ordering of representations.

An **early** slot sits upstream of most computation. If an early seed begins to dominate traffic, much of the downstream network is effectively trained on *seed-produced features* rather than the host’s original features. This can look like “host collapse” in snapshot terms, but it is not inherently parasitic: it may be a valid step in metamorphic succession if later seeds (or the downstream host blocks) learn to exploit the new representation and the final organism is better after consolidation.

By contrast, seeds in **adjacent** depths (e.g., early+mid) are more likely to tug on similar feature sets and gradients at roughly the same level of abstraction. This increases the risk of mutual interference: negative interaction that is not a transitional phase but a stable failure mode.

A **late** slot is positioned to act as a consolidator. Late seeds receive gradients that reflect end-to-end task performance and can learn to “consume” upstream signals—whether those signals originate from the host, from earlier seeds, or from a mixture. In metamorphosis, this is the point where a successor circuit can become end-to-end meaningful.

### 10.2 Traffic-flow hypothesis

We hypothesise that slot depth influences regime outcomes primarily through **traffic flow** and its effect on optimisation dynamics.

Let (T_{L}(t)) denote a traffic measure for slot (L) at time (t). “Traffic” may be defined as any consistent utilisation proxy (activation magnitude, routing probability, gradient flow share); the specific choice must be reported (Section 13.4). The qualitative hypothesis is:

1. **Upstream dominance induces representation capture.**
   When an early seed carries a large share of traffic, downstream layers adapt to its representation. This increases conditional dependence: later modules learn on top of the seed’s features.

2. **Adjacent slots increase competition.**
   Seeds at neighbouring depths receive similar gradient signals and may attempt to solve overlapping subproblems. This increases the chance that interaction terms remain persistently negative.

3. **Downstream slots enable consolidation.**
   Late seeds can learn to fuse and refine upstream signals, potentially converting an initially turbulent multi-seed system into a coherent successor circuit.

This framing is deliberately regime-agnostic. The same phenomenon (“later layers depend on early seed features”) can be:

* **parasitic** if it produces sticky interference and net regression, or
* **metamorphic** if it leads to an improving successor topology that later stabilises and can be consolidated via hygiene (Section 11).

### 10.3 Depth-pair priors and testable predictions

We propose the following *priors* (not guarantees) about slot-pair dynamics. These are intended to be tested and falsified via interaction telemetry (Section 13.6) and post-hoc regime classification (Section 13.8).

**Early + Late (separated by the host mid-stack)**

* *Prior:* lower direct competition; higher opportunity for composition.
* *Prediction:* interaction terms are closer to zero on average, or show negative-then-recovering behaviour in metamorphic runs.

**Early + Mid (adjacent)**

* *Prior:* higher interference risk due to overlapping representational scope.
* *Prediction:* higher incidence of persistently negative interaction (“sticky negative”), especially when both blueprints have high capacity relative to the host region.

**Mid + Late (adjacent but downstream-biased)**

* *Prior:* mixed: can interfere, but late slot has consolidation leverage.
* *Prediction:* more cases of negative interaction that resolves, particularly when the late blueprint is capacity-limited and forced to compose rather than dominate.

These predictions are deliberately phrased in interaction language because interaction provides a better lens than marginals alone: a large marginal contribution can coexist with either synergy or cannibalisation.

### 10.4 Measurement: stratified interaction telemetry and traffic diagnostics

To connect depth ordering to regime outcomes we recommend a stratified analysis plan.

**(1) Stratify by slot pairing and blueprint capacity**
For each run, record:

* slot occupancy pattern (e.g., early+late, early+mid, mid+late)
* blueprint IDs and parameter counts
* relative capacity: (\text{params(seed)}/\text{params(host region)}) where defined

Then compare regime frequencies and interaction statistics across strata.

**(2) Use interaction decomposition where feasible ((n \le 3))**
For diagnostic runs with (n \le 3), compute and log:

* (acc_{\varnothing}) (host-only)
* (acc_{S_i}) standalones
* (acc_{S_i,S_j}) combinations
* interaction terms (I_t), smoothed (\tilde{I}_t), and derivative (\Delta \tilde{I}_t)

Primary comparison variables:

* distribution of (I_t) by slot pair
* distribution of (\Delta \tilde{I}_t) by slot pair
* fraction of runs with “sticky negative” vs “recovering turbulence”

**(3) Track traffic share over time**
Log traffic metrics per slot:

* (T_{\text{early}}(t)), (T_{\text{mid}}(t)), (T_{\text{late}}(t))
* seed vs host traffic share within each slot (where blending supports separation)

Key questions:

* Do metamorphic runs show an early-seed traffic rise followed by late-seed consolidation?
* Do parasitic runs show traffic capture without subsequent recovery in (\Delta \tilde{I}_t)?
* Do certain slot pairs exhibit systematic oscillation (traffic sloshing) that correlates with collapse?

**(4) Connect traffic to Emrakul decisions**
Because Emrakul prunes low-traffic components, traffic telemetry also determines which “larval tissue” is likely to be removed after turbulence resolves. For metamorphosis, a clean story is:

* early transition: traffic shifts, interaction negative
* mid transition: interaction derivative becomes positive
* consolidation: traffic stabilises on successor paths
* hygiene: low-traffic host remnants are pruned, improving efficiency without performance loss

This closes the loop between depth ordering (where composition happens) and hygiene (what remains at the end).

**Placeholders:**

* [Figure: interaction trajectories grouped by slot pair]
* [Figure: traffic share per slot over time, annotated with lifecycle events]
* [Table: regime frequencies by slot pair and blueprint capacity]
* [Table: interaction derivative statistics by slot pair]

### 10.5 Implications for controller design

Depth ordering is not only an analysis tool; it can inform controller features and action constraints.

* **Feature design:** include slot depth and adjacency indicators in Tamiyo’s observation vector (Section 3.2), so the policy can learn that some pairings are riskier than others.
* **Action masking or priors (optional):** in early training, the system may restrict certain high-risk combinations (e.g., two high-capacity adjacent seeds) unless stability signals are strong.
* **Turbulence budgeting:** adjacency-heavy configurations may require a larger turbulence budget to allow recoverable transitions, but also stricter rollback triggers when recovery fails (Section 9).

The important point is that depth is not merely “where you plug the seed in”. It shapes the optimisation landscape and the probability that turbulence resolves. Treating depth explicitly makes the metamorphosis/parasitism distinction more tractable and makes growth policies more learnable.

---

## 11. Emrakul: Model Hygiene and Detritus Consumption

Morphogenesis is messy by design. During germination, blending, and succession, the network temporarily contains redundant pathways, partially connected structures, and components that receive little or no traffic. If left unaddressed, this **detritus** creates three practical problems:

1. **Inefficiency:** dead parameters and low-traffic branches waste compute, memory, and bandwidth.
2. **Attribution confusion:** counterfactual contribution becomes less trustworthy when unused pathways distort baselines or interact with normalisation state.
3. **Instability:** stale statistics and noisy gradients accumulate in rarely-used components, increasing the risk of brittle behaviour during later integration.

Esper introduces **Emrakul** as a hygiene subsystem whose job is to make the evolving topology *legible, efficient, and safe to keep modifying* by removing detritus and consolidating the “adult” architecture that emerges after metamorphic transitions.

### 11.1 What counts as detritus?

We treat detritus as structure that is either **unreachable** (graph-dead) or **persistently low-traffic** (connected but functionally unused).

**Unreachable structure.** A component is unreachable if it lies on no path from the model’s inputs to the loss under the current wiring. Unreachability is typically introduced by topology edits: a seed is culled, a bypass path is rewired, a slot’s blend operator changes, or a gating configuration is updated. These components should be removed aggressively: they do not affect behaviour and only impose overhead and bookkeeping complexity.

**Persistently low-traffic structure.** A component can be connected yet unused because its output is effectively ignored. In Esper, “traffic” may be measured in multiple ways depending on the architecture and blend mechanism, but the common requirement is that traffic be:

* **cheap** to estimate and log,
* **stable** enough to support thresholding,
* and **meaningfully correlated** with “this component participates in computation”.

Concrete traffic proxies include:

* mean absolute activation (or activation energy) through the component,
* routing or gating probability (for mixture / gated paths),
* gradient-flow proxies (e.g., norm of gradients entering or leaving the component),
* token/patch share for routed architectures.

In experiments, we treat one proxy as the **default** and any others as sensitivity checks. This reduces “threshold tuning” ambiguity and supports reproducibility.

### 11.2 Emrakul’s responsibilities

Emrakul performs hygiene operations that are conceptually separate from Tamiyo’s policy decisions. Tamiyo decides *what to try*; Emrakul ensures the result remains clean enough to measure and efficient enough to keep training.

Emrakul’s responsibilities are:

1. **Graph pruning (reachability cleanup).**
   Remove unreachable subgraphs after topology changes. This includes orphaned modules, disconnected buffers, and “dead” branches left behind after culls or rewires.

2. **Traffic-based pruning (detritus consumption).**
   Remove components whose traffic falls below a threshold (\tau) for long enough to be confident the low usage is not transient. This is the “eating away” behaviour: detritus is consumed as the adult topology stabilises.

3. **Cleanup around lifecycle events.**
   Culls and fossilisation can change routing patterns and expose unreachable or low-traffic structure. Hygiene passes are naturally triggered near these events. Where normalisation layers are sensitive, hygiene can also optionally trigger a lightweight recalibration pass (without weight updates) to reduce measurement artefacts in subsequent counterfactual evaluations.

This separation of concerns matters for the metamorphosis framing: host displacement is not inherently harmful if the system actively removes dissolved tissue and consolidates the successor circuit into a coherent topology.

### 11.3 Hygiene metrics (reporting standard)

Because “cleanliness” is otherwise hand-wavy, we recommend reporting explicit hygiene metrics, including before/after deltas around hygiene passes.

Let (\mathrm{params}(\cdot)) be the number of parameters in a component and (T(c)) be its measured traffic. Define:

* **Disconnected mass:**
  [
  M_{\text{dead}} = \sum_{c \in \mathcal{C}_{\text{unreachable}}} \mathrm{params}(c)
  ]

* **Low-traffic mass:**
  [
  M_{\text{low}}(\tau) = \sum_{c \in \mathcal{C}} \mathbf{1}[T(c) < \tau] \cdot \mathrm{params}(c)
  ]
  with an additional minimum-age constraint (Section 11.4) to avoid pruning newborn circuits.

* **Detritus ratio:**
  [
  R_{\text{detritus}} = \frac{M_{\text{dead}} + M_{\text{low}}}{M_{\text{total}}}
  ]

* **Hygiene delta:** report (R_{\text{detritus}}) before and after hygiene passes.

In addition to topology cleanliness, hygiene should be evaluated on outcomes that matter for a morphogenetic system:

* **Efficiency deltas:** latency, FLOPs, VRAM, throughput changes post-hygiene (at matched evaluation settings).
* **Attribution stability:** reduction in variance of counterfactual marginals across ablation protocols pre/post hygiene.
* **Behavioural stability:** collapse rate and run-to-run variance under matched conditions.

**Placeholders:**

* [Figure: detritus ratio over time with hygiene events marked]
* [Table: efficiency deltas post-hygiene]
* [Table: counterfactual protocol sensitivity pre/post hygiene]

### 11.4 Safety and timing: don’t prune the baby

The central risk of hygiene is removing structure that is temporarily low-traffic but would become useful if given time—particularly during blending or early succession. Emrakul therefore requires explicit guardrails.

**Minimum age.** Components are not eligible for traffic-based pruning until they have been present for at least (A_{\min}) steps (or epochs), and ideally until they have spent a minimum time outside the most fragile integration stages.

**Stability gating.** Hygiene should be more aggressive when training is stable and more conservative when volatile. This naturally connects to the stability signal (S_t) used for shared energy budgeting (Section 12): high stability permits higher pruning cadence and lower traffic thresholds.

**Connectivity constraints.** Pruning must preserve at least one valid path from inputs to loss, and should avoid removing the last remaining instance of a capability unless a successor is already carrying traffic.

**Rollback compatibility.** If adaptive checkpointing/rollback is enabled (Section 9), hygiene decisions should be logged and replayable so a rollback does not produce a divergent topology-history that confounds attribution comparisons.

These constraints are less about “being cautious” in general and more about ensuring hygiene is compatible with metamorphosis: in succession, short-lived redundancy is expected, but it must be removed only once the successor circuit has genuinely stabilised.

### 11.5 Why hygiene changes the interpretation of takeover

Without hygiene, host displacement can look like parasitism even when it is a productive transition: unused host tissue lingers, detritus inflates compute cost, and counterfactual baselines become noisy. With hygiene, metamorphosis becomes legible:

* dissolved tissue is removed,
* the final topology is smaller and clearer,
* and contribution estimates stabilise because “dead branches” no longer distort evaluation.

In this sense, Emrakul is not merely an optimisation. It is a prerequisite for safely scaling morphogenetic systems: if the architecture is allowed to grow dynamically, it must also be able to *clean up after itself*.

**Placeholders:**

* [Evidence: comparison of morphogenesis with vs without Emrakul on detritus ratio, latency, and counterfactual stability]

## 12. Shared Energy Budget: Stability-Coupled Allocation Between Growth and Pruning

Tamiyo (growth/control) and Emrakul (hygiene/pruning) both consume scarce “attention” and compute. Germination, counterfactual evaluation, and staged blending are expensive. So are hygiene passes, pruning sweeps, and any recalibration needed to keep counterfactuals honest. More importantly, Tamiyo and Emrakul represent competing priorities:

* when training is going poorly, the system usually needs **corrective capacity and intervention** (growth);
* when training is going well, the system benefits from **consolidation and compaction** (pruning).

Hard-coding a fixed schedule (“grow for X, prune for Y”) is brittle across tasks, architectures, and phases of learning. We therefore propose a shared budget mechanism that dynamically allocates resources between growth and hygiene as a function of measured stability.

### 12.1 Stability as a control signal

Let (S_t \in [0, 1]) denote a stability estimate computed from telemetry at controller time (t), where larger values indicate more stable learning dynamics. The precise definition is implementation-dependent, but it should be:

* cheap to compute;
* robust to noise and transient spikes;
* correlated with “safe to prune” versus “need to grow”.

A stability estimate can be built from any combination of the following (logged anyway for control and diagnostics):

* windowed variance of training loss and/or validation score;
* oscillation measures (e.g., sign changes in smoothed improvement slope);
* gradient norm spikes or NaN/Inf sentinels;
* plateau confidence (stable but not improving);
* controller uncertainty proxies (optional).

Because these signals are noisy, we treat (S_t) as a **smoothed** quantity. In practice we compute (S_t) from windowed aggregates and apply an exponential moving average (EMA), and we apply hysteresis to prevent rapid mode switching (Section 12.4).

**Reporting commitment (placeholders):**

* stability definition: ([,])
* window length / EMA factor: ([,])
* hysteresis thresholds: ([,])

### 12.2 Budget allocation rule

Let (E) be the total available “energy” per control window (a unitless budget that can be mapped to compute allowance, evaluation calls, pruning operations, or any bounded resource the system cares about). We allocate this energy between Emrakul and Tamiyo as:

[
E_{\text{emrakul}}(t) = E \cdot S_t,\qquad
E_{\text{tamiyo}}(t) = E \cdot \bigl(1 - S_t\bigr).
]

Interpretation:

* **stable learning ((S_t \approx 1)) → Emrakul gets budget:** prune detritus, consolidate topology, reclaim compute;
* **unstable learning ((S_t \approx 0)) → Tamiyo gets budget:** germinate corrective seeds, adjust integration, “stop the bleeding”.

This “sloshing” behaves like a thermostat: stability funds compaction; instability funds growth. Importantly, this is not only about compute efficiency—budget allocation changes which *operations* are available and how aggressively they run.

### 12.3 What the budget actually buys

The shared pool can be implemented in several concrete ways. The paper’s core requirement is that **budget has enforceable consequences**, rather than being a purely diagnostic number.

**Tamiyo-side budget uses** (growth/control):

* permitting **GERMINATE** actions only when (E_{\text{tamiyo}}(t)) is above a minimum;
* increasing the cadence or resolution of counterfactual measurement (e.g., more frequent (O(n+1)) marginals) when intervention is needed;
* allowing additional diagnostic sampling windows (e.g., temporarily enabling (O(2^n)) decomposition for (n \le 3));
* funding adaptive checkpointing frequency during turbulence (Section 9.3).

**Emrakul-side budget uses** (hygiene/consolidation):

* enabling pruning sweeps (unreachable subgraphs, low-traffic components) more frequently when stable;
* allowing “deeper” hygiene (e.g., more aggressive thresholds, more components evaluated) when stability is high;
* funding post-prune recalibration passes (when necessary for attribution stability);
* triggering consolidation actions after successful metamorphic transitions (prune dissolved host tissue).

The key design property is that stability-coupled budgeting yields the behaviour we want without hard-coded phase schedules: when the system is “thrashing”, it spends budget on recovery and safety; when it is calm, it spends budget on cleanup and efficiency.

### 12.4 Coupling to turbulence budgets and risk management

The shared pool complements the turbulence framework from Section 9. When the policy takes a high-turbulence bet (e.g., tolerating negative interaction because (\Delta \tilde{I} > 0) and seed gradient health is good), the system can temporarily reserve energy for:

* higher checkpointing frequency;
* richer diagnostics (where feasible);
* reduced pruning aggressiveness (avoid pruning nascent successor circuits);
* extended blending windows (avoid abrupt transitions).

When turbulence resolves—e.g., interaction recovers, gradients stabilise, and performance trends improve—the energy naturally shifts toward Emrakul, which can prune dissolved tissue and consolidate the adult topology.

This interaction is intentional: metamorphosis is allowed to be turbulent, but the system should spend *more* on safety during turbulence and *more* on compaction after turbulence.

### 12.5 Guardrails: preventing oscillation and starvation

A naive budget split can oscillate in response to noise: small fluctuations in stability can repeatedly flip the system between “grow” and “prune” modes. It can also starve one subsystem entirely, creating failure modes:

* **growth starvation:** no budget to germinate even when the model is failing;
* **hygiene starvation:** detritus accumulates until attribution and efficiency collapse.

We therefore recommend three guardrails.

**(1) Smoothing and hysteresis**
Compute (S_t) from windowed aggregates, smooth it, and apply hysteresis bands so that mode changes require sustained evidence.

**(2) Minimum allocation floors**
Enforce:

[
E_{\text{tamiyo}}(t) \ge E_{\text{tamiyo}}^{\min},\qquad
E_{\text{emrakul}}(t) \ge E_{\text{emrakul}}^{\min},
]

so neither subsystem is ever completely disabled.

**(3) Switching penalties / cooldowns (optional)**
Introduce a cooldown after large budget shifts, or penalise frequent switching to encourage stable allocation.

**Reporting commitment (placeholders):**

* floors: [E_{\text{tamiyo}}^{\min}=]( ), [E_{\text{emrakul}}^{\min}=]( )
* cooldown policy: [ ]
* hysteresis bands: [ ]

### 12.6 Failure modes and expected trade-offs

A shared pool does not remove the underlying trade-offs; it makes them explicit and controllable.

* If pruning is too aggressive under “stable” conditions, Emrakul can delete useful redundancy or nascent successor circuits (Section 11.4).
* If growth is over-funded during mild instability, Tamiyo can over-germinate and bloat the topology.
* If stability signals lag, allocation can arrive too late (pruning during fragile transitions, or growing after collapse has begun).

These failure modes are testable and should be evaluated directly via the experimental grid (Section 13), particularly with ablations comparing fixed schedules versus sloshing budgets.

### 12.7 Why this matters for scaling

As concurrency increases (more slots, more seeds, potentially multiple controllers), both detritus and decision complexity scale up. Fixed heuristics become increasingly brittle. A stability-coupled budget provides a simple coordinating principle that generalises:

* don’t over-prune while the organism is still forming;
* don’t grow indefinitely once the organism is stable;
* allocate scarce diagnostic attention to the moments where it matters most.

In short, the shared energy pool turns “growth versus hygiene” from a hand-tuned engineering constant into an adaptive systems variable—exactly the kind of move a morphogenetic framework should make.

**Placeholders:**

* [Figure: stability (S_t) and budget split over time, annotated with lifecycle events]
* [Table: fixed vs sloshing budget comparisons—Pareto outcomes, collapse rates, overhead]

---

## 13. Experiments

This section specifies the experimental programme and telemetry required to evaluate (i) symbiosis vs parasitism vs metamorphosis, (ii) Tamiyo’s control behaviour under uncertainty, (iii) interaction-trajectory diagnostics, (iv) Emrakul hygiene effects, and (v) the shared stability-coupled energy budget. Numerical outcomes and plots are reported in Section 14.

### 13.1 Research questions

**RQ1 — Regimes.** Under what conditions do runs converge to symbiosis, parasitism, or metamorphosis?

**RQ2 — Control and learnability.** Can Tamiyo learn effective lifecycle policies with the minimal reward, and does the over-shaped reward collapse learning?

**RQ3 — In-flight detection.** Do trajectory statistics—especially the interaction-term derivative—separate early metamorphosis from parasitism *in expectation*?

**RQ4 — Hygiene.** Does Emrakul reduce detritus and improve efficiency and attribution stability without suppressing useful growth or succession?

**RQ5 — Budget thermostat.** Does stability-coupled energy allocation reduce collapse rate and improve Pareto outcomes compared to fixed schedules/splits?

> **Definition (collapse).** We treat a run as “collapsed” if [insert criterion: e.g., NaNs/divergence; sustained validation accuracy below random baseline; irrecoverable loss explosion; etc.]. The exact operational threshold is reported with results and held fixed across conditions.

### 13.2 Tasks, datasets, and evaluation splits

All tasks are specified as versioned *TaskSpecs* with frozen preprocessing and evaluation protocols.

* **Vision:** [e.g., CIFAR-10 / CIFAR-100], standard train/val/test splits.
* **Language:** [e.g., TinyStories], fixed held-out eval prompts with a declared decoding protocol.
* **Optional stress tests:** [e.g., distribution shift, corruption suites, staged curricula].

**Must report (placeholders):**

* Dataset versions and preprocessing: [ ]
* Augmentation (if any): [ ]
* Evaluation cadence and eval set size: [ ]
* Hardware, precision, batch sizes: [ ]
* Training budget definition (steps/tokens and wall-clock): [ ]

### 13.3 Baselines and comparison families

To avoid “wins by moving the goalposts”, all comparisons include matched-resource baselines.

1. **Fixed host (no morphogenesis).** Identical optimiser, training budget, and evaluation protocol.
2. **Full-fat baseline.** A monolith matched on a declared budget axis (either FLOPs or wall-clock), with deltas on the other axes reported.
3. **Morphogenesis − Emrakul.** Identical control stack but hygiene disabled, isolating pruning effects.
4. **Morphogenesis + Emrakul.** Hygiene enabled.
5. **Controller variants.**

   * Tamiyo with **minimal reward** (≈64-line design)
   * Tamiyo with **over-shaped reward** (historical ≈974-line design)
   * Optional heuristic controller (Tamiyo-off) for sanity checks
6. **Energy budget variants.**

   * fixed split / fixed schedules
   * stability-coupled sloshing budget

**Matching rule (must be explicit):** comparisons are matched on training steps (or tokens/images seen) and one of FLOPs or wall-clock time; deltas for the remaining axes are reported.

### 13.4 Experimental grid

The grid is designed to separate architectural effects from controller effects.

* **Seed blueprints:** [depthwise], [attention-lite], [conv-heavy], …
* **Slot depth positions:** early (post-block1), mid (post-block2), late (post-block3)
* **Concurrency:** (n \in {1,2,3,10})
* **Blending schedules:** [alpha schedules], [STE/no-STE], [gating variants]
* **Hygiene cadence:** none / periodic / stability-triggered
* **Traffic metrics for pruning:** [activation magnitude], [routing probability], [gradient-flow proxy]
* **Action masking rules:** versioned and logged
* **Checkpoint policies:** fixed vs adaptive (warning-triggered)
* **Diagnostics policy:** full decomposition enabled only when (n \le 3), or on sampled windows.

Each run logs: config hash, RNG seeds, host architecture hash, seed blueprint IDs, controller checkpoint IDs, and (if enabled) energy-pool allocation traces.

### 13.5 Counterfactual measurement protocols

Counterfactual contribution is defined with respect to an explicit ablation protocol. All runs report contributions under at least two ablation modes:

* **Gate-to-zero (soft excision).** Seed output forced to zero while preserving graph shape.
* **Bypass/identity (structural bypass).** Seed path replaced by a defined bypass where valid.

Optional third protocol (recommended when normalisation artefacts are suspected):

* **Recalibrated ablation.** After ablation, refresh normalisation statistics on a calibration set *without weight updates*.

For a model (M) with active seed set (S={s_1,\dots,s_n}) and evaluation function (\mathcal{E}(\cdot)), define the training-time marginal for seed (s_i) as:

[
m_i ;=; \mathcal{E}(M);-;\mathcal{E}(M \setminus s_i),
]

where (M \setminus s_i) denotes ablation of (s_i) under the declared protocol.

Two computation regimes are used:

* **Training attribution (scales as (O(n))).** Compute (\mathcal{E}(M)) once and (\mathcal{E}(M \setminus s_i)) for each active seed, yielding (n+1) evaluations.
* **Diagnostic decomposition (scales as (O(2^n)), capped).** For (n \le 3), evaluate all subsets of seeds and compute interaction terms (Section 6).

**Must report (placeholders):**

* Evaluation batch/stream used for counterfactuals: [ ]
* Counterfactual cadence (per step/epoch/stage boundary): [ ]
* Whether recalibration is used, and how: [ ]
* Whether ablations preserve RNG state and data order: [ ]

### 13.6 Interaction telemetry (diagnostic runs, (n \le 3))

When diagnostic decomposition is enabled, log at each diagnostic tick:

* (acc_{\varnothing}): host-only (no seeds active)
* (acc_{S1}), (acc_{S2}), (acc_{S3}): standalones
* (acc_{S1,S2}), (acc_{S1,S3}), (acc_{S2,S3}), (acc_{S1,S2,S3}): combinations
* Interaction terms (I_t) (pairwise and, if used, higher-order)
* Smoothed interaction (\tilde{I}_t) (EMA or equivalent)
* Windowed derivative (\Delta \tilde{I}_t = \tilde{I}*t - \tilde{I}*{t-k})

This directly supports the in-flight detection hypothesis: parasitism tends to produce persistently negative (“sticky”) interaction; metamorphosis tends to produce negative interaction that recovers ((\Delta \tilde{I}_t>0)).

### 13.7 Seed and host “health” telemetry

To support leading indicators (Section 8) and turbulence management (Section 9), log:

* **Seed-path health (per seed/slot):** loss, gradient norms, update magnitudes, and stage-specific deltas.
* **Host-path health:** the same metrics on the host pathway.
* **Representation structure proxies** (choose ≥1; report method and layers):

  * activation covariance spectrum summaries
  * singular value statistics of key activations
  * effective rank / participation ratio
* **Traffic flow signals:** per slot/component traffic estimates for hygiene decisions.
* **Stability signal (S_t):** the scalar used for budget sloshing (Section 12), including smoothing/hysteresis configuration if any.

### 13.8 Post-hoc regime labelling protocol

Because regimes can be ambiguous mid-run, regime classification is performed post-hoc using a published rubric with declared thresholds ([,]). To avoid conflating measurement with hindsight, labels are computed from logged metrics using a fixed rule set.

* **Symbiosis.** Final net improvement with host-only retention above threshold; interactions near-zero or positive; no sustained turbulence.
* **Parasitism.** Net performance not improved (or regressed) while a seed becomes indispensable; interactions persistently negative; no recovery; brittle dependence signatures.
* **Metamorphosis.** Final organism improves Pareto metrics; host-only may degrade; successor composition evidenced by interaction recovery and downstream consumption; post-hygiene larval host tissue becomes dispensable.

**Sensitivity requirement:** report how regime labels change under small threshold perturbations (rule robustness), and ensure thresholds are not re-tuned per task without declaring it.

### 13.9 Statistical reporting

All results report:

* [N=]( ) independent runs per condition (distinct RNG seeds).
* Mean ± [confidence interval method], plus distribution plots where appropriate.
* Effect sizes for key comparisons (accuracy, latency, params, detritus ratio, collapse rate, seed economy).
* Multiple-comparisons corrections where grid sweeps are performed.

**Reproducibility requirement:** publish the exact config hashes and telemetry schema versions used to produce each figure/table in Section 14.

---

## 14. Results *(placeholders; insert evidence here)*

This section reports results aligned to the research questions in Section 13. Where numeric values, confidence intervals, and plots are not yet final, we use explicit placeholders. All performance evaluations use the task-specific evaluation function (\mathcal{E}(\cdot)) (e.g., accuracy for classification, task score for RL), computed on the declared evaluation split and cadence (Section 13.2). Counterfactual measurements are reported under at least two ablation protocols (gate-to-zero and bypass/identity), with recalibration where used (Section 13.5).

Unless otherwise stated, results are aggregated over [N=]( ) independent runs per condition (distinct RNG seeds), reported as mean ± [CI method], and accompanied by distribution plots for heavy-tailed or multimodal outcomes.

---

### 14.1 Overview: regime frequencies and exemplar trajectories

We begin with a bird’s-eye view of how often Esper converges to each integration regime (symbiosis, parasitism, metamorphosis), stratified by task family, concurrency, and slot placement. Regime labels are assigned post-hoc using the published rubric and thresholds in Section 13.8.

**Table 1.** Regime counts and proportions across all runs, stratified by:

* task family ([vision/language/…]),
* concurrency (n),
* slot depth combinations (early/mid/late),
* blueprint families ([depthwise/attention-lite/conv-heavy/…]),
* hygiene configuration (Emrakul on/off),
* budget strategy (fixed vs stability-coupled).

**Figure 1.** Representative trajectories for each regime:

* (\mathcal{E}(M_t)) over training time,
* lifecycle stage markers (germinate/blend/probation/fossilise/cull),
* parameter growth and/or seed count,
* hygiene events (prune passes, detritus reductions),
* (where enabled) stability signal (S_t) and energy allocation.

**Figure 2.** Interaction trajectories (diagnostic runs, (n \le 3)):

* interaction term (I_t) (pairwise and, where used, higher-order),
* smoothed interaction (\tilde{I}_t) (EMA),
* derivative feature (\Delta \tilde{I}_t = \tilde{I}*t - \tilde{I}*{t-k}) for window [k=]( ).

**Placeholders (to replace with results):**

* “Across [tasks], [X%] of runs were labelled symbiotic, [Y%] parasitic, and [Z%] metamorphic.”
* “Metamorphic runs exhibited an early negative interaction phase followed by recovery; median (\Delta \tilde{I}) over window (k) during the transition was [ ].”
* “Parasitic runs exhibited persistently negative interaction with (\Delta \tilde{I} \le 0) for [ ]% of the blending/probation window.”

---

### 14.2 Reward ablation: over-shaped vs minimal reward

We evaluate the controller’s learnability and downstream task outcomes under two reward designs:

* **Over-shaped reward** (historical ~974-line design; includes stacked penalties and specialised failure-mode detection).
* **Minimal reward** (~64-line design): counterfactual-grounded contribution, rent, terminal outcome, and germination-only budget pressure (Section 4).

**Figure 3.** Controller learning curves:

* episodic return and value loss (where applicable),
* policy entropy (or action distribution concentration),
* germination frequency and lifecycle operation rates over time,
* (\mathcal{E}(M_t)) trajectory overlays for representative runs.

**Table 2.** Outcome summary by reward design:

* final (\mathcal{E}(M)) and best-seen (\mathcal{E}(M)),
* collapse rate (definition: [insert formal collapse criteria]),
* seed economy (gain per seed; marginal returns),
* average parameter growth ratio,
* action distribution summary (germinate/wait/cull/fossilise proportions).

**Placeholders:**

* “The over-shaped reward converged to [do-nothing / oscillatory / collapse] behaviour, with median germinations per run = [ ], compared to [ ] under minimal reward.”
* “Final (\mathcal{E}(M)) under minimal reward exceeded over-shaped by [ ] (absolute) / [ ]% (relative), with effect size [ ].”
* “Over-shaped reward reduced baseline host performance from [ ] to [ ] (measured as (\mathcal{E}(M_{\varnothing})) under identical training history).”

---

### 14.3 Counterfactual attribution stability across ablation protocols

Counterfactual contribution is foundational to both training-time control and offline interpretation. Here we quantify how sensitive marginal contributions are to ablation protocol choice.

For seed (s_i), the training-time marginal is:
[
m_i = \mathcal{E}(M) - \mathcal{E}(M \setminus s_i)
]
computed under each ablation protocol.

**Figure 4.** Marginal contributions by protocol:

* per-seed (m_i) under gate-to-zero vs bypass/identity,
* protocol deltas (\Delta m_i),
* (optional) recalibrated ablation results where used.

**Table 3.** Protocol sensitivity statistics:

* rank stability (Spearman ρ) of ({m_i}) across protocols,
* correlation and absolute deviation distributions,
* stability stratified by architecture family, normalisation configuration, and stage.

**Placeholders:**

* “Marginal rankings were [stable/unstable] across protocols (median Spearman ρ = [ ]).”
* “Protocol sensitivity increased during [BLENDING/PROBATION], consistent with [normalisation/routing] effects.”
* “Recalibration reduced spurious deltas by [ ], improving protocol agreement to ρ = [ ].”

---

### 14.4 Interaction decomposition: synergy vs interference vs independence

For diagnostic runs with (n \le 3), we compute full subset evaluations and decompose interactions (Section 6). For two seeds (s_1, s_2), we report:

* host-only (acc_{\varnothing}),
* standalones (acc_1, acc_2),
* combined (acc_{12}),
* interaction term:
  [
  I_{12} = acc_{12} - acc_{\varnothing} - (acc_1 - acc_{\varnothing}) - (acc_2 - acc_{\varnothing})
  ]

**Figure 5.** Interaction scatter:

* points in ((g_1, g_2, I)) space, where (g_i = acc_i - acc_{\varnothing}),
* coloured by final regime label.

**Figure 6.** Depth-stratified interaction distributions:

* (I) distributions grouped by slot pairs (early+mid, mid+late, early+late),
* optionally stratified by blueprint pair types and concurrency.

**Placeholders:**

* “Symbiotic runs clustered near (I \approx 0) with (g_1, g_2 > 0).”
* “Parasitic runs exhibited persistently negative (I), often with one standalone near-zero/negative.”
* “Metamorphic runs exhibited early negative (I) with subsequent recovery to (I \approx 0) or (I > 0) after successor formation.”
* “The slot pairing [early+mid/mid+late/early+late] showed the highest interference incidence (median (I) = [ ]).”

---

### 14.5 In-flight detection: interaction derivative and leading indicators

We test whether trajectory statistics—and specifically the interaction derivative—provide a tractable in-flight signal distinguishing early metamorphosis from parasitism.

For diagnostic runs ((n \le 3)), define:

* smoothed interaction (\tilde{I}_t) (EMA with decay [ ]),
* derivative over window (k):
  [
  \Delta \tilde{I}_t = \tilde{I}*t - \tilde{I}*{t-k}
  ]

We evaluate candidate leading indicators (Section 8.2), including:

* seed gradient health (seed-path loss slope, gradient norm stability),
* representation structure proxies ([effective rank / spectrum summaries]),
* host degradation shape (accelerating vs stabilising),
* downstream consumption signals (traffic routing into later slots),
* interaction trajectory features ((\tilde{I}_t), (\Delta \tilde{I}_t)).

**Figure 7.** Trajectory separability:

* (\tilde{I}_t) and (\Delta \tilde{I}_t) overlays for runs that end parasitic vs metamorphic,
* distributions of (\Delta \tilde{I}) in early windows.

**Table 4.** Early-window predictive utility:

* AUC / PR metrics for indicator-only classifiers trained on early slices ([first X%] of training),
* ablations: interaction magnitude alone vs derivative alone vs combined indicators.

**Placeholders:**

* “Using only the first [ ]% of training, (\Delta \tilde{I}) predicted eventual metamorphosis vs parasitism with AUC = [ ].”
* “Seed gradient health + (\Delta \tilde{I}) improved early discrimination compared to interaction magnitude alone (ΔAUC = [ ]).”
* “False-positive ‘metamorphosis’ predictions were concentrated in [blueprint/slot] conditions, suggesting [ ].”

---

### 14.6 Turbulence budget and adaptive checkpointing / rollback

We evaluate two safety mechanisms designed to make “tolerate turbulence” a viable strategy:

1. **Turbulence budget:** bounded tolerance for negative interaction/negative blending under healthy local signals (Section 9.1).
2. **Adaptive checkpointing and rollback:** increased checkpoint frequency under early warning signals and rollback when recovery fails (Section 9.3).

**Figure 8.** Turbulence budget ablation:

* premature cull rate (culling eventual metamorphic winners),
* recovery rate (turbulence resolves into stable successor),
* collapse rate (per formal definition),
* distribution of time spent in turbulent states.

**Figure 9.** Checkpointing and rollback trade-offs:

* rollback frequency vs final (\mathcal{E}(M)),
* compute/storage overhead vs avoided collapses,
* time-to-recovery distributions after rollback.

**Placeholders:**

* “Turbulence budgeting reduced premature culls by [ ]% and increased metamorphic wins by [ ]%, with collapse rate change of [ ].”
* “Adaptive checkpointing reduced catastrophic collapses from [ ] to [ ], at overhead [ ] (wall-clock/FLOPs/storage).”
* “Rollback primarily triggered under [indicator thresholds], consistent with [ ].”

---

### 14.7 Emrakul hygiene: detritus reduction, efficiency, and attribution clarity

We quantify Emrakul’s hygiene effects on structure, efficiency, and measurement clarity.

**Reported hygiene metrics (Section 11.3):**

* disconnected mass,
* low-traffic mass (threshold [\tau=]( ), minimum age = [ ]),
* detritus ratio,
* hygiene delta pre/post prune,
* efficiency deltas (latency/FLOPs/VRAM/throughput),
* attribution stability pre/post hygiene (protocol sensitivity).

**Figure 10.** Detritus dynamics:

* detritus ratio over time with hygiene events marked,
* stratified by regime and concurrency.

**Table 5.** Efficiency outcomes:

* latency / FLOPs / VRAM deltas post-hygiene at matched (\mathcal{E}(M)),
* separate reporting for symbiotic vs metamorphic runs (where metamorphosis aims to improve Pareto points).

**Figure 11.** Attribution clarity:

* protocol sensitivity of marginals before vs after hygiene passes,
* rank stability improvements after detritus pruning.

**Placeholders:**

* “Emrakul reduced detritus ratio from [ ] to [ ] (median), with hygiene delta [ ].”
* “Latency improved by [ ]% at matched performance, with VRAM reduction [ ].”
* “Post-hygiene counterfactual protocol sensitivity decreased (variance reduction = [ ]), improving interpretability.”

---

### 14.8 Shared energy budget: stability-coupled allocation outcomes

We evaluate the stability-coupled shared energy budget (“sloshing”) versus fixed splits/schedules.

Let (S_t \in [0,1]) be the stability estimate (definition in Section 12.1; smoothing/hysteresis [ ]). Allocation follows:
[
E_{\text{emrakul}} = E \cdot S_t,\quad E_{\text{tamiyo}} = E \cdot (1 - S_t)
]

**Figure 12.** Budget dynamics:

* (S_t), (E_{\text{tamiyo}}), (E_{\text{emrakul}}) over time,
* annotated lifecycle events (germinations, culls, fossilisation),
* annotated hygiene events (prunes, recalibration passes),
* indicator-triggered checkpointing/rollback events (where enabled).

**Table 6.** Budget strategy outcomes:

* Pareto summary (accuracy vs params/latency/FLOPs),
* collapse rate and recovery time,
* detritus ratio and final model compactness,
* seed economy and churn (germinate→cull frequency).

**Placeholders:**

* “Stability-coupled allocation improved Pareto outcomes versus fixed splits, yielding [ ] at matched [compute axis].”
* “Sloshing reduced oscillatory failure modes (stability variance reduced from [ ] to [ ]).”
* “Energy allocation responded to turbulence episodes by shifting budget to Tamiyo, then back to Emrakul during consolidation.”

---

### 14.9 Pareto frontier: morphogenesis vs full-fat baselines

Finally, we evaluate whether morphogenesis—particularly metamorphosis with hygiene—achieves better Pareto trade-offs than matched baselines.

**Figure 13.** Pareto plots:

* (\mathcal{E}(M)) vs parameter count,
* (\mathcal{E}(M)) vs latency,
* (\mathcal{E}(M)) vs FLOPs / throughput,
  with separate markers for:
* fixed host baseline,
* full-fat monolith baseline,
* morphogenesis without hygiene,
* morphogenesis with hygiene,
* morphogenesis with shared energy budget (where enabled).

**Placeholders:**

* “Metamorphic runs achieved (\mathcal{E}(M)=[ ]) at [ ]% fewer params and [ ]% lower latency than matched full-fat baselines.”
* “Symbiotic runs improved (\mathcal{E}(M)) without substantial compute growth; parasitic runs increased dependency without net gains.”
* “Across [tasks], the best morphogenetic Pareto points dominated the best baseline points in [ ]% of comparisons.”

---

## 15. Discussion

### 15.1 Indispensability is not value (and sometimes it is a transition)

Counterfactual ablation is still the right primitive for modular morphogenesis: if removing a seed causes a sharp drop in evaluation score, the system is presently relying on it. The key result of Esper’s framing, however, is that **reliance is underdetermined**. A seed can become indispensable because it *adds* capability (symbiosis), because it *displaces* and weakens the host (parasitism), or because it is part of a successor circuit that is legitimately replacing larval host tissue (metamorphosis).

This matters because the “obvious” safety heuristic—protect host-only competence—silently bakes in a symbiosis-only objective. In metamorphic succession, host-only degradation is not necessarily a failure; it may be the price of forming a successor topology that ultimately wins on a Pareto frontier (accuracy vs latency/params/seed count). In other words: **the moral status of displacement depends on the end state and the path to it**, not on a snapshot.

A practical takeaway is that evaluation must be regime-aware. In symbiosis, host-only retention is a constraint. In metamorphosis, the constraint shifts to successor robustness and post-hygiene coherence. Parasitism is the case where neither constraint is satisfied: the host is damaged, and no stable successor emerges.

### 15.2 Trajectory signals beat snapshot gating

A consistent theme across the taxonomy and the controller design is that **snapshot gating rules are brittle**. Parasitism and early metamorphosis can share the same instantaneous phenotype (host degradation, load-bearing seeds, negative interactions), so any policy that deterministically culls on those symptoms will either:

* cull productive metamorphic transitions (false positives), or
* tolerate destructive parasitism too long (false negatives).

The paper’s proposed escape hatch is to move from snapshot classification to **trajectory statistics**. Interaction decomposition (when available) turns “are seeds valuable?” into “are seeds cooperating, interfering, or independent?”, and the **interaction derivative** turns “is it bad right now?” into “is it recovering?”. That is precisely the discriminator that matters in-flight.

This does not require perfect separability. Tamiyo’s job is not to label regimes, but to make decisions under uncertainty that maximise expected downstream return. Trajectory features (e.g., (\Delta \tilde{I}) alongside gradient health) are attractive because they convert regime ambiguity into a learnable prediction problem: “turbulence like this tends to resolve” versus “turbulence like this tends to stick”.

### 15.3 Tamiyo learns strategy when the reward remains learnable

The reward-engineering history can be summarised bluntly: *clever penalties are not free*. The over-shaped reward attempted to encode every anticipated pathology (parasitism, farming, stage hacks, ransomware-like dynamics) and in doing so produced an environment where “do nothing” was the safest policy.

The minimal reward works because it is aligned with what the controller actually needs to learn:

* **counterfactual-grounded contribution** gives seed-specific credit,
* **rent** creates an efficiency gradient without hand-coding architecture preferences,
* **terminal outcome** anchors behaviour to real task performance,
* **germination-only budget pressure** discourages runaway growth without punishing integration itself.

In this framing, many “failure-mode protections” are better implemented as **measurement and systems mechanisms** (interaction diagnostics, hygiene, checkpoint/rollback) rather than as dense reward shaping. The agent can learn how to avoid parasitism and exploit metamorphosis if the learning landscape remains navigable and the telemetry makes those outcomes legible.

### 15.4 Emrakul turns messy takeover into clean metamorphosis

A subtle risk in morphogenetic systems is that we can confuse a measurement artefact for a behavioural pathology. If the evolving model is cluttered with unreachable or low-traffic structures, counterfactual baselines become noisy and hard to interpret, and the system’s compute footprint grows even when true functional capacity does not.

This is where Emrakul materially changes the story. Hygiene makes metamorphosis operationally distinct from parasitism:

* In parasitism, displacement leaves the organism worse and brittle, with no coherent consolidation.
* In metamorphosis, displacement is transitional: once the successor circuit stabilises, Emrakul can prune dissolved tissue and dead branches, leaving a coherent adult topology.

Put differently: Emrakul doesn’t merely “save compute”. It improves the *legibility* of the system by removing detritus that would otherwise masquerade as “the host still matters” or distort counterfactuals through stale pathways. That legibility is a prerequisite for making regime-aware decisions and for meaningfully claiming Pareto improvements.

### 15.5 Energy sloshing behaves like a thermostat for growth vs consolidation

The shared energy budget proposal is a systems-level control idea: when training is unstable, give Tamiyo budget to “stop the bleeding” via growth and corrective interventions; when training is stable, give Emrakul budget to prune and consolidate.

The main value here is not the exact budget equation, but the principle: **growth and pruning should be dynamically coupled to stability**, not hard-coded as a fixed schedule. That principle aligns with the empirical reality of morphogenesis:

* pruning during unstable transitions can kill nascent circuits,
* unlimited growth during stable phases produces bloat and detritus,
* fixed schedules tend to be task-specific and fragile.

The thermostat framing also plays nicely with turbulence management: if the controller chooses to tolerate turbulence, the system can temporarily reserve energy for increased diagnostics and checkpointing, then shift toward pruning once recovery indicators turn positive.

### 15.6 Scaling: what survives when (n) grows?

The interaction decomposition that underpins the cleanest regime signals does not scale beyond small (n). That limitation is real, but it does not erase the broader control lessons:

* **Training-time control** can rely on (O(n)) marginals for reward and lifecycle decisions.
* **Diagnostics** can be done in sampled windows or restricted regimes (e.g., during critical blending stages, or on a subset of seeds).
* **Trajectory features** can be constructed from cheap proxies (traffic trends, gradient health, host stability) even when full interaction terms are unavailable.
* **Hygiene becomes more important**, not less, as concurrency grows: detritus accumulation and measurement confusion are multiplicative with scale.
* **Budget coupling becomes coordination**, especially if future systems deploy multiple Tamiyos controlling different regions concurrently. Without a shared stabilising mechanism, concurrent “turbulence bets” can compound into system-wide collapse.

Overall, the paper’s framing suggests that scaling morphogenesis safely is less about finding the perfect rule for parasitism and more about building a system where (i) value signals are cheap and grounded, (ii) ambiguous transitions are treated probabilistically, (iii) risk is managed via rollback, and (iv) the adult topology is kept clean through active hygiene.

**Placeholders:** [Insert references to the main empirical figures supporting each point once numbered.]

---

## 16. Limitations and Failure Modes

This work is intentionally systems-heavy: it combines dynamic architecture growth, counterfactual measurement, an RL controller, and an explicit hygiene/pruning subsystem. That breadth is a strength, but it also creates multiple surfaces where methodology choices and implementation details can dominate outcomes. This section enumerates the most important limitations and the failure modes we have observed or expect, along with the practical implications for interpretation and future work.

### 16.1 Counterfactual attribution is protocol-dependent

Counterfactual contribution is central to both control (reward) and analysis (regime diagnostics), but ablation is not a single operation—it is a family of interventions. Gate-to-zero, bypass/identity, and recalibrated ablations can yield materially different marginals, especially in models with normalisation layers or routing/gating mechanisms.

**Implications:**

* Marginals should be treated as *measurement outputs under a declared protocol*, not as a universal causal truth.
* Reporting at least two ablation protocols is a requirement for interpretability; single-protocol attribution invites artefacts.
* Where normalisation sensitivity is high, recalibrated ablations can reduce spurious deltas, but introduce additional methodological choices (calibration set, cadence).

**Failure mode:** a seed appears “highly causal” under one ablation protocol due to normalisation or graph-shape artefacts rather than robust representational contribution.

### 16.2 Interaction decomposition does not scale

Full interaction decomposition is exponential in the number of concurrent seeds. Capping diagnostics to (n \le 3) (or sampled windows) makes the approach practical, but it limits visibility into higher-order interactions and rare emergent conflicts that can occur when many seeds are active.

**Implications:**

* In high-concurrency regimes, the controller primarily sees marginals; synergy/interference beyond pairwise effects may be missed.
* Interaction-derivative signals are most directly supported in low-(n) diagnostic settings; scaling requires approximations (sampling subsets, periodic diagnostic windows, or learned interaction proxies).

**Failure mode:** a system exhibits stable-looking marginals while higher-order interference (not measured) accumulates and eventually triggers collapse or silent performance degradation.

### 16.3 Leading indicators are suggestive, not guarantees

Trajectory-based indicators—seed gradient health, representation structure proxies (e.g., effective rank), host degradation shape, and especially the interaction-term derivative—are intended to help separate parasitism from early metamorphosis. However, these indicators are noisy and can be brittle under distribution shift, heavy regularisation, delayed credit assignment, or abrupt curriculum changes.

**Implications:**

* Indicators should be used as probabilistic features for control and risk management, not as hard gates.
* “Recovering turbulence” is an empirical hypothesis that must be validated per task family and architecture class.
* False positives are expected: some runs will look like recoverable metamorphosis early and still fail.

**Failure mode:** Tamiyo tolerates turbulence because (\Delta \tilde{I} > 0) and gradients look healthy, but recovery never completes (e.g., due to later-stage interference or budget starvation), leading to avoidable regressions.

### 16.4 Minimal reward is learnable, but not “complete”

The minimal reward design is deliberately spartan to preserve learnability. That choice has a cost: it may not penalise subtle or rare pathologies until they manifest repeatedly, and it may under-shape towards behaviours that humans consider desirable (e.g., interpretability, smoothness, monotonic integration).

**Implications:**

* Reward design should be iterative: start minimal, instrument heavily, add constraints only when specific failures repeat.
* Some objectives (e.g., strict host retention) are incompatible with metamorphosis and should be enforced only in host-preserving modes.
* Minimal reward can still yield conservative equilibria in some tasks if exploration pressure is too low or budgets are too constrained.

**Failure mode:** the policy converges to a “safe” local optimum (limited germination, limited pruning) that avoids collapses but under-explores productive metamorphic transitions.

### 16.5 Hygiene can remove useful redundancy

Emrakul treats persistently low-traffic structure as detritus, but low traffic does not necessarily imply uselessness. Redundant paths can improve robustness, calibration, or recovery from distribution shift. Over-aggressive pruning can therefore trade away resilience for efficiency.

**Implications:**

* Hygiene should be conservative on newborn circuits (minimum age), stability-gated, and compatible with rollback.
* Pruning thresholds should be reported and stress-tested; “wins” that depend on tuned (\tau) values are fragile.
* Where robustness matters, detritus definitions may need to include “rarely used but critical” pathways.

**Failure modes:**

* **Premature pruning:** nascent successor circuits are removed before they attract stable traffic.
* **Bottleneck creation:** pruning concentrates computation into a narrow path, increasing brittleness and variance.

### 16.6 Checkpointing and rollback incur real cost and can distort dynamics

Adaptive checkpointing and rollback reduce the cost of “wrong bets” in turbulence, but they introduce overhead and can change optimisation dynamics. Frequent checkpoints increase wall-clock and storage, while rollbacks create discontinuities that can affect learning curves and credit assignment.

**Implications:**

* Evaluate controller variants on total cost (wall-clock, energy, storage), not only final accuracy.
* Rollback policy should be treated as part of the environment dynamics; training with rollback can differ meaningfully from training without it.
* Care is required to avoid pathological loops (e.g., repeated partial progress followed by rollback).

**Failure mode:** the system enters an oscillatory regime where it repeatedly explores a turbulent transition, fails to recover, rolls back, and repeats—consuming budget without net progress.

### 16.7 Stability signals and energy sloshing can oscillate

The shared energy budget is conceptually simple, but stability estimates are noisy. Without smoothing and hysteresis, energy allocation can chatter: rapid switching between growth and pruning can destabilise both Tamiyo and Emrakul.

**Implications:**

* (S_t) (stability) should be smoothed (e.g., EMA) and/or computed over windows.
* Budget allocation should use hysteresis and minimum allocation floors to prevent starvation.
* Switching frequency should be monitored and penalised if necessary.

**Failure mode:** budget oscillation starves either growth (unable to germinate corrective seeds) or hygiene (detritus explodes), leading to worse stability rather than better.

### 16.8 External validity and portability

Esper’s dynamics may differ across modalities, scales, and architectural families. In particular, very large transformers, sparse MoE systems, and heavy alignment/feedback loops introduce routing dynamics and non-stationarities that may invalidate assumptions about traffic, pruning, and counterfactual measurement.

**Implications:**

* Claims should be scoped to the tasks, architectures, and training regimes actually tested.
* Porting requires re-validating: ablation protocol sensitivity, traffic metrics, pruning safeguards, and controller observation design.
* Some environments may require different “default” success criteria (e.g., robustness-heavy settings may value redundancy more than efficiency).

**Failure mode:** mechanisms that are benign in small-to-mid scale experiments become pathological at scale (e.g., traffic metrics saturate, pruning removes rare-but-important experts, or controller policies exploit logging artefacts).

### 16.9 Summary: what to trust, and what to verify

The central conceptual contributions—three integration regimes, the in-flight ambiguity framing, and hygiene as an enabler of metamorphosis—are intended to be robust. The operational conclusions (which indicators work best, how much pruning helps, and whether energy sloshing improves Pareto outcomes) are inherently empirical and must be validated per configuration.

**Placeholders:** In Section 14, we report (i) protocol sensitivity analyses, (ii) decomposition limits and sampling strategies, (iii) indicator predictive utility, (iv) pruning threshold stress tests, and (v) budget oscillation diagnostics.
