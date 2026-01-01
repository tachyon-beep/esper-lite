# Esper: Architectural Ecology in Neural Networks

**A Draft Discussion Paper on Morphogenetic AI (Revised)**

## Abstract

Contemporary deep learning largely follows a paradigm of **architectural engineering**: models are statically designed, initialised, and trained as monolithic blocks. Neural Architecture Search (NAS) automates parts of this process, but typically remains a discrete, offline optimisation loop.

**Esper** proposes a shift to **architectural ecology**: an online, continuous process where neural modules (seeds) germinate, compete, and stabilise inside a living host network during training. Esper separates concerns across a **substrate** that executes and prices decisions, and an **organism layer** that grows and prunes structure online:

**Organism (agents and organs)**

1. **Kasmina (The Body):** A differentiable host that supports controlled growth via SeedSlots and blending.
2. **Tamiyo (The Brain):** A policy that germinates and develops *Kasmina seeds* during training.
3. **Emrakul (The Immune System):** Custodian of committed structure, enforcing long-term ROI via lysis and pruning. *(Planned; currently Tamiyo handles both roles.)*

**Substrate (infrastructure and physics)**

4. **Tolaria (The Metabolism):** The training engine for Model Alpha: high-throughput execution, deterministic replay, and flight-recorded data capture. Includes an internal safety governor for NaNs and divergence rollback.
5. **Simic (The Selective Pressure):** The economy and accounting layer that prices complexity (Rent and Churn) and provides credit signals (including Shapley audits in Phase 1).

**Kasmina seeds** and **Phages** are peer primitives: seeds are creative tissue (can become anything, Tamiyo-managed), while phages are infrastructure wrappers (measured, priced, lysable, Emrakul-managed). Fossilisation rewraps a seed as a phage and transfers custodianship.

*Narset (speculative)* is a slow-timescale coordinator that allocates budgets across zones using telemetry only (not yet part of the system).

A key clarification: Esper is intended to be **two-timescale**. In *Phase 1 ("train the trainer")*, we spend significant compute on audited credit assignment (including exact, full-retrain Shapley values on small candidate sets) to teach Tamiyo what "good growth" looks like under a defined protocol. In *deployment*, a trained Tamiyo is intended to grow many new models **without** the Shapley harness, using only cheap online signals and learned critics, amortising the cost of the training scaffold across downstream runs.

**Key point:** Shapley values are used as *training-only teacher signals* to shape credit assignment and learning updates; they are **not** included in Tamiyo's observation space, which remains identical to deployment telemetry. This ensures no privileged-information leakage between training and deployment.

---

## 1. Introduction: From Fossils to Flora

A standard ResNet or Transformer is a **fossil**: its skeletal structure is fixed before the first gradient is calculated. If the task proves more complex than anticipated, the model cannot grow. If the task is simpler, the model wastes compute on redundant parameters.

We argue that the topology of a neural network should be a function of its training history, not solely its creator’s intuition. Biology does not build a brain and then switch it on; the brain grows in response to stimuli while paying ongoing metabolic costs.

Esper aims to build an environment where networks can grow safely, and where growth is not free. Seeds are introduced, evaluated under selective pressure, and either integrated (**Committed**) or removed (**Pruned**). The focus is not "the perfect architecture", but the **physics of a system** in which useful architectures can emerge under cost and stability constraints.

### 1.1 Two-timescale learning: training models vs training Tamiyo

A common failure mode in describing morphogenetic systems is accidentally selling them as “we trained many expensive models”. Esper is not primarily about producing thousands of partially-trained CIFAR-10 models.

Esper is about training a **reusable architect policy**:

* **Inner loop (model training):** Kasmina’s weights learn the task while the architecture changes online.
* **Outer loop (policy training):** Tamiyo learns *how to cause beneficial architectural change* across runs, seeds, checkpoints, and hosts.

In Phase 1, expensive audits (e.g. Shapley on small candidate sets) act as teacher labels. The intended outcome is a Tamiyo that can later be deployed to grow new models **without** repeating the full audit machinery.

---

## 2. System Overview and Terminology

### 2.1 The loop (online morphogenesis)

At a high level, Esper runs a closed feedback loop across substrate and organism:

**Substrate (executes and prices):**
1. **Tolaria** executes training and evaluation, enforces determinism, records events, and rolls back on divergence.
2. **Simic** defines rewards and costs, computes rent and churn, and provides credit labels.

**Organism (decides and acts):**
3. **Kasmina** performs forward and backward passes while supporting dormant and active SeedSlots.
4. **Tamiyo** observes training dynamics and slot states, then chooses **growth** actions (germinate, blend, commit).
5. **Emrakul** observes phage states and chooses **decay** actions (lysis, extraction). *(Planned; currently unified with Tamiyo.)*

A rough schematic:

```
Data -> Tolaria (training engine incl. governor) -> Kasmina (Host + Seeds/Phages) -> Loss / Metrics
                     |                                          ^
                     v                                          |
              Slot and wrapper state                  Tamiyo (growth) + Emrakul (decay)
                     |                                          ^
                     v                                          |
                        Simic (Reward + Credit Assignment)
```

### 2.2 “Train the trainer” vs “Deploy the trainer”

Esper has two operational modes:

* **Train-the-trainer (Phase 1):**
  Instrumented runs generate high-quality credit signals (including expensive audits on small candidate sets) and a rich flight-recorded dataset. Tamiyo and its critics are trained to predict value/credit and choose actions reliably.

* **Deployment (Phase 2):**
  Tamiyo grows new models with **cheap online signals** (loss trends, gradient statistics, gate stability, cost signals, critic predictions), without full retrain Shapley. Periodic audits may still exist as diagnostics, but are not assumed to be part of the runtime budget.

A useful cost framing is:
[
C_{\text{total}} = C_{\text{meta}} + N\cdot C_{\text{deploy}}
]
Phase 1 deliberately pays a large one-time (C_{\text{meta}}) to reduce (C_{\text{deploy}}) and increase reliability across (N) downstream training runs.

### 2.3 Glossary (working definitions)

**Substrate terms:**
* **Host:** The base backbone (e.g. ResNet, Transformer) plus insertion points for seeds and phage-wrapped infrastructure.
* **Tolaria:** The training engine for Model Alpha, responsible for high-throughput execution, determinism and replay, and event-sourced telemetry capture.
* **TolariaGovernor:** A safety subsystem inside Tolaria that monitors training health (NaNs, divergence, loss explosions) and can trigger rollback or emergency interventions.
* **Rent:** The ongoing cost of keeping a structure active (compute, memory, latency proxy). Defined by Simic.
* **Churn:** A penalty for structural volatility (rapid add/remove, oscillating gates, frequent state flips). Defined by Simic.
* **Shapley value (φ):** A principled attribution of marginal contribution that shares interaction gains across participants.

**Organism terms:**
* **SeedSlot:** A pre-allocated insertion point. When dormant, it behaves as an identity (or no-op) under a defined contract. The "heavy" wrapper for growth.
* **Kasmina seed:** A growable module instantiated in a SeedSlot during exploration and development. Creative tissue: can become anything, learns and morphs, Tamiyo-managed.
* **Phage (wrapper):** A standard "infrastructure wrapper" that exposes telemetry and a lysis interface. Committed seeds are rewrapped as phage-wrapped infrastructure.
* **Blueprint:** The family/type of module a seed can become (the "genome library").
* **Alpha (α):** A continuous gate controlling a seed's contribution to the host forward path (SeedSlot only; phages run at effective α=1).

**Lifecycle terms:**
* **Committed (Tamiyo-locked):** A seed state where Tamiyo will not modify the slot further; custodianship transfers to Emrakul.
* **Compacted:** A physical rewrite step (optional) that removes edit machinery and fuses the committed structure at a safe boundary (end of training or checkpoint compaction).
* **Lysis:** Controlled dissolution via a **sedation mask** (distinct from α). Phages expose `apply_sedation_mask()` for soft gating and `physical_lysis()` for reclamation at safe boundaries.
* **Emrakul:** The immune system: a peer policy to Tamiyo that controls phages and decides when to trigger lysis. *(Planned; currently Tamiyo handles both growth and decay.)*

**Value terms:**
* **Online return:** The discounted reward used by PPO (γ, GAE) for short-horizon control and stability.
* **Audit value v(S):** Used for Phase 1 Shapley labels: a fixed training budget under a specified protocol with a final evaluation metric (e.g. validation loss).

### 2.4 Seed lifecycle (state machine)

Esper treats growth as a controlled state machine rather than graph surgery:

1. **Dormant:** Slot is identity; no effect (α = 0).
2. **Germinated:** Module instantiated, sanity checks complete, ready for training.
3. **Training (Incubator):** Seed receives surrogate gradients via Straight-Through Estimator; host forward-path unchanged (α = 0 in blend).
4. **Blending:** Seed is gradually introduced via 0 < α < 1.
5. **Holding:** Final validation period before commitment.
6. **Committed (Tamiyo-locked):** Tamiyo will not modify this slot thereafter. The module is rewrapped as a Phage and becomes subject to Emrakul's long-horizon pruning. The structure is not yet **compacted** (physically fused); compaction occurs at safe boundaries (end of training or checkpoint).
7. **Pruned → Embargoed → Resetting:** Seed faded out, slot enters cooldown, then returns to Dormant.

```
DORMANT → GERMINATED → TRAINING → BLENDING → HOLDING → COMMITTED (Tamiyo-locked)
   ^                      │          │          │               │
   │                      │          │          │               v
   │                      │          │          │        (rewrap) PHAGE-WRAPPED
   │                      │          │          │               │
   │                      │          │          │        (Emrakul) LYSIS
   │                      │          │          │               │
   │                      └──────────┴──────────┴───────────────┴─→ PRUNED → EMBARGOED → RESETTING ─┘
```

**Authority over lifecycle transitions:**

| Actor | Can modify α? | Can change lifecycle? | Scope |
|-------|---------------|----------------------|-------|
| **Tamiyo** | Yes, pre-commit | Yes, up to Commit (including PRUNE) | Seeds in DORMANT…HOLDING (pre-commit only) |
| **Emrakul** | No (uses sedation mask) | Yes (lysis only) | Committed seeds (phage-wrapped) + baseline phage infrastructure |
| **Tolaria** | Yes (emergency) | Yes (rollback/emergency lysis) | Anything, but only on catastrophic triggers |

---

## 3. Physiology: Kasmina as a Morphogenetic Host

### 3.1 The morphogenetic plane (SeedSlots)

Instead of cutting and pasting network graphs mid-training, Esper uses pre-allocated **SeedSlots** placed at structurally meaningful points (e.g. residual branches, MLP blocks, attention subpaths). Each SeedSlot is an identity when dormant, ensuring the host’s baseline function is intact.

A typical residual-style integration is:
[
y = h(x) + \alpha \cdot s(x)
]

* `h(x)` is the host path
* `s(x)` is the seed module
* `α` gates how "real" the seed is

This turns “adding a module” into “activating a pre-existing dormant organ”.

### 3.2 The incubator: gradient isolation via Straight-Through Estimator

A recurring failure mode in naïve morphogenesis is the bull-in-a-china-shop problem: a newly initialised seed produces noisy outputs that destabilise the host.

Esper mitigates this via **Gradient Isolation**: during the TRAINING stage, the seed receives the same inputs as it would in production, but its output does not affect the host's forward computation.

The implementation uses a **Straight-Through Estimator (STE)**:

```python
# Forward: returns host_features (seed contribution cancels out)
# Backward: gradients flow to both host AND seed parameters
return host_features + (seed_features - seed_features.detach())
```

This uses a surrogate gradient so the seed receives task-aligned gradient signal while its forward contribution is cancelled. TRAINING behaves like α = 0 in the forward pass and approximately α = 1 in the backward pass. The seed learns task-relevant features in isolation without destabilising the host.

**Additional gradient isolation:** Host input into the seed path is detached during TRAINING, ensuring host gradients remain identical to a host-only model.

**Maturity gate (G2):** A `GradientHealthMonitor` tracks the seed's gradient-to-host ratio. Maturity gating uses gradient health plus a bounded activation-stat probe under a small test blend to reduce blend-time distribution shock. Once the seed passes stability criteria (bounded norm, healthy gradients, acceptable blend probe), it transitions to **Blending**.

### 3.3 Blending and gate schedules

Blending is the controlled ramp of α from 0 toward an operational value. This is the main safety valve against destabilisation. Practical notes:

* α can be scheduled (handcrafted ramps) or policy-controlled (Tamiyo selects step sizes).
* Rent can be made α-dependent to prevent "gate gaming" (e.g. paying almost nothing at tiny α while still extracting benefit).

---

## 4. Metabolism: Tolaria and High Throughput

### 4.1 Inverted control flow

Morphogenesis multiplies evaluation. Python stepping and the GIL become bottlenecks.

Tolaria implements **inverted control flow**: the high-performance execution engine drives training and evaluation rather than a Python agent stepping the environment. Multiple environments, candidate sets, and evaluation branches can be executed in parallel, treating the GPU as a batch-processing metabolic substrate.

### 4.2 Vectorised determinism (the determinism contract)

A core engineering goal is replayability: if a specific architecture emerges, we should be able to reproduce the exact history that produced it.

**Determinism contract (intended):** given fixed seeds, fixed data order/augmentations, and deterministic kernel settings, the morphogenetic stack (including seed germination decisions and policy updates) replays identically.

**Boundary conditions:** in practice, determinism depends on framework/kernel determinism flags, distributed reduction order, and careful RNG handling. Esper treats determinism as a first-class system feature: RNG states, action events, and relevant execution settings are logged so that “replay drift” becomes diagnosable rather than mysterious.

Determinism is a prerequisite for the Flight Recorder (Section 8) and for validating that Tamiyo’s improvements are behavioural, not just lucky trajectories.

---

## 5. Selection Pressure: Simic’s Economy

### 5.1 The rent and churn economy (with units)

Tamiyo is not rewarded solely for accuracy. It plays a game of ROI (Return on Investment): improve task performance while paying ongoing costs.

A generic shaping form:
[
R_{\text{total}} = R_{\text{task}} - W_{\text{rent}}\cdot C_{\text{rent}} - W_{\text{churn}}\cdot C_{\text{churn}}
]

To make this auditable, costs are expressed in parameter-based units with logarithmic scaling:
```
C_rent = log(1 + (BaseSlotRent + α·P_seed) / P_host)
```
Note: The weight W_rent is applied once in R_total, not inside C_rent.
where:
* `BaseSlotRent ≈ 0.0039 × P_host`: a fixed overhead per occupied slot, calibrated to set a default "occupied slot" cost comparable to a small residual block under the reference host
* `P_seed`: the seed's parameter count
* `P_host`: the host's parameter count
* Logarithmic scaling prevents runaway penalties while maintaining sensitivity to growth

The rent penalty is capped (`max_rent`) to avoid overwhelming the task signal.

*Current rent uses a parameter-based proxy; future versions will incorporate measured step-time and memory pressure for hardware-aligned pricing.*

**Churn** penalises structural volatility (edits per window, oscillations, frequent state flips). Its purpose is to discourage tax-loophole behaviours such as rapid add/remove cycles or α oscillation around thresholds.

### 5.2 Value functions and horizons

Esper uses two value definitions on different timescales:

1. **Online return** is the discounted reward used by PPO (γ = 0.995, GAE λ = 0.98, current implementation values) for short-horizon control and stability. Episode boundaries are 150 epochs.

2. **Audit value v(S)** is used for Phase 1 Shapley labels: a fixed training budget under a specified protocol with a final evaluation metric (e.g. validation loss or accuracy).

Phase 1 learns critics that predict audit credit from cheap online telemetry, enabling deployment without repeated Shapley harness runs.

### 5.3 Credit assignment: exact Shapley as a calibration oracle (Phase 1)

Credit assignment is hard because contributions are contextual: a seed can be useful only in combination with others, or only at a particular stage of training.

For small candidate sets, Esper uses exact Shapley values as an **oracle-style calibration signal**. Given a candidate set (C) of size (n) and a value function (v(S)) over subsets (S \subseteq C), the Shapley value for seed (i) is:
[
\phi_i = \sum_{S \subseteq C\setminus{i}} \frac{|S|!(n-|S|-1)!}{n!}\left[v(S \cup {i}) - v(S)\right]
]

**Phase 1 protocol (e.g. CIFAR-10, small (n)):**
We compute Shapley over up to 3 seeds at a time using **full retrain Shapley**:

* For each subset (S), train the model under a fixed training budget with only the seeds in (S) enabled (others gated off).
* Evaluate (v(S)) via a fixed metric (e.g. validation accuracy or loss).
* Compute φᵢ across all subsets.

This is expensive, but it provides an audit trail: did Tamiyo choose the seed because it was truly useful under the protocol, or because of noise?

**Important note:** even "full retrain Shapley" is not metaphysical ground truth; it is a high-quality label *under a specific protocol*. Where feasible, Shapley labels should be treated as expectations over controlled randomness (multiple seeds) rather than single-point estimates.

**Shapley-as-teacher (no observation leakage).** During Phase 1, Shapley values (φ) are treated as privileged *labels*, not state. Tamiyo's policy receives only deployment-available telemetry in its observation vector. Shapley is computed offline under the audit protocol and then joined to recorded events, where it shapes **learning targets** (reward relabelling, advantage correction, critic supervision). This ensures that:

1. **No deployment mismatch:** the policy's input distribution matches deployment exactly; it never relies on signals it won't have later.
2. **Anti-cheating:** Shapley cannot become an "oracle feature" the policy learns to game; it is a supervisory signal only.

The causal structure is:
* Tamiyo **acts** on telemetry (oₜ).
* Shapley is computed **later** and joined to the trajectory as a label.
* **Learning** uses Shapley; **acting** does not.

### 5.4 From oracle to deployment: distillation and critics

The intended path is to turn expensive oracle labels into cheap deployment behaviour:

* Shapley-labelled runs become a dataset.
* Critics learn to predict credit/advantage (or cost-adjusted value deltas).
* Tamiyo's policy is trained to act using critic outputs and cheap online features.
* Deployment uses the learned policy/critics, not the Shapley harness.

**Intuition:** Tamiyo learns to read raw telemetry well enough to foresee the same outcomes that Shapley attribution reveals. Phase 1 calibrates this predictive capacity; deployment relies on it.

---

## 6. Tamiyo: Policies and Scaling

Tamiyo uses a **factored action space** with 8 independent heads, enabling compositional control:

| Head | Options | Purpose |
|------|---------|---------|
| `slot` | N slots | Which slot to target |
| `blueprint` | 13 types | Module type to germinate (NOOP, CONV_LIGHT, ATTENTION, NORM, etc.) |
| `style` | 4 styles | Germination style combining blend algorithm + alpha algorithm |
| `tempo` | 3 speeds | Blending tempo (FAST: 3 epochs, STANDARD: 5, SLOW: 8) |
| `alpha_target` | 3 values | Target alpha amplitude (0.5, 0.7, 1.0) |
| `alpha_speed` | 4 speeds | Schedule speed in controller ticks |
| `alpha_curve` | 5 curves | Schedule curve (LINEAR, COSINE, SIGMOID variants) |
| `op` | 6 operations | Lifecycle operation |

The lifecycle operations (`op` head) are:

* `WAIT`: no action this tick
* `GERMINATE`: instantiate a seed in the target slot
* `SET_ALPHA_TARGET`: adjust the alpha target for an active seed
* `ADVANCE`: push seed to the next lifecycle stage
* `PRUNE`: begin lysis (fade out and remove)
* `COMMIT`: graduate seed to phage-wrapped infrastructure (Tamiyo-locked; custody transfers to Emrakul)

Scaling to large networks is treated as a curriculum problem. A policy that does “2 seeds, 3 slots on CIFAR-10” well is not automatically a policy that can manage 1000 seeds across diverse hosts.

A plausible scaling path:

1. **Oracle bootcamp:** small (n) with Shapley audits across varied host initialisations and checkpoints.
2. **Distillation:** train critics to predict φ (or cost-adjusted advantages), not just the policy.
3. **Candidate selection:** introduce Scouts that propose small candidate sets (K \ll N) for expensive evaluation.
4. **Approximate credit:** Monte Carlo Shapley, leave-one-out, short-rollout proxies, or learned critics, validated against periodic exact audits.
5. **Full scale:** large seed libraries and many slots, with compute bounded by shortlists and occasional audits.

---

## 7. Case Study: The Rise of the Norm (Phase 1, CIFAR-10)

In early CIFAR-10 runs we observed an emergent behaviour: **Norm Dominance**. Given a library of blueprints (Conv, Attention, Norm), Tamiyo strongly preferred normalisation layers.

* **Low rent:** few parameters and low FLOP overhead.
* **High stability:** stabilises gradient variance quickly.
* **Interpretation:** the policy learned to exploit optimisation physics. Instead of building risky and expensive feature extractors, it smoothed the optimisation landscape.

This is both a warning and a validation: it validates that the economy is real (Tamiyo optimises ROI), and it warns that local optima and reward hacking are default behaviours.

To ground this in measurements, we propose reporting not only best-run accuracy, but **reliability under fixed budget**, e.g. median and lower-quantile accuracy across seeds at matched compute.

| Setting           | Task metric (median / q25) | Added params | FLOPs delta | Blueprint distribution | Notes              |
| ----------------- | -------------------------: | -----------: | ----------: | ---------------------- | ------------------ |
| Baseline host     |                        TBD |            0 |           0 | none                   |                    |
| Esper (low rent)  |                        TBD |          TBD |         TBD | TBD                    | expect more growth |
| Esper (high rent) |                        TBD |          TBD |         TBD | TBD                    | expect norm-heavy  |
| No churn penalty  |                        TBD |          TBD |         TBD | TBD                    | expect oscillation |

---

## 8. Engineering for Learning: The Flight Recorder

Because Tolaria is designed for replayability, Esper treats runs as dataset generation, not just online training.

### 8.1 Event-sourced logging

Every Tamiyo decision is logged as an immutable event. Currently implemented:

* explicit action (seed, slot, blueprint, gate changes)
* slot state at decision time
* policy outputs (confidence, per-head entropies)
* expected value V(s) and TD advantage (computed later)
* delayed teacher labels joined post-hoc (Shapley φ, oracle rewards, long-horizon outcomes)

**Note on delayed labels:** Teacher labels (including Shapley φ) are joined post-hoc to the event log and used for reward relabelling, advantage correction, and critic supervision; they are **not** emitted as online observations and do not appear in Tamiyo's state. This is the mechanism that enables Phase 1's expensive attribution to improve learning without leaking privileged information into the policy's input distribution.

Planned but not yet implemented:

* observation vectors (pre-action and post-action): needed for "obs off by one" diagnosis
* action masks and full logit vectors
* identifiers for specs and versions (obs spec, reward spec, seed library, host manifest)

### 8.2 Forensic replay and counterfactuals

Combined with determinism, the Flight Recorder is designed to enable forensic replay:

* reload the exact universe state at decision step (t)
* inject a counterfactual action
* replay forward to observe how topology and training dynamics diverge

*Current implementation:* Determinism infrastructure exists (per-environment RNG seeding, determinism tests). Counterfactual analysis exists for episode-end attribution. Full forensic replay (state serialization, counterfactual injection, forward replay) is planned but not yet implemented.

This turns "black box evolution" into a debuggable process. It also turns expensive oracle runs into long-lived assets: future Tamiyos can be trained offline on archived experience, and approximations can be benchmarked against stored ground truth.

---

## 9. Emrakul: The Immune System (via Phage wrappers)

Esper's growth story is incomplete without decay. **Emrakul** is the peer policy to Tamiyo: while Tamiyo controls growth (seeds), Emrakul controls decay (phages).

### 9.1 The Tamiyo/Emrakul split

The two policies have distinct domains and a clear handoff:

| Policy | Controls | Actions | Focus |
|--------|----------|---------|-------|
| **Tamiyo** | Seeds | germinate, blend, commit, advance | Growth: when to add capacity |
| **Emrakul** | Phages | lysis, extraction, redundancy prune | Decay: when to remove capacity |

**Custodianship handoff:** When Tamiyo commits a seed, the decision is **irreversible at the policy level**: Tamiyo will not edit that slot again. The structure is rewrapped as a Phage and custodianship transfers to Emrakul. If the architecture later becomes redundant or stops paying rent, Emrakul can lysis it.

**Custody boundary:** Tamiyo prunes her own seeds at any point before commitment. Emrakul has no authority over in-progress development, only graduated structure that Tamiyo has blessed.

This separation allows each policy to specialise: Tamiyo optimises for growth timing and blueprint selection; Emrakul optimises for infrastructure health and efficiency.

*Current implementation note:* Tamiyo currently handles both roles (prune action in factored action space). The Emrakul separation is planned for Phase 2.

### 9.2 Seeds and Phages: peer primitives

Kasmina seeds and Phages are **peer tools**, not a hierarchy. They represent two fundamentally different kinds of tissue:

| Primitive | Nature | Manager | Purpose |
|-----------|--------|---------|---------|
| **Seed (SeedSlot)** | Creative tissue | Tamiyo | Can become anything; learns, morphs, explores |
| **Phage** | Conservative tissue | Emrakul | Measured, priced, lysable; "you can stay, but I'm watching" |

**Wrapper overhead:**

| Wrapper | Overhead | Features |
|---------|----------|----------|
| **SeedSlot** | Heavy | α blending, gradient isolation, lifecycle state machine, full telemetry |
| **Phage** | Light | ROI monitoring, lysis interface only (no α computation) |

At **commitment**, the wrapper swaps: SeedSlot → Phage. This is a **custody transfer**, not destruction:
* The seed becomes **committed** (Tamiyo will not touch it)
* The wrapper becomes lightweight (no blend overhead)
* Emrakul gains jurisdiction (can lysis if ROI drops)
* The structure is not yet **compacted** (physically fused) until a safe boundary

A **Phage** provides:

* **ROI telemetry:** contribution vs rent monitoring
* **Lysis interface:** `apply_sedation_mask(target_id)` for reversible gating; `physical_lysis(target_id)` for reclamation at safe boundaries (invoked by a governor, not Emrakul directly)
* **No α blending:** the module runs directly in the forward path

### 9.3 Functional lysis (online) and physical lysis (safe boundary)

* **Functional lysis:** Emrakul monitors a phage's ROI. If contribution drops below rent (or confidence falls), Emrakul triggers lysis via a **sedation mask** (not α; phages have no α). This encourages the network to re-route around the fading structure while training continues.

* **Physical lysis:** Once the sedation mask → 0 (fully gated), the structure can be reclaimed at a safe boundary. The slot transitions through PRUNED → EMBARGOED → RESETTING → DORMANT (ready for reuse).

### 9.4 Redundant takeover and "lazy paths"

Emrakul targets redundancy:

* If an expensive module behaves like an identity (but pays rent), Emrakul pressures the network to route around it, then lyses.
* If multiple modules provide overlapping function, Emrakul encourages winner-takes-more dynamics, simplifying the organism.

*Current implementation:* Redundancy detection is learned via Simic's reward shaping (contribution < rent → negative reward), not rule-based. Explicit ROI monitoring and proactive Emrakul are planned.

### 9.5 Safety rollback (Tolaria)

Separate from Emrakul's ROI-driven pruning, **Tolaria** includes built-in safety for **catastrophic failure**:

* Monitors for NaN, gradient explosion, loss divergence
* Triggers emergency rollback to last known good state
* Can force-prune failed seeds via `governor_nan`, `governor_lobotomy` events

This is an implementation detail within Tolaria (the training engine for model-alpha), not a separate conceptual component.

### 9.6 Scaling considerations

For large models, a single Tamiyo–Emrakul pair may become a bottleneck. The natural extension is **multiple pairs**, each owning a disjoint region. Tamiyo is always the leaf-level controller; scaling grows the C2 chain *upward* via coordinator layers, not downward via Tamiyo subdivision.

See §11 (Future Directions) for speculative extensions including Narset coordination, surgical lysis, and fractal growth.

---

## 10. Limitations and Failure Modes

Esper’s promises come with sharp edges:

* **Reward hacking and local optima:** Norm dominance can be rational under the economy but may block richer growth.
* **Non-stationarity:** the environment changes as the architecture changes; naïve RL can destabilise.
* **Credit assignment cost:** exact Shapley does not scale; approximations must be audited.
* **Objective mismatch:** long-horizon Shapley labels and short-horizon control signals can disagree.
* **Protocol overfitting:** Tamiyo can over-specialise to a host family, optimiser, dataset, or cost model. Diversity is not optional.
* **Gate gaming:** continuous α can be exploited unless rent/churn are designed to close loopholes.
* **Determinism brittleness:** reproducibility depends on careful control of kernels, RNG, and distributed execution.

These failure modes are not footnotes; they are the constraints that shape curriculum design, reward shaping, and engineering choices.

---

## 11. Future Directions

The following extensions are speculative and not required for core Esper. They represent potential scaling paths if the simpler ecology proves stable.

### 11.1 Narset: meta-coordination

**Narset** is a slow-timescale coordinator that allocates per-zone budgets using telemetry only (performance trends, cost, churn, health). Narset does not observe the architecture or seed inventory, and cannot directly select modules.

**Allocation heuristic:** Energy flows based on zone state. Stable zones receive more energy for Emrakul (optimise, prune redundancy); volatile or sluggish zones receive more energy for Tamiyo (experiment, adapt). Narset doesn't dictate tactics; she adjusts metabolic budgets.

**Simpler alternative (flag-based coordination):** Before introducing Narset, a lightweight mechanism may suffice: a flag indicating "the other policy acted this tick." If Emrakul learns (via penalty for destabilization) to defer when Tamiyo is active, explicit budget arbitration may be unnecessary.

### 11.2 Surgical lysis

At scale, module-level lysis becomes too coarse. The asymmetric granularity principle: Emrakul's atomic unit of destruction should be *smaller* than Tamiyo's atomic unit of creation.

| Unit | Blast Radius (1B Host) | Blast Radius (1M Tiny Host) |
|------|------------------------|----------------------------|
| **Module** | ~0.25% | ~3-5% (catastrophic) |
| **Head** | ~0.02% | ~0.4% (surgical) |
| **Channel** | <0.001% | ~0.05% (micro-surgery) |

**Two-stage lysis protocol:** (1) Sedation: apply mask, enter probation. (2) Resection: if probation passes without loss spike, physically delete. Safety interrupt: if loss spikes, instantly lift mask and mark structure as CRITICAL (immune for N steps).

### 11.3 Fractal growth

A longer-term vision: a seed can be a container for another morphogenetic model, enabling "zoom-in" growth at bottlenecks. This depends on stabilising the simpler non-recursive ecology first.

### 11.4 Tamiyo expansion

If Tamiyo is observed hacking around limitations, those behaviors should be promoted to first-class controls:

| Knob | Options |
|------|---------|
| head_count | 1, 2, 4, 8 |
| channel_width | 0.25x, 0.5x, 1x, 2x |
| warmup_epochs | 0, 1, 3, 5 |
| lr_multiplier | 0.1x, 0.5x, 1x, 2x |

### 11.5 Blueprint meta-loop

Another policy observes Tamiyo's telemetry and generates new blueprints: sub-graphs discovered through offline evolutionary search. This turns Tamiyo into a composer selecting from a library of pre-evolved components.

---

## 12. Conclusion

Esper proposes a move away from the intelligent-designer model of architecture. The goal is not to build the perfect network by hand. The goal is to build a **world with rules and let ecology happen inside it**.

The system has two layers:

**Substrate (infrastructure and physics):**
* **Tolaria:** execution engine for model-alpha: high throughput, determinism, replay, safety
* **Simic:** economy and accounting: rent, churn, Shapley-based credit

**Organism (agents and organs):**
* **Kasmina:** the body: SeedSlots and Phages as peer primitives for growth and decay
* **Tamiyo:** the growth policy: controls seeds via factored RL
* **Emrakul:** the decay policy: controls phages via ROI-driven lysis

Fossilisation is not destruction but **custody transfer**: a seed becomes *committed* (Tamiyo-locked), rewrapped as a phage, and handed to Emrakul. *Compaction* (physical fusion) occurs later at safe boundaries.

In Phase 1, we deliberately spend compute on audited attribution (full retrain Shapley over small candidate sets) to ensure the feedback signal is trustworthy while Tamiyo learns. The longer-term plan is to convert that expensive truth into scalable behaviour via distillation, approximation, candidate selection, and durable flight-recorded experience, so a trained Tamiyo can reliably grow many future models without the Phase 1 harness.

The bet is simple: if we define the rules of growth carefully enough, architectures can emerge that no human would think to design, while an explicit accounting system keeps them honest.

---

## Appendix A: Pending Validation (Do Not Publish)

> **Status:** This appendix contains draft observations and claims that require validation before inclusion in any public-facing document. The content here represents early interpretations of experimental results that may be incomplete, overstated, or incorrect.

---

### A.1 Emergent Restoration: The "Lobotomy" Stress Test

To rigorously test Tamiyo's capacity for structural independence, we subjected the system to a catastrophic failure condition termed the **Lobotomy Scenario**.

In this configuration, the host network is severely compromised: a blind 1x1 Convolutional host with broken residual paths, effectively severing gradient flow and spatial awareness. Unlike standard residual tuning (where the seed refines a competent host), this scenario requires the seed to assume full load-bearing responsibility for the task.

**Observed Results (requires validation):**

Rather than collapsing, the policy exhibited a distinct **rescue behaviour**:

1. **Panic Search (Epochs 0-10):** High-entropy exploration characterized by rapid germination and lysis of unsuitable candidates (`depthwise`, `norm`, `conv_small`), leading to a high "infant mortality" rate in the Seed Graveyard.
2. **Identification (Epoch 11):** Tamiyo identified a `conv_light` blueprint (approx. 2.3k params) capable of restoring spatial vision.
3. **Bypass Grafting:** Control was routed almost exclusively through the seed. Counterfactual analysis confirms the seed alone achieves **46.1%** accuracy, compared to the host's **0.0%**.

**Tentative Interpretation:**

This result may validate a critical hypothesis: **Esper does not require a competent host.** If the "Gradient Isolation" mechanism is robust enough to train fully independent modules from scratch, the system can perform functional organ transplants on dead infrastructure. The agent did not just *tune* the model; it *rebuilt* it.

If confirmed, this would constitute evidence of **Architectural Self-Repair**: break the spine, and the organism grows an exoskeleton to hold itself up.

---

### A.2 Draft Figure Caption (for TUI screenshot)

**Figure X: Forensic Analysis of a Rescue Operation.**

*Telemetry from the "Lobotomy" stress test on CIFAR-10. Top Right: Best run achieves 46.1% accuracy despite a functional host baseline of 0.0%. Center Right: The "Seed Graveyard" shows the policy rejecting 11 candidate structures (red) before committing the successful `conv_light` module (green). Bottom: Counterfactual analysis confirms the seed provides 100% of the effective signal (+35.5 individual contribution vs 0.0 baseline).*

---

### A.3 Engineering Sanity Checks

Before any publication, validate the following:

1. **Rent Calculation for Broken Host:**
   - **Risk:** If the host has 0 gradient/accuracy, does the Rent formula (which usually scales based on host size or performance) behave correctly?
   - **Check:** Ensure `P_host` in the rent formula `log(1 + (Base + P_seed)/P_host)` is the *physical* size of the broken host, not zero. If `P_host` is read as 0, rent explodes to infinity.
   - **Expected:** Since it's a 1x1 ConvNet, it has parameters (just useless ones), so the math should hold.

2. **Reproducibility:**
   - Can the rescue behaviour be reproduced across multiple seeds?
   - Is the "panic search" phase consistent or stochastic?

3. **Counterfactual Validity:**
   - Confirm the 46.1% vs 0.0% split is from proper ablation (seed zeroed vs host zeroed), not a reporting artifact.

4. **Gradient Isolation Verification:**
   - Confirm gradients actually flow through the seed in isolation when the host is broken.
   - Check that the seed isn't accidentally receiving signal through some other path.
