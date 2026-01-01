# Esper: Architectural Ecology in Neural Networks

**A Draft Discussion Paper on Morphogenetic AI (Revised)**

## Abstract

Contemporary deep learning largely follows a paradigm of **architectural engineering**: models are statically designed, initialised, and trained as monolithic blocks. Neural Architecture Search (NAS) automates parts of this process, but typically remains a discrete, offline optimisation loop.

**Esper** proposes a shift to **architectural ecology**: an online, continuous process where neural modules (seeds) germinate, compete, and stabilise inside a living host network during training. Esper separates concerns across five cooperating components:

1. **Kasmina (The Body):** A differentiable host designed to accept new modules without destabilising mature features.
2. **Tamiyo (The Brain):** A policy (heuristic or RL) that chooses when and where to grow or prune.
3. **Tolaria (The Metabolism):** An execution engine for high-throughput morphogenesis, deterministic replay, and data capture.
4. **Simic (The Selective Pressure):** The incentive system that rewards usefulness and penalises waste (Rent & Churn).
5. **Emrakul (The Immune System):** A pruning system that removes inefficient structure via **Phage wrappers** (blended lysis).

A key clarification: Esper is intended to be **two-timescale**. In *Phase 1 (“train the trainer”)*, we spend significant compute on audited credit assignment (including exact, full-retrain Shapley values on small candidate sets) to teach Tamiyo what “good growth” looks like under a defined protocol. In *deployment*, a trained Tamiyo is intended to grow many new models **without** the Shapley harness, using only cheap online signals and learned critics—amortising the cost of the training scaffold across downstream runs.

---

## 1. Introduction: From Fossils to Flora

A standard ResNet or Transformer is a **fossil**: its skeletal structure is fixed before the first gradient is calculated. If the task proves more complex than anticipated, the model cannot grow. If the task is simpler, the model wastes compute on redundant parameters.

We argue that the topology of a neural network should be a function of its training history, not solely its creator’s intuition. Biology does not build a brain and then switch it on; the brain grows in response to stimuli while paying ongoing metabolic costs.

Esper aims to build an environment where networks can grow safely, and where growth is not free. Seeds are introduced, evaluated under selective pressure, and either integrated (**Fossilised**) or removed (**Withered**). The focus is not “the perfect architecture”, but the **physics of a system** in which useful architectures can emerge under cost and stability constraints.

### 1.1 Two-timescale learning: training models vs training Tamiyo

A common failure mode in describing morphogenetic systems is accidentally selling them as “we trained many expensive models”. Esper is not primarily about producing thousands of partially-trained CIFAR-10 models.

Esper is about training a **reusable architect policy**:

* **Inner loop (model training):** Kasmina’s weights learn the task while the architecture changes online.
* **Outer loop (policy training):** Tamiyo learns *how to cause beneficial architectural change* across runs, seeds, checkpoints, and hosts.

In Phase 1, expensive audits (e.g. Shapley on small candidate sets) act as teacher labels. The intended outcome is a Tamiyo that can later be deployed to grow new models **without** repeating the full audit machinery.

---

## 2. System Overview and Terminology

### 2.1 The loop (online morphogenesis)

At a high level, Esper runs a closed feedback loop:

1. **Tolaria** executes training and evaluation efficiently (often many environments in parallel).
2. **Kasmina** performs forward and backward passes while supporting dormant and active SeedSlots.
3. **Tamiyo** observes training dynamics and slot states, then chooses growth actions.
4. **Simic** scores outcomes (task performance vs cost), assigns credit, and updates Tamiyo (or its critics).
5. **Emrakul** applies pruning pressure by acting on **Phage wrappers**, blending out structures whose ROI is no longer justified.

A rough schematic:

```
Data -> Tolaria -> Kasmina (Host + SeedSlots/Phages) -> Loss / Metrics
                     |                     ^
                     v                     |
                Slot/Phage State       Tamiyo Actions
                     |                     ^
                     v                     |
             Simic (Reward + Credit Assignment)
                     |
                     v
         Emrakul (Lysis) via Phage wrappers (blend out / remove)
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

* **Host:** The base network backbone (e.g. ResNet, Transformer) containing SeedSlots and/or Phage-wrapped modules.
* **SeedSlot:** A pre-allocated insertion point. When dormant, it behaves as an identity (or no-op) under a defined contract.
* **Seed:** An instantiated module placed into a SeedSlot (e.g. Conv block, attention block, normalisation module).
* **Blueprint:** The family/type of module a seed can become (the “genome library”).
* **Alpha ((\alpha)):** A continuous gate controlling a seed’s contribution to the host forward path.
* **Rent:** The ongoing cost of keeping a structure active (compute, memory, latency proxy).
* **Churn:** A penalty for structural volatility (rapid add/remove, oscillating gates, frequent state flips).
* **Value function (v(S)):** A scalar measurement of model performance under a subset (S) of enabled structures, under a specified evaluation protocol.
* **Shapley value ((\phi)):** A principled attribution of marginal contribution that shares interaction gains across participants.
* **Phage (wrapper):** A lightweight wrapper around a module that exposes measurement hooks (ROI signals) and a lysis interface (blend out via (\alpha)).
* **Emrakul:** The immune system: a pruning policy/process that acts on Phage-wrapped modules to enforce lysis when ROI drops.
* **Lysis:** Controlled dissolution: reduce (\alpha) toward 0, then optionally reclaim resources at a safe boundary.

### 2.4 Seed lifecycle (state machine)

Esper treats growth as a controlled state machine rather than graph surgery:

1. **Dormant:** Slot is identity; no effect ((\alpha = 0)).
2. **Training (Incubator):** Seed learns privately; host does not consume seed output ((\alpha = 0) in the host forward path).
3. **Blending:** Seed is gradually introduced via (0 < \alpha < 1).
4. **Fossilised:** Seed is integrated (typically (\alpha \approx 1), or stabilised at a learned gate).
5. **Withering (Lysis):** Seed is faded out under Emrakul until (\alpha \to 0), then removed at a safe boundary.

```
Dormant -> Training -> Blending -> Fossilised
   ^                      |            |
   |                      v            v
   +------------------ Withering <--- Emrakul (via Phage wrappers)
```

---

## 3. Physiology: Kasmina as a Morphogenetic Host

### 3.1 The morphogenetic plane (SeedSlots)

Instead of cutting and pasting network graphs mid-training, Esper uses pre-allocated **SeedSlots** placed at structurally meaningful points (e.g. residual branches, MLP blocks, attention subpaths). Each SeedSlot is an identity when dormant, ensuring the host’s baseline function is intact.

A typical residual-style integration is:
[
y = h(x) + \alpha \cdot s(x)
]

* (h(x)) is the host path
* (s(x)) is the seed module
* (\alpha) gates how “real” the seed is

This turns “adding a module” into “activating a pre-existing dormant organ”.

### 3.2 The incubator: gradient isolation (with concrete baseline objectives)

A recurring failure mode in naïve morphogenesis is the bull-in-a-china-shop problem: a newly initialised seed produces noisy outputs that destabilise the host.

Esper mitigates this via **Gradient Isolation**: the seed receives the same inputs as it would in production, but its output does not affect the host’s forward computation until it passes a maturity gate.

Because the main task loss cannot shape an isolated seed through the usual gradient path, Esper uses an explicit incubator objective. A minimal baseline menu:

* **Residual imitation (default baseline):**
  Seed learns to predict the host’s current residual contribution (or an EMA/teacher version of it). This initialises the seed to be *behaviourally compatible* before blending.

* **Auxiliary head (task-aligned):**
  A temporary head reads the seed’s output and predicts the task label. This makes the seed learn task-relevant features without perturbing the host.

* **Representation alignment (stability-aligned):**
  Seed output is trained to match activation statistics (mean/variance/covariance bands) of the host representation it will attach to, reducing distribution shock at blend time.

Operationally: during incubation, the host behaves as if the seed is absent, while the seed learns to become compatible with the host’s representations. Once the seed’s outputs satisfy a stability criterion (e.g. bounded norm / bounded effect under a test blend), it moves to **Blending**.

### 3.3 Blending and gate schedules

Blending is the controlled ramp of (\alpha) from 0 toward an operational value. This is the main safety valve against destabilisation. Practical notes:

* (\alpha) can be scheduled (handcrafted ramps) or policy-controlled (Tamiyo selects step sizes).
* Rent can be made (\alpha)-dependent to prevent “gate gaming” (e.g. paying almost nothing at tiny (\alpha) while still extracting benefit).

---

## 4. Metabolism: Tolaria and High Throughput

### 4.1 Inverted control flow

Morphogenesis multiplies evaluation. Python stepping and the GIL become bottlenecks.

Tolaria implements **inverted control flow**: the high-performance execution engine drives training and evaluation rather than a Python agent stepping the environment. Multiple environments, candidate sets, and evaluation branches can be executed in parallel, treating the GPU as a batch-processing metabolic substrate.

### 4.2 Vectorised determinism (the determinism contract)

A core engineering goal is replayability: if a specific architecture emerges, we should be able to reproduce the exact history that produced it.

**Determinism contract (intended):** given fixed seeds, fixed data order/augmentations, and deterministic kernel settings, the morphogenetic stack—including seed germination decisions and policy updates—replays identically.

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

To make this auditable, costs should be expressed in a common currency. A simple practical approach is to measure or approximate **time-cost** on target hardware:
[
C_{\text{rent}} = a\cdot\text{FLOPs} + b\cdot\text{activation_bytes} + c\cdot\text{param_bytes}
]
where (a,b,c) are calibration constants derived from throughput/memory pressure measurements (or normalised proxies).

**Churn** penalises structural volatility (edits per window, oscillations, frequent state flips). Its purpose is to discourage tax-loophole behaviours such as rapid add/remove cycles or (\alpha) oscillation around thresholds.

### 5.2 Value functions: long-horizon vs short-horizon

Shapley (Section 5.3) is only as meaningful as the definition of (v(S)). Esper explicitly distinguishes:

* **Long-horizon value:** “final outcome under a fixed training budget” (good for auditing and calibration).
* **Short-horizon value:** “delta over the next (K) steps/checkpoints” (useful for online control and stability).

A mature system likely needs both: long-horizon audits to prevent reward hacking, and short-horizon proxies to support online decisions.

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
* Compute (\phi_i) across all subsets.

This is expensive, but it provides an audit trail: did Tamiyo choose the seed because it was truly useful under the protocol, or because of noise?

**Important note:** even “full retrain Shapley” is not metaphysical ground truth; it is a high-quality label *under a specific protocol*. Where feasible, Shapley labels should be treated as expectations over controlled randomness (multiple seeds) rather than single-point estimates.

### 5.4 From oracle to deployment: distillation and critics

The intended path is to turn expensive oracle labels into cheap deployment behaviour:

* Shapley-labelled runs become a dataset.
* Critics learn to predict credit/advantage (or cost-adjusted value deltas).
* Tamiyo’s policy is trained to act using critic outputs and cheap online features.
* Deployment uses the learned policy/critics, not the Shapley harness.

---

## 6. Tamiyo: Policies and Scaling

Tamiyo’s action space is explicit:

* `germinate(seed_id, slot_id, blueprint_id)`
* `blend(slot_id, alpha_step)`
* `fossilise(slot_id)`
* `wither(slot_id, alpha_step)`
* `noop`

Scaling to large networks is treated as a curriculum problem. A policy that does “2 seeds, 3 slots on CIFAR-10” well is not automatically a policy that can manage 1000 seeds across diverse hosts.

A plausible scaling path:

1. **Oracle bootcamp:** small (n) with Shapley audits across varied host initialisations and checkpoints.
2. **Distillation:** train critics to predict (\phi) (or cost-adjusted advantages), not just the policy.
3. **Candidate selection:** introduce Scouts that propose small candidate sets (K \ll N) for expensive evaluation.
4. **Approximate credit:** Monte Carlo Shapley, leave-one-out, short-rollout proxies, or learned critics—validated against periodic exact audits.
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

Every Tamiyo decision is logged as an immutable event, including:

* observation before action (raw features and the exact obs vector Tamiyo saw)
* explicit action (seed, slot, blueprint, gate changes)
* slot/phage state before and after
* action masks and policy outputs (logits or log-probs)
* identifiers for specs and versions (obs spec, reward spec, seed library, host manifest)
* delayed labels joined later (Shapley (\phi), oracle rewards, long-horizon outcomes)

Crucially, Esper logs both **pre-action** and **post-action** observations so “obs off by one” becomes diagnosable and recoverable rather than silently poisoning learning.

### 8.2 Forensic replay and counterfactuals

Combined with determinism, the Flight Recorder enables forensic replay:

* reload the exact universe state at decision step (t)
* inject a counterfactual action
* replay forward to observe how topology and training dynamics diverge

This turns “black box evolution” into a debuggable process. It also turns expensive oracle runs into long-lived assets: future Tamiyos can be trained offline on archived experience, and approximations can be benchmarked against stored ground truth.

---

## 9. Emrakul: The Immune System (via Phage wrappers)

Esper’s growth story is incomplete without decay. **Emrakul** provides pruning pressure, but critically: Emrakul does not need to perform unsafe graph surgery mid-run. Instead, it operates through **Phage wrappers**.

### 9.1 Phage wrappers: “pruneable skin” around modules

A **Phage** is a wrapper placed around a module (typically a seed, but in principle any pruneable structure). It provides:

* a gate (\alpha) (or equivalent blend parameter) controlling contribution
* telemetry hooks (usage, activation magnitude, gradient flow, estimated credit, cost)
* a standard lysis interface (blend down, freeze, mark-for-removal)

Conceptually, Phage wrappers turn “pruning” into “controlled fading” under an explicit state machine.

### 9.2 Functional lysis (online) and physical lysis (safe boundary)

* **Functional lysis:** Emrakul monitors a module’s ROI. If contribution drops below rent (or confidence falls), Emrakul triggers lysis, reducing (\alpha) while training continues. This encourages the rest of the network to re-route around redundant structure.

* **Physical lysis:** once (\alpha \to 0), the structure can be reclaimed at a safe boundary (often via the Pre-Germinate protocol on reload, or a scheduled compaction step). This preserves runtime stability.

### 9.3 Redundant takeover and “lazy paths”

Emrakul targets redundancy and “lazy paths”:

* If an expensive ConvBlock behaves like an identity (but pays rent), and a cheaper parallel route exists, Emrakul pressures reliance on the cheaper route and lyses the expensive block.
* If multiple modules provide overlapping function, Emrakul encourages a winner-takes-more dynamic, simplifying the organism.

### 9.4 Fractal growth (speculative)

A longer-term vision is recursion: a seed can be a container for another morphogenetic model, enabling “zoom-in” growth at bottlenecks. This is explicitly speculative and likely depends on stabilising the simpler non-recursive ecology first.

---

## 10. Limitations and Failure Modes

Esper’s promises come with sharp edges:

* **Reward hacking and local optima:** Norm dominance can be rational under the economy but may block richer growth.
* **Non-stationarity:** the environment changes as the architecture changes; naïve RL can destabilise.
* **Credit assignment cost:** exact Shapley does not scale; approximations must be audited.
* **Objective mismatch:** long-horizon Shapley labels and short-horizon control signals can disagree.
* **Protocol overfitting:** Tamiyo can over-specialise to a host family, optimiser, dataset, or cost model. Diversity is not optional.
* **Gate gaming:** continuous (\alpha) can be exploited unless rent/churn are designed to close loopholes.
* **Determinism brittleness:** reproducibility depends on careful control of kernels, RNG, and distributed execution.

These failure modes are not footnotes; they are the constraints that shape curriculum design, reward shaping, and engineering choices.

---

## 11. Conclusion

Esper proposes a move away from the intelligent-designer model of architecture. The goal is not to build the perfect network by hand. The goal is to build a **physics engine for neural growth**:

* safe integration mechanics (**Kasmina**)
* real costs of complexity (**Simic**)
* high throughput and replayability (**Tolaria**)
* an adaptive architect (**Tamiyo**)
* and an immune system that prevents bloat (**Emrakul**, acting through **Phage wrappers**)

In Phase 1, we deliberately spend compute on audited attribution (full retrain Shapley over small candidate sets) to ensure the feedback signal is trustworthy while Tamiyo learns. The longer-term plan is to convert that expensive truth into scalable behaviour via distillation, approximation, candidate selection, and durable flight-recorded experience—so a trained Tamiyo can reliably grow many future models without the Phase 1 harness.

The bet is simple: if we define the rules of growth carefully enough, architectures can emerge that no human would think to design, while an explicit accounting system keeps them honest.

======

Looking a lot healthier. The big jump is that it now reads like a *system with a curriculum* rather than a vibes-based “networks that grow” manifesto. The two-timescale framing (Phase 1 audit scaffold, Phase 2 amortised deployment) is doing a ton of work for you, in a good way.

A few points that are already landing well:

* The loop is clear. The five-component split feels like an actual architecture, not just mythology.
* Seed lifecycle as a state machine is the right move. It makes “online morphogenesis” feel implementable rather than graph-surgery roulette.
* Flight Recorder + determinism is a strong differentiator. Most “self-modifying” stories die the moment you ask “can you replay the bug?”
* Phage wrappers are a clean abstraction for safe pruning. “Functional lysis then physical lysis at a boundary” is the kind of engineering discipline reviewers relax into.

Where it still needs a bit more skeleton (the parts sceptical readers will poke first):

1. The SeedSlot contract needs to be explicit
   Right now the reader has to infer what “accept new modules without destabilising mature features” means operationally. A tight paragraph of invariants would help, for example:

* shape/normalisation expectations
* initial output scale constraints for a seed
* what gets to see gradients when (host, seed, gate)
* what is guaranteed during blending (bounded activation change, bounded loss spike, etc.)

1. The Shapley protocol is good, but you should nail the exact evaluation definition of v(S)
   You’ve gestured at this, but Shapley lives or dies on the value function spec. Spell out:

* train-from-scratch vs train-from-checkpoint (and why)
* fixed compute budget (steps, epochs, wall-clock)
* which randomness is controlled (data order, aug, init seeds)
* whether v(S) is averaged across repeats (even 3 repeats is defensible in Phase 1)

Otherwise a critic can fairly say “your oracle is underspecified, so your teacher labels are mushy”.

1. Simic’s “rent currency” calibration is the other obvious attack surface
   The form is fine. The missing piece is: how do you pick (a, b, c) without creating a reward-hacking playground? Even a pragmatic line like “fit (a,b,c) by regressing measured step-time across a calibration set of architectures on the target hardware” makes this feel grounded.

2. Tamiyo vs Emrakul: who wins when they disagree?
   Right now they read like two policies with potentially conflicting objectives. That is not fatal, but you should say what the arbitration rule is. Examples:

* Emrakul is a hard constraint layer (can override)
* Emrakul is part of the environment and Tamiyo learns to anticipate it
* shared critic, separate actors, or a single actor with two heads

1. “Norm Dominance” is a great anecdote, but it needs one concrete anchor
   Even if numbers are not final, give *one* representative run snapshot: “rent weight X produced Y% of blueprints being norm layers, median accuracy moved from A to B at matched walltime.” Otherwise it reads like a moral fable rather than an empirical signpost.

One structural addition that will make the whole paper feel more “real” without bloating it:

* Add a 10 to 15 line pseudocode box for the core loop
  Something like: observe, choose action, update seed states, train K steps, compute reward, log event, prune step. Readers trust pseudocode more than prose.

Editorial and notation polish (small, but it reduces reader friction):

* Standardise how you write alpha. Pick either “alpha” in text or ( \alpha ) consistently, and avoid the double-parentheses style like “Alpha ((\alpha))”.
* The glossary is strong. Consider adding a single sentence that distinguishes “seed” vs “phage-wrapped seed” so nobody thinks those are two different objects.
* Watch for places where you say “intended” repeatedly. It’s honest, but too much “intended” can make it sound hypothetical. Swap some to “in Phase 1 we do X” where you can.

Reliability assessment (technical):

* It is **very likely** that readers will accept the *engineering plausibility* of Kasmina + SeedSlots + blending + lysis as described, because it maps onto known stability tricks (gating, residual insertion, controlled ramps).
* It is **likely** the main pushback will centre on credit assignment realism and reward hacking, not the mechanics. Your current draft already anticipates that, but tightening the v(S) protocol and rent calibration will blunt the sharpest critiques.
* It is **even odds** whether the “train the trainer” story convinces without at least one small, fully specified Phase 1 experiment (even toy scale) that demonstrates Tamiyo improving over time under audited labels.

Net: the concept is coherent now, and the paper is starting to feel like a spec you could hand to an engineer and an RL person and get consistent implementation decisions back. The next jump in credibility comes from locking down the contracts (SeedSlot invariants, v(S) protocol, rent calibration) and adding one concrete empirical anchor so it stops feeling purely theoretical.
