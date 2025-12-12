## Companion “Beta Paper” Skeleton

### Hierarchical Control and Recursive Morphogenesis in Seed-Grafted Neural Networks

*(Tamiyo & Narset; seeded blueprints; signal locality; promotion path)*

**Status:** Beta / companion note (structurally complete; evidence and some definitions intentionally deferred)

**Authors:** [Author 1], [Author 2], …
**Affiliation:** [Org / Lab]
**Artefacts:** [repo], [telemetry schema], [example runs], [controller configs]
**Version / Date:** [v0.1], [YYYY-MM-DD]

---

## Abstract

Large-scale morphogenetic systems face two scaling pressures: (i) **control bottlenecks**, where a single controller cannot manage the combinatorics of many seed locations, and (ii) **representation recursion**, where a “seed” may itself become a host by fossilising a blueprint that contains internal seed slots. We sketch a hierarchical control architecture comprising a global strategic controller (**Tamiyo**) and many local tactical controllers (**Narset**), and a recursive growth mechanism where blueprints can contain embedded seed slots (“seeded blueprints”). We argue that hierarchy is primarily an **information design** problem: local controllers should observe local signals (marginals, local gradient health, local traffic) while global controllers observe regional summaries (stability, budget requests, anomaly flags). We also outline a development strategy (“building Narset by stealth”) in which today’s flat controller is intentionally constrained to region-local observations so it can later be promoted to a tactical role without relearning. This document is a structurally complete proposal with placeholders for formalisation, evidence, and design refinements.

**Evidence placeholders:** [hierarchical scaling experiments], [recursion case studies], [signal locality ablations].

---

## 1. Introduction

Seed-grafted morphogenesis enables models to grow, integrate, and prune modules during training. The “flat” variant (single controller, fixed slots) is workable at modest scale, but two behaviours emerge as the system scales:

1. **Control does not scale linearly** with the number of slots/seeds; decision complexity and credit assignment become bottlenecks.
2. **Architectural recursion becomes natural:** a seed can fossilise into a new host-like structure, and if that structure itself contains seed slots, morphogenesis becomes hierarchical in topology as well as control.

This companion paper focuses on two forward-looking extensions:

* **Recursive seeds:** blueprints may contain embedded seed slots (“blueprints can contain seeds”).
* **Hierarchical control:** a strategic controller allocates budgets and sets constraints, while local controllers manage tactical integration decisions.

We treat these as a *proposal* rather than a confirmed result. The intent is to make the design legible and testable.

**Open question:** Which parts of this hierarchy should be learned vs rule-based at each layer?

---

## 2. Background and Relationship to the Main Paper

This document assumes familiarity with the base Esper concepts: host, slots, seeds, blending, fossilisation/culling, counterfactual marginals, interaction terms, hygiene, and budget control.

### 2.1 What this paper adds

* A definition of **seeded blueprints** (recursive morphology).
* A definition of a **two-layer controller** (Tamiyo + Narset) and candidate interfaces.
* A hypothesis about **signal locality**: global signals harm local policies.
* A practical development plan: **promote current Tamiyo into Narset** later.

### 2.2 What this paper does *not* claim (yet)

* That recursion is always beneficial.
* That two layers are sufficient (could require >2).
* That the proposed interfaces are optimal.
* That signal locality always dominates (cross-region coupling may break locality).

---

## 3. Recursive Morphogenesis: Seeded Blueprints

### 3.1 Definition: “Blueprints can contain seeds”

A *seeded blueprint* is a blueprint that instantiates a subgraph which includes **internal seed slots**. When the outer blueprint fossilises, the internal slots become active attachment points for future growth.

**Working semantics (draft):**

* A seed is not merely a module; it is a **growth anchor** that may contain further growth anchors.
* Fossilisation locks weights **and may register new local anchors**.

**Wobbly bit to fix later:** whether internal anchors are created at blueprint instantiation vs only upon fossilisation.

### 3.2 Recursion depth and why it should be rare

Recursive structure is powerful but dangerous:

* telemetry explosion (nested seeds multiply logs),
* credit assignment ambiguity (which controller gets credit/blame?),
* debugging complexity (more moving parts),
* risk of uncontrolled capacity growth.

**Design knob:** cap recursion depth at (d \le [,]), or require explicit strategic approval for deeper nesting.

### 3.3 Metamorphosis reframed as “seed becomes new host”

In the base framing, metamorphosis is “seed replaces host”. With seeded blueprints, it becomes “seed **becomes** a new host with its own slots”.

This reframing changes what “success” looks like:

* the adult topology may have **new anchors** not present in the original host,
* long-term growth can be concentrated in regions that have proven expandable.

**Evidence placeholder:** [case study where seeded blueprint creates a stable successor hierarchy].

---

## 4. Hierarchical Control: Strategic Tamiyo and Tactical Narset

### 4.1 Motivation: the scaling problem

At large parameter counts and many growth anchors, a single controller faces:

* combinatorial action space (slot × blueprint × schedule × lifecycle op),
* sparse credit assignment,
* observation overload (too many seeds to track),
* and the “attention problem” (which region matters right now?).

### 4.2 Architecture sketch (two-layer)

```
Strategic Tamiyo (global)
  - allocates budgets to regions/controllers
  - sets constraints or goals
  - handles rare, high-impact actions (region creation/reorg)

Tactical Narset (local, per region)
  - manages seeds within a region
  - fast-loop blend/cull decisions
  - reports summaries and exceptions upward
```

**Wobbly bit to fix later:** what counts as a “region”? (static partition, learned partition, topology-defined partition).

### 4.3 Interface design options

Tamiyo → Narset can send:

* **goals** (high autonomy, hard credit assignment),
* **constraints** (lower autonomy, clearer accountability),
* **budgets** (resource-bounded autonomy; simplest and attractive).

Narset → Tamiyo can send:

* **summaries** (scalable but lossy),
* **exceptions** (efficient but can hide slow failures),
* **budget requests** (supports “help me” dynamics).

**Candidate MVP interface (draft):**

* Downward: budget + constraint flags (e.g., “avoid germination”, “prioritise pruning”, “stabilise”).
* Upward: region health summary + anomaly flags + emergency budget request.

---

## 5. Credit Assignment Across Layers

Hierarchical control is mostly a credit assignment problem with better marketing.

### 5.1 Candidate approaches (non-exclusive)

* **Feudal:** Tamiyo reward = sum of Narset rewards; Narset reward = task return + local efficiency.
* **Options framework:** Narset policies are options; Tamiyo selects among them; credit flows through selection.
* **Shared global reward + local shaping:** both layers receive global task reward; Narset receives local shaping (budget adherence, local stability).

**Wobbly bit to fix later:** how to avoid local shaping creating perverse incentives (e.g., Narset hoards budget, refuses risky metamorphosis).

### 5.2 A conservative “learned local, rule-based global” alternative

Sometimes the tractable version is:

* Narset learned (tactical decisions),
* Tamiyo rule-based (budget allocation heuristics),
  then later make Tamiyo learned once the interface is proven.

This is explicitly listed as a potential stepping stone.

---

## 6. Signal Locality: Observation Design at Scale

### 6.1 Claim (hypothesis): global signals are noise for local control

At regional scale, global loss spikes can be unactionable: Narset can’t know whether the spike was caused by her region, another region, or unrelated host drift.

**GDP analogy:** giving Narset global gradient norms is like a shop owner using GDP to decide whether to hire one employee.

### 6.2 What each layer should observe (draft)

**Narset (local) needs:**

* per-seed counterfactual marginals (local),
* local loss/gradient health,
* local traffic/utilisation,
* local budget remaining,
* possibly neighbour stability *if* cross-region coupling exists.

**Tamiyo (global) needs:**

* per-region health summaries,
* global resource utilisation,
* anomaly reports,
* cross-region interaction proxies (if any).

### 6.3 Protecting the promotion path

If today’s controller is later promoted into Narset, it must not depend on signals that won’t exist at Narset scale.

**Rule of thumb:** keep today’s observation space region-local even if you could give more.

**Evidence placeholder:** [ablation: adding global signals worsens local policy learning].

---

## 7. “Building Narset by Stealth”: A Development Strategy

### 7.1 Strategy

* Build and validate the tactical controller first (today’s “Tamiyo”).
* Constrain it to region-local observations and responsibilities.
* Later: promote it to **Narset**, add a new strategic Tamiyo above.

### 7.2 Why this is attractive

* avoids designing the hierarchy interface before knowing what matters,
* avoids rewriting the tactical policy at scale-up time,
* makes debugging today directly useful for tomorrow’s scaling.

### 7.3 Practical guidance: turn debugging intuition into specs

Track what you actually look at:

* “is this region healthy?” → define the 3-number summary,
* “should I intervene?” → define triggers,
* “was that good?” → define the regional outcome metric.

---

## 8. Interaction, Turbulence, and Hierarchy

This section connects hierarchy to the in-flight detection problem.

### 8.1 Local turbulence vs global turbulence

Local controllers may see negative interactions within a region that are recoverable; global controllers see aggregate instability across regions.

**Hypothesis:** hierarchy helps because it separates:

* local “recovering turbulence” (Narset manages with turbulence budgets),
* global “system is unstable” (Tamiyo reallocates energy).

### 8.2 Candidate local features for Narset

* local (\Delta \tilde{I}) (interaction derivative, when feasible),
* local gradient health,
* local traffic shifts,
* local collapse warnings.

### 8.3 Cross-region coupling (the awkward part)

If regions are not independent, local decisions can create global externalities.

**Wobbly bit to fix later:** define and measure cross-region interaction proxies without exponential diagnostics.

---

## 9. System Design: Budgets, Hygiene, and Hierarchical Allocation

### 9.1 Budget “sloshing” generalised

In the base paper, budget sloshes between growth and hygiene. In hierarchy, budgets slosh:

* between regions (Tamiyo → Narsets),
* between growth and hygiene within regions (Narset ↔ local Emrakul processes).

### 9.2 Local hygiene: “Emrakullets” (optional term; probably rename later)

A region may run local hygiene passes, reporting detritus deltas upward.

**Wobbly bit:** naming aside, define whether hygiene is:

* centrally scheduled,
* region-controlled,
* or jointly constrained (Tamiyo sets cadence bounds; Narset chooses times).

---

## 10. Experimental Plan (Placeholders)

### 10.1 Experiments for recursion

* seeded blueprint vs non-seeded blueprint variants,
* capped recursion depth sweeps,
* telemetry overhead vs performance trade-offs,
* successor stability and hygiene outcomes.

### 10.2 Experiments for hierarchy

* flat controller vs 2-layer controller on increasing slot counts,
* local-only observation vs mixed (global leakage) observation,
* interface variants (budgets vs constraints vs goals),
* learned local + heuristic global vs learned both.

### 10.3 Metrics (draft)

* Pareto: accuracy vs latency/params vs seed count,
* stability: collapse rate, oscillation, rollback frequency,
* control: action entropy, budget utilisation efficiency,
* observability: region summary fidelity vs performance,
* overhead: telemetry volume, diagnostic cost.

**Evidence placeholders:** [plots], [tables], [ablation results].

---

## 11. Discussion (Beta)

### 11.1 Why this matters

If morphogenesis is to scale, both topology and control likely need to become hierarchical. Recursive seeds make metamorphosis more expressive; hierarchical control makes it manageable.

### 11.2 Main risks

* runaway complexity (telemetry, debugging, credit assignment),
* perverse incentives across layers,
* cross-region externalities,
* policy brittleness under distribution shift.

### 11.3 What we expect to change after real experiments

* The MVP interface will likely simplify (budgets-only is a strong candidate).
* Some signals proposed here will prove useless (especially global leaks into local policies).
* Recursion depth limits will become stricter and more formal.

---

## 12. Limitations (Beta)

* This document is primarily architectural; it does not yet present definitive empirical validation.
* “Region” is underspecified; different partition schemes may change everything.
* Cross-region coupling is acknowledged but not solved.
* Credit assignment proposals are not yet tested.

---

## 13. Conclusion

We presented a structurally complete proposal for scaling seed-grafted morphogenesis through (i) recursive seeded blueprints and (ii) hierarchical controllers (Tamiyo + Narset), motivated by signal locality and control bottlenecks. We outlined candidate interfaces, promotion strategy, and an experimental programme to validate or falsify the key hypotheses.

**Evidence placeholder:** [summarise validated claims once results exist.]

---

## Appendices (optional)

### A. Draft interface schemas

* Narset → Tamiyo: health summary tuple, anomaly flags, budget request
* Tamiyo → Narset: budget allocation, constraint bits

### B. Glossary

* seeded blueprint, region, local controller, strategic controller, option, feudal reward, etc.

### C. Open questions checklist

* region partition definition
* cross-region interaction proxy
* recursion depth gating policy
* learned-vs-rule split per layer

---

### To fix later (quick list)

* Formal definition of “region” and anchor registration timing
* How local Emrakul interacts with global hygiene policy
* Credit assignment that doesn’t create bad incentives
* Measuring cross-region coupling cheaply
* Whether recursion should require explicit strategic approval
* Whether the hierarchy should be 2-layer or deeper
