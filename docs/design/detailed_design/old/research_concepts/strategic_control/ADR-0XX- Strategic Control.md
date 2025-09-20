Here’s an expanded, prose-first ADR that folds in **Narset**, scaffolds, and the Emrakul/Elesh coupling while keeping your original intent and tone. I’ve preserved your numbering, tightened some language, and added the missing connective tissue so evaluators can read it straight through.

---

# ADR-XXX: Federated Controller Architecture for Large-Scale Models

**Version:** 1.0
**Status:** PROPOSED
**Date:** 2025-09-11
**Author:** System Architecture Conclave
**Component:** Control Plane — Strategic Control Subsystem

> **Reading note:** When reviewing the scaffold component of this ADR, agents should also read **Urza** (blueprint library), **Karn** (blueprint generator), and **Tezzeret** (blueprint compiler) for context.

---

### 1.0 Executive Summary

Esper needs a control plane that scales as the model scales. One monolithic strategist (Tamiyo) becomes a bottleneck once we’re shepherding hundreds of billions of parameters and thousands of concurrent adaptations. This ADR proposes a **Federated Controller Architecture**: a single global strategist (**Tamiyo Prime**) that sets risk posture and budgets, and a fleet of **Narset** controllers that act tactically within clearly bounded regions of the model.

Two mechanisms make this both safe and powerful:

* A **Stability (Energy) Budget** that Tamiyo allocates and Narset spends, turning “how much change is acceptable right now?” into an explicit, audited resource.
* A **Strategic Opportunity Proposal (SOP)** lane that lets Narset argue for extra budget when data indicates outsized upside.

We further integrate **Emrakul** (offline structured pruning planner) and **Elesh** (offline analysis workers) with a simple rule: **Emrakul’s pruning budget is the inverse of Narset’s near-term spend**. When Narset is executing a lot of change (especially with temporary **scaffolds**), Emrakul quiets down; when Narset is quiet, Emrakul gets a freer hand.

This turns risk management from a tangle of heuristics into an emergent, self-mediating economy of change.

---

### 2.0 Core Architectural Decision

#### Hierarchical strategic control with an explicit Stability/Energy Budget

* **Control model.** Tamiyo Prime owns global strategy and risk; Narset instances own tactical execution within disjoint regions (e.g., blocks of layers, vision encoder, modality adapters).
* **Budget model.** Tamiyo allocates a finite **Stability/Energy Budget** per cycle. Narset spends from it to germinate seeds, run grafting ramps, and—when warranted—deploy **scaffolds** (temporary, self-sunsetting blueprints).
* **Opportunity model.** Narset can submit **SOPs** to request more budget for high-impact moves; Tamiyo approves or defers based on global posture and unallocated capacity.
* **Compaction coupling.** Emrakul is subordinate to Tamiyo and throttled **inversely** to Narset’s active/forecast spend, so pruning never fights stabilization.

---

### 3.0 System Architecture & Roles

This is a simple chain of intent and evidence.

#### 3.1 Tamiyo Prime — strategist and central bank

Tamiyo holds the global risk dial. It reads cross-system telemetry, decides “how much change the whole model can tolerate,” and mints that capacity as a numeric budget for the next cycle. Tamiyo’s output is policy, not micromanagement: per-region allocations, guardrails (e.g., embargo rings), and approvals or denials of SOPs. Tamiyo also determines when we shift posture (explore vs exploit) and when emergency energy should be released for a scaffolded stabilization.

#### 3.2 Narset — many tactical lieutenants

Each Narset is responsible for one well-bounded region. It watches local stability, chooses which seeds to run in shadow or graft, requests scaffolds when needed, and spends its allocation deliberately. Narset speaks the language of **AdaptationCommands** to Kasmina and **ScaffoldRequests**/**SOPs** to Tamiyo. It is judged by outcomes: stability restored, capability gained, budget respected.

#### 3.3 Emrakul & Elesh — quiet strategists of removal

Elesh computes importance and vitality offline at checkpoint windows. Emrakul turns that into concrete **structured pruning** plans (channels, heads, sometimes layers). Emrakul only plans within the budget Tamiyo allows and never within **Protected Zones** that Tamiyo has granted around active scaffolds or volatile regions. When Narset is active, Emrakul stands down; when Narset stands down, Emrakul compacts.

*(Diagrams omitted by design; the mental model is a single strategist, many field officers, and a careful gardener who prunes only when the field is calm.)*

---

### 4.0 The Stability/Energy Budget

Think of this as the system’s “capacity for change” for a cycle. We spend it where the ROI is highest and where the risk is lowest.

#### 4.1 Cost of change = risk × disruption

An adaptation has an **inherent risk** (from Urabrask’s BSDS on the blueprint) and an architectural **disruption** (how widely it perturbs the graph). We price both. High-risk micro-changes can be cheap; low-risk macro-changes can be expensive. The exact weights live in Tamiyo policy; the principle is simple: we pay most for the things that shake the graph the hardest.

#### 4.2 Emergent slowdown without heuristics

Budgets add themselves up. If Tamiyo funds one Narset’s ambitious SOP, everyone else automatically has less room to move. That scarcity forces incremental, conservative choices elsewhere—without a single hardcoded “limit changes to N” heuristic.

#### 4.3 Tamiyo’s policy, not hand-tuned rules

Tamiyo’s allocator is a policy (trained by Simic, evaluated by Jace) that reads volatility, recent win/loss ratios, confidence from Urabrask, backlog pressure from Narset, and tail-latency/oom risk from SRE. It sets allocations to maximize expected improvement subject to a stability constraint. We keep a manual override, but the happy path is policy-driven.

---

### 5.0 SOPs: let the field argue for upside

Narset can propose a move that exceeds its local budget. An SOP is short and economic:

* **Opportunity.** A concrete, data-backed hypothesis (“layers 9–16 show a representational bottleneck”).
* **Plan.** The specific blueprint(s) and grafting plan (or a request for a **scaffold**, see §7).
* **Ask.** The incremental budget, priced by risk × disruption.
* **ROI.** Predicted capability increase and expected parameter efficiency.
* **Mitigations.** How we will keep drift controlled (staged α, KD weaning, tighter clipping).

Tamiyo plays venture capitalist: it compares SOPs across Narset regions, funds the best bets, and reserves enough budget for safety and compaction.

---

### 6.0 Integration & System Impact

Nothing here changes Kasmina’s core role as a pure executor with strict gradient isolation. Narset just becomes the main *voice* telling Kasmina what to try, when, and by how much. Tolaria routes the right telemetry to the right Narset. Simic provides the lieutenant policy that runs Narset’s day-to-day choices (seed selection, α schedules, local risk tolerance). Jace adapts curriculum so that different parts of the model are not asked to learn conflicting things at once.

---

### 7.0 Scaffolds — temporary capacity that teaches, then leaves

Scaffolds are **a class of blueprint** (not a new seed type) that any seed can germinate into by setting `class=scaffold` and sunset metadata. They exist to stabilize flaky regions or accelerate learning during a rough patch, then get out of the way.

A scaffold trains **on-graph** after a short sync, follows the **same lifecycle** as any seed (Dormant → Train Isolated → Grafting → Stabilisation → Active/Fossilised), but with three policy differences:

1. **Priority energy during crisis.** Narset can pull from a small emergency pool to give a scaffold room to work when telemetry says the region is melting down.
2. **Success is stability, not just parity.** We watch jitter, drift, and feature diversity as well as task parity. When those hold, we **wean**: the scaffold’s α ramps down while the host is encouraged (via light KD and small L1 anchoring) to reproduce the scaffold’s helpful behavior.
3. **Self-sunset.** When gates are green for K windows and a weaning dwell is clean, the scaffold converts to a **no-op** kernel and returns its unused energy. If a minimum TTL expires without success, it sunsets anyway and the region is escalated to Tamiyo for a different tactic.

While a scaffold is active, Tamiyo grants a **Protected Zone** around that region. Emrakul does not plan compaction inside it; Elesh still measures but marks those stats as “for diagnostics only.”

This is the “Mary Poppins” pattern: arrive, fix, teach, leave.

---

### 8.0 Emrakul/Elesh coupling and the inverse throttle

Emrakul operates only at checkpoint boundaries, planning **structured pruning** that keeps tensors dense (channels, heads, sometimes layers). That plan is constrained twice:

* **Protected Zones.** No surgery in, or too close to, an active scaffold’s region.
* **Inverse budget.** Emrakul’s pruning allowance for a cycle is a decreasing function of Narset’s current and forecast spend. If Narset is busy, pruning quiets; if Narset is quiet, pruning proceeds.

This interplay prevents “right hand builds while left hand tears down” incidents and aligns removal with periods of calm.

*(We keep the option to shard Emrakul later if checkpoint planning time approaches the budget; start with one Emrakul Prime for simplicity.)*

---

### 9.0 Safety, rollback, and invariants

Nothing here relaxes safety:

* **Gradient isolation** remains absolute until joint training is explicitly enabled; all grafting blends use `host_activations.detach()`.
* **Two optimizers** from day one; Narset never steps the host optimizer during isolated/grafting phases.
* **Circuit breakers** protect serving; all asserts become breakers with telemetry and backoff.
* **Rollback-first.** Every alpha ramp has a one-touch “α=0 identity” escape and a hot checkpoint to restore.

Scaffolds add one more invariant: **while a scaffold is stabilising or weaning, surgery is embargoed** in a ring around its region until one full clean checkpoint passes after sunset.

---

### 10.0 Rollout plan

* **Phase A (control wiring).** Tamiyo exposes budgets; Narset consumes them, files SOPs and ScaffoldRequests; Oona carries the new messages; Tolaria routes per-region telemetry.
* **Phase B (execution wiring).** Kasmina honors `class=scaffold` metadata and sunset conversion; Elesh filters embargoed regions; Emrakul reads protected zones and inverse budget.
* **Phase C (policy tuning).** Simic trains Tamiyo’s allocator; Jace tunes the Narset lieutenant policy; SRE rehearses scaffold thrash breakers and protected-zone expiry.

We gate promotion on simple, observable outcomes: fewer regressions during active adaptation, cleaner post-scaffold checkpoints, and measurable speedups when Emrakul is allowed to compact.

---

### 11.0 Open questions for the Conclave

* **One Emrakul or many?** Start with one; shard by Narset scope only when checkpoint planning threatens the window and cross-scope interference is demonstrably low.
* **Protected zone radius.** Default one block wide? Two for MoE or long-skip models?
* **Emergency pool sizing.** Fixed per model size, or adaptive to recent volatility?
* **Sunset certainty.** One clean checkpoint after weaning, or two, before lifting embargo?

---

### 12.0 Why this architecture now

We’re already seeing the limits of central planning at scale. This design lets us move many parts of the model forward in parallel without losing control of risk. It gives us a graceful way to **add temporary complexity when the graph needs help** (scaffolds) and to **remove permanent complexity when it’s safe** (structured pruning), all under a single budgeted policy.

It is principled, observable, and humane to operate. And it aligns perfectly with Esper’s north star: **maximum capability per parameter**.
