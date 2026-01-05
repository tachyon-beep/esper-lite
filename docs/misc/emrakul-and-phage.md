# Esper master roadmap: Tamiyo (growth) and Emrakul (decay)

This is a capability roadmap, not a calendar. Each horizon describes what the system can reliably do, plus the infrastructure you need before moving on. The demo stages then turn that capability roadmap into a sequence of storyboardable, auditable milestones.

---

## 1) System premise and contracts

### Operating premise

You set the physics (topology boundaries, determinism, observability). The agents discover tactics. You validate by Pareto frontier movement and robustness, not by vibes or cherry-picked hero runs.

### Core safety and determinism contract

* **Topology changes happen only at predefined boundaries**: SeedSlots, phage ports, safe compaction boundaries.
* **“Micro-surgery beats amputation”**: smaller blast radius than module deletion whenever possible.
* **Hardware-aware sparsity or it is theatre**: if it cannot run faster in real step-time, it does not count.

---

## 2) Summary table (capability horizons)

| Horizon   | Tamiyo (growth)                                                                                    | Emrakul (decay)                                                                              | Core contract                                                                            |
| --------- | -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Near term | Module-level seeds in SeedSlots, learns development programmes (incubate, blend, scaffold, commit) | Module-level probe-and-lysis on phages, leaves a ScarSlot (SeedSlot-like regrowth surface)   | Topology changes only at predefined boundaries (SeedSlots, phage ports, safe compaction) |
| Mid term  | Submodule morphogenesis (heads, channels, neuron groups, width multipliers), zone-level control    | Submodule surgery (head/channel pruning, low-rank swaps, merges, quantisation), evidence-led | “Micro-surgery beats amputation”: smaller blast radius than module deletion              |
| Far term  | Parameter-level placement and rewiring (dynamic sparse growth), mostly research mode               | Parameter-level sedation and deletion (saliency ranked), plus compaction pipelines           | Hardware-aware sparsity or it is theatre                                                 |

---

## 3) Near term: make the ecology stable and legible

### Tamiyo v1: module-based morphogenesis (what you largely have now)

**Capabilities**

* Choose slot, blueprint, and a development protocol (incubator → blending → commit).
* Use scaffolding motifs: train-behind, blend-in then blend-out, staged replacement.
* Learn placement and ordering under Simic pricing (rent/churn), with Phase 1 teacher labels (Shapley / full counterfactual harness).

**Deliverables**

* Stable behavioural competence: improves the Pareto frontier (performance vs cost) across many seeds, not just best-run.
* Developmental motifs become repeatable and measurable (style curves, tempo, hold time, swap patterns).

**Telemetry you need (non-negotiable)**

* For each action: sampled action, resolved config actually applied (blend algorithm, curve, tempo), and resulting state transition.
* Per-head logits, masks, probabilities, entropy, logprob (so you can prove a head is real and not dead).
* Slot-local stats: activation moments, gradient norms, gradient alignment at the slot boundary, gate stability.

---

### Emrakul v0: module-level probe-and-lysis plus scar

**Capabilities**

* Operate only on committed phages (plus baseline infrastructure phages).
* Do reversible sedation probes, maintain an immune ledger (debt, antibodies, criticality lock).
* Schedule physical lysis only at safe boundaries.
* On lysis, collapse the wrapper into a **ScarSlot** (identity SeedSlot with metadata plus quarantine) that Tamiyo may regrow into.

**Coordination mechanism (no direct policy chat needed)**

* Either enrich slot state with:

  * `slot_kind = SCAR`, `scar_age`, `last_removed_blueprint`, `probe_outcome_summary`, `quarantine_remaining`
* Or add a fixed-size dropshute event buffer (8–16 events) as extra observation tokens.

**Success criteria**

* Emrakul can delete some modules without quality loss (after probation), and does not oscillate.
* ScarSlots do not proliferate uncontrollably (expiry timers, rent ramps, quarantine).

**Estimative probability:** very likely feasible and high value. This fits determinism and boundary contracts.

---

## 4) Mid term: submodule control and evidence-led efficiency

This is where both agents operate on structured subtargets, not whole modules.

### Tamiyo v2: submodule morphogenesis

**What she controls**

* Attention: head count, head selection, KV dims, projection width.
* Conv: channel groups, kernel families, depthwise plus pointwise staging.
* MLP: neuron-group growth, width multipliers, gated branches.
* Blueprint specialisation: seeds become containers that can spawn internal substructure under a contract.

**Policy shape**

* Hierarchical or token-based policy (transformer over slot tokens, possibly with inbox tokens for recent events).
* Curriculum training: start with a limited action set, then expand submodule levers once stable.

**Telemetry upgrades**

* Define a subtarget inventory per phage/seed: list of deletable or growable units with stable IDs (heads, channels, groups).
* Boundary gradients: dL/d(mask) or dL/d(alpha) style signals for marginal utility estimates without full counterfactuals.

---

### Emrakul v1: submodule surgery and maintenance

**What she controls**

* Micro-surgery: head and channel sedation/lysis, neuron-group pruning.
* Maintenance: low-rank factorisation swaps, branch merges (internal distil), quantisation of stable phages.
* Evidence engine: sedation probes become the primary data source, optional short ablation sweeps remain diagnostics.

**Why this matters**

* Smaller blast radius means she can act more often and learn faster.
* It also gives hardware-aligned efficiency wins (structured pruning beats random sparsity).

**Success criteria**

* Shift from delete whole module to shave cost while preserving function.
* Learn the trade: growth now, prune later, without requiring expensive counterfactual loops.

**Estimative probability:** likely feasible. Main risks are action-space explosion and the need for clean, persistent subtarget IDs and compaction boundaries.

---

## 5) Far term: parameter-level morphogenesis and surgery (research frontier)

This can work, but it will try to eat your time budget.

### Tamiyo v3: parameter-level placement and rewiring

**What “placing individual parameters” means in practice**

* Not literal arbitrary single-weight edits in the live graph.
* More realistic: dynamic sparse training and regrowth rules (block-sparse masks, RigL-style regrowth, learned sparse patterns), or per-parameter gating inside a guild that owns the tensor.

**Guardrails**

* Restrict to hardware-friendly sparsity (block, N:M, structured) or you will get slow sparse matmuls that only look good in spreadsheets.
* Maintain determinism and compaction protocols.

**Success criteria**

* At matched accuracy, deployed model is smaller or faster than mid-term structured approaches.
* Patterns generalise across runs and do not require constant counterfactual supervision.

---

### Emrakul v2: parameter-level sedation and deletion

**What she actually does**

* Rank parameters or blocks by saliency proxies (first-order Taylor, Fisher-ish approximations, learned critics).
* Apply reversible sedation masks, accumulate evidence, then compact at safe boundaries.
* Treat module-level lysis as a last resort.

**Infrastructure prerequisites**

* A real compaction pipeline (tensor resize, weight packing, codegen or kernel selection).
* Per-parameter work must be measurable in step-time, not just “parameters saved”.

**Estimative probability:** roughly even odds this beats staying structured, unless deployment strongly rewards sparsity and you can exploit real hardware support.

---

## 6) Cross-cutting milestones (unlock each step)

1. **Telemetry integrity (near-term prerequisite)**
   If you cannot prove “sampled action → resolved config → executed behaviour”, you will chase ghosts.

2. **Stable identifiers for everything (mid-term prerequisite)**
   Slots, phages, and subtargets need persistent IDs so ledgers, probes, and flight recorder joins stay correct.

3. **Compaction and safe-boundary discipline (mid and far prerequisite)**
   If physical rewrites happen outside controlled boundaries, determinism and debugging die.

4. **Pareto evaluation as the primary scoreboard**
   Accuracy alone rewards bloat and hides local optima. Track frontier movement and reliability (median/q25), not just champions.

5. **Anti-oscillation primitives**
   Quarantine, cooldown, criticality locks, age-based rent ramps. Otherwise multi-agent behaviour becomes churn theatre.

---

## 7) Demo-stage roadmap (storyboarded milestones)

These stages translate the capability horizons into demo-grade exits. Rough mapping:

* Stages 0–3: near term
* Stage 4: mid term
* Stage 5: far term (optional)

### Stage 0: demo foundation

**Goal:** you can trust what you are seeing.

**Must-haves**

* Deterministic replay works often enough to debug (same run, same decisions, same outcomes within tight tolerances).
* Telemetry is end-to-end correct: sampled action → resolved config → executed behaviour.
* Flight recorder can reconstruct an episode at the “why did this happen” level.

**Exit criteria**

* You can take a suspicious screenshot, jump to the exact event, and prove whether it was a policy choice, a mask bug, or an env wiring issue.

---

### Stage 1: Tamiyo demo (module-level growth)

**Goal:** Tamiyo can grow useful structure safely and repeatably.

**Capabilities**

* Seeds are module-level blueprints in fixed SeedSlots.
* Incubator plus blending prevents destabilisation.
* Tamiyo reliably improves task performance under rent/churn across many runs (not just cherry-picked).

**Exit criteria (demo-grade, not SOTA)**

* Across a fixed benchmark setup, median performance improves vs baseline under matched or better cost profile.
* Behavioural sanity: no endless oscillation, no NOOP collapse, no dead action heads.

---

### Stage 2: Emrakul demo (module-level decay plus ScarSlot)

**Goal:** Emrakul can remove redundant modules without wrecking the run, and it is auditable.

**Capabilities**

* Operates on committed phages only.
* Uses reversible sedation probes plus immune ledger (debt, antibodies, criticality lock).
* Schedules physical lysis only at safe boundaries.
* On lysis, leaves a ScarSlot (identity regrowth surface) with quarantine metadata.

**Exit criteria**

* Emrakul performs N successful prunes (module removals) in live training where:

  * loss does not permanently spike
  * criticality locks prevent repeated bad probes
  * the system ends up cheaper (rent proxy down) without losing achieved accuracy

---

### Stage 3: handoff demo (Emrakul cuts, Tamiyo heals)

**Goal:** demonstrate the “trauma surgeon” loop end-to-end.

**Capabilities**

* When Emrakul lyses a module, Tamiyo notices the ScarSlot and can:

  * leave it dormant (identity is sufficient), or
  * regrow a replacement seed inside it and stabilise it

**Exit criteria**

* At least one repeatable scenario where:

  * a module is removed
  * a scar remains
  * the run stays stable
  * Tamiyo regrows capacity at that site and recovers the lost metric (or proves the module was not needed)

This is the first genuinely compelling “ecology” demo.

---

### Stage 4: submodule work (only if Stage 3 is boringly stable)

**Goal:** finer granularity without adding chaos.

**Tamiyo**

* Adjust submodule structure through pre-defined knobs (head count, channel groups, width multipliers).

**Emrakul**

* Submodule surgery (heads/channels/neuron groups) with the same probe-and-ledger safety.

**Exit criteria**

* Demonstrable reduction in cost at the same performance, with better stability than module-level amputation.

---

### Stage 5: parameter-level work (optional, likely unnecessary for a demo)

Only pursue if you need it for a specific narrative or deployment constraint. It explodes engineering scope and kernel complexity.

---

## 8) The meta-rule for a tech demo (applies to every stage)

Each stage needs:

1. A storyboarded scenario (for example: underfit host → Tamiyo grows; redundant phage → Emrakul trims; cut → scar → regrow).
2. A single page of metrics (median/q25, cost proxy, churn, failure counts).
3. A forensic trace you can replay to explain one run.

---

## 9) Glossary (compact)

* **SeedSlot**: a fixed insertion point where growth can occur under a strict contract.
* **Seed**: a blueprint instance placed into a SeedSlot (module-level in near term).
* **Phage**: a committed module wrapper (a unit of structure subject to maintenance or removal).
* **Sedation**: reversible masking or dampening for probes.
* **Lysis**: physical removal at safe boundaries.
* **ScarSlot**: an identity SeedSlot left after lysis, with metadata and quarantine, used as a regrowth surface.
* **Rent/churn (Simic pricing)**: cost pressure and penalties that shape when to keep, replace, or remove structure.
* **Safe boundary / compaction boundary**: the only allowed places to physically rewrite topology or pack tensors.

---
