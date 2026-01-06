## Narset Design Note

### Budget allocator for scalable Tamiyo and Emrakul coordination

**Author role:** Esper Research Architect
**Status:** Design note (planning, no code)
**Purpose:** Capture the Narset concept as a stable, scalable control-layer design that coordinates multiple Tamiyo and Emrakul pairs using coarse telemetry signals, with safety and stability as first-class goals.

---

## 1. Executive summary

Narset is a slow-timescale coordinator that allocates “energy” (budget, authority, and bandwidth) across many Tamiyo and Emrakul pairs. Each pair manages a local region of a model (a Kasmina cell) using fine-grained module/submodule actions. Narset does not see enough detail to make per-module decisions. Instead, she receives a coarse “vibe” signal (progress, stability, efficiency) from each region and converts it into budget allocations that shift the system between two broad modes:

* **Repair/exploration mode:** more energy to Tamiyo (growth and restructuring)
* **Consolidation/efficiency mode:** more energy to Emrakul (probe, sedation, lysis/compaction scheduling)

Narset’s core value is stability: she prevents the ecology from becoming an “ant farm” of simultaneous growth and decay thrash. She enforces smooth, auditable, deterministic budget shifts and makes extreme interventions possible but progressively expensive.

---

## 2. Role boundaries and why Narset is “budget-only”

Narset is intentionally constrained. She does not choose slots, blueprints, alpha targets, or subtargets. She does not run probes. She does not interpret lifecycle semantics beyond coarse outcomes. She acts only by allocating budget to local agents.

**Narset is:**

* A budget allocator and attention director
* A slow-timescale controller (acts at safe boundaries: batch/episode/epoch windows)
* A determinism-friendly coordinator

**Narset is not:**

* Another growth or pruning agent
* A backdoor path for “Simic does everything” at a higher level
* A module-level decision maker

This separation preserves legibility and prevents the system from regressing into a god-object architecture.

---

## 3. Inputs: what Narset should sense (coarse but sufficient)

Narset should operate on a small set of hard-to-spoof aggregate signals per region. The goal is to be robust against noise and against agents learning to trigger budget flips cheaply.

Narset’s inputs per region should be smoothed (EWMA) and windowed to avoid flapping:

### A. Progress (“are we improving?”)

* Rolling delta in accuracy or loss (trend, not instantaneous)
* Plateau indicators (no improvement over N windows)

### B. Stability (“are we safe?”)

* Rollback rate (TolariaGovernor events)
* Anomaly rate (ratio explosions/collapses, value collapse, NaN/Inf)
* LSTM health spikes (if relevant)
* Persistent rather than single-step spikes

### C. Efficiency pressure (“are we bloated or thrashing?”)

* Growth ratio / rent proxy
* Churn proxy (edits per window, alpha shock)
* Seeds active vs fossilised distribution (coarse summary)

Narset does not need to understand which slot or blueprint caused instability. That’s local. Narset only needs to decide “this region needs repair” vs “this region is stable enough to optimise”.

---

## 4. Outputs: what Narset controls

Narset allocates an **intensity** to Tamiyo and Emrakul per region (or per pair). Intensity represents how much bandwidth/authority the local agent receives over a window.

Examples of what intensity can modulate:

* action frequency limits
* allowed blast radius (probe-only vs sedation vs irreversible lysis)
* cost multipliers / thresholds local agents use
* how aggressively a local agent is permitted to act (eg max ops per window)

This is not micromanagement. It is budget shaping.

---

## 5. Budget mechanics: from fixed-sum to “convex overdrive”

### 5.1 Original concept: conserved energy (fixed sum)

Initial design: (e_T + e_E = 1)

Benefit: prevents both growth and decay being “excited” simultaneously, reducing thrash.

Risk: in emergencies, you may need to push one agent very hard while still allowing the other to do minimal probing/maintenance. Fixed-sum makes that impossible.

### 5.2 Improved concept: soft budget with convex costs

Narset has a total pool larger than 1.0 (e.g. **1.5**). Intensities can exceed 1.0 (enter the red zone), but doing so costs more than linear.

Let Narset choose intensities (x_T, x_E) where:

* (x=1.0) is nominal full operating level
* (x>1.0) is overdrive

Define a convex cost function (c(x)) such that:

* For normal range: (c(x)=x) when (x \le 1)
* For overdrive: (c(x)=1 + \alpha(x-1)^2) when (x > 1)

Constraint per region (or per global pool if allocating across regions):

[
c(x_T) + c(x_E) \le 1.5
]

Interpretation:

* Narset can push Tamiyo into overdrive, but it burns disproportionate budget.
* Often, pushing Tamiyo to **1.2** is “almost as good” as 1.5, and leaves budget for Emrakul to keep a minimal probe/ledger heartbeat.
* Overdrive becomes a deliberate emergency tool, not a default posture.

This is DRL-friendly because it creates a smooth trade-off surface rather than brittle “if/then” rules.

---

## 6. Stability: smoothing, inertia, and hysteresis

The primary engineering hazard is oscillation: Narset flapping budgets in response to noisy metrics and causing churn.

Narset must behave like a slow controller:

### A. Smoothing

* Inputs are EWMA’d over a window
* Use trend measures rather than instantaneous spikes

### B. Inertia on outputs

Budgets should not jump sharply. Apply an update rule like:

[
x_T \leftarrow (1-\beta)x_T + \beta x_{T,new}
]

Same for (x_E). This makes Narset resistant to short-term noise and harder to exploit.

### C. Hysteresis in mode shifts

Prefer “sticky” transitions:

* require sustained evidence to switch from consolidation to repair mode
* require sustained stability to switch back

This can be as simple as different thresholds for entering vs leaving “repair-heavy” mode.

---

## 7. Blast radius control: probe-only vs irreversible actions

A crucial safety rule that is a “physics constraint”, not a strategic hard-code:

* **Irreversible actions require authority.**
* Low Emrakul intensity should allow **probe/sedation only**, not physical lysis.
* Physical lysis and compaction scheduling require higher allocated intensity and safe boundaries.

This prevents dangerous edge cases where Narset starves Emrakul completely (losing all maintenance) or where Emrakul amputates during unstable phases.

This also aligns with the ecology metaphor: you can always “poke” with low energy, but you don’t do surgery without anaesthesia and staff.

---

## 8. “No Emrakuls” as an emergent regime, not a rule

Hard-coded rules like “if unstable then e_E = 0” are brittle and exploitable. Instead:

* let the convex budget + stability signals naturally reduce Emrakul intensity during high instability
* keep the possibility of minimal probing
* treat “near zero Emrakul” as an emergent outcome during genuine crises, not a hard switch

If a minimum maintenance heartbeat is desired, it should be implemented as a change in blast radius (probe-only) rather than a strict floor.

---

## 9. Determinism and auditability requirements

Narset is part of the substrate control plane and must preserve determinism and explainability.

Narset should emit telemetry for every allocation decision:

* (x_T, x_E)
* (c(x_T), c(x_E)) and total spent
* overdrive flags (whether either exceeded 1.0)
* the smoothed input signals that drove the decision (progress, stability, efficiency)
* region identifiers (which Tamiyo/Emrakul pair)

This ensures:

* allocation decisions are replayable
* “Narset got gaslit” is diagnosable
* budget dynamics can be tuned without guessing

---

## 10. Summary: what Narset enables

Narset provides the missing scaling primitive:

* Many local ecosystems (Kasmina cells) can run in parallel
* Each cell has a Tamiyo/Emrakul pair doing fine-grained work
* Narset allocates budget using coarse signals, keeping the global system stable
* Overdrive is possible but expensive, encouraging balanced operation
* Maintenance can be throttled without being forbidden, avoiding brittle rules

This design turns “lots of morphogenesis” from chaos into a controllable, auditable ecology, and creates a clean path to scaling from dozens of slots to thousands across a mesh of local cells.
