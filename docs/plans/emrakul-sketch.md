# High-Level Design: The Esper Immune System (Phase 4)

**Status:** Draft Specification (Concept Locked)
**Date:** 2025-12-31
**Context:** Following the success of the "Seed" (Tamiyo/Growth), we introduce its counterpart: the "Phage" (Emrakul/Death). This design defines the immune subsystem responsible for removing obsolete host structure after successful takeovers.

## 1. Purpose and Boundaries

**Goal:** Create an immune subsystem that autonomously audits and removes dead code ("cruft") from the model topology, maintaining efficiency without destabilizing training.

**Non-Goals:**

* **No new seed logic:** Kasmina remains the sole operator of seed lifecycle.
* **No runtime graph surgery:** No `del module`, no FX graph rewrites, no dynamic layer removal during the training loop.
* **No per-op hooks:** Gating happens strictly at the segment/block level to preserve `torch.compile` compatibility.

---

## 2. The Core Concept: Metabolic Equilibrium

The system is defined by two opposing forces that maintain equilibrium, regulated by a central resource allocator.

| Entity | **The Seed (Kasmina/Tamiyo)** | **The Phage (Emrakul)** |
| --- | --- | --- |
| **Biological Role** | Stem Cell / Spore | Bacteriophage / Lysosome |
| **Primary Action** | **Grafting** (Integration) | **Lysis** (Dissolution) |
| **Alpha Trajectory** |  (Fade In) |  (Fade Out) |
| **Success Condition** | High Utility (Keep) | Zero Utility (Delete) |
| **Failure Condition** | Low Utility (Prune the Seed) | High Utility (Restore the Host) |
| **Payload** | New Weights (Parameters) | **Sensors & Gates** (Telemetry) |

---

## 3. The Subsystems

### 3.1 Narset (The Endocrine System)

* **Role:** Strategic Resource Allocator.
* **Input:** Global System Health (from Nissa). Specifically: `Loss Stability`, `Gradient Norm Volatility`.
* **Output:** The **Energy Budget** (scalar vector `[Growth_Budget, Hygiene_Budget]`).
* **Policy:**
* **High Instability (Stress):** "We are dying."  Prioritize growth control (Tamiyo). Zero hygiene budget (Emrakul disabled).
* **High Stability (Peace):** "We are stagnant."  Allocate hygiene budget to Emrakul. Authorized to audit and wither.

### 3.2 Emrakul (The Removal Controller)

* **Role:** The Policy Agent for Deletion. She is the "Anti-Tamiyo."
* **Domain:** She never touches seeds directly. She sees **Phages**.
* **Action Space:**
* **Wither:** Decrease Phage Alpha (fade out host block).
* **Restore:** Increase Phage Alpha (abort audit).
* **Necrosis:** Lock alpha at 0.0 and mark segment as disabled.

* **Policy:** Conservative. Unlike Tamiyo (who rewards risk), Emrakul is rewarded for **Traffic Reduction** but heavily penalized for **Loss Spikes**.

### 3.3 Phage (The Tactical Interface)

* **Role:** The Viral Vector / Mechanism of Action.
* **Implementation:** A `PhageWrapper` module that latches onto existing Host Blocks.
* **Mechanism:** The **Bypassed Gate** (Soft Pruning).

* *Where `Bypass` is `Identity()` or a `1x1 Conv` for shape matching.*

* **Safety:** This guarantees topological validity. Even if Emrakul aggressively deletes a vital organ, the signal routes around it (ResNet style) rather than crashing to zero.

### 3.4 Lysis (The Terminal State)

* **Role:** The Equivalent of **Fossilization**.
* **Trigger:** When `Phage.alpha == 0.0` AND `Stability == High`.
* **Effect:**

1. **Functional Lysis (Runtime):** The `HostBlock` parameters are frozen and gradient-detached. They effectively cease to exist in the backward pass.
2. **Physical Lysis (Offline):** During the next offline save/load cycle, the architecture definition is physically rewritten to remove the `HostBlock`, leaving only the `Bypass`.

---

## 4. The Lifecycle: "Infection & Digestion"

This architecture introduces a secondary lifecycle running parallel to the Seed lifecycle.

### Step 1: Infection (The Marker)

When Tamiyo successfully **Fossilizes** a Seed (integrating new knowledge), the system flags the *original* host block at that location as "Suspect."

* **Action:** A **Phage** is attached to the original Host Block.
* **State:** `DORMANT` (). The block operates normally, but under observation.

### Step 2: The Audit (Narset's Call)

Narset detects a period of high stability and positive hygiene budget.

* **Action:** Emrakul wakes up. She observes that the Phage's host block has low gradient flow (because the Fossilized Seed is doing the work).
* **Command:** `WITHER` ().

### Step 3: The Stress Test

The system runs with the Host Block at 50% capacity. Emrakul measures the **Churn** (Loss Delta).

* **Scenario A (Redundancy):** Loss remains stable. The Seed has truly replaced the Host.
* *Result:* Emrakul continues `WITHER` ().

* **Scenario B (Dependency):** Loss spikes. The Host Block was doing something the Seed missed.
* *Result:* Emrakul executes `DETACH` (). The Phage goes dormant; the block is marked "Essential."

### Step 4: Necrosis (Reclamation)

If  hits  and stability holds, Emrakul triggers **Necrosis**.

* The Host Block is chemically dissolved (weights frozen/zeroed).
* The Phage remains as a permanent scar (Bypass).
* **Outcome:** The model size shrinks back down, freeing up `param_budget` for Tamiyo to plant new seeds.

---

## 5. Host Gating Surface (Implementation)

Emrakul requires a minimal, compile-friendly interface. We do not wrap every op.

**Design Constraints:**

* **Granularity:** One gate per segment (O(n_blocks)), not per layer.
* **Explicit Graph:** Gating must be explicit in the forward pass (no hooks).
* **Ownership:** `HostGateSpec` lives in `leyline`.

**Minimal Interface (Conceptual):**

```python
interface HostHygieneProtocol:
    def gate_specs() -> list[HostGateSpec]
    def set_gate_alpha(gate_id: str, alpha: float) -> None
    def get_gate_alpha(gate_id: str) -> float

```

**Attachment Strategy:**

* **CNN Host:** Wrap each residual block output.
* **Transformer Host:** Wrap the MLP block and Attention output projection.

---

## 6. Safety Guardrails & Validation

**Guardrails:**

1. **Turbulence Lock:** Do not audit segments when `loss_variance` is high.
2. **Conflict Prevention:** Never delete a segment that is currently hosting an active (non-fossilized) Seed.
3. **Rate Limiting:** Cap the number of simultaneous Necrosis events per window to prevent topological collapse.

**Validation Plan:**

* **A/B Testing:** Compare runs with Emrakul OFF vs. ON. Metric: `detritus_ratio` vs. `throughput`.
* **Audit Correctness:** Verify that alpha decay does not trigger loss spikes in known-redundant blocks.
* **Hygiene Conservatism:** Verify zero necrosis events during unstable training phases.
