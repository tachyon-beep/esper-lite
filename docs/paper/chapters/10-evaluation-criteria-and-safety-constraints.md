---
title: EVALUATION CRITERIA AND SAFETY CONSTRAINTS
source: /home/john/esper-lite/docs/paper/draft_paper.md
source_lines: 604-655
split_mode: consolidated
chapter: 10
coauthors:
  - John Morrissey
  - Codex CLI (OpenAI)
generated_by: scripts/split_paper.py
---

# Evaluation Criteria and Safety Constraints
The introduction of seed-based local evolution mechanisms within frozen neural architectures presents novel evaluation challenges. Because the global model remains static, traditional training metrics are insufficient: functional gain must be measured relative to localised intervention, and safety guarantees must be enforced to prevent unintended cross-model effects. This section outlines the formal criteria under which a morphogenetic architecture is to be assessed.
## 10.1 Evaluation Domains
Seed-enabled systems must be evaluated across multiple axes to ensure correctness, stability, and effective governance by the policy controller.

| Domain                  | Goal                                                    | Metrics / Methods                                                  |
|-------------------------|---------------------------------------------------------|--------------------------------------------------------------------|
| Functional Gain         | Validate that a fossilised seed yields measurable gain. | Δ Accuracy, local loss reduction, representation quality           |
| Gradient Isolation      | Ensure no gradient flows into the frozen base network. | Parameter delta checks, backward hook assertions                   |
| Interface Integrity     | Confirm I/O shape and signal consistency at grafts.    | Forward shape checks, activation variance monitoring               |
| Behavioural Stability   | Detect and prevent post-germination drift.             | Output similarity (cosine, JS), regression test suite              |
| Controller Policy Quality | Verify Tamiyo’s decision quality.                    | Policy reward, blueprint selection accuracy, FP/FN trigger rates   |
| Reproducibility         | Ensure deterministic outcomes under repeats.           | Seeded trials, versioned blueprints, checksums in germination logs |
## 10.2 Safety Constraints
To prevent uncontrolled or undesirable behaviour, the following safety constraints are respected during design, training, and deployment.
• 10.2.1 Gradient Containment: No gradient may propagate into the frozen base model. This is enforced via requires_grad = False and backward hooks.
• 10.2.2 Interface Contract Preservation: A germinated module must not alter tensor shapes or distributions in a way that breaks downstream compatibility. This is validated during the SHADOWING phase of the lifecycle.
• 10.2.3 Bounded Germination: Seed growth must be capped. This is enforced by the Tamiyo controller, which is trained on a curriculum with a finite energy budget (ATP) and learns to conserve resources.
• 10.2.4 Deterministic Execution: Given identical inputs and seeds, germination outcomes must be deterministic. This is ensured through rigorous seeding and versioning of all components, including blueprints.
## 10.3 Evaluation Pipeline
A reference evaluation pipeline aligned to the formal seed lifecycle:

| Step | Action                         | Description                                                                                                             |
|-----:|--------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| 1    | Baseline Capture               | Train and freeze the base model; save output signature; archive activation statistics at all seed sites                 |
| 2    | Policy‑Driven Germination      | Deploy Tamiyo; on bottleneck detection, request germination via SeedManager (seed enters GERMINATED queue)              |
| 3    | Lifecycle Progression & Checks | Progress through TRAINING and BLENDING; validate at SHADOWING (internal stability) and PROBATIONARY (systemic impact)  |
| 4    | Post‑Fossilisation Audit       | After successful lifecycle and FOSSILIZED state, compare pre/post accuracy; confirm interface invariants; compute Δ gain |
## 10.4 Failure Modes and Mitigations
The new framework provides specific mitigations for key failure modes.
Failure Mode Cause Mitigation Strategy
Unbounded Parameter Growth A naive or poorly trained Tamiyo controller. The controller is trained on a curriculum with a finite energy budget (ATP), explicitly teaching it resource management.
Functional Drift A germinated module alters the host's representations. The PROBATIONARY lifecycle state acts as a final validation gate, where overall system performance is checked. Failure leads to the seed being CULLED.
Policy-Level Failure Tamiyo triggers germination unnecessarily or chooses a poor blueprint. The controller's RL reward function is shaped to penalize low-value growth and incorrect blueprint selections.
Redundant Adaptation Tamiyo selects a blueprint that replicates an existing function. Karn's discovery process is rewarded for finding diverse blueprints. Tamiyo's policy can be trained with an auxiliary loss to select for functional novelty.
## 10.5 Recommended Auditing Practices
• Maintain a complete, versioned germination log from the SeedManager, detailing all state transitions, blueprints used, and reasons for any culled seeds.
• Periodically re-evaluate the frozen components of the model against an archival test set to provide a stable baseline for drift detection.
• Tag and version each blueprint from Karn's library and each fossilised module for complete lineage tracking.
## 10.6 Hardware Realization and Constraints
The choice of germination strategy has direct hardware implications. The design of blueprints by Karn should be co-developed with a target hardware profile.
| Blueprint Type (from Karn) | Target Hardware | Kernel Strategy                  |
|----------------------------|-----------------|----------------------------------|
| Adapter                    | MCU             | Lookup-table fusion              |
| Germinal Module (GM)       | Edge TPU        | Pre-compiled binaries            |
| Surgical                   | FPGA            | Dynamic partial reconfiguration  |
## 10.7 Adversarial Robustness and Security
The framework's layered defences are resilient to adversarial manipulation.

| Category          | Description                                                                                                                |
|-------------------|----------------------------------------------------------------------------------------------------------------------------|
| Attack Vector     | Inputs crafted to create pathological conditions (e.g., low activation variance) to trick Tamiyo into unnecessary growth    |
| Defence Mechanism | Stateful policy (GRU) conditions on telemetry sequences; robust to single‑step adversarial inputs                           |
| Defence Mechanism | Lifecycle validation rejects manufactured problems during SHADOWING/PROBATIONARY; seed is CULLED and change embargoed      |

### 10.7.1 Threat Model

| Element            | Details                                                                                 |
|--------------------|-----------------------------------------------------------------------------------------|
| Assets             | Base model integrity, interface invariants, blueprint library, germination logs         |
| Trust Assumptions  | Frozen base is trusted; Tamiyo and SeedManager run in trusted runtime with audit logs   |
| Attack Surfaces    | Inputs (crafted sequences), telemetry spoofing, blueprint tampering, log manipulation   |
| Adversary Goals    | Force unnecessary germination; degrade performance; cause drift; exhaust resources     |
| Defender Controls  | Stateful policy (telemetry sequences), two-stage validation (shadowing/probationary),   |
|                    | bounded germination budgets, rollback/culling, signed/versioned blueprints, TLS logs    |
| Detection          | Drift metrics (cosine), anomaly detection on activation/telemetry, policy outlier flags |
| Response           | Automatic culling + embargo; revert to last good; alerting and forensics on logs       |
## 10.8 Long-Term Stability and Cumulative Drift
To simulate a long deployment lifecycle, a ResNet-18 model was subjected to an accelerated aging process over 5,000 training cycles, with the Tamiyo policy controller permitted to trigger up to 20 germination events. The results indicate that the system maintains high stability, with cumulative interface drift and regression on the original core task remaining minimal and well-bounded.
