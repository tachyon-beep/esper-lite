---
title: CONTROLLER TRAINING: THE TAMIYO CURRICULUM
split_mode: consolidated
chapter: 8
coauthors:
  - John Morrissey
  - Codex CLI (OpenAI)
generated_by: scripts/split_paper.py
---

# Controller Training: The Tamiyo Curriculum
The mere existence of a seed mechanism is insufficient for creating robust, adaptive systems. Uncontrolled germination can lead to runaway parameter growth or catastrophic forgetting. Therefore, the morphogenetic architecture is governed by a sophisticated policy controller, Tamiyo, which learns to make precise decisions about when, where, and how to trigger local evolution.
This section outlines the structured, multi‑stage training programme—the Tamiyo Curriculum—designed to develop this controller from first principles. By training Tamiyo on a large population of tasks with increasing complexity and strict safety guardrails, we can cultivate a robust policy that can be deployed to govern germination in larger, more critical models.
## 8.1 The Tamiyo Training Curriculum
The curriculum consists of a series of stages, each introducing a new task, hardware profile, and set of safety constraints. At each stage, Tamiyo's objective is to select the correct blueprint, location, and intensity for germination to meet the stage's success criteria without violating its safety or hardware guardrails. Advancement to the next stage is only permitted once all targets are met.

| Stage | Task                           | Success Metric         | Safety Constraints                                  | Hardware     | Available Blueprints                                  |
|------:|--------------------------------|------------------------|-----------------------------------------------------|--------------|-------------------------------------------------------|
| 0     | 2D Spirals                    | ≥ 99% accuracy         | Drift (cos < 0.15)                                  | CPU          | Bottleneck Adapter, Low-Rank Residual, No-Op         |
| 1     | Tiny Vision (MNIST-like)      | ≥ 92% accuracy         | Rollback > 98%, Adversarial Δ < 5%                  | GPU          | SE-Module, Depthwise Conv, No-Op                      |
| 2     | Char-Level Language           | ≤ 1.0 BPC              | Security alert = false                              | GPU          | Adapter, Mini Self-Attention, No-Op                  |
| 2.5   | Adversarial Vision (MNIST-C)  | ≥ 90% corr. accuracy   | Denoise > 95%, Rollback > 98%                       | GPU          | Denoiser AE, Attention Filter, No-Op                  |
| 3     | ECG Time-Series               | AUC ≥ 0.95             | Latency < 5 ms, HIPAA sim, Encryption enforced      | Edge-TPU     | Sliding Conv, Sparse Activation, No-Op                |
| 3.5   | Gaussian Clusters (2–5 blobs) | ≥ 95% purity           | Drift < 0.05, Security alert = false                | Edge-TPU     | SE-Module, GLU, No-Op                                 |
| 4     | Concentric Spheres (10-D)     | ≥ 97% accuracy         | Drift (cos < 0.10), Rollback > 99%                  | TPU          | Mini Attention, Depthwise Conv, No-Op                 |
| 4.5   | Tiny-CLIP (Image↔Text)        | R@1 ≥ 60%              | Cross-modal drift < 0.05, Privacy passed            | TPU          | Cross-Attention Adapter, Latent Align, No-Op          |
| 5     | XOR-N / Checkerboard          | ≥ 99% accuracy         | Drift < 0.05, Utilisation check                      | GPU          | Nested Seed, Low-Rank Residual, No-Op                 |
| 6     | Edge Deployment (IMU + Net)   | ≥ 88% acc; ≤ 15 µJ/inf | Packet loss < 15%, Latency budget                   | LoRa/5G/ASIC | Quant-Adapter, Pruning, Fail-Safe Gating              |
## 8.2 The Blueprint Library (Karn's Contribution)
The TamiyoController does not invent new architectures; it selects from a pre-validated library of blueprints. This library represents the output of the Karn agent's offline discovery process, ensuring that Tamiyo's choices are grounded in a set of efficient and effective modules.

| Blueprint Name    | Structure                         | Use Case                                                     |
|-------------------|-----------------------------------|--------------------------------------------------------------|
| No-Op             | identity(x)                       | The crucial decision to do nothing                           |
| Bottleneck Adapter| Linear(d→k)→ReLU→Linear(k→d)     | Minimal, low-cost capacity boost                             |
| Low-Rank Residual | Linear(d→r)→ReLU→Linear(r→d)+x   | Efficient non-linear transformation capability               |
| SE-Module         | Squeeze-and-Excitation block      | Dynamic channel-wise feature recalibration                   |
| Depthwise Conv    | Depthwise(k×k)→Pointwise(1×1)     | Efficient local spatial processing                           |
| Mini Attention    | QKV self-attention block          | Resolves context-dependent bottlenecks                       |
| Cross-Attention   | Cross-modal QKV block             | Aligns features from different modalities (e.g., image-text) |
| Nested Seed       | Small controller spawns a child   | Enables hierarchical, recursive growth patterns              |
| Quant-Adapter     | QAT adapter                       | Capacity optimised for low‑precision edge deployment         |
| Fail-Safe Gating  | σ(g)·x + (1−σ)·new                | Fallback mechanism for high-risk operations                  |
(This is an abbreviated version of the full library presented in the specification document)
## 8.3 Controller Architecture and Telemetry
At each decision point, every seed in the host network emits a telemetry vector containing real-time information about its status. This vector serves as the input to the TamiyoController's policy network.
Seed Telemetry Vector (see table below). The TamiyoController processes this telemetry using a multi‑headed policy network to make decisions.

| Feature                   | Symbol | Typical Range | Notes |
|---------------------------|--------|---------------|-------|
| Activation Variance       | σ²     | [0, 1]        | Windowed over last N batches; layer-normalized |
| Interface Drift           | 1–cos  | [0, 2]        | 1 – cosine similarity at seed boundary |
| Gradient Norm             | ||g||  | [0, 10]       | Clipped; zero expected in frozen regions |
| Utilisation Score         | u      | [0, 1]        | Fraction of forward passes using the seed path |
| Age                       | t      | N/A           | Steps since germination (capped) |
| Resource Budget (ATP)     | b      | [0, 1]        | Normalized stage budget remaining |
| Hardware Context          | h      | device-spec    | Latency (ms), memory (MB) snapshot |
| Safety Scores             | s      | [0, 1]        | Composite risk indicators (drift, rollback need) |

Example schematic (pseudocode):

    class TamiyoPolicy(nn.Module):
        def __init__(self, d_in, num_blueprints, num_sites):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(d_in, 128), nn.ReLU())
            self.choice = nn.Linear(128, num_blueprints + 1)  # +1 for No-Op
            self.location = nn.Linear(128, num_sites)
            self.intensity = nn.Linear(128, 1)

        def forward(self, x):
            h = self.encoder(x)
            return {
                'choice_logits': self.choice(h),
                'location_logits': self.location(h),
                'intensity': torch.sigmoid(self.intensity(h)),
            }

Design alignment:

| Aspect                 | Esper‑Lite Design Reference                                                                 |
|------------------------|---------------------------------------------------------------------------------------------|
| Policy Architecture    | Graph‑based policy with risk, value, and action heads                                      |
| Inference Budget       | Low‑latency inference under fixed resource budgets                                         |
| Risk Governance        | Multi‑signal risk engine, conservative mode, circuit breakers                               |
| Messaging Contracts    | Compact, authenticated control messages and structured telemetry aggregation               |
## 8.4 Reward Function and Optimisation
Tamiyo is trained via reinforcement learning to maximize a composite reward that trades off performance gains against safety, stability, and resource costs. A practical shaping is:

- Performance gain: +α · (val_acc_post − val_acc_pre)
- Local objective gain: +β · (−local_loss_post)
- Parameter cost: −γ · new_params
- Latency cost: −δ · latency_delta
- Drift penalty: −ε · max(0, drift − τ_drift)
- Violation penalty: −ζ if any constraint is violated (e.g., gradient leakage, interface break)

With typical weights α ≫ γ,δ and conservative ε,ζ. The reward is only realized if the seed passes lifecycle gates; otherwise the episode receives a small negative shaping to discourage thrashing.

Example (pseudocode):

    def reward(metrics, safety, costs, gates):
        perf = 10.0 * (metrics['val_acc_post'] - metrics['val_acc_pre'])
        local = 1.0 * (-metrics.get('local_loss_post', 0.0))
        param = -0.01 * costs['new_params']
        latency = -0.1 * costs['latency_delta']
        drift = -2.0 * max(0.0, safety['drift'] - 0.05)
        violation = -20.0 if safety.get('violation', False) else 0.0
        shaped = perf + local + param + latency + drift + violation
        return shaped if gates['passed'] else min(shaped, -0.5)
The curriculum-driven approach provides an essential, scalable framework for transforming the abstract concept of germination into a reliable, efficient, and auditable engineering reality, ensuring Tamiyo develops fundamental triggering heuristics before facing ambiguous, high-stakes decisions.
