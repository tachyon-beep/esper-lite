# Phase 1 Final Exam: The Reward Efficiency Protocol

**Status:** DRAFT
**Date:** 2025-12-19
**Objective:** Validate the "Rent & Churn Economy" and determine the optimal reward signal for Phase 3 (Transformers).

---

## 1. Context

We have successfully trained `cifar_blind`, a model that achieves ~60% accuracy on CIFAR-10 with only +10% parameter growth using a heuristic/random strategy. This sets the **Baseline for Competence**.

For the RL agent (`Simic`) to justify its existence, it must outperform this baseline not just in accuracy, but in **structural efficiency** (getting more accuracy *per unit of growth*).

We have observed that the current 7-component `SHAPED` reward might be creating an "unlearnable landscape" due to conflicting signals (e.g., attribution vs. rent).

---

## 2. The Exam Protocol

We will run a concurrent A/B/C test on the CIFAR-10 task to determine the winning reward signal.

### 2.1 The Cohorts

| Cohort | Description | Reward Function | Hypothesis |
| :--- | :--- | :--- | :--- |
| **Control** | `cifar_blind` | Heuristic / Random | **Baseline.** The floor for performance. |
| **A (Shaped)** | Current Default | 7-component (PBRS + Attribution + Warnings + Rent...) | **Over-engineered.** likely to cause confusion/instability. |
| **B (Simplified)** | **The Challenger** | 3-component (PBRS + Intervention Cost + Terminal Bonus) | **Optimal.** Cleanest gradient for temporal credit assignment. |
| **C (Sparse)** | Hard Mode | Terminal Accuracy - Rent | **The Truth.** Hardest to learn, but theoretically perfect alignment. |

### 2.2 Configuration

- **Task:** `cifar_blind` topology (ResNet host + 2 seed slots).
- **Duration:** 100 Episodes.
- **Envs:** 8-12 concurrent environments (split evenly across cohorts).
- **Seed Budget:** Max 2 active seeds.

---

## 3. Success Metrics

We are measuring **Return on Investment (ROI)**, not just profit.

1.  **Accuracy ROI:** $\frac{\text{Final Accuracy} - \text{Baseline Accuracy}}{\text{Added Parameters}}$
    *   *Did we spend our parameter budget wisely?*
2.  **Decision Decisiveness:** Entropy trends.
    *   *Is the agent confident, or flailing?*
3.  **Lifecycle Efficiency:** Ratio of `FOSSILIZED` to `PRUNED` seeds.
    *   *Does the agent kill bad seeds early (high precision)?*

### 3.1 Pass Criteria
*   **Essential:** Cohort B (Simplified) must outperform Cohort A (Shaped) in Accuracy ROI.
*   **Essential:** Cohort B must outperform Control (`cifar_blind`) in Final Accuracy.
*   **Stretch:** Cohort C (Sparse) learns *anything* better than random chance.

---

## 4. Execution Plan

1.  **Implement `SIMPLIFIED` Reward:** Execute `docs/plans/2025-12-18-reward-ab-testing.md`.
2.  **Run True A/B:** Use `--dual-ab` to train separate policies per reward mode.
    *   Run 1: `shaped-vs-simplified`.
    *   Run 2: `simplified-vs-sparse`.
3.  **Analyze with Overwatch:** Use the new TUI to watch entropy collapse and decision quality in real-time.
4.  **Verdict:** Select the winning reward mode as the default for **Phase 3 (TinyStories)**.

---

## 5. Why This Matters for Phase 3

Transformers are expensive. We cannot afford "confused" agents that waste parameters on low-value attention heads. We need a reward signal that is **sharp, clean, and ruthless**. This exam determines that signal.
