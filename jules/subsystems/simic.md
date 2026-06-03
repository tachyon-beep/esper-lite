# Simic Subsystem

**Role:** Selection Pressure, PPO Loop, and Accounting
**Location:** `src/esper/simic/`

## Overview
Simic is the environment and reward mechanism that trains the Tamiyo policy. It defines the "selection pressure" that dictates whether a seed is successful or should be pruned. It houses the PPO (Proximal Policy Optimization) algorithm.

## Key Components

*   **Training & Agent (`training/`, `agent/`):** The RL training loop (PPO) that updates Tamiyo's weights based on collected trajectories.
*   **Rewards (`rewards/`):** The logic that translates host performance into RL rewards. It supports various modes like "shaped", "sparse", and "basic_plus" (which includes the Drip Mechanism).
*   **Attribution (`attribution/`):** Calculates how much a specific seed contributed to the host's overall performance (counterfactual contribution vs. direct loss delta).
*   **Control & Telemetry (`control/`, `telemetry/`):** Manages the environment interactions and emits metrics related to the RL process.

## Responsibilities
*   **The Economy:** Calculates rent (cost of parameters) and contribution signals for seeds.
*   **Reward Signal:** Provides the temporal credit assignment to Tamiyo (e.g., did germinating that seed 50 steps ago actually improve accuracy?).
*   **Optimization:** Runs the PPO updates using the collected batches of episodes.
