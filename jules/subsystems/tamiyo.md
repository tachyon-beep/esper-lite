# Tamiyo Subsystem

**Role:** Growth Policy and Decision-Maker
**Location:** `src/esper/tamiyo/`

## Overview
Tamiyo represents the "agent" in Esper's RL framework. It is the policy that observes the state of the host (via Kasmina) and decides which lifecycle actions to take for each SeedSlot.

## Key Components

*   **Policy (`policy/`):** The core decision-making logic. It can be a learned neural network (used in PPO) or a set of hardcoded rules.
*   **Networks (`networks/`):** The PyTorch architectures for the learned policies. Currently features a **512-dim feature net + 512 hidden LSTM**, designed for ~150-step horizons.
*   **Decisions & Actions (`decisions.py`, `action_enums.py`):** Translates observations into concrete commands (WAIT, GERMINATE, ADVANCE, PRUNE, FOSSILISE).
*   **Heuristic (`heuristic.py`):** A baseline, rule-based policy used for testing or when learning is not required.
*   **Tracker (`tracker.py`):** Maintains internal state and observations across steps.

## Responsibilities
*   **Observation:** Takes a 128-dim input (116 non-blueprint dims + 12 blueprint embedding dims) representing the host's health, current topology, and seed states.
*   **Action Selection:** Outputs a discrete action for each available slot.
*   **Long-Horizon Planning:** Utilizes recurrent structures (LSTM) to understand the delayed consequences of germinating or pruning a seed over a 150-step episode.
