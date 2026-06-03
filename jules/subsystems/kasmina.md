# Kasmina Subsystem

**Role:** Host Network + SeedSlots + Lifecycle Mechanics
**Location:** `src/esper/kasmina/`

## Overview
Kasmina is the subsystem responsible for managing the physical structure of the neural network. It contains the logic for the "host" network and the predefined "slots" where "seeds" (new structural modules) can be attached.

It handles the actual execution of the morphogenetic changes requested by the Tamiyo policy, ensuring strict adherence to the seed lifecycle contracts.

## Key Components

*   **Host (`host.py`):** The primary neural network being trained and modified.
*   **SeedSlots (`slot.py`):** Specific hook points in the host's computation graph where seeds can be inserted.
*   **Isolation & Blending (`isolation.py`, `blending.py`, `blend_ops.py`):** Mechanisms to ensure that newly germinated seeds can train on task gradients without immediately disrupting the host's output. They manage the `alpha` ramp, smoothly integrating the seed's output.
*   **Alpha Controller (`alpha_controller.py`):** Controls the schedules for blending.
*   **Blueprints (`blueprints/`):** Definitions for what types of seeds can be created and inserted into slots.

## Responsibilities
*   **State Machine Enforcement:** Validates and executes state transitions (e.g., GERMINATE, FOSSILISE) as dictated by the lifecycle.
*   **Topology Mutation:** Physically adding or removing PyTorch modules based on commands.
*   **Gradient Flow Control:** Ensuring safe gradient propagation, particularly during the isolated `TRAINING` and `BLENDING` phases.
