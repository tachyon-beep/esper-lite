# Nissa Subsystem

**Role:** Telemetry Backends
**Location:** `src/esper/nissa/`

## Overview
Nissa handles the emission, routing, and storage of structured diagnostics and run artifacts. Telemetry in Esper is treated as a strict contract, ensuring reliable and typed data for analysis.

## Key Components

*   **Tracker & Config (`tracker.py`, `config.py`):** Manages what gets tracked and how it's configured.
*   **Analytics (`analytics.py`):** Processes and aggregates raw telemetry data.
*   **Backends (`wandb_backend.py`, `output.py`):** Routes telemetry to various sinks, such as local files or Weights & Biases.

## Responsibilities
*   **Structured Emission:** Emits typed payloads and validates them against schemas defined in `Leyline`.
*   **Artifact Routing:** Saves checkpoints, configuration files, and logs to the designated telemetry directories.
*   **Integration:** Connects Esper runs to external monitoring tools like W&B.
