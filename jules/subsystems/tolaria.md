# Tolaria Subsystem

**Role:** Execution Engine, Determinism, and Safety Rollback
**Location:** `src/esper/tolaria/`

## Overview
Tolaria acts as the high-throughput, deterministic execution substrate. It manages the parallel execution of environments and ensures that operations are safe and reproducible.

## Key Components

*   **Environment (`environment.py`):** Wraps the host training and policy interaction into an RL-compatible environment interface.
*   **Governor (`governor.py`):** The safety mechanism that oversees the execution and handles rollbacks.

## Responsibilities
*   **Vectorized Execution:** Orchestrates parallel environments efficiently, optimizing for GPU usage.
*   **Inverted Control Flow:** Ensures that batches drive environments, not the other way around.
*   **Determinism:** Maintains strict reproducible environments for consistent RL training.
*   **Safety Rollbacks:** Reverts the host to a safe state if a morphogenetic action causes a catastrophic failure (e.g., NaN losses or explosive gradients).
