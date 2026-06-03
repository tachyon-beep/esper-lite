# Leyline Subsystem

**Role:** Shared Enums, Contracts, and Schemas
**Location:** `src/esper/leyline/`

## Overview
Leyline acts as the central source of truth for types, ordering invariants, and communication protocols across all other subsystems. Because Esper is highly decoupled, Leyline prevents integration bugs by defining strict boundaries.

## Key Components

*   **Protocols (`*_protocol.py`):** Abstract base classes and interfaces defining how subsystems communicate (e.g., `host_protocol.py`, `policy_protocol.py`, `governor_protocol.py`).
*   **Schemas & Types (`schemas.py`, `types.py`):** Pydantic models or strictly typed dataclasses for data passing.
*   **Enums (`actions.py`, `stages.py`):** Definitions for discrete states (e.g., the seed lifecycle stages) and actions (e.g., GERMINATE, WAIT).
*   **Configurations (`*_config.py`):** Shared configuration structures (e.g., `reward_config.py`, `slot_config.py`).

## Responsibilities
*   **Type Safety:** Ensuring that data passed between Kasmina, Tamiyo, and Simic is correctly formatted.
*   **Contract Enforcement:** Defining the expected behavior and APIs of each decoupled module.
*   **Single Point of Definition:** Preventing duplication of core concepts like action spaces or observation shapes.
