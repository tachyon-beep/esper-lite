# Karn Subsystem

**Role:** Operator UI and Analytics
**Location:** `src/esper/karn/`

## Overview
Karn is the user interface and analytics dashboard suite for Esper. It provides visibility into the complex, concurrent operations of the system, acting as a "flight recorder".

## Key Components

*   **Sanctum (`sanctum/`):** A Textual-based Terminal UI (TUI) for real-time debugging and observation directly in the terminal.
*   **Overwatch (`overwatch/`):** A web-based dashboard for deeper analytics and visualization.
*   **Data Handling (`collector.py`, `ingest.py`, `store.py`):** Collects data emitted by Nissa, processes it, and stores it for the UIs to consume.
*   **Pareto (`pareto.py`):** Analytics related to the frontier of quality vs. cost vs. stability.

## Responsibilities
*   **Visibility:** Exposing the internal state of the nested training loops, seed lifecycles, and reward signals in an understandable format.
*   **Aggregation:** Summarizing parallel environment data into coherent operator views.
*   **Real-time Monitoring:** Providing instant feedback during long training runs.
