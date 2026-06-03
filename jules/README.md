# Esper Knowledge Base

Welcome to the Esper Knowledge Base. This directory serves as a comprehensive guide to understanding, mapping, and navigating the **Esper** framework.

**Esper** is a framework for **morphogenetic AI**, enabling neural networks to grow, prune, and adapt their topology during training. Instead of a static architecture, Esper uses a lifecycle-driven approach to germinate "seeds", train them safely alongside the host network, and blend them into the host.

## Table of Contents

### High-Level Concepts
* **[Architecture Overview](architecture.md):** The core control flow, data flow, nested training loops, and the seed lifecycle state machine.
* **[Terminology](terminology.md):** A glossary of terms, especially clarifying RL domain vs. neural network domain overlaps.
* **[Development Guide](development_guide.md):** Common commands, testing, PPO training flags, and system configuration.

### Subsystems Deep Dives
Esper is composed of highly decoupled subsystems. Detailed documentation on each can be found here:

* **[Kasmina](subsystems/kasmina.md):** Host network, SeedSlots, and the lifecycle mechanics.
* **[Tamiyo](subsystems/tamiyo.md):** Growth policy (heuristic or learned) managing seed actions.
* **[Simic](subsystems/simic.md):** Selection pressure, PPO loop, reward logic, and accounting.
* **[Tolaria](subsystems/tolaria.md):** Execution engine governing deterministic training and safety rollbacks.
* **[Nissa](subsystems/nissa.md):** Telemetry backends.
* **[Karn](subsystems/karn.md):** Operator UI, Sanctum TUI, Overwatch dashboard, and analytics.
* **[Leyline](subsystems/leyline.md):** Shared contracts, schemas, enums, and types.

---

*This knowledge base maps out the codebase as it stands and is intended to help contributors quickly grasp Esper's concepts and subsystem responsibilities.*
