# JANK Template

- **Title:** SignalTracker history_window hardcoded to 10 with no task-aware defaults
- **Category:** ergonomics / correctness-risk
- **Symptoms:** SignalTracker uses a fixed `history_window=10` unless manually set. Tasks with shorter episodes or different timescales (e.g., TinyStories vs CIFAR) might need different windows to detect plateau/stabilization accurately. There’s no task-aware defaulting or config plumbing.
- **Impact:** Lower but real—plateau detection and best_val_loss tracking may be noisy or unresponsive depending on task; users must know to override manually.
- **Triggers:** Running Tamiyo on tasks with different epoch counts or loss dynamics; using default tracker config.
- **Root-Cause Hypothesis:** Simplicity; task-specific tuning deferred.
- **Remediation Options:**
  - A) Thread task config into SignalTracker (e.g., via TaskSpec) to set history_window per topology/task.
  - B) Expose tracker config on CLI/config to avoid hidden defaults; document guidance.
  - C) Add adaptive windowing based on max_epochs or plateau detection needs.
- **Validation Plan:** Tests ensuring tracker picks task-appropriate window when provided; documentation update; optional adaptive unit test.
- **Status:** Open
- **Links:** `src/esper/tamiyo/tracker.py` (history_window default), `src/esper/runtime/tasks.py` (task configs)
