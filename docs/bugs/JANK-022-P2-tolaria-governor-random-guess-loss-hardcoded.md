# JANK Template

- **Title:** Governor random_guess_loss defaults to CIFAR-10, not task-aware
- **Category:** correctness-risk / configurability
- **Symptoms:** TolariaGovernor defaults `random_guess_loss` to ln(10) unless explicitly set. For other tasks (e.g., TinyStories, other class counts), lobotomy detection tolerance and panic triggers are mismatched, reducing effectiveness.
- **Impact:** Medium – governor may miss or falsely trigger lobotomy detection outside CIFAR-10; safety guarantees weaken on non-CIFAR tasks.
- **Triggers:** Running governor on non-CIFAR tasks without setting random_guess_loss via config.
- **Root-Cause Hypothesis:** Default tuned for CIFAR; task-specific override not wired through task specs/config.
- **Remediation Options:**
  - A) Thread task-specific random_guess_loss (e.g., ln(num_classes) or LM baseline) from TaskSpec into governor init.
  - B) Add a helper to compute from task_type/topology; document defaults.
  - C) Emit telemetry warning when using default and task != cifar10.
- **Validation Plan:** Test governor init for TinyStories sets random_guess_loss appropriately; ensure lobotomy detection logic uses task-aware values.
- **Status:** Open
- **Links:** `src/esper/tolaria/governor.py` (random_guess_loss default), TaskSpec wiring
