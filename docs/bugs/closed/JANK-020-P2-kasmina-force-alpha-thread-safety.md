# JANK Template

- **Title:** `force_alpha` remains non-thread-safe and non-DDP-safe despite TODO
- **Category:** correctness-risk / concurrency
- **Symptoms:** `SeedSlot.force_alpha` mutates `state.alpha` and `alpha_schedule` without guards; docstring warns “NOT THREAD-SAFE” and “Nested calls NOT supported,” and a TODO references DDP support. Counterfactuals in vectorized validation rely on this, so concurrent use across streams/threads/ranks can produce inconsistent alpha or schedule state.
- **Impact:** Medium – risk of race conditions and divergent state under multi-env/stream or future DDP use; could corrupt alpha for in-flight forwards, especially with torch.compile graph specialization.
- **Triggers:** Concurrent forwards with force_alpha (e.g., multiple envs, DataLoader workers, future DDP).
- **Root-Cause Hypothesis:** Context manager mutates shared state; no synchronization or per-call isolation; TODO left unresolved.
- **Remediation Options:** 
  - A) Make force_alpha functional (take alpha override as arg to forward) instead of mutating slot state.
  - B) Add per-call lock or assert single-threaded eval; guard against DDP usage.
  - C) Provide a per-call clone/copy of seed slot for counterfactual passes to avoid shared mutation.
- **Validation Plan:** Add tests to ensure force_alpha doesn’t leak state across concurrent calls; guard against DDP usage; ensure counterfactual runs remain correct.
- **Status:** Closed (Not a bug under current architecture)
- **Resolution:** `force_alpha` is explicitly documented as not thread/DDP-safe, and current production usage is safe because counterfactual evaluation runs single-threaded per model instance (vectorized training uses one model per env). Future DDP-safe counterfactuals are tracked by the in-code TODO; this ticket is redundant until DDP is actually implemented.
- **Links:** `src/esper/kasmina/slot.py` (`force_alpha` + DDP TODO), `src/esper/simic/training/vectorized.py` (counterfactual evaluation)
