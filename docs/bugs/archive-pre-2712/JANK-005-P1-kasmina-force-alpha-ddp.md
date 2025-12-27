# JANK Template

- **Title:** `force_alpha` mutates slot state and is not DDP-safe
- **Category:** correctness-risk | maintainability
- **Symptoms:** `SeedSlot.force_alpha` temporarily overwrites `state.alpha` and disables `alpha_schedule`, mutating shared module state. Under DDP or concurrent eval, ranks can diverge (different alpha values and schedules) and torch.compile guards may churn when the context is used mid-graph.
- **Impact:** Risk of cross-rank divergence during counterfactual validation, especially if called on only a subset of ranks. Also increases guard specialization churn for compiled graphs since the schedule toggle flips per context.
- **Triggers:** Using counterfactual evaluation or debugging tools that wrap `force_alpha` while running with DDP/multi-process or while compiled. Even single-process concurrent calls (e.g., dataloader workers) can race.
- **Root-Cause Hypothesis:** `force_alpha` mutates instance fields without synchronization or functional API. No rank barrier or copy-on-write; schedule is set to None then restored, so any overlapping forward sees inconsistent alpha.
- **Remediation Options:**
  - A) Introduce a functional counterfactual forward that takes `alpha_override` as an argument (no mutation) and is excluded from compiled graphs.
  - B) Add DDP-safe guards: assert `world_size==1` or broadcast alpha overrides to all ranks, and refuse nested calls.
  - C) Separate immutable schedule state from runtime alpha so temporary overrides don't touch schedule objects.
- **Risks of Change:** API churn for callers; need to keep torch.compile friendliness and avoid extra syncs in PPO fast paths.
- **Stopgap Mitigation:** Document DDP restriction (currently only noted in comments) and add a runtime guard that raises if `torch.distributed.is_initialized()` while using `force_alpha`.
- **Validation Plan:** Add DDP unit/integration test where `force_alpha` is invoked uniformly across ranks and ensure no parameter/state divergence; add compile test to confirm guard count stability.
- **Status:** Open
- **Links:** `src/esper/kasmina/slot.py` (`force_alpha` context manager, DDP warning comment)
