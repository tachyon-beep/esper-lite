# JANK Template

- **Title:** DDP gate consensus can deadlock when ranks diverge in stage calls
- **Category:** correctness-risk / distributed stability
- **Symptoms:** `_sync_gate_decision` in `src/esper/kasmina/slot.py` calls `torch.distributed.all_reduce` on every gate check, assuming all ranks invoke gates identically. If ranks differ (e.g., one has no active seed in a slot, or seeds are at different stages), some ranks skip the gate while others call it, leading to a hang.
- **Impact:** P1 – any multi-GPU/DDP experiment risks indefinite hangs during lifecycle advancement; also blocks planned DDP enablement in Simic.
- **Triggers:** Divergent seed states across ranks (different germination timing, failures, or slot availability), or control-flow differences that skip `step_epoch`/`advance_stage` on some ranks.
- **Root-Cause Hypothesis:** Consensus reduction requires strict call symmetry, but lifecycles are per-env and can diverge without an explicit barrier/sync of stage state before gate checks.
- **Remediation Options:** 
  - A) Synchronize slot/seed stage metadata across ranks before gate checks (broadcast or all_gather), ensuring every rank executes the same gate order.
  - B) Centralize gate decisions on rank 0 and broadcast results to others.
  - C) Add watchdog/timeout around `_sync_gate_decision` and fail fast with diagnostics.
- **Risks of Change:** Extra comms overhead; need to ensure determinism with async streams; potential redesign of per-slot state machine for DDP.
- **Stopgap Mitigation:** Document DDP as unsupported; guard `_sync_gate_decision` behind a feature flag to force single-rank behavior; add timeout to prevent indefinite hang.
- **Validation Plan:** Simulate divergent seeds across ranks in a DDP test harness; ensure gate synchronization completes or times out cleanly. Verify multi-GPU PPO smoke passes without hang.
- **Status:** Open
- **Links:** `src/esper/kasmina/slot.py:1207-1250` (`_sync_gate_decision`), PyTorch specialist review noting DDP deadlock risk
