# BUG Template

- **Title:** TolariaGovernor rollbacks restore weights but not optimizer state
- **Context:** Tolaria governor (`src/esper/tolaria/governor.py`) snapshots only `model.state_dict()` to CPU; optimizer states are not captured/restored.
- **Impact:** P1 – after rollback, optimizer momentum/adam moments are stale relative to restored weights, causing training instability or divergence. Rollback purports to restore Last Known Good but leaves optimizer in a mismatched state.
- **Environment:** Main branch; any training with governor rollbacks and momentum/Adam optimizers.
- **Reproduction Steps:** Trigger a governor rollback; observe optimizer state (e.g., momentum buffers) not reset, leading to immediate large steps.
- **Expected Behavior:** Rollback should restore both model and optimizer (or reinitialize optimizer) to a consistent state.
- **Observed Behavior:** Only model weights are restored; optimizer state remains from the pre-panic trajectory.
- **Hypotheses:** Snapshot kept minimal for memory; optimizer state omitted inadvertently.
- **Fix Plan:** Snapshot optimizer state alongside model (CPU-stored) and restore on rollback, or explicitly reinit optimizer after rollback; document memory tradeoff.
- **Validation Plan:** Add test simulating rollback with SGD/Adam and ensure parameters/optimizer state match snapshot; verify no jump in loss post-rollback.
- **Status:** Open
- **Links:** `src/esper/tolaria/governor.py` (snapshot/execute_rollback), optimizer usage in Simic/Tolaria
