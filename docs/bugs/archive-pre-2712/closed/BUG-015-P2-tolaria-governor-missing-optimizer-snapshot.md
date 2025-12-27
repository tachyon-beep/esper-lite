# BUG-015: TolariaGovernor rollbacks don't restore optimizer state

- **Title:** TolariaGovernor rollbacks restore weights but not optimizer state
- **Context:** Tolaria governor (`src/esper/tolaria/governor.py`)
- **Impact:** P2 – Design improvement, low practical impact
- **Environment:** Main branch
- **Status:** Closed (Fixed)
- **Resolution:** Fixed by updating `TolariaGovernor.execute_rollback` to accept an `optimizer` argument and zero out its momentum/variance buffers (Option C). Call sites in `simic/vectorized.py` updated to pass the host optimizer.
- **Links:** `src/esper/tolaria/governor.py` (snapshot/execute_rollback), `src/esper/simic/training/vectorized.py`
