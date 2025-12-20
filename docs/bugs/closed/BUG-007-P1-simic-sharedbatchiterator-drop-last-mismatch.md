# BUG-007: SharedBatchIterator drops tail batches unevenly (superseded)

- **Title:** SharedBatchIterator drops tail batches unevenly, desynchronizing envs/metrics
- **Context:** Simic data loading (`SharedBatchIterator` in `src/esper/utils/data.py`, used by `src/esper/simic/training/vectorized.py` when `gpu_preload=False`)
- **Impact:** P1 (historical) – this described a potential mismatch between per-env deliveries and `num_train_batches=len(shared_train_iter)`.
- **Environment:** Main branch; PPO vectorized path with SharedBatchIterator.
- **Reproduction Steps:**
  1. Create a toy dataset with 10 samples, `n_envs=3`, `batch_size_per_env=2` ⇒ total batch=6.
  2. Observe `len(shared_train_iter)` returns 1 with `drop_last=True` (tail dropped).
  3. Observe envs process exactly 1 batch (no mismatch vs `num_train_batches`).
- **Expected Behavior:** No silent desync between env batch deliveries and averaging denominators.
- **Observed Behavior:** Under current defaults, the described desync does not reproduce: training uses `drop_last=True`, so every yielded batch is full-size and splits evenly across envs.
- **Hypotheses:** This ticket captured an earlier iteration (or a hypothetical `drop_last=False` training mode) where partial combined batches could produce fewer per-env deliveries than expected.
- **Fix Plan:** None (ticket closed). If we ever enable `drop_last=False` for training, we should rescope into a new ticket to define remainder semantics and make loss aggregation per-sample.
- **Validation Plan:** Existing unit coverage for partial-batch splitting exists in `tests/utils/test_data.py` (SharedBatchIterator with `drop_last=False`).
- **Status:** Closed (Superseded by current training semantics)
- **Resolution:** Vectorized training enforces `drop_last=True` for the train iterator and computes `num_train_batches = len(shared_train_iter)`, so all envs receive the same number of train steps and the denominator matches the actual number of iterations.
- **Links:** `src/esper/utils/data.py::SharedBatchIterator`, `src/esper/simic/training/vectorized.py` (`num_train_batches`, metric aggregation)
