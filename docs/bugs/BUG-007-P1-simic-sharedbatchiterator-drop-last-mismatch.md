# BUG Template

- **Title:** SharedBatchIterator drops tail batches unevenly, desynchronizing envs/metrics
- **Context:** Simic data loading (`SharedBatchIterator` in `src/esper/utils/data.py`, used by `vectorized.py` when `gpu_preload=False`)
- **Impact:** P1 – when dataset size isn’t divisible by `batch_size_per_env * n_envs`, `drop_last=True` drops the tail of the combined loader. Some envs may receive fewer batches than `num_train_batches=len(shared_train_iter)`, leading to desynced accumulators and incorrect averaging (train_loss divided by an inflated `num_train_batches`). In mixed GPU/CPU or uneven device assignment, envs can under-train silently.
- **Environment:** Main branch; PPO vectorized path with SharedBatchIterator; any dataset size not evenly divisible by total batch.
- **Reproduction Steps:**
  1. Create a toy dataset with 10 samples, `n_envs=3`, `batch_size_per_env=2` ⇒ total batch=6.
  2. `len(shared_train_iter)` returns 1 (6 samples) with `drop_last=True`, leaving 4 samples unused; envs process 1 batch but loop may still expect `num_train_batches` > delivered batches.
  3. Metrics divide by `num_train_batches` even if StopIteration hit early for some envs.
- **Expected Behavior:** Either cover all samples (pad/uneven split) or adjust `num_train_batches`/averaging per env based on actual batches delivered; no silent desync.
- **Observed Behavior:** Tail batches are dropped; num_train_batches is based on loader length, not actual per-env deliveries, risking skewed metrics and learning rate schedules tied to epochs.
- **Hypotheses:** `drop_last=True` was used to keep even splits, but it implicitly discards data and assumes divisibility; no guard for remainder.
- **Fix Plan:** Option A: allow final uneven batch (pad or smaller per-env batches) and adjust masks; Option B: compute `num_train_batches` based on actual iterations and accumulate per-env batch counts; add a guard/telemetry warning when data is dropped.
- **Validation Plan:** Add a unit test with non-divisible dataset to assert per-env batch counts match `num_train_batches` and metrics aren’t inflated; run a PPO smoke with synthetic data to ensure no StopIteration mismatches.
- **Status:** Open
- **Links:** `src/esper/utils/data.py::SharedBatchIterator`, `src/esper/simic/vectorized.py` (`num_train_batches`, accumulator logic)
