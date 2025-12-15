# BUG Template

- **Title:** Validation SharedBatchIterator drops tail batches, skewing val metrics
- **Context:** Simic vectorized validation uses `SharedBatchIterator` with default `drop_last=True` (`src/esper/utils/data.py`) when `gpu_preload=False`. Validation/test iterators in `vectorized.py` are constructed without overriding `drop_last`, so the final partial batch is discarded.
- **Impact:** P1 – validation accuracy/loss are computed on a truncated dataset, biasing metrics and counterfactual baselines; seeds may be culled or advanced based on incomplete validation.
- **Environment:** Main branch; PPO vectorized path with standard CPU DataLoader (no gpu_preload).
- **Reproduction Steps:**
  1. Use a dataset size not divisible by `batch_size_per_env * n_envs`.
  2. Run validation; last partial batch is dropped silently because `drop_last=True` in SharedBatchIterator.
  3. Val accuracy/loss differ from a full-dataset pass.
- **Expected Behavior:** Validation should evaluate all available data (or explicitly report drop), especially for reward/counterfactual computation.
- **Observed Behavior:** Tail batch is dropped; metrics are biased without warning.
- **Hypotheses:** `SharedBatchIterator` default `drop_last=True` was set for even train splits but reused for validation without override.
- **Fix Plan:** Set `drop_last=False` for validation iterator; optionally warn if train iterator drops data; add tests to ensure val iter covers full dataset.
- **Validation Plan:** Unit test with non-divisible dataset verifying val iterator length covers all samples and metrics match a reference DataLoader; PPO smoke to ensure no regression.
- **Status:** Open
- **Links:** `src/esper/utils/data.py::SharedBatchIterator` (drop_last default), `src/esper/simic/vectorized.py` validation iterator creation (~650)
