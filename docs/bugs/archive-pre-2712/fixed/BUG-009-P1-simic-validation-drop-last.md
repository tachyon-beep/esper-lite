# BUG-009: Validation SharedBatchIterator drops tail batches

- **Title:** Validation SharedBatchIterator drops tail batches, skewing val metrics
- **Context:** Simic vectorized validation (`src/esper/simic/vectorized.py`)
- **Impact:** P1 – validation metrics computed on truncated dataset, biasing seed decisions
- **Environment:** Main branch; SharedBatchIterator path (gpu_preload=False)
- **Status:** FIXED (2025-12-17)

## Root Cause Analysis

**Line 1109 in vectorized.py:**
```python
shared_test_iter = SharedBatchIterator(
    dataset=test_dataset,
    ...
    drop_last=True,  # BUG: drops tail samples for validation
)
```

The training iterator (line 1095) correctly uses `drop_last=True` for even batch sizes.
But the test/validation iterator copied this setting, causing data loss.

### Impact Quantification (CIFAR-10 test set: 10,000 samples)

| n_envs | batch_size | total_batch | dropped | % data lost |
|--------|------------|-------------|---------|-------------|
| 2 | 32 | 64 | 16 | 0.16% |
| 4 | 32 | 128 | 16 | 0.16% |
| 8 | 32 | 256 | 16 | 0.16% |
| 4 | 64 | 256 | 16 | 0.16% |
| **8** | **64** | **512** | **464** | **4.64%** |

Worst case loses nearly 5% of validation data silently.

### Comparison with gpu_preload Path

The `gpu_preload=True` path uses `load_cifar10_gpu()` which creates DataLoaders
with PyTorch's default `drop_last=False`. So gpu_preload does NOT have this bug.

### Downstream Effects

Biased validation metrics affected:
1. Validation accuracy/loss - not representative of full test set
2. Counterfactual baselines - uses same iterator
3. Seed culling decisions - based on incomplete validation
4. Reward computation - based on biased accuracy delta

## Fix

Changed line 1109 from `drop_last=True` to `drop_last=False`.

`torch.chunk()` handles uneven splits gracefully - last batch environments get
fewer samples, but `val_totals[i] += total` correctly tracks per-env sample
counts for proper averaging.

## Validation

- Existing tests pass (no change in behavior for evenly-divisible datasets)
- The fix ensures all validation samples are processed

## Links

- Fix: `src/esper/simic/vectorized.py` (line 1109)
- Related: `src/esper/utils/data.py::SharedBatchIterator`
