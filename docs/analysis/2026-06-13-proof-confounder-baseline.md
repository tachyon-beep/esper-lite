# Proof Confounder Baseline

Date: 2026-06-13

## Branch State

- Branch: `confounder-drain`
- Upstream: `origin/confounder-drain`
- Commit: `40eb87e7027c4ee0b4bf540f945d2d0499bc8a18`
- Starting worktree: clean
- Concurrent UX changes observed in this worktree: none. The branch was clean before implementation work began.

## Tracker State

Filigree session snapshot:

- Dashboard: `http://localhost:9189`
- Ready items: 1
- Blocked items: 0
- Only ready item: `P4 esper-lite-88a71d93f9 [release] "Future"`
- `work_ready` reports that release as non-startable with next action `complete child issues`.

## Loomweave State

The first `project_status_get` reported stale index state:

- Last analyzed commit: `7e85197f15a30bf3a97faa5a65b5296207c2787f`
- Current commit: `40eb87e7027c4ee0b4bf540f945d2d0499bc8a18`
- Staleness: `stale`

`loomweave analyze /home/john/esper-lite` was run before code archaeology. It completed at commit `40eb87e7027c4ee0b4bf540f945d2d0499bc8a18` with a fresh index:

- Run id: `f6c5f75b-2d1a-4634-9784-651a177990ed`
- Entities: 10107
- Edges: 24041
- Subsystems: 83
- Findings: 79

The analyze run recorded six Pyright timeout warnings while resolving references/calls in:

- `src/esper/simic/telemetry/emitters.py`
- `src/esper/simic/training/vectorized_trainer.py`
- `src/esper/utils/data.py`

These are archaeology quality warnings, not failing repository tests.

## Source-Verified Task Names

Command:

```bash
PYTHONPATH=src uv run python - <<'PY'
from esper.runtime.tasks import VALID_TASKS
print(sorted(VALID_TASKS))
PY
```

Output:

```text
['cifar_baseline', 'cifar_deep_impaired', 'cifar_deep_multichannel', 'cifar_highres', 'cifar_impaired', 'cifar_minimal', 'cifar_multichannel', 'cifar_postpool', 'cifar_scale', 'tinystories']
```

The proof rehearsal should use `cifar_impaired`, not stale `cifar_blind` wording.

## Focused Health Checks

Command:

```bash
uv run --python 3.11 pytest tests/tamiyo/networks/test_op_value_consistency.py tests/simic/training/test_bootstrap_consistency.py -q
```

Result:

```text
10 passed in 1.62s
```

Command:

```bash
uv run --python 3.11 pytest tests/karn -q
```

Result:

```text
707 passed, 4 deselected, 6 warnings in 10.94s
```

Command:

```bash
uv run python scripts/lint_defensive_patterns.py
```

Result:

```text
Checked 185 files in strict mode
Total patterns found: 848
Patterns checked: 71
Allowed: 71
Violations: 0
Stale whitelist entries: 0
All defensive patterns are whitelisted or absent.
```

## Baseline Findings

- `docs/bugs/investigations/CRITICAL-op-value-mismatch.md` is resolved and the current focused regressions pass.
- `docs/bugs/investigations/CRITICAL-telemetry-freeze.md` already names learnable fraction as the next required enhancement to separate mask-collapse zeros from broken telemetry.
- `src/esper/simic/training/dual_ab.py` describes a lockstep A/B architecture but still trains policy groups sequentially. Its own limitations section states later groups may benefit from GPU warm-up and cached compilation. This remains a known proof confounder unless the proof packet marks it or the implementation changes it.
- `src/esper/scripts/train.py` exposes `--rounds`, `--envs`, `--episode-length`, and `--dual-ab shaped-vs-simplified`; the proof plan's `cifar_impaired` command shape is supported by the current parser.

## Baseline Conclusion

The branch starts from a clean, green Karn and op/value baseline. The next implementation work should fail closed on proof validity rather than add passive charts: first create a run-level confounder ledger, then surface learnability and freshness data, then close reward-accounting loopholes before running the proof exam.
