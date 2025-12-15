# BUG Template

- **Title:** Heuristic training API throws when slots use default None
- **Context:** Simic / `train_heuristic` entrypoint (`src/esper/simic/training.py`)
- **Impact:** Medium – programmatic callers importing `train_heuristic()` hit an immediate ValueError before training starts; diverges from CLI defaults (`--slots` defaults to `["mid"]`), so samples in notebooks or quick API calls fail.
- **Environment:** main branch, Python 3.11, `PYTHONPATH=src`; device-agnostic (repro on CPU)
- **Reproduction Steps:**
  1. `PYTHONPATH=src uv run python - <<'PY'\nfrom esper.simic.training import train_heuristic\ntrain_heuristic(n_episodes=1, max_epochs=1, max_batches=1)\nPY`
  2. Observe failure before any training loop executes.
- **Expected Behavior:** With no explicit `slots` provided, heuristic training should default to the same slot list as the CLI (`["mid"]`), initialize the model, and run a 1-episode loop without raising.
- **Observed Behavior:** Raises `ValueError("slots parameter is required and cannot be empty")` from `run_heuristic_episode` because `train_heuristic` passes `slots=None` through (`src/esper/simic/training.py` around lines 526-588).
- **Logs/Telemetry:** None emitted; exception thrown before telemetry wiring.
- **Hypotheses:** Default argument mismatch between CLI parser (defaults to `["mid"]`) and API (`slots=None`), leaving `run_heuristic_episode` to enforce a non-empty slots list without providing a default.
- **Fix Plan:** Set `train_heuristic` default `slots` to `["mid"]` (or reuse `ordered_slots` for deterministic ordering), mirror CLI behavior, and keep duplicate/empty validation in `run_heuristic_episode`.
- **Validation Plan:** Re-run reproduction snippet (should complete without exception), plus a fast smoke: `PYTHONPATH=src uv run python -m esper.scripts.train heuristic --episodes 1 --max-epochs 1 --max-batches 1` (should remain unaffected).
- **Status:** Open
- **Links:** README quickstart (heuristic), `src/esper/scripts/train.py` `--slots` defaults
