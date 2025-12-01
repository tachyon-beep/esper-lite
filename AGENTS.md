# Repository Guidelines

## Project Structure & Module Organization
- `src/esper/leyline`: Shared contracts (stages, commands, telemetry).
- `src/esper/kasmina`: Seed mechanics and host models (blueprints, slots, isolation, blending).
- `src/esper/tamiyo`: Strategic controller logic (heuristics, tracking).
- `src/esper/simic`: Policy learning (features, rewards, PPO/IQL); `src/esper/simic_overnight.py` is the batch runner.
- `src/esper/nissa`: Telemetry profiles/configuration; `src/esper/scripts/` and `scripts/` hold CLIs (e.g., `train_ppo.sh`).
- Tests live in `tests/`; `docs/` + `notebooks/` are research notes; `data/` holds generated artifacts—avoid committing large outputs.

## Architecture Notes
- Leyline defines the schema surface; Kasmina hosts seeds and enforces stage transitions; Tamiyo chooses actions; Simic learns a policy to improve Tamiyo; Nissa is optional telemetry wiring.
- Core loop: training signals -> Tamiyo/Simic action -> Kasmina seed update -> Leyline reports back into signals; keep vector sizes and enums consistent across layers.

## Build, Test, and Development Commands
- Create env: `python -m venv .venv && source .venv/bin/activate`.
- Install: `pip install -e .[dev]` (pytest, jupyter; Python 3.11+).
- Proof-of-concepts: `PYTHONPATH=src python src/esper/poc.py` (fixed) or `PYTHONPATH=src python src/esper/poc_tamiyo.py` (Tamiyo-driven).
- Policy training: `./scripts/train_ppo.sh -e 50 --single` (writes `data/models/ppo_tamiyo.pt`; `--help` for options); GPU recommended, CPU supported but slower.
- Batch policy learning: `PYTHONPATH=src python src/esper/simic_overnight.py --episodes 50`.
- Tests: `PYTHONPATH=src pytest -q` before PRs.

## Coding Style & Naming Conventions
- 4-space indentation, type hints; prefer `@dataclass(slots=True)` or `NamedTuple` for hot paths.
- Functions/variables `snake_case`; classes `CamelCase`; constants `UPPER_SNAKE`.
- Keep public APIs re-exported via package `__init__.py`; add docstrings and only brief comments where logic is non-obvious.
- Maintain deterministic tensor shapes/vector sizes when updating feature schemas (e.g., `TensorSchema`, `TrainingSnapshot.vector_size()`).

## Testing Guidelines
- Add pytest cases under `tests/` using `test_*.py`; extend existing fixtures instead of duplicating scaffolding.
- Use `tempfile` for file/model writes; prefer deterministic seeds where feasible.
- When changing feature schemas (`TensorSchema`, `TrainingSnapshot.vector_size()`), update vector/serialization tests.

## Commit & Pull Request Guidelines
- Follow Conventional Commit style: `feat: …`, `fix(scope): …`, `chore: …` (see recent history).
- PRs should include summary, motivation/linked issue, commands/tests run (exact invocations), and resulting metrics or artifacts (paths, not binaries). Keep diffs focused and update docs/tests alongside behavior changes.

## Security & Configuration Tips
- Keep `PYTHONPATH=src` for all local runs; prefer `.venv/bin/python` in scripts.
- Avoid committing checkpoints or generated data; if essential, keep under `data/`, note sizes, and prefer gitignore/LFS for large files.

## No Legacy Code Policy
- **Strict:** No backwards compatibility, shims, adapters, or dual code paths. When behavior changes, delete old code entirely and update all call sites in the same change.
- **Never add:** Version checks, feature flags for old behavior, compatibility modes, deprecated stubs, commented-out old code, `_legacy`/`_old` helpers, or migration helpers supporting both old and new.
- **Default stance:** If it’s removed or replaced, it’s gone—rely on git history, not in-repo shims.

## hasattr Authorization
- **Strict:** Every `hasattr()` must have an inline authorization comment with operator name, ISO 8601 datetime (UTC), and justification.
- **Format example:**  
  `# hasattr AUTHORIZED by John on 2025-11-30 14:23:00 UTC`  
  `# Justification: Serialization handling of external polymorphic payloads`
- **Allowed cases (still require authorization):** External/serialization polymorphism, cleanup guards (`__del__/close`), external feature detection. All others should be refactored away.
