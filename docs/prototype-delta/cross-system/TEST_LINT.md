# Test & Lint Warning Remediation Plan (Prototype)

## Context
- Scope: resolve the warning classes observed in `TAMIYO_DEVICE=cpu TAMIYO_ENABLE_COMPILE=0 .venv/bin/python -m pytest --timeout=180` (2025-03-03 run).
- Goals: keep the suite warning-clean, ensure forward-compatibility with PyTorch 2.8+ and pytest 9, and preserve prototype-delta constraints (strict dependencies, fail-fast behaviour).
- Owners: Tamiyo (policy/service), Tolaria (trainer), Core (dependencies), Oona (messaging), Weatherlight (async fixtures).

## Warning Inventory
1. `UserWarning: The CUDA Graph is empty …` (torch.cuda.graphs) — emitted 126× while `tests/tamiyo/test_service.py` exercises `TamiyoPolicy` with compile enabled on CPU.
2. `FutureWarning: torch.cuda.amp.GradScaler` / ``torch.cuda.amp.autocast`` deprecated — emitted from `src/esper/tolaria/trainer.py` during CUDA-path initialisation.
3. `FutureWarning: The pynvml package is deprecated` — emitted during `torch.cuda` import because we pin `pynvml`.
4. `DeprecationWarning: close() -> aclose()` — emitted by `fakeredis` Redis clients in `esper.oona.messaging` and `tests/oona/test_messaging_integration.py`.
5. `PytestRemovedIn9Warning: async fixture 'fake_redis' without plugin` — emitted for Weatherlight tests that register an `async def` fixture without `pytest_asyncio`.

## Remediation Actions

### 1. Tamiyo torch.compile UserWarnings
- **Design choice:** keep compile-on by default for CUDA only; skip CUDA graph capture for CPU execution to avoid backend warnings.
- **Implementation steps:**
  1. **Device-sensitive enablement**
     - In `TamiyoPolicy.__init__` ensure `enable_compile` is coerced to `False` for non-CUDA devices, regardless of config toggle; emit a structured telemetry event (`tamiyo.gnn.compile_disabled_cpu`) and increment `tamiyo.gnn.compile_fallback_total` so visibility is retained.
     - Surface the reason via `logger.info("tamiyo_gnn_compile_disabled_cpu", ...)` to aid log-based diagnostics.
  2. **Warm-up / graph capture guards**
     - Wrap `_warmup_compiled_model` so it only runs when `self._device.type == "cuda"` and `torch.cuda.is_available()`; return early for CPU to avoid scheduling CUDA graphs.
     - In `select_action`, gate the `torch.compile` path behind the same predicate and skip the CUDA graph context (`torch.cuda.graphs`) on CPU.
     - Add a fast path that demotes to eager once when a backend raises `torch.cuda.CudaError`, ensuring we do not repeatedly attempt graph capture.
  3. **Telemetry alignment**
     - Ensure telemetry metrics (`tamiyo.gnn.compile_enabled`, `tamiyo.gnn.compile_fallback_total`, `tamiyo.gnn.compile_warm_ms`) reflect the CPU demotion so downstream monitoring reads 0/disabled without warning spam.
     - Document the behaviour shift in `docs/prototype-delta/tamiyo/pytorch-2.8-upgrades.md`.
  4. **Test coverage**
     - Extend `tests/tamiyo/test_service.py::test_compile_fallback_counter_exposed` to assert the new telemetry/log behaviour (compile disabled on CPU).
     - Add a regression test in `tests/tamiyo/test_policy_gnn.py` that verifies `policy.compile_enabled` is `False` on CPU even when `enable_compile=True`, and that `select_action` does not emit warnings (use `pytest.warns(None)` context).
     - Retain an integration-style test with CUDA (guarded by `ESPER_RUN_CUDA_TESTS=1`) to ensure compile still activates on GPU hardware.
- **Validation:** `pytest tests/tamiyo/test_service.py::test_compile_fallback_counter_exposed`, full `pytest --timeout=180` with `TAMIYO_DEVICE=cpu TAMIYO_ENABLE_COMPILE=0`, confirm no `torch/cuda/graphs.py` warnings.

### 2. PyTorch AMP API Migration (Tolaria)
- **Design choice:** migrate to `torch.amp.autocast(device_type="cuda", dtype=…)` and `torch.amp.GradScaler(device_type="cuda", ...)` to align with PyTorch ≥2.8 guidance.
- **Implementation steps:**
  1. Replace the legacy `torch.cuda.amp.autocast` context manager in `TolariaTrainer._inference_autocast_ctx`.
  2. Swap `torch.cuda.amp.GradScaler` for `torch.amp.GradScaler(device_type="cuda", enabled=…)` and preserve existing configuration.
  3. Update unit tests in `tests/tolaria/test_tolaria_trainer.py` that patch/scenario the GradScaler to expect the new API.
- **Validation:** `pytest tests/tolaria/test_tolaria_trainer.py`, `pytest tests/tolaria -m "not integration"`, confirm no FutureWarnings; run GPU smoke (optional) if hardware available.

### 3. NVML Dependency Refresh
- **Design choice:** replace `pynvml` with `nvidia-ml-py` per PyTorch deprecation notice; keep strict dependency contract.
- **Implementation steps:**
  1. Update `pyproject.toml` (core deps and optional `sysmetrics`) to depend on `nvidia-ml-py` instead of `pynvml`.
  2. Adjust import sites (`esper.core.hw` or wherever NVML is touched) to prefer `nvidia_ml_py` module names (if needed) and fail fast when missing.
  3. Regenerate lockfiles/constraints if maintained; document change in `docs/prototype-delta/cross-system/STRICT_DEPENDENCIES_PLAN.md`.
- **Validation:** `pip freeze | rg nvidia-ml-py`, run `pytest tests/nissa tests/weatherlight` (NVML stubs), confirm no FutureWarning on `torch.cuda` import.

### 4. Redis Close Deprecation
- **Design choice:** standardise on `await redis.aclose()` with sync fallback for older fakeredis releases.
- **Implementation steps:**
  1. Update `esper.oona.messaging.OonaClient.close` (and any other shutdown helpers) to call `aclose()` when available; keep `close()` fallback with explicit comment.
  2. Mirror the change in `tests/oona/test_messaging_integration.py` and other test fixtures.
  3. Add a focused regression test that ensures the async close path runs without raising (FakeRedis + real aioredis).
- **Validation:** `pytest tests/oona/test_messaging.py tests/oona/test_messaging_integration.py`, ensure no DeprecationWarning.

### 5. Pytest Async Fixture Compliance (Weatherlight)
- **Design choice:** annotate async fixtures with `@pytest_asyncio.fixture` and ensure tests opt into `pytest.mark.asyncio` where needed.
- **Implementation steps:**
  1. Add `import pytest_asyncio` in `tests/weatherlight/test_service_runner.py` and convert `@pytest.fixture` on `fake_redis` to `@pytest_asyncio.fixture`.
  2. Audit other test files for async fixtures; apply the same decorator where found.
  3. Bump `pytest-asyncio` minimum in `pyproject.toml` if current pin predates the new API.
- **Validation:** `pytest tests/weatherlight/test_service_runner.py`, confirm no `PytestRemovedIn9Warning`.

## Verification & Rollout
- Run `TAMIYO_DEVICE=cpu TAMIYO_ENABLE_COMPILE=0 .venv/bin/python -m pytest --timeout=180` locally → expect zero warnings.
- Run targeted GPU job (if available) to ensure Tamiyo compile still engages correctly.
- Update CI notes: ensure GitHub Actions environment exports `TAMIYO_DEVICE=cpu` and `TAMIYO_ENABLE_COMPILE=0` (already applied) so warning-free baseline is enforced.
- Document the changes in subsystem delta files (`docs/prototype-delta/tamiyo/pytorch-2.8-upgrades.md`, `docs/prototype-delta/tolaria/pytorch-2.8-upgrades.md`, `docs/prototype-delta/oona/README.md`, etc.) as applicable.

## Tracking
- Link this plan from future PRs touching the above warnings; mark items complete once CI runs without warnings for two consecutive builds.
