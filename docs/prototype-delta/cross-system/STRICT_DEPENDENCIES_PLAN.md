# Cross‑System Plan — Mandatory Dependencies + Strict Preflight

Status: In Progress (Weatherlight/Nissa/Tamiyo/Urza implemented)

Objective
- Remove backward‑compatibility/fallback clutter for core dependencies in the pre‑production prototype. Fail fast when mandatory deps are missing or unhealthy. Simplify codepaths and unlock small optimisations.

Scope (Subsystems)
- Weatherlight (supervisor; preflight guard)
- Nissa (observability service)
- Tamiyo (controller)
- Urza (catalog + artifacts)
- Tolaria (trainer; system metrics)
- Urabrask (bench/producers)
- Oona/Kasmina/others unaffected functionally, but benefit from clearer startup failures and consistent metrics.

Mandated Dependencies
- Core ML stack: `torch`, `torch-geometric`, `torch-scatter`, `torch-sparse`, `torch-cluster`, `torch-spline-conv`.
- Serialization: `orjson`.
- Observability: `elasticsearch` (service reachable and responsive).
- System metrics: `psutil` (always), `nvidia-ml-py` (GPU present only; exposes the `pynvml` module).
- Internal modules: Urza must import cleanly where used.

Design Principles
- Preflight once, fail fast: Validate presence and basic health at service start (Weatherlight/Nissa). Do not propagate optionality into live code.
- Prototype posture: Prefer simplicity and determinism over compatibility toggles. Avoid growing “optional paths” unless tied to a clear prototype delta.
- Explicit GPU gating: Require NVML only when CUDA is available (or when a GPU probe is configured).
- No partial degradation: Assume all subsystems are available; otherwise treat the system as fully degraded. Weatherlight exposes `system_mode ∈ {operational, degraded}` in telemetry.

What’s Already Implemented
- Nissa: Removed in‑memory Elasticsearch stub. Runner now imports ES and requires a successful `ping()` at startup.
  - File: `src/esper/nissa/service_runner.py`
- Urza: `orjson` made mandatory; removed JSON fallbacks.
  - File: `src/esper/urza/library.py`
- Tamiyo: Removed optional `UrzaLibrary` import guard.
  - File: `src/esper/tamiyo/service.py`

Next Changes (by subsystem)
1) Weatherlight — Strict Preflight Guard [Implemented]
   - Add a `check_mandatory_dependencies()` called in `WeatherlightService.start()` that verifies:
     - Imports: `torch`, `torch_geometric`, `torch_scatter`, `torch_sparse`, `torch_cluster`, `torch_spline_conv`, `psutil`.
     - If CUDA available: `pynvml` import (provided by `nvidia-ml-py`) + `pynvml.nvmlInit()`.
     - Elasticsearch: delegate to Nissa runner if managed separately, or perform a quick ping if Weatherlight launches Nissa co‑resident.
   - Behavior: Raise a clear `RuntimeError` with actionable remediation text on failure.
   - File: `src/esper/weatherlight/service_runner.py`

2) Nissa — Keep strict posture [Implemented]
   - Optionally, add bulk ingestion via `elasticsearch.helpers.bulk` to reduce per‑packet `index()` overhead.
   - Files: `src/esper/nissa/observability.py`

3) Tolaria — Tighten system metrics imports [Implemented]
   - Treat `psutil` as mandatory: import directly; remove try/except. Always export CPU utilisation when step enrichment is enabled.
   - NVML: When CUDA present, require `pynvml` (module installed via `nvidia-ml-py`; init once and reuse); if init fails, raise at startup instead of silently continuing.
   - File: `src/esper/tolaria/trainer.py`

4) Urabrask — Bench coherence [Implemented]
   - Retain runtime vs fallback provenance (runtime kernel vs synthetic), but remove the codepath that assumes `torch` is not importable. Given `torch` is required project‑wide, fallback should only mean “runtime not available”, not “torch missing”.
   - File: `src/esper/urabrask/benchmarks.py`

Code Simplifications & Dead Code Removal
- Delete MemoryElasticsearch stub and builder fallback [done].
- Remove `orjson` speed‑path guards and JSON fallbacks [done].
- Remove `UrzaLibrary` optional import branch in Tamiyo [done].
- Remove `psutil` try/except blocks; collapse NVML’s nested exception handling to a single startup probe; reuse handle.
- Trim Tamiyo `None` guards that assumed Urza might not be present (metadata paths can remain guarded by command type).
- Drop references/comments claiming “optional dependency” where we now enforce mandatory deps.

Optimisations Unlocked
- Nissa: ES bulk ingestion for telemetry batches from Oona drains.
- Tolaria: NVML handle reuse; fewer branches per metrics snapshot.
- Urza: `orjson` everywhere → faster extras persistence and WAL emissions.
- Tamiyo: Slightly leaner metadata/telemetry path with fewer defensive branches.

Acceptance Criteria
- Weatherlight/Nissa fail fast at startup if mandatory deps are missing or unhealthy; errors include remediation hints.
- Tolaria exports CPU/GPU metrics consistently (given enrichment enabled); GPU metrics require NVML when CUDA present.
- Urabrask benchmarks require torch; “fallback” only reflects lack of runtime kernel, not missing torch.
- No “optional dependency” fallbacks remain in live code for the above packages; unit/integration tests are updated accordingly.

Risks & Mitigations
- Developer friction (missing ES/NVML locally): mitigate with clear error messages and a short section in the operator runbook; avoid re‑introducing optionality.
- CI health checks: ensure ES is available in CI, or skip Nissa service startup tests where not applicable.
- GPU variance: Gate NVML requirement on `torch.cuda.is_available()` to avoid false negatives on CPU‑only hosts.

Test Strategy
- Unit tests: preflight raises when a dependency import is monkey‑failed; Weatherlight/Nissa startup passes when deps healthy.
- Integration: Emergency signal path and telemetry drain unaffected; Tamiyo/Kasmina/Urza codepaths continue to pass targeted tests.
- Benchmarks: Urabrask tests updated to assume torch present and focus on runtime vs synthetic provenance.

Timeline & Effort
- Preflight guard + tests: 0.5 day
- Tighten Tolaria imports + NVML handling: 0.5 day
- Urabrask torch‑mandate cleanup: 0.25 day
- ES bulk ingestion + minor optimisations: 0.25 day
- Docs/tests polish: 0.25 day
- Total: ~1.75 days

Change Log Hook
- Track this work under the cross‑system enhancements label and link in the PR body for visibility across subsystems.
