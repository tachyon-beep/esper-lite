# Kasmina — Traceability Map

Mapping of design assertions to implementation artefacts and tests.

| Design Assertion | Source | Implementation | Tests |
| --- | --- | --- | --- |
| Lifecycle uses Leyline enums; aligns to unified design | `docs/design/detailed_design/02-kasmina-unified-design.md` | `src/esper/kasmina/lifecycle.py` | `tests/kasmina/test_lifecycle.py` |
| Seed germination→training fast path exists | `docs/design/detailed_design/02-kasmina-unified-design.md` | `src/esper/kasmina/lifecycle.py` (allowed transitions) | `tests/kasmina/test_lifecycle.py` |
| Kernel fetch via Urza, with latency budget and fallback | `docs/design/detailed_design/02.1-kasmina-kernel-execution.md` | `src/esper/kasmina/seed_manager.py` (`_graft_seed`, `_load_fallback`) | `tests/kasmina/test_seed_manager.py` (latency/fallback), `tests/integration/test_control_loop.py` |
| Export seed states via Leyline messages | `docs/design/detailed_design/02-kasmina-unified-design.md` | `src/esper/kasmina/seed_manager.py` (`export_seed_states`) | Used by Tolaria state assembly (`src/esper/tolaria/trainer.py`) |
| Gradient isolation sanity check | `docs/design/detailed_design/02-kasmina-unified-design.md` | `src/esper/kasmina/seed_manager.py` (host parameter ID overlap) | `tests/kasmina/test_seed_manager.py::test_gradient_isolation_detects_overlap` |
| Structured telemetry with health status | `docs/design/detailed_design/02-kasmina-unified-design.md` | `src/esper/core/telemetry.py`; emissions in `seed_manager.py` | `tests/kasmina/test_seed_manager.py` (telemetry presence) |
| Performance validation thresholds | `docs/design/detailed_design/02.5-kasmina-performance-validation.md` | — | — |
