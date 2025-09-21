# Tezzeret — Prototype Delta (Compilation Forge)

Executive summary: the prototype includes a stub compiler that writes `CompiledBlueprint` modules to disk with a simple WAL, and a `TezzeretForge` that enumerates Karn templates and compiles them into Urza, resuming from a WAL on failure. The full design specifies polling, multiple compilation strategies (Fast/Standard/Aggressive/Emergency) on top of torch.compile, circuit breakers with conservative mode, resource monitoring, TTL cleanup, telemetry, and signing/versioning of artifacts.

Documents in this folder:
- `delta-matrix.md` — requirement‑by‑requirement status with evidence
- `traceability-map.md` — mapping of design assertions to code/tests
- `implementation-roadmap.md` — plan to close gaps without tech debt

Design sources:
- `docs/design/detailed_design/06-tezzeret.md`
- `docs/design/detailed_design/06.1-tezzeret-compilation-internals.md`

Implementation evidence (primary):
- `src/esper/tezzeret/compiler.py`, `src/esper/tezzeret/runner.py`
- Tests: `tests/tezzeret/test_compiler.py`, `tests/tezzeret/test_runner.py`

