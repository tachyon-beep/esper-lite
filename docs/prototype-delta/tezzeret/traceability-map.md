# Tezzeret — Traceability Map

| Design Assertion | Source | Implementation | Tests |
| --- | --- | --- | --- |
| Enumerate blueprints and compile into Urza | `06-tezzeret.md` | `src/esper/tezzeret/runner.py::TezzeretForge.run()` | `tests/tezzeret/test_runner.py::test_forge_compiles_catalog` |
| Skip existing artifacts | `06-tezzeret.md` | Forge checks Urza before compile | `tests/tezzeret/test_runner.py::test_forge_skips_existing` |
| Resume from WAL after failure | `06.1` | Forge WAL (`forge_wal.json`); compiler WAL | `tests/tezzeret/test_runner.py::test_forge_resumes_from_wal` |
| Retry failed jobs | `06.1` | Compiler retries | `tests/tezzeret/test_runner.py::test_forge_retries_failed_job` |
| torch.compile strategies, breakers, telemetry | `06.1` | — | — |

