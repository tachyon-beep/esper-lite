# Tezzeret — Delta Matrix

Status and evidence per requirement. See rubric in `../rubric.md`.

| Area | Design Source | Expected Behaviour | Prototype Evidence | Status | Severity | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Blueprint polling & queueing | `06-tezzeret.md` | Enumerate blueprints from Urza/Karn at startup; enqueue | `src/esper/tezzeret/runner.py::TezzeretForge._discover_jobs()` | Implemented | Should‑have | Enumerates Karn; no periodic refresh. |
| Compilation pipeline (torch.compile) | `06.1` | Deterministic pipeline; resource/time metrics; strategies | `src/esper/tezzeret/compiler.py` (stub save of nn.Module) | Missing | Must‑have | No torch.compile, no metrics/strategies. |
| WAL & crash recovery | `06.1` | WAL for forge and compiler; resume in‑flight jobs | `runner.py` WAL for pending; `compiler.py` WAL per job; tests resume | Partially Implemented | Must‑have | JSON WAL; no CRC/O_DSYNC; minimal.
| Circuit breakers & conservative mode | `06.1` | Breakers around timeouts/resources; throttle to Fast pipeline | — | Missing | Must‑have | Not implemented. |
| Telemetry | `06-tezzeret.md` | `tezzeret.compilation.*`, breaker state | — | Missing | Should‑have | No Oona/Nissa integration. |
| Resource monitoring | `06.1` | GPU utilisation ≤25 %, memory guards, TTL cleanup | — | Missing | Should‑have | Not present. |
| Signing/versioning | `06-tezzeret.md` | Sign artifacts, include version in Urza | — | Missing | Nice‑to‑have | Not present. |
| Leyline as canonical | `00-leyline` | Use Leyline messages where applicable | Indirect; uses Karn (Leyline descriptors) | Implemented | Must‑have | Descriptor inputs align to Leyline via Karn. |

