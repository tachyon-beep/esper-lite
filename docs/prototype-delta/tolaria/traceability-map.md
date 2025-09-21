# Tolaria — Traceability Map

Mapping of design assertions to implementation artefacts and tests.

| Design Assertion | Source | Implementation | Tests |
| --- | --- | --- | --- |
| End‑of‑epoch boundary is tight (≤18 ms) | `01.1-tolaria-epoch-lifecycle.md` | `src/esper/tolaria/trainer.py` (hook measured, warns) | `tests/tolaria/test_tolaria_trainer.py` (budget check) |
| SystemStatePacket assembly and Tamiyo handshake | `01.1`, `01.4` | `trainer._emit_state()` and `tamiyo.evaluate_epoch(state)` | `tests/tolaria/test_tolaria_trainer.py` (states exist, Tamiyo stub used) |
| Apply AdaptationCommand to Kasmina | `01.4` | `kasmina.apply_command(command)` | `tests/integration/test_control_loop.py` (Kasmina probe) |
| Checkpoint + WAL for rollback | `01.2` | `trainer._checkpoint()`, `rollback_to_last_checkpoint()` | `tests/tolaria/test_tolaria_trainer.py::test_tolaria_checkpoint_and_rollback` |
| Telemetry emission and publish to Oona | `01.1`, `01.4` | `trainer._emit_telemetry()`, `publish_history()` | `tests/tolaria/test_tolaria_trainer.py::test_tolaria_publish_history_to_oona` |
| Unified LR controller with circuit breakers | `01.3` | — | — |
| Two‑tier rollback (fast + full) | `01.2` | — | — |
| Emergency escalation protocol | `01.2` | — | — |
| Multi‑seed gradient aggregation | `01.1` | — | — |

