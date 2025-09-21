# Tolaria — Traceability Map

Mapping of design assertions to implementation artefacts and tests.

| Design Assertion | Source | Implementation | Tests |
| --- | --- | --- | --- |
| End‑of‑epoch boundary is tight (≤18 ms) | `01.1-tolaria-epoch-lifecycle.md` | `src/esper/tolaria/trainer.py` (hook measured, warns) | `tests/tolaria/test_tolaria_trainer.py` (budget check) |
| SystemStatePacket assembly and Tamiyo handshake | `01.1`, `01.4` | `trainer._emit_state()` and `tamiyo.evaluate_epoch(state)` | `tests/tolaria/test_tolaria_trainer.py` (states exist, Tamiyo stub used) |
| Apply AdaptationCommand to Kasmina | `01.4` | `kasmina.apply_command(command)` | `tests/integration/test_control_loop.py` (Kasmina probe) |
| Checkpoint + WAL for rollback | `01.2` | `trainer._checkpoint()`, `rollback_to_last_checkpoint()` | `tests/tolaria/test_tolaria_trainer.py::test_tolaria_checkpoint_and_rollback` |
| Two‑tier rollback with deadline | `01.2` | `src/esper/tolaria/rollback.py`, `trainer` integration | `tests/tolaria/test_trainer_knobs.py::test_rollback_snapshot_cadence`, `tests/tolaria/test_emergency_halt.py::test_l4_halt_on_rollback_deadline_exceeded` |
| Unified LR controller | `01.3` | `src/esper/tolaria/lr_controller.py`, `trainer` wiring/metrics | — (exercised indirectly via metrics) |
| Optimizer rebuild manager | `01.3` | `src/esper/tolaria/optimizer_manager.py`, `trainer` wiring | `tests/tolaria/test_trainer_knobs.py::test_optimizer_rebuild_storm_guard_steps` |
| Emergency protocol L4 (halt) | `01.2` | `src/esper/tolaria/emergency.py`, `trainer` gating | `tests/tolaria/test_emergency_halt.py::{test_l4_halt_on_failed_epochs_streak,test_l4_halt_on_rollback_deadline_exceeded}` |
| Profiler trace emission | `01.1` | `src/esper/tolaria/profiler.py`, `trainer` hooks | `tests/integration/test_profiler_trace.py::test_profiler_emits_chrome_trace` |
| Telemetry emission and publish to Oona | `01.1`, `01.4` | `trainer._emit_telemetry()`, `publish_history()` | `tests/tolaria/test_tolaria_trainer.py::test_tolaria_publish_history_to_oona` |
| Unified LR controller with circuit breakers | `01.3` | — | — |
| Two‑tier rollback (fast + full) | `01.2` | — | — |
| Emergency escalation protocol | `01.2` | — | — |
| Multi‑seed gradient aggregation | `01.1` | `src/esper/tolaria/{aggregation.py,trainer.py}` (registry masks; attribution split; PCGrad) | `tests/tolaria/test_aggregation_attribution.py` |

| PyTorch 2.8 upgrades (compile/AMP/TF32/pinned/foreach) | `01.1` | `trainer.__init__` (compile/AMP/TF32/pin/foreach), `_eager_train_step` | `tests/tolaria/test_tolaria_trainer.py::{test_tolaria_compile_fallback,test_tolaria_amp_metrics_disabled_on_cpu}` |
| Advance α during BLENDING | `01.1` | `trainer._train_single_epoch()` (export + `advance_alpha`) | `tests/tolaria/test_tolaria_trainer.py::test_tolaria_advances_alpha_during_blending` |
