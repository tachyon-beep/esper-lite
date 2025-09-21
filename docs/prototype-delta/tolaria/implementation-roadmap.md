# Tolaria — Implementation Roadmap (Closing the Delta)

Goal: close the gap to the design without tech debt, keeping Leyline as the single source of truth.

| Order | Theme | Key Tasks | Outcome |
| --- | --- | --- | --- |
| 1 | End‑of‑epoch enforcement | Add strict ≤18 ms guard with circuit breaker; on breach enter conservative mode and skip non‑essential work | Predictable epoch boundary behaviour |
| 2 | Circuit breakers + conservative mode | Introduce breaker utility; wrap timing, integrity checks; degrade features (e.g., skip Tamiyo on repeated breaches) | Robustness under stress |
| 3 | Unified LR controller | Implement central LR authority; register optimiser groups; integrity monitors; rebuild API | Single LR truth, safer optimiser changes |
| 4 | Dynamic optimiser manager | Rebuild with momentum preservation; LR‑group registration; telemetry of changes | Safe hot‑swaps |
| 5 | Multi‑seed gradient aggregation | State‑aware weighting; stabilisation and optional conflict handling (e.g., PCGrad) | Proper host/seed blending in training |
| 6 | Two‑tier rollback | Fast (≤500 ms) rollback with LRU cache + shared‑memory signalling; full (≤12 s) rollback orchestrator | Recovery guarantees match spec |
| 7 | Emergency protocol | Four‑level escalation; shared‑memory broadcast; deadline handling | Consistent incident response |
| 8 | Telemetry priorities | CRITICAL emergency bypass; timing/gate events; health semantics | Operator‑friendly observability |
| 9 | Profiling harness | Integrate profiler hooks; optional Chrome traces for epoch/hook timings | Clear performance visibility |

Status update:
- Steps 1–2: Implemented in `src/esper/tolaria/trainer.py` (budgets, breaker, conservative mode, timeouts, telemetry).
- Step 9: Partially implemented via external harness `scripts/profile_tolaria.py`; optional in‑trainer hooks remain.
- Steps 3–8: Outstanding.

Notes:
- All lifecycle/contract artefacts must use Leyline enums & messages directly; do not create local mirrors.
- Coordinate changes with Kasmina/Tamiyo where control‑loop contracts touch (AdaptationCommand handling, rollback orchestration).
