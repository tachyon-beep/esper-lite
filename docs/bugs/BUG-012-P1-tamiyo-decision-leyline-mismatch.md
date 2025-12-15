# BUG Template

- **Title:** TamiyoDecision to_command uses `CommandType.REQUEST_STATE` for WAIT, deviating from Leyline command intents
- **Context:** Tamiyo decisions (`src/esper/tamiyo/decisions.py`) map `WAIT` to `CommandType.REQUEST_STATE` with `target_stage=None`.
- **Impact:** P1 – down-chain consumers expecting semantic WAIT (no-op) may mis-handle it as a state request. If/when commands are wired (Leyline has TODOs), this mapping will be wrong and could trigger unintended operations. Risk grows as command system comes online.
- **Environment:** Main branch; any future use of `AdaptationCommand` produced from heuristic/Tamiyo decisions.
- **Reproduction Steps:** Convert a WAIT decision via `to_command()`; observe `command_type=REQUEST_STATE`.
- **Expected Behavior:** WAIT should map to a neutral/no-op command or a dedicated WAIT command type; REQUEST_STATE implies an active query/operation.
- **Observed Behavior:** WAIT maps to REQUEST_STATE.
- **Hypotheses:** Command system not fully implemented; placeholder mapping chosen without strong contract.
- **Fix Plan:** Align `to_command` with Leyline contract—either introduce a WAIT/NOOP command type or ensure consumers treat REQUEST_STATE as safe noop; document mapping and add tests to lock semantics.
- **Validation Plan:** Add a unit test for WAIT → AdaptationCommand mapping; ensure downstream (when command handling is added) doesn’t act on WAIT.
- **Status:** Open
- **Links:** `src/esper/tamiyo/decisions.py::TamiyoDecision.to_command`, `src/esper/leyline/telemetry.py` command TODOs
