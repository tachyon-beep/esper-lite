# BUG-012: TamiyoDecision command mapping mismatch (resolved)

- **Title:** TamiyoDecision to_command uses `CommandType.REQUEST_STATE` for WAIT, deviating from Leyline command intents
- **Context:** Tamiyo decisions (`src/esper/tamiyo/decisions.py`) map `WAIT` to `CommandType.REQUEST_STATE` with `target_stage=None`.
- **Impact:** P1 – down-chain consumers expecting semantic WAIT (no-op) may mis-handle it as a state request. If/when commands are wired (Leyline has TODOs), this mapping will be wrong and could trigger unintended operations. Risk grows as command system comes online.
- **Environment:** Main branch; any future use of `AdaptationCommand` produced from heuristic/Tamiyo decisions.
- **Reproduction Steps:** Convert a WAIT decision via `to_command()`; observe `command_type=REQUEST_STATE`.
- **Expected Behavior:** WAIT should map to a neutral/no-op command or a dedicated WAIT command type; REQUEST_STATE implies an active query/operation.
- **Observed Behavior:** WAIT maps to REQUEST_STATE.
- **Hypotheses:** Command system not fully implemented; placeholder mapping chosen without strong contract.
- **Fix Plan:** ~~Align `to_command` with Leyline contract—either introduce a WAIT/NOOP command type or ensure consumers treat REQUEST_STATE as safe noop; document mapping and add tests to lock semantics.~~
- **Validation Plan:** ~~Add a unit test for WAIT → AdaptationCommand mapping; ensure downstream (when command handling is added) doesn't act on WAIT.~~
- **Status:** Closed (Resolved by removal)
- **Links:** `src/esper/tamiyo/decisions.py::TamiyoDecision.to_command`, `src/esper/leyline/telemetry.py` command TODOs

## Resolution (2025-12-17)

**Root Cause Analysis:** The entire command system (`to_command()`, `AdaptationCommand`, `CommandType`, `RiskLevel`) was speculative infrastructure that was never wired up. Git history shows it was added 3 weeks ago but the actual execution path bypasses it entirely:

- **Actual flow:** `TamiyoDecision.action` → `FactoredAction` → direct Kasmina method calls (`germinate_seed()`, `advance_stage()`, `cull_seed()`)
- **Dead code flow:** `TamiyoDecision.to_command()` → `AdaptationCommand` → dispatcher (never built)

**Resolution:** Instead of fixing the bug in dead code, the entire command system was removed:
- ~100 lines production code removed
- ~600 lines test code removed (including property tests, unit tests)
- 9 files affected

**Files Modified:**
- `src/esper/tamiyo/decisions.py` - removed `to_command()` method
- `src/esper/leyline/schemas.py` - removed `AdaptationCommand` class
- `src/esper/leyline/stages.py` - removed `CommandType`, `RiskLevel` enums
- `src/esper/leyline/telemetry.py` - removed `COMMAND_*` event types
- `src/esper/leyline/__init__.py` - removed exports
- `tests/tamiyo/properties/test_command_properties.py` - deleted entirely
- `tests/tamiyo/test_decisions_unit.py` - removed command conversion tests
- `tests/tamiyo/properties/test_tamiyo_properties.py` - removed command property tests
- `tests/integration/test_tamiyo_kasmina.py` - removed `to_command()` tests
- `tests/leyline/test_leyline.py` - removed `TestAdaptationCommand` class

**Rationale:** Classic case of speculative infrastructure. The code was designed before knowing if it would be needed, sat unused for 3 weeks, and the simpler direct execution pattern works fine. Follows "No Legacy Code Policy" - delete completely rather than maintain dead code.
