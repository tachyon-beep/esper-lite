# BUG Template

- **Title:** Heuristic blueprint rotation ignores enabled slots/topology and can select invalid actions
- **Context:** Tamiyo heuristic (`src/esper/tamiyo/heuristic.py`) uses a static `blueprint_rotation` list (conv_light, conv_heavy, attention, norm, depthwise) regardless of topology/slot availability. In non-CNN topologies or when certain blueprints aren’t registered for a slot, `getattr(Action, GERMINATE_<BP>)` can fail or return NOOP.
- **Impact:** P1 – heuristic germination can raise `AttributeError` or select dead actions in transformer/other host configs, blocking runs or silently degrading policy behavior.
- **Environment:** Main branch; heuristic runs on transformer tasks or custom hosts without matching blueprints.
- **Reproduction Steps:**
  1. Run heuristic on a transformer topology task (`--task tinystories`) with default `blueprint_rotation`.
  2. `build_action_enum` lacks GERMINATE_* entries for CNN blueprints; getattr raises or maps to NOOP.
- **Expected Behavior:** Blueprint rotation should be derived from available blueprints for the host/topology; heuristic should not attempt germination actions that don’t exist.
- **Observed Behavior:** Static CNN blueprint list is used unconditionally.
- **Hypotheses:** Heuristic config was written for CNN-only; multi-topology support added later without updating rotation.
- **Fix Plan:** Build `blueprint_rotation` from `build_action_enum`/registry for the current topology or validate the config; default rotation per topology (CNN vs transformer).
- **Validation Plan:** Heuristic run on transformer task germinates valid blueprints and doesn’t error; add tests for topology-specific rotations.
- **Status:** Open
- **Links:** `src/esper/tamiyo/heuristic.py` (`HeuristicPolicyConfig.blueprint_rotation`, `_get_next_blueprint`), `esper.kasmina.blueprints`
