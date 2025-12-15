# BUG Template

- **Title:** Kasmina shallow CNN slots crash/skip due to hardcoded segment mapping
- **Context:** Kasmina CNNHost fallback segment names (`block0`, `block1`, …) and MorphogeneticModel slot ordering when `n_blocks < 3`
- **Impact:** High — shallow CNN hosts either throw at segment boundaries or silently ignore configured slots, blocking seed usage for small models and making experiments misreport slot coverage.
- **Environment:** HEAD @ workspace, Python 3.11, CPU; reproducible with `PYTHONPATH=src python` scripts below.
- **Reproduction Steps:**
  1. Trigger the crash on fallback segment names:
     ```bash
     PYTHONPATH=src python - <<'PY'
     import torch
     from esper.kasmina.host import CNNHost

     host = CNNHost(n_blocks=2)
     x = torch.randn(1, 3, 32, 32)
     host.forward_to_segment("block0", x)
     PY
     ```
     → `KeyError: 'block0'` from `segment_to_block` in `CNNHost.forward_to_segment`.
  2. Observe silent seed bypass in MorphogeneticModel with a fallback slot:
     ```bash
     PYTHONPATH=src python - <<'PY'
     import torch
     from esper.kasmina.host import CNNHost, MorphogeneticModel

     x = torch.randn(1, 3, 32, 32)
     host = CNNHost(n_blocks=2)
     baseline = host(x)
     model = MorphogeneticModel(host, device="cpu", slots=["block0"])
     out = model(x)
     print("outputs equal:", torch.allclose(out, baseline))
     PY
     ```
     → prints `outputs equal: True` because `_slot_order` filters to `["early", "mid", "late"]`, so the configured slot is never executed.
- **Expected Behavior:** Fallback segment ids exposed via `segment_channels` should be routable end-to-end (or rejected up front) and MorphogeneticModel should honor configured slot ids instead of silently dropping them.
- **Observed Behavior:** `forward_to_segment`/`forward_from_segment` KeyError on fallback ids; MorphogeneticModel silently behaves like a bare host when slots are not `early/mid/late`.
- **Logs/Telemetry:** KeyError above; silent no-op shown by identical baseline/model outputs.
- **Hypotheses:** `segment_to_block` mapping in `CNNHost.forward_to_segment`/`forward_from_segment` is hardcoded to early/mid/late, and `_slot_order` in `MorphogeneticModel` is fixed to canonical slots so any non-canonical id is ignored even if `segment_channels` exposes it.
- **Fix Plan:** Unassigned — derive segment/block mapping from `host.segment_channels`, fail fast on unsupported slot ids, and build `_active_slots` from the host’s available segments instead of a fixed `["early", "mid", "late"]` list. Add coverage for shallow CNNs.
- **Validation Plan:** Unit tests for `CNNHost(n_blocks=2)` exercising `forward_to_segment`/`forward_from_segment` with `block*` ids, and a MorphogeneticModel test that asserts a `block0` slot actually runs (output differs from baseline when alpha>0) instead of being filtered out.
- **Status:** Open
- **Links:** `src/esper/kasmina/host.py` (segment mapping), `src/esper/kasmina/host.py` MorphogeneticModel `_slot_order`
