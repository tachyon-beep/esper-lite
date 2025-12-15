# BUG Template

- **Title:** Gradient telemetry segfaults in SeedSlot.capture_gradient_telemetry
- **Context:** Kasmina isolation monitor (`SeedSlot.capture_gradient_telemetry`) on CNN host during incubator (TRAINING) stage
- **Impact:** Blocker — process crashes (exit 139) as soon as gradient telemetry runs, aborting training/rollouts
- **Environment:** HEAD @ workspace, Python 3.11, CPU (torch default); repro below
- **Reproduction Steps:**
  1. Run:
     ```bash
     PYTHONPATH=src python - <<'PY'
     import torch
     from esper.kasmina.host import CNNHost, MorphogeneticModel
     from esper.leyline import SeedStage

     x = torch.randn(4, 3, 32, 32)
     labels = torch.randint(0, 10, (4,))
     model = MorphogeneticModel(CNNHost(num_classes=10), device="cpu", slots=["mid"])
     model.germinate_seed("norm", "seed-1", slot="mid")
     slot = model.seed_slots["mid"]
     slot.state.stage = SeedStage.TRAINING
     slot.isolate_gradients = True

     out = model(x)
     loss = torch.nn.functional.cross_entropy(out, labels)
     loss.backward()
     slot.capture_gradient_telemetry()  # <-- crashes here
     print("done", slot.state.metrics.isolation_violations)
     PY
     ```
  2. Process exits with `Segmentation fault (core dumped)` before printing.
- **Expected Behavior:** Gradient telemetry should return cleanly, updating `seed_gradient_norm_ratio` and `isolation_violations` without crashing.
- **Observed Behavior:** Hard segfault during `_foreach_norm` inside `capture_gradient_telemetry`, aborting the process.
- **Logs/Telemetry:** Exit 139; no Python traceback (native crash). Code path: `src/esper/kasmina/slot.py` inside `capture_gradient_telemetry() -> isolation_monitor.check_isolation() -> torch._foreach_norm`.
- **Hypotheses:** Interaction between `torch._foreach_norm` on CPU tensors and the isolated seed/host graphs; possibly missing grad tensors or stale parameter refs causing UB inside foreach kernels.
- **Fix Plan:** Unassigned — investigate isolation_monitor internals and replace `_foreach_norm` with a safe fallback if needed; add guard rails for empty/None grads.
- **Validation Plan:** Unit test mirroring the repro to ensure telemetry returns without crashing and updates metrics; run under ASAN build of PyTorch if possible.
- **Status:** Open
- **Links:** `src/esper/kasmina/slot.py` (`capture_gradient_telemetry`, isolation monitor usage)
