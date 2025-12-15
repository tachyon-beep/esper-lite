# JANK Template

- **Title:** Blend alpha schedules may live on wrong device after slot/device moves
- **Category:** correctness-risk / maintainability
- **Symptoms:** `start_blending` moves gated blend modules to the seed’s device once, but subsequent device moves (`MorphogeneticModel.to()`) or checkpoint loads don’t re-sync `alpha_schedule` to the slot’s current device. Linear/sigmoid schedules are pure tensors; gated blend is an nn.Module that can remain on CPU.
- **Impact:** Medium – mixing CPU alpha_schedule with GPU seeds can trigger host-device errors or silent CPU fallback; alpha evolution may break under torch.compile/streams.
- **Triggers:** Calling `.to(device)` after germination/start_blending, or loading checkpoints to a different device.
- **Root-Cause Hypothesis:** Device sync only occurs at start_blending; no hooks on `to()`/load_state to move alpha_schedule.
- **Remediation Options:**
  - A) Override `MorphogeneticModel.to()`/SeedSlot hooks to move alpha_schedule to the slot device when present.
  - B) Store device metadata and rebind on load.
- **Validation Plan:** Test moving a model with active blend schedule from CPU→GPU and ensure blending runs without device mismatch.
- **Status:** Open
- **Links:** `src/esper/kasmina/slot.py::start_blending`, device move handling
