# JANK Template

- **Title:** Blend alpha schedules may live on wrong device after slot/device moves
- **Category:** correctness-risk / maintainability
- **Symptoms:** Earlier blending schedules had device affinity and could drift out of sync across device moves/checkpoint loads (especially for the gated blend network).
- **Impact:** Medium – mixing CPU alpha_schedule with GPU seeds can trigger host-device errors or silent CPU fallback; alpha evolution may break under torch.compile/streams.
- **Triggers:** Calling `.to(device)` after germination/start_blending, or loading checkpoints to a different device.
- **Root-Cause Hypothesis:** Schedules were not consistently treated as registered submodules with device-aware alpha tensor generation.
- **Remediation Options:**
  - A) Override `MorphogeneticModel.to()`/SeedSlot hooks to move alpha_schedule to the slot device when present.
  - B) Store device metadata and rebind on load.
- **Validation Plan:** Test moving a model with active blend schedule from CPU→GPU and ensure blending runs without device mismatch.
- **Status:** Closed (Resolved by refactor)
- **Resolution:** Blend schedules are now `nn.Module` instances (`BlendAlgorithm`) and are registered as submodules when assigned to `SeedSlot.alpha_schedule`, so `model.to(...)` moves gated schedule parameters automatically. For linear/sigmoid, alpha tensors are generated on-demand on the input tensor’s device/dtype (thread-local cached), avoiding stale-device issues after moves.
- **Links:** `src/esper/kasmina/blending.py` (`BlendAlgorithm`, `LinearBlend`, `SigmoidBlend`, `GatedBlend`), `src/esper/kasmina/slot.py` (`start_blending`, `set_extra_state`)
