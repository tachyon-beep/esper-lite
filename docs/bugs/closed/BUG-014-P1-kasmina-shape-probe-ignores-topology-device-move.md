# BUG Template

- **Title:** Shape probe cache ignores topology/device moves, can return wrong tensor shape
- **Context:** Kasmina SeedSlot `_shape_probe_cache` (`src/esper/kasmina/slot.py`) caches probe tensors keyed only by topology and device at creation; `to()` clears cache only when `self.device` changes. After a `.to()` that updates device, new probes are created, but if topology changes (e.g., switching host types or reusing slots) the cache may serve a tensor with mismatched shape/channels.
- **Impact:** P1 – incorrect shape probes can bypass validation or raise runtime errors when germinating seeds on reused slots with different topologies/channels; potential torch.compile specialization errors.
- **Environment:** Reusing SeedSlot instances across models/topologies, or checkpoint load with changed topology/channels.
- **Reproduction Steps:** Reuse a slot with cached CNN probe on a transformer host (same device); `_shape_probe_cache` returns the old probe because device matches, causing shape mismatch in validation.
- **Expected Behavior:** Shape probe cache should key on topology and channel shape (and device), or clear on topology/channel changes; probes must match current slot configuration.
- **Observed Behavior:** Cache key only uses topology string and device equality; channel count changes don’t clear cache; topology changes on same device reuse stale probe.
- **Hypotheses:** Cache was added for perf; invalidation not updated for multi-topology/channel reuse.
- **Fix Plan:** Include channels/topology in cache key; clear cache on germinate/host change; add assertions.
- **Validation Plan:** Add test reusing a slot across topologies/channels to ensure probes regenerate with correct shape.
- **Status:** Closed (Fixed)
- **Resolution:** Fixed by updating `_shape_probe_cache` key to `(topology, channels)` tuple in `SeedSlot`. This ensures unique probes for different channel configurations even on the same device.
- **Links:** `src/esper/kasmina/slot.py::_shape_probe_cache`, `_get_shape_probe`, `to()`
