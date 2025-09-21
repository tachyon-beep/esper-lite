# Leyline — Prototype Delta (Canonical Contracts)

Executive summary: Leyline is the single authoritative source of truth for all cross‑subsystem data classes and enums. The prototype uses generated bindings (`leyline_pb2`) consistently for `SystemStatePacket`, `TelemetryPacket`, `AdaptationCommand`, `FieldReport`, `PolicyUpdate`, `BusEnvelope`, and enums (e.g., `SeedLifecycleStage`, `BusMessageType`). The key delta to close is aligning the lifecycle to the full 11 stages (with gates G0–G5) as canonical enums in the schema, eliminating any need for local overlays or mappings. Message budgets (Option B: native maps) are respected in most paths; HMAC signing/nonce/freshness is partially adopted (Oona implements optional HMAC; other producers/consumers vary).

Documents in this folder:
- `delta-matrix.md` — requirement‑by‑requirement status with evidence
- `traceability-map.md` — where contracts are used across subsystems
- `implementation-roadmap.md` — single, batched schema change + adoption plan (no tech debt)

Design sources:
- `docs/design/detailed_design/00-leyline.md`
- `docs/design/detailed_design/00-leyline-shared-contracts.md`

Implementation evidence (primary):
- Generated: `src/esper/leyline/_generated/leyline_pb2.py`
- Widespread usage in: Kasmina, Tolaria, Tamiyo, Simic, Oona, Nissa, Urza, Tezzeret, Karn

