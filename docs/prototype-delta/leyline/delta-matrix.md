# Leyline — Delta Matrix

Status and evidence per requirement. See rubric in `../rubric.md`.

| Area | Design Source | Expected Behaviour | Prototype Evidence | Status | Severity | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Canonical lifecycle enums | `00-leyline.md` | Full 11‑stage `SeedLifecycleStage` present in schema | `leyline_pb2.SeedLifecycleStage` lacks DORMANT/EMBARGOED/RESETTING/TERMINATED | Missing | Must‑have | Requires schema update; currently some states proxied via app logic. |
| Lifecycle gates | `00-leyline-shared-contracts.md` | `SeedLifecycleGate` (G0–G5) enums and optional `GateEvent` | Not present | Missing | Must‑have | Needed for gate‑level telemetry and alignment. |
| Option B performance | `00-leyline-shared-contracts.md` | Native map fields; <80 µs serialisation; <280 B messages | Used widely (`metrics`, `annotations`) | Implemented | Should‑have | Budget validation tests partial only. |
| Bus envelope | `00-leyline-shared-contracts.md` | `BusEnvelope` wraps payloads with type | Oona uses `BusEnvelope` | Implemented | Must‑have | Verified by tests. |
| Security envelope | `00-leyline.md` | HMAC signature + nonce + freshness window | Oona supports HMAC signing; nonce/freshness not universal | Partially Implemented | Should‑have | Extend across producers; add nonce/timestamp fields if absent. |
| Versioning | `00-leyline.md` | Single `uint32 version` across messages | Present across messages | Implemented | Must‑have | Canonical pattern in repo. |
| Enum canonicalisation tests | `00-leyline.md` | Repo‑wide tests assert enum alignment | `tests/leyline/*` cover some areas | Partially Implemented | Should‑have | Add tests for new lifecycle/gates once schema updated. |

