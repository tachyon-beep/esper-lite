# Security — Prototype Delta (Signing, Replay Protection)

Executive summary: Security controls in the prototype are partially adopted. Oona supports optional HMAC‑SHA256 signing and verification of Leyline `BusEnvelope` payloads using the shared secret `ESPER_LEYLINE_SECRET`. Elsewhere, command/control paths (e.g., Tamiyo→Kasmina/Tolaria) do not sign messages or enforce nonce/freshness checks. Telemetry packets are generally unsigned. The goal is a consistent, Leyline‑first security envelope: HMAC + nonce + freshness window enforced across all producers/consumers, with clear operational guidance for key management and rotation.

Documents in this folder:
- `delta-matrix.md` — adoption status by area
- `traceability-map.md` — where signing is implemented/used today
- `implementation-roadmap.md` — concrete steps to reach consistent enforcement without tech debt

References:
- Shared secret var: `ESPER_LEYLINE_SECRET` (see `.env.example`)
- Code: `src/esper/security/signing.py`, `src/esper/oona/messaging.py`

