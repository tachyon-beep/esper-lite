# BSDS‑Lite (Speculative Prototype)

BSDS‑Lite is a minimal, JSON‑encoded blueprint safety sheet used to unblock Tamiyo’s risk‑aware decision making in the prototype. Canonical contracts are provided in Leyline from day 1; this JSON mirrors the `BSDS` message and is carried via Urza `extras` for transport until direct protobuf persistence is wired end‑to‑end.

Quick Start
- Produce a BSDS‑Lite dict per blueprint (see `schema.md`).
- Attach to Urza via `extras["bsds"]` when saving/upserting a record.
- Tamiyo consumes the block and annotates + gates decisions accordingly.

Notes
- This spec evolves toward the full Leyline `BSDS` message; keep fields stable.
- Add only monotonic, optional fields; never break existing consumers.

Prototype Extensions
- When signing is enabled, Urabrask attaches a signature block alongside the BSDS mirror in Urza extras:
  - `extras["bsds_sig"] = { "algo": "HMAC-SHA256", "sig": <base64>, "prev_sig": <base64|empty>, "issued_at": <rfc3339> }`
- The JSON schema for BSDS‑Lite intentionally does not include the signature block; treat it as prototype metadata carried next to the BSDS payload.
