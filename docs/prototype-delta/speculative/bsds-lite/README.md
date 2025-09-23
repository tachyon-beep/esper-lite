# BSDS‑Lite (Speculative Prototype)

BSDS‑Lite is a minimal, JSON‑encoded blueprint safety sheet used to unblock Tamiyo’s risk‑aware decision making in the prototype. Canonical contracts are provided in Leyline from day 1; this JSON mirrors the `BSDS` message and is carried via Urza `extras` for transport until direct protobuf persistence is wired end‑to‑end.

Quick Start
- Produce a BSDS‑Lite dict per blueprint (see `schema.md`).
- Attach to Urza via `extras["bsds"]` when saving/upserting a record.
- Tamiyo consumes the block and annotates + gates decisions accordingly.

Notes
- This spec evolves toward the full Leyline `BSDS` message; keep fields stable.
- Add only monotonic, optional fields; never break existing consumers.
