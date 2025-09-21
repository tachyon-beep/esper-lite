# Security — Implementation Roadmap (Leyline‑First)

Goal: consistent, enforced message authenticity across the prototype without tech debt.

| Order | Theme | Key Tasks | Outcome |
| --- | --- | --- | --- |
| 1 | Standard envelope fields | Extend Leyline to include nonce + issued_at (or use existing), and optional signature field in a common metadata block | Canonical fields for replay/freshness |
| 2 | Producer signing | Add HMAC signing to Tamiyo (AdaptationCommand), Tolaria (SystemStatePacket), Simic (PolicyUpdate), and telemetry producers | All control/updates signed |
| 3 | Consumer enforcement | Verify signature, nonce table (TTL), and freshness window at Kasmina/Tolaria/Nissa/Oona boundaries; drop/telemetry on failure | Replay/forgery resistance |
| 4 | Rotation & ops | Support dual‑secret validation; document rotation runbook; expose `security.signature.fail_total` metrics | Operational readiness |
| 5 | Telemetry policy | Decide per‑stream: require signature or allow unsigned; document and enforce | Clear risk posture |

Notes
- Reuse `src/esper/security/signing.py` helpers; wire through code paths.
- Nonce TTL (e.g., 5 min) and freshness (e.g., ±60 s) recommended defaults.

Acceptance Criteria
- All producers sign; all consumers enforce; replay attempts rejected; rotation documented; failure metrics present.

