# Security — Delta Matrix

Status and evidence per requirement. See rubric in `../rubric.md`.

| Area | Expected Behaviour | Prototype Evidence | Status | Severity | Notes |
| --- | --- | --- | --- | --- | --- |
| HMAC signing of bus envelopes | All published bus payloads signed when secret configured | `OonaClient._generate_signature()` signs, `_verify_payload()` verifies | Implemented (Oona only) | Must‑have | Enforcement optional; only Oona path covered. |
| Signature enforcement on consume | Reject unsigned/invalid bus messages when secret configured | `OonaClient.consume()` verifies and drops invalid payloads | Partially Implemented | Must‑have | Works for Oona; other consumers bypass Oona. |
| Nonce and freshness window | Include per‑message nonce + timestamp; reject replays/stale | — | Missing | Must‑have | Not present in Oona or producers. |
| Command path signing | Tamiyo→Kasmina/Tolaria commands signed; verified before action | — | Missing | Must‑have | Add signing at producer and verification at consumers. |
| PolicyUpdate authenticity | Simic→Tamiyo updates authenticated | Oona signs envelope only | Partially Implemented | Should‑have | End‑to‑end signature inside payload recommended. |
| Telemetry authenticity | Tamiyo/Tolaria/Kasmina telemetry signed (optional) | Oona envelope signature | Partially Implemented | Nice‑to‑have | Envelope may suffice; can keep unsigned if low risk. |
| Secret management & rotation | Document `ESPER_LEYLINE_SECRET`, rotation plan, disable unsigned | `.env.example` has var; no rotation docs | Missing | Should‑have | Add ops guidance; support dual‑secret rotation. |

