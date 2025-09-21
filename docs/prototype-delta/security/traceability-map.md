# Security — Traceability Map

| Concern | Implementation | Files |
| --- | --- | --- |
| HMAC signing primitives | Provided as helpers | `src/esper/security/signing.py` |
| Bus envelope signing/verification | Implemented in Oona client | `src/esper/oona/messaging.py` (`_generate_signature`, `_verify_payload`, `consume`) |
| Secret provisioning | Environment variable and example | `.env.example` (`ESPER_LEYLINE_SECRET`) |
| Command path signing | Not implemented | Tamiyo/Tolaria/Kasmina command flows use raw protobufs |
| Replay/freshness | Not implemented | No nonce/timestamp enforcement across producers |
| PolicyUpdate authenticity | Envelope‑level only | `src/esper/simic/trainer.py` publishes via Oona (envelope signed if secret set) |

