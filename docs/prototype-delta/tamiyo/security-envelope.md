# Tamiyo — Security Envelope (Signing, Freshness, Nonce)

Purpose
- Ensure all AdaptationCommands are authenticated and fresh so Kasmina accepts them.

Flow
1) Build AdaptationCommand
2) Assign `command_id` (UUID v4) and `issued_at` (now)
3) Serialize command bytes
4) HMAC sign bytes with `ESPER_LEYLINE_SECRET`
5) Store signature in `annotations["signature"]`

Verifier (Kasmina)
- Verifies HMAC; rejects if invalid
- Rejects missing `command_id` or reused command_id (nonce replay)
- Rejects stale `issued_at` (outside freshness window) or future timestamps

Implementation
- TamiyoService holds a `SignatureContext` created from env; fallback: reject with telemetry if missing
- Sign immediately before returning from `evaluate_step`

Acceptance
- Signed commands pass verification; replays/stale fail as expected
- No `command_rejected` due to {missing_signature, invalid_signature, nonce_replayed, stale_command}

Tests
- Unit: sign/verify; replay; stale timestamp; future issued_at

References
- Kasmina verifier: `src/esper/kasmina/security.py`, `src/esper/kasmina/seed_manager.py:1265`

## Checklist (for PR)
- [ ] command_id and issued_at set before signing
- [ ] HMAC signature present in annotations["signature"]
- [ ] Replay and stale timestamps rejected in tests
- [ ] No command_rejected due to signature/freshness in integration
