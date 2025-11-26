# Tamiyo — Decision Taxonomy (Reasons, Severity, Routing)

Purpose
- Canonicalize decision reasons, their severity, and expected routing (priority) for telemetry.

Reasons (examples)
- `bp_quarantine` → CRITICAL (emergency)
- `rollback_deadline` → CRITICAL (emergency)
- `loss_spike` → HIGH
- `isolation_violation` → HIGH
- `hook_budget` → HIGH
- `timeout_inference` → HIGH
- `timeout_urza` → HIGH
- `conservative_entered` → INFO/HIGH on entry
- `conservative_exited` → INFO
- `policy_update_applied` → INFO
- `policy_update_rejected` → WARNING

Routing
- CRITICAL/HIGH map to Oona emergency stream; include `priority` indicator in packet

Acceptance
- Every action carries a reason code (annotation + telemetry event)
- Packet priorities align with severity

Tests
- Assert severity→priority mapping and Oona routing

## Checklist (for PR)
- [ ] Each decision includes a reason annotation
- [ ] Severity→priority mapping correct
- [ ] Oona routes HIGH/CRITICAL to emergency in tests
