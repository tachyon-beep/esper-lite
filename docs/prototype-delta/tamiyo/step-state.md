# Tamiyo — Step‑Level State Payload (Reuse SystemStatePacket)

Purpose
- Define the minimal, stable fields Tolaria provides to Tamiyo at step cadence without introducing new contracts.

Packet: `SystemStatePacket`
- `current_epoch`: epoch index
- `global_step`: true step index (set by trainer)
- `training_metrics`: include a small subset per step:
  - `loss` (batch average), `gradient_norm`, `samples_per_s` (if available), `hook_latency_ms` (from previous step)
- `seed_states`: optional export via Kasmina for BLENDING/alpha/etc.

Notes
- Do not rely on epoch‑only fields in step decisions
- Keep payload lean to minimize overhead

Acceptance
- Tamiyo `evaluate_step` consumes only documented fields; trainer fills them consistently

## Checklist (for PR)
- [ ] Trainer populates minimal metrics each step
- [ ] Tamiyo evaluate_step does not rely on undefined fields
