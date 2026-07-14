# r9 archival record (SEALED) — the K=1 / obs-v3 / shaped evidence base for the 13-round permanence investigation

Date: 2026-07-14. **This is the durable, hash-sealed record of the r9 telemetry that underpins PDR-0069→0079.** r9 becomes non-extendable the moment a K>1 / obs-v4 run happens (different regime), so every finding below was mined *before* that regime change. **Any re-run of an r9 read MUST assert the regime stamp and fail loudly off-regime.**

## Regime stamp (asserted by every read's extractor)
`recurrent_n_epochs=1` (**K=1**) / **obs-v3** / `reward_mode=shaped` / `chunk_length=max_epochs=150` / `per_head_advantage_norm=False` / `max_seeds=3` / γ=0.995 / `cifar_baseline`. Decisions: 1,079,932 (s41), 1,079,950 (s42).

## A4 — Provenance / immutability (tamper-evident)
Path: `telemetry/stage2_on_longdiag/seed{41,42}/telemetry_2026-07-12_161217/events.jsonl`.
| file | SHA-256 | size | mtime |
|---|---|---|---|
| seed41 events.jsonl | `3bfd1f834ba1712837784c12c2811beef3d7e7410a2c6924ef4e8a30e4ad4823` | 6,843,572,929 | 2026-07-13 12:42:58 +1000 |
| seed42 events.jsonl | `35a4171e36b8a616879369cf3fd0b5fdfcb8e1a136452efc0eaadcea6d72e79c` | 6,816,748,352 | 2026-07-13 12:11:46 +1000 |
Read-script SHA-256 manifest (session scratchpad `a4_hashes.txt`): perm_lib 52543c63, extract_perm 92c5161b, extract_heads 2395a96b, read1 a7b75946, read2 c6fb8dc0, read3_pr15 6839389b, read3_v2 93a0d9cd, read4_mechanism d62c7e5b, read_archival 8db3b873.

## A1 — Per-head dead-zone blast radius: THREE distinct pathologies (measured, was inferred)
Caveat: `head_telemetry.{head}_confidence` logs MAX-probability (argmax), not the chosen-action prob — so the direct chosen-at-floor dead-zone is measured only for OP (top-level `action_confidence`); the other 7 heads are inferred from max-prob structure. Both seeds identical.
- **OP — genuine floor-binding dead-zone:** ~43% of real |H|=1 choices floored at 0.15 (the round-8 measure; the only directly-measured head), gradient-censored.
- **`alpha_speed` + `alpha_curve` — COLLAPSED:** max-prob 1.00, one action always, 100% — dead/degenerate heads, no exploration.
- **blueprint — NEAR-UNIFORM:** max-prob 0.13 (7/13 realized), near-random selection — the head barely prefers any blueprint (chosen prob ≈ floor → gradient-weak).
- slot (0.68–0.75, moderate), style (0.26, soft), tempo (0.34, moderately-confident single), alpha_target (0.36–0.39, soft).

## A2 — OP |H|=1 (pinned-simplex) rate over training: CONFIDENCE-TRAP CONFIRMED (round-11 Seam-2, first time)
The OP head's pinned rate RISES monotonically over training — s41 0.195→0.260 (corr +0.37), s42 0.195→0.274 (corr +0.48), plateauing after batch ~150–250, both seeds. As training proceeds the head becomes "confidently" pinned, so the gradient-frozen fraction *grows over time* (~+33% relative). The dead-zone is not static — it deepens.

## A3 — Expressive in RANGE, weak in PREFERENCE (N1 partially refuted)
- `alpha_target` @ SET_ALPHA (n≈33k, enum n=3): real 3-way spread — 0.7: 43%, 0.5: 33–37%, 1.0: 21–24% (not collapsed, not uniform). `alpha_algorithm` ADD 64–74%.
- blueprint @ GERMINATE (n≈89k, enum n=13): 7/13 realized, max-share 18–20% (uniform 8%) — the set is used.
- **Verdict:** the enum action *ranges* are expressive (not clamped to one — refutes the "dominant action executes a random target" collapse concern), but combined with A1 the *selection confidence is weak* (blueprint near-uniform; alpha_target soft). Expressive in range, under-differentiated in preference.

## Bottom line for the floor repair (PDR-0079 coupled bet)
The floor pathology is not one thing: OP floor-binding (deepening over training, A2) + two collapsed alpha heads + a near-uniform blueprint head. The forced-PRUNE destruction (PDR-0079: 55% of good held seeds pruned within ~13 epochs) and the OP floor-binding are the two facets most relevant to the coupled repair. Recorded for the eventual floor-repair design; not decision-gating on its own.
