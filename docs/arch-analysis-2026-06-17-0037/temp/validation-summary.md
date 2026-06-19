# Validation Summary (mandatory multi-subsystem gate)

Each of the 10 domain catalog entries was independently spot-checked by an `analysis-validator` agent reading source. **Result: 3 PASS, 7 WARN, 0 BLOCK** — no major unsupported claims; all WARNs carried specific corrections that have been merged into `02-subsystem-catalog.md` (marked ✎).

| Unit | Status | Issues | Most material correction merged |
|------|:------:|:------:|---------------------------------|
| leyline | WARN | 4 | Class is `LossRewardConfig` not `LossPrimaryRewardConfig`; `PolicyBundle` not `PolicyProtocol`; `_LEGACY_NAMES` is a clean-break rejection guard |
| kasmina | WARN | 3 | Removed non-existent `simic/rewards/types.py` import edge (docstring-only ref); `registry.py`=140 LOC |
| tamiyo | WARN | 2 | Dropped stray "fan-in 76" rebuttal; 116/128-dim figures confirmed correct |
| simic-rl | WARN | 4 | Inbound fixed: removed `policy_protocol.py`/`config.py`; added emitters/epoch_runner/vectorized_trainer; key is `ppo_update_performed` |
| simic-rewards | PASS | 3 | Removed malformed `reward_telemetry.py:861` citation; "9+" vs in-code "7" reconciled |
| simic-training | PASS | 6 | `_run_batch` tuple at L2899-2908; `AnalyticsSnapshotPayload` at telemetry.py:1668; "factory" metaphor reworded |
| tolaria | WARN | 4 | **Commandment renumbering** — Train Anything=C5, Governor=C7; No-Legacy/No-Defensive/Telemetry are CLAUDE.md rules, not ROADMAP commandments; `vectorized_trainer.py` not `vectorized.py` |
| nissa | WARN | 3 | Added 4 missing inbound edges incl. **tolaria→nissa**; console else-branch renders unhandled events (C1 caveat) |
| karn | PASS | 3 | `RewardComponents` at schema.py:1253; largest widget under `widgets/tamiyo/`; function is `copy_snapshot()` |
| support | WARN | 3 | Removed `proof_packet` from inbound (not a caller); alloc-conf citation split |

**Most consequential validation catch:** the tolaria validator detected that the workflow prompt's commandment numbering conflated ROADMAP's Nine Commandments with CLAUDE.md's hard rules. This was reconciled across all deliverables — the compliance *evidence* was sound; only the labels/numbers were corrected. This is the validation gate doing exactly its job.

**Cross-cutting passes (5):** RL soundness (11 findings, 2 high / 4 positive), GPU-first (9, all positive/low — claim CONFIRMED), commandments+rules audit (11, 2 high / 1 positive), tech-debt (12, 2 critical / 5 high), dependency map (19, 17 positive). All findings traced to `04`/`05`/`06`.
