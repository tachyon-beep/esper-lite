# Tamiyo — Traceability Map

Mapping of design assertions to implementation artefacts and tests.

| Design Assertion | Source | Implementation | Tests |
| --- | --- | --- | --- |
| Tamiyo evaluates SystemStatePacket and returns AdaptationCommand | `03.4-tamiyo-integration-contracts.md` | `service.evaluate_epoch()`; `policy.select_action()` | `tests/tamiyo/test_service.py::test_tamiyo_service_generates_command` |
| Inference budget (<45 ms) | `03-tamiyo.md` | Timing in `evaluate_epoch()` and telemetry metric | `tests/tamiyo/test_service.py` (assert ≤45 ms) |
| Risk gating and conservative mode | `03.3-tamiyo-risk-modeling.md` | Loss‑spike check and blueprint risk; conservative flag | `tests/tamiyo/test_service.py::test_conservative_mode_overrides_directive` |
| Field report generation and durable store | `03-tamiyo.md` | `generate_field_report()`; WAL with fsync | `tests/tamiyo/test_service.py::test_field_report_generation` |
| Oona publish/consume for telemetry and policy updates | `03.4` | `publish_history()`, `consume_policy_updates()` | `tests/tamiyo/test_service.py` publish/consume cases |
| Blueprint metadata integration (Urza) | `03.4` | `_resolve_blueprint_info()` | `tests/tamiyo/test_tamiyo_annotations_include_blueprint_metadata` |
| PPO/IMPALA learning + replay | `03.2` | — | — |
| Circuit breakers, quotas, deadlines | `03.3`, `03.4` | — | — |

