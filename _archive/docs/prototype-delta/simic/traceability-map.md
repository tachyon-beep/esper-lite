# Simic — Traceability Map

| Design Assertion | Source | Implementation | Tests |
| --- | --- | --- | --- |
| Ingest FieldReports and build experiences | `04.2-simic-experience-replay.md` | `FieldReportReplayBuffer.add()` / `.extend()` / `.ingest_from_oona()` | `tests/simic/test_replay.py` (buffer behaviours) |
| TTL pruning and capacity | `04.2` | `_prune()` and `_enforce_capacity()` | `tests/simic/test_replay.py::test_buffer_enforces_ttl/capacity` |
| Sample batches for training | `04.2` | `sample_batch()` returns tensors | `tests/simic/test_replay.py::test_sample_batch_returns_tensors` |
| PPO/IL training loop | `04.1` | `SimicTrainer.run_training()` | `tests/simic/test_trainer.py::test_trainer_updates_policy_parameters` |
| Validation gates publishing | `04-simic.md` | `create_policy_update()` blocks on failed validation | `tests/simic/test_trainer.py::test_trainer_blocks_policy_update_on_validation_failure` |
| Telemetry of training metrics | `04-simic.md` | `build_metrics_packet()` | `tests/simic/test_trainer.py` (metric presence) |
| Version publish via Oona | `04-simic.md` | `publish_policy_updates()` | Covered indirectly in Tamiyo tests; Simic tests assert update objects |
| IMPALA/V‑trace, breakers, prioritised replay | `04.1`, `04.2` | — | — |

