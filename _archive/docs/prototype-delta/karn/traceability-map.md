# Karn — Traceability Map (Phase‑1)

| Design Assertion | Source | Implementation | Tests |
| --- | --- | --- | --- |
| Leyline‑backed descriptors for templates | `05-karn.md` | `src/esper/karn/templates.py`, `catalog.py` | `tests/karn/test_catalog.py` |
| Parameter bounds validation | `05.1` | `catalog._validate_parameters` and `validate_request` | `tests/integration/test_blueprint_pipeline_integration.py` (through pipeline) |
| Tier filtering and conservative selection | `05.1` | `catalog.choose_template()` | `tests/karn/test_catalog.py` |
| Blueprint request → Urza pipeline | `05.1` | `src/esper/urza/pipeline.py` | `tests/urza/test_pipeline.py` |
| Circuit breaker/telemetry/TTL | `05.1` | — | — |

