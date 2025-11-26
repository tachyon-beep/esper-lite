# Esper-Lite Capability Matrix

This matrix translates the HLD and detailed design documents into implementation-ready guidance. Each subsystem lists its primary responsibilities, canonical references, and the remaining tasks or open questions to resolve before coding begins.

## Core Functional Subsystems

### Tolaria (Training Orchestrator)
| Aspect | Details |
| --- | --- |
| Responsibilities | Epoch-driven training loop, end-of-epoch Tamiyo handshake (<18 ms), checkpoint/rollback orchestration, rollback safety nets. |
| Key References | `docs/design/detailed_design/01-tolaria.md` and sub-docs `01.1`–`01.4`; HLD “Training & Execution Layer”. |
| Implementation Notes & Gaps | Decide host-model skeleton for prototype (resnet-lite vs transformer-lite); finalize checkpoint storage location and rollback dataset; confirm rollback trigger matrix; create contract tests for `SystemStatePacket` emission. Reference `docs/design/detailed_design/old/01-tolaria.md` for the authoritative control-loop (epoch handshake, WAL, rollback stack). |

### Kasmina (Execution Layer Controller)
| Aspect | Details |
| --- | --- |
| Responsibilities | Seed lifecycle management, kernel grafting, gradient isolation enforcement, telemetry publication, KD safeguards (disabled by default). |
| Key References | `docs/design/detailed_design/02-kasmina.md` and `02.1`–`02.5`; HLD “Training & Execution Layer”. |
| Implementation Notes & Gaps | Specify seed registration API surface for Tolaria integration; confirm memory budgets for prototype GPUs; implement telemetry schema sample payloads; define fallback behavior for KD-disabled mode. Eleven-stage lifecycle remains mandatory (see `docs/design/detailed_design/old/02-kasmina.md`). |

### Tamiyo (Strategic Controller & Telemetry Hub)
| Aspect | Details |
| --- | --- |
| Responsibilities | GNN policy inference (<45 ms), risk governance, adaptation command issuance, telemetry aggregation, field report generation. |
| Key References | `docs/design/detailed_design/03-tamiyo.md` and `03.1`–`03.4`; HLD “Strategic Control Layer”. |
| Implementation Notes & Gaps | Choose concrete graph schema and feature extractors from Tolaria/Kasmina; set initial risk thresholds and conservative-mode configs; define field-report persistence path; identify hardware requirements for inference (GPU vs CPU). `old/03-tamiyo.md` details risk gates, telemetry aggregation, and field-report lifecycle. |

### Simic (Offline Policy Trainer)
| Aspect | Details |
| --- | --- |
| Responsibilities | Ingest Tamiyo field reports, maintain graph replay buffer, run IMPALA/PPO training, validate and publish policy checkpoints. |
| Key References | `docs/design/detailed_design/04-simic.md` and `04.1`–`04.2`; HLD “Strategic Control Layer”. |
| Implementation Notes & Gaps | Decide replay data source (synthetic vs captured runs); configure single-node PPO + LoRA training resources (IMPALA optional for later); script validation pipeline (chaos/property tests) at prototype scale; define policy checkpoint exchange format. Follow replay/validation rules in `old/04-simic.md`. |

### Urza (Central Blueprint Library)
| Aspect | Details |
| --- | --- |
| Responsibilities | Store BlueprintIR metadata and compiled kernel artifacts, serve low-latency queries (<10 ms p50), manage WAL durability and local caching. |
| Key References | `docs/design/detailed_design/08-urza.md` and `08.1`; HLD “Blueprint Management Layer”. |
| Implementation Notes & Gaps | Use SQLite (or Postgres) plus local filesystem storage; provide in-process cache (no Redis tier); seed catalog with initial blueprint set and versioning. Honour catalogue governance in `old/08-urza.md`. |

### Tezzeret (Compilation Engine)
| Aspect | Details |
| --- | --- |
| Responsibilities | Startup compilation of predefined blueprints, WAL-backed job recovery, simple retry handling. |
| Key References | `docs/design/detailed_design/06-tezzeret.md` and `06.1`; HLD “Blueprint Management Layer”. |
| Implementation Notes & Gaps | Implement single Standard pipeline with basic retry logic; limit concurrency to 1 job; prepare torch.compile flags for PyTorch 2.8; mock Urza API for testing. See `old/06-tezzeret.md` for WAL and telemetry requirements. |

### Karn (Static Blueprint Service)
| Aspect | Details |
| --- | --- |
| Responsibilities | Expose 50-template tiered blueprint catalog, enforce safety approvals, respond to Tamiyo/Kasmina queries, log usage telemetry. |
| Key References | `docs/design/detailed_design/05-karn.md` and `05.1`; HLD “Blueprint Management Layer”. |
| Implementation Notes & Gaps | Produce actual template definitions (metadata + allowed parameters); implement basic request/response flow with logging (no breakers); script telemetry emission aligned with Leyline metrics; confirm integration auth requirements. Safety tiers/fallback behaviour per `old/05-karn.md`. |

## Shared Infrastructure Services

### Leyline (Shared Contracts)
| Aspect | Details |
| --- | --- |
| Responsibilities | Provide protobuf schemas (`SystemStatePacket`, `AdaptationCommand`, `FieldReport`, `EventEnvelope`, etc.), enforce schema governance, guarantee serialization latency/size budgets. |
| Key References | `docs/design/detailed_design/00-leyline.md` and `00.1`–`00.3`; HLD “Infrastructure Services”. |
| Implementation Notes & Gaps | Establish dedicated repo or module for `.proto` files; automate serialization benchmarks in CI; publish schema versioning policy; document generated code packaging for Python services. |

### Oona (Message Bus)
| Aspect | Details |
| --- | --- |
| Responsibilities | Redis Streams-based publish/subscribe, NORMAL/EMERGENCY priority routing, at-least-once delivery, breaker-protected operations, TTL cleanup. |
| Key References | `docs/design/detailed_design/09-oona.md` and `09.1`; HLD “Infrastructure Services”. |
| Implementation Notes & Gaps | Define concrete stream names and consumer groups per subsystem; provision Redis with required TTL/maxlen settings; implement health endpoint & metrics exporter; script conservative-mode scenarios for testing. Messaging guarantees described in `old/09-oona.md` remain in force. |

### Nissa (Observability Stack)
| Aspect | Details |
| --- | --- |
| Responsibilities | Ingest telemetry via Oona, store in Prometheus/Elasticsearch, evaluate alert/SLO rules, expose mission-control API & dashboards. |
| Key References | `docs/design/detailed_design/10-nissa.md` and `10.1`–`10.3`; HLD “Infrastructure Services”. |
| Implementation Notes & Gaps | Stand up Prometheus/Elasticsearch instances (single-node); define dashboard layouts and alert routing (Slack/email stubs); ensure telemetry packet schemas are concrete; create runbook for conservative-mode and breaker incidents. Align with observability requirements in `old/10-nissa.md`. |

## Cross-Cutting Items
- **Telemetry Topic & Metric Catalog:** Compile definitive mapping of Leyline `TelemetryPacket` categories to Oona streams and Nissa dashboards.
- **Security & Secrets Handling:** Decide prototype-level auth (shared tokens, signed messages) consistent with detailed designs.
- **Deployment Topology:** Choose target orchestration (docker-compose vs Kubernetes) and document port/service dependencies across subsystems.
- **Testing Strategy:** Outline integration test matrix covering control-loop happy path, failure/breaker scenarios, and schema compatibility checks.

This matrix should be reviewed alongside the prototype charter. Any open items should convert into backlog stories before implementation begins.
