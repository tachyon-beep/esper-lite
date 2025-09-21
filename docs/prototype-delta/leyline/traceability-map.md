# Leyline — Traceability Map

| Contract/Enum | Used By | File Examples |
| --- | --- | --- |
| `SystemStatePacket` | Tolaria (produce), Oona (bus), Nissa (ingest) | `src/esper/tolaria/trainer.py`, `src/esper/oona/messaging.py`, `src/esper/nissa/observability.py` |
| `TelemetryPacket` | All subsystems for metrics | `src/esper/core/telemetry.py`, Oona/Nissa/Tamiyo/Tolaria |
| `AdaptationCommand` | Tamiyo (produce), Kasmina/Tolaria (consume) | `src/esper/tamiyo/service.py`, `src/esper/kasmina/seed_manager.py`, `src/esper/tolaria/trainer.py` |
| `FieldReport` | Tamiyo (produce), Simic (ingest) | `src/esper/tamiyo/service.py`, `src/esper/simic/replay.py` |
| `PolicyUpdate` | Simic (produce), Tamiyo (consume) | `src/esper/simic/trainer.py`, `src/esper/tamiyo/service.py` |
| `BusEnvelope` + `BusMessageType` | Oona | `src/esper/oona/messaging.py` |
| `SeedLifecycleStage` | Kasmina lifecycle/telemetry | `src/esper/kasmina/lifecycle.py`, `seed_manager.py`, tests/leyline |

