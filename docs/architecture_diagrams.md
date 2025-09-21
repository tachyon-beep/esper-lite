# Esper‑Lite Architecture Diagrams

This document contains Mermaid diagrams that illustrate major integration points and end‑to‑end flows. View directly on GitHub or paste into any Mermaid‑enabled viewer.

## 1) System Context & Integrations

```mermaid
flowchart TD
  %% Subsystems
  TLR[Tolaria\ntraining loop]
  TMY[Tamiyo\npolicy + risk]
  KSM[Kasmina\nlifecycle + kernel]
  KRN[Karn\nblueprint catalog]
  TEZ[Tezzeret\ncompiler]
  URZ[Urza Library\nartifacts + runtime]
  ONA[Oona\nRedis Streams]
  SMC[Simic\noffline trainer]
  NSS[Nissa\nobservability]
  WTH[Weatherlight\nsupervisor]

  %% Runtime control loop
  TLR -- SystemStatePacket --> TMY
  TMY -- AdaptationCommand --> KSM
  KSM -- SeedState/Alpha (optional) --> TLR

  %% Kernel fetch path
  KSM -- fetch_kernel --> URZ
  KSM -- KernelPrefetchRequest --> ONA
  ONA -- KernelPrefetchReady/Error --> KSM

  %% Blueprint pipeline
  KRN -. descriptors .-> TEZ
  TEZ -- artifact + update --> URZ
  TEZ -. KernelCatalogUpdate .-> ONA

  %% Offline policy loop
  TMY -- FieldReport --> ONA
  ONA -- FieldReport --> SMC
  SMC -- PolicyUpdate --> ONA
  ONA -- PolicyUpdate --> TMY

  %% Telemetry/state
  TLR -- Telemetry --> ONA
  TMY -- Telemetry --> ONA
  KSM -- Telemetry --> ONA
  TEZ -- Telemetry --> ONA
  URZ -- Telemetry --> ONA
  ONA -- Metrics --> ONA
  WTH -- Telemetry --> ONA
  ONA -- Telemetry/State/Reports --> NSS

  %% Supervisor
  WTH --- ONA
  WTH --- URZ
  WTH --- KSM
  WTH --- TMY
```

## 2) Epoch Sequence (Training → Decision → Execution)

```mermaid
sequenceDiagram
  participant Tolaria as Tolaria
  participant Tamiyo as Tamiyo
  participant Kasmina as Kasmina
  participant Urza as UrzaRuntime
  participant Oona as Oona
  participant Nissa as Nissa

  Tolaria->>Tamiyo: SystemStatePacket
  Tamiyo->>Tamiyo: Policy inference + risk gating
  Tamiyo->>Kasmina: AdaptationCommand
  Kasmina->>Urza: fetch_kernel(blueprint_id)
  Urza-->>Kasmina: CompiledBlueprint + latency_ms
  Kasmina->>Kasmina: Lifecycle gates (G0..G4), attach/fallback
  Kasmina-->>Tolaria: export_seed_states() (optional)
  par Telemetry
    Tolaria->>Oona: TelemetryPacket
    Tamiyo->>Oona: TelemetryPacket
    Kasmina->>Oona: TelemetryPacket
  and Delivery
    Oona-->>Nissa: consume telemetry/state
    Nissa->>Nissa: index to ES + update Prom + alerts/SLO
  end
```

## 3) Kernel Prefetch Flow

```mermaid
sequenceDiagram
  participant Ksm as Kasmina PrefetchCoordinator
  participant Oon as Oona
  participant Wkr as Urza Prefetch Worker
  participant Lib as UrzaLibrary

  Ksm->>Oon: KernelPrefetchRequest
  Oon-->>Wkr: consume request
  Wkr->>Lib: get(blueprint_id)
  alt Artifact exists and checksum ok
    Wkr-->>Oon: KernelArtifactReady (p50/p95, checksum, guard_digest)
  else Error case
    Wkr-->>Oon: KernelArtifactError(reason)
  end
  Oon-->>Ksm: deliver Ready/Error
  Ksm->>Ksm: attach kernel or handle failure
  Ksm->>Oon: TelemetryPacket
```

## 4) Tezzeret Compilation Pipeline

```mermaid
sequenceDiagram
  participant Forge as TezzeretForge
  participant Karn as KarnCatalog
  participant Comp as TezzeretCompiler
  participant Urza as UrzaLibrary
  participant Oona as Oona (optional)

  Forge->>Karn: get(blueprint_id)
  Karn-->>Forge: BlueprintDescriptor
  Forge->>Comp: compile(descriptor, params)
  Comp-->>Forge: artifact_path + KernelCatalogUpdate
  Forge->>Urza: save(metadata, artifact, update, extras)
  opt notify
    Forge->>Oona: publish KernelCatalogUpdate
  end
  Forge->>Forge: telemetry metrics/events
```

## 5) Offline Policy Improvement (Simic)

```mermaid
sequenceDiagram
  participant Tamiyo as Tamiyo
  participant Oona as Oona
  participant Simic as Simic Trainer

  Tamiyo->>Oona: publish FieldReport
  Oona-->>Simic: consume FieldReport
  Simic->>Simic: PPO training + validation
  alt Validation passed
    Simic->>Oona: publish PolicyUpdate
    Oona-->>Tamiyo: deliver PolicyUpdate
    Tamiyo->>Tamiyo: ingest_policy_update()
  else Validation fail
    Simic->>Oona: Telemetry (validation failed)
  end
```

## 6) Observability & SLOs

```mermaid
sequenceDiagram
  participant Subsys as Subsystems (Tolaria/Tamiyo/Kasmina/Urza/Tezzeret/Simic/Weatherlight/Oona)
  participant Oona as Oona
  participant Nissa as Nissa
  participant ES as Elasticsearch
  participant Prom as Prometheus

  Subsys->>Oona: publish Telemetry / State / FieldReport
  Oona-->>Nissa: consume streams
  Nissa->>ES: index documents
  Nissa->>Prom: inc counters / update gauges
  Nissa->>Nissa: AlertEngine rules + SLO tracker
```

## Notes

- Edge labels in the context diagram indicate primary message types.
- All inter‑service payloads are Leyline protobuf messages; avoid shadow enums.
- Circuit breakers and conservative modes are omitted for clarity in diagrams but are enforced in code paths (see architecture summary).
