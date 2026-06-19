# 03 — Architecture Diagrams (C4)

All edges below are evidence-backed from the dependency-map cross-cutting pass (concrete import/call sites verified). Diagrams use Mermaid.

---

## C1 — System Context

Esper-lite is a single-process research platform. The "users" are an operator (CLI + live dashboards) and an analyst (post-hoc SQL).

```mermaid
graph TB
    operator["👤 Operator<br/>(runs training, watches live)"]
    analyst["👤 Analyst<br/>(post-hoc analysis)"]
    esper["⬛ Esper-lite<br/>Morphogenetic NN training platform<br/>(PPO controls another network's training)"]
    cifar[("CIFAR-10 / TinyStories<br/>datasets")]
    wandb["Weights & Biases<br/>(optional)"]
    jsonl[("events.jsonl<br/>telemetry artifact")]

    operator -->|"train ppo/heuristic CLI"| esper
    operator -->|"Sanctum TUI / Overwatch web"| esper
    esper -->|"loads"| cifar
    esper -->|"emits metrics (optional)"| wandb
    esper -->|"writes"| jsonl
    analyst -->|"esper-karn MCP: SQL over DuckDB"| jsonl
```

---

## C2 — Container / Domain View

The 8 domains as containers, annotated with the **Protocol** each cross-domain edge implements. leyline is the shared contracts substrate (no outbound runtime arrows).

```mermaid
graph TB
    subgraph contracts["leyline — Contracts (DNA)"]
      direction LR
      proto["HostProtocol · GovernorProtocol<br/>PolicyBundle · OutputBackend · SeedSlotProtocol<br/>SeedStage/VALID_TRANSITIONS · TelemetryPayload union<br/>factored actions · slot_id space · DEFAULT_* constants"]
    end

    simic["<b>Simic</b> — Evolution<br/>RL orchestration apex<br/>(PPO, rewards, vectorized trainer)"]
    tamiyo["<b>Tamiyo</b> — Brain<br/>policy (LSTM + heuristic)<br/>features · action masks"]
    kasmina["<b>Kasmina</b> — Stem Cells<br/>MorphogeneticModel<br/>SeedSlot lifecycle FSM"]
    tolaria["<b>Tolaria</b> — Metabolism<br/>TolariaGovernor (safety)<br/>model/device factory"]
    nissa["<b>Nissa</b> — Senses<br/>NissaHub telemetry bus<br/>DiagnosticTracker"]
    karn["<b>Karn</b> — Memory<br/>Sanctum TUI · Overwatch web<br/>DuckDB MCP · TelemetryStore"]
    support["scripts/train · runtime/tasks · utils<br/>(composition glue)"]

    support -->|"builds + dispatches"| simic
    simic -->|"implements PolicyBundle<br/>trains policy weights"| tamiyo
    simic -->|"acts on (germinate/blend/prune)<br/>SeedSlotProtocol"| kasmina
    simic -->|"drives passively<br/>GovernorProtocol"| tolaria
    simic -->|"emits TelemetryEvent"| nissa
    simic -->|"HealthMonitor"| karn
    tolaria -->|"builds host"| kasmina
    tolaria -->|"best-effort telemetry"| nissa
    tamiyo -->|"SignalTracker telemetry"| nissa
    karn -->|"OutputBackend consumer"| nissa

    simic -.->|implements/consumes| contracts
    tamiyo -.-> contracts
    kasmina -.-> contracts
    tolaria -.-> contracts
    nissa -.-> contracts
    karn -.-> contracts
    support -.-> contracts
```

**Key reading:** Simic is the single integration apex (imports tamiyo ×20, kasmina ×7, tolaria ×4, nissa ×6, karn ×2, runtime ×7, utils ×7). Every other domain depends only on leyline plus small, well-justified cross-links (tamiyo→nissa, tolaria→{nissa, kasmina}, karn→nissa). **Zero import cycles** across 27,976 edges.

---

## C3 — Component View: the Telemetry Path

Telemetry flows one-way from every domain into the Nissa bus, then fans out to pluggable sinks. This is the realization of the "telemetry as a contract" principle.

```mermaid
graph LR
    domains["All domains<br/>(simic, kasmina, tolaria, tamiyo…)"]
    hub["NissaHub<br/>(main worker thread)"]
    subgraph backends["OutputBackends (pluggable sinks)"]
      console["ConsoleOutput"]
      filebk["FileOutput / DirectoryOutput<br/>→ events.jsonl"]
      wandbk["WandbBackend"]
      analytics["BlueprintAnalytics"]
      sanctum["SanctumBackend<br/>→ AggregatorRegistry"]
      overwatch["OverwatchBackend<br/>→ WebSocket"]
      karncol["KarnCollector<br/>→ TelemetryStore"]
    end
    aggregator["SanctumAggregator<br/>folds stream → SanctumSnapshot"]
    tui["Sanctum TUI"]
    web["Overwatch web (Vue)"]
    duck["DuckDB (in-memory)<br/>esper-karn MCP"]

    domains -->|"get_hub().emit(TelemetryEvent)"| hub
    hub -->|"bounded queue per backend<br/>(drop-on-overflow)"| console
    hub --> filebk
    hub --> wandbk
    hub --> analytics
    hub --> sanctum
    hub --> overwatch
    hub --> karncol
    sanctum --> aggregator
    overwatch --> aggregator
    aggregator --> tui
    aggregator --> web
    filebk -.->|"offline load"| duck
```

**Two read paths from one contract:** the *live* path (Sanctum/Overwatch share one `AggregatorRegistry`+`SanctumSnapshot`) and the *offline* path (DuckDB SQL over `events.jsonl`). Both originate from the same leyline `TelemetryEvent`. **Note:** the MCP/DuckDB surface is currently *orphaned* — a standalone post-hoc CLI, not wired as a live backend.

---

## C4 — Dynamic View: One PPO Update (the nested-loop spine)

The system's defining feature: each RL step *is* one host training epoch. Verified call chain.

```mermaid
sequenceDiagram
    participant CLI as scripts/train.py:main
    participant V as vectorized.train_ppo_vectorized
    participant T as VectorizedPPOTrainer.run()
    participant B as _run_batch
    participant E as _run_epoch (1 RL step)
    participant Host as Kasmina host (Tolaria-built)
    participant Gov as TolariaGovernor
    participant Pol as Tamiyo policy.get_action
    participant Rew as simic.rewards.compute_reward
    participant Co as PPOCoordinator
    participant Ag as PPOAgent.update

    CLI->>V: train_ppo_vectorized(config)
    V->>T: build EnvFactoryContext, trainer.run()
    loop outer: per batch (= 1 PPO update)
        T->>B: _run_batch(envs)
        loop inner: per epoch (= 1 RL step), ×max_epochs
            B->>E: _run_epoch()
            E->>Host: _run_train_pass (host trains 1 epoch)
            E->>Gov: check_vital_signs(val_loss) [un-disableable]
            E->>Host: _run_fused_val_pass (val + counterfactual)
            E->>Pol: get_action(obs) → masked factored action
            E->>Host: execute_actions → germinate/blend/fossilize/prune
            E->>Rew: compute_reward (accuracy − rent + shaping)
        end
        B->>Co: run_update()
        Co->>Ag: agent.update()
        Note over Ag: GAE → per-head clipped surrogate<br/>+ value loss + entropy floor<br/>backward → grad-clip → optimizer.step
    end
```

**The load-bearing seam (Commandment 3, CONFIRMED):** the entire inner loop runs in **one sequential thread** with the PPO policy embedded inline. There is **no producer/consumer queue, `threading.Lock`, or multiprocessing primitive** between host-training (Tolaria) and the policy (Simic). Per-env work runs on persistent per-env CUDA streams, accumulates into device-resident tensors, and synchronizes **once per phase** before any `.item()`. The single irreducible D2H sync is `actions_stacked.cpu().numpy()` at action dispatch (the on-policy control boundary), already consolidated into one transfer for all heads/envs.

---

## Seed lifecycle FSM (botanical metaphor — the only place it applies)

```mermaid
stateDiagram-v2
    [*] --> DORMANT
    DORMANT --> GERMINATED: Germinate
    GERMINATED --> TRAINING: Advance (G1)
    TRAINING --> BLENDING: Advance (G2 gate)
    BLENDING --> HOLDING: Advance (G3)
    HOLDING --> FOSSILIZED: Fossilise (commit)
    TRAINING --> PRUNED: Prune
    BLENDING --> PRUNED: Prune
    PRUNED --> EMBARGOED: Cleanup
    EMBARGOED --> RESETTING: Cooldown
    RESETTING --> DORMANT: Recycle
```

Single source of truth: `leyline/stages.py::VALID_TRANSITIONS`. Enforced by `SeedState.transition()` (kasmina) and mirrored into Tamiyo's action masks — so the legal action set is *derived from* the FSM, never hand-coded. (`SHADOWING`, value 5, was removed; the gap is intentional, documented, not shimmed.)

---

## Diagram provenance & confidence

- **C2/C4 spine and the telemetry path: High** — call sites and method bodies were read directly (train.py → vectorized → trainer.run → _run_batch → _run_epoch → ppo_coordinator.run_update → agent.update).
- **C2 full fan-out: Medium** — domain-level edge *counts* are grep-derived; the load-bearing edges were source-verified, but not all ~110 simic-internal edges were opened.
- The `tamiyo→kasmina` edge (tamiyo imports kasmina once, in `action_masks` under TYPE_CHECKING for `SeedState`) is intentionally **not** drawn as a first-class runtime edge in C2.
