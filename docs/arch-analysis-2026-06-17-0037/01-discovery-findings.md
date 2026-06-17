# 01 — Discovery Findings (Holistic Assessment)

**Analyzed at**: 2026-06-17 · git `e259169` · Loomweave index of same SHA
**Confidence**: High for structure/sizing (measured); Medium for intent-vs-implementation (deferred to per-domain review).

---

## 1. What this system is

**Esper** is a **morphogenetic neural-network training platform**: it grows, prunes, and adapts a host network's *topology during training* rather than committing to a fixed architecture. The distinctive twist is a **nested-loop meta-RL design** — a PPO agent (Tamiyo policy, trained by Simic) controls *another network's training process*. The RL "environment" is literally a host network mid-training; each RL step == one host training epoch.

The system is organized via two deliberate, non-mixed metaphors (enforced by CLAUDE.md):
- **Body/organism** for system architecture (organs = domains).
- **Plant/botanical** for the seed lifecycle state machine only.

---

## 2. Technology stack

| Layer | Choice |
|-------|--------|
| Language | Python ≥3.11 |
| ML | PyTorch ≥2.8, torchvision, transformers, datasets |
| RL | Custom PPO (recurrent, factored action space, Q(s,op) critic) |
| Config/validation | Pydantic v2, PyYAML, strict JSON configs |
| Telemetry store | DuckDB (Karn analytics), MCP server for SQL access |
| UI | Rich/Textual TUI (Sanctum), FastAPI + websockets web (Overwatch) |
| Tooling | uv (package mgr), ruff, mypy, pytest, hypothesis, mutmut |

---

## 3. Directory organization — by domain (organ), not by layer

Organization is **domain/feature-oriented**, each domain mapping to a biological organ:

| Domain | LOC | Organ role | One-line responsibility |
|--------|-----|-----------|--------------------------|
| **simic** | 23,643 | Evolution | RL infra: PPO agent, rollout buffer, reward economy, vectorized trainer, telemetry emitters |
| **karn** | 22,539 | Memory | Operator UI (Sanctum TUI + Overwatch web), DuckDB store, MCP analytics server, health/aggregation |
| **leyline** | 7,277 | DNA/Genome | Shared enums, protocols, tensor schemas, telemetry contracts — the dependency apex |
| **kasmina** | 5,821 | Stem Cells | Morphogenetic host model, SeedSlot lifecycle state machine, blending, blueprints |
| **tamiyo** | 5,146 | Brain/Cortex | Decision policy (heuristic + learned LSTM), action masks, feature extraction |
| **nissa** | 3,153 | Sensory | Telemetry hub/output, analytics backends, W&B integration |
| **utils** | 1,285 | — | Data loading (CIFAR), loss helpers |
| **scripts** | 1,154 | — | `train.py` CLI entry point (ppo/heuristic subcommands) |
| **tolaria** | 869 | Metabolism | Training execution environment + safety Governor |
| **runtime** | 704 | — | Task/topology presets (`tasks.py`) |

**Planned-but-not-shipped** (designed, referenced in docs, not first-class controllers): **Emrakul** (immune/decay), **Narset** (endocrine/allocator), **Esika** (host superstructure).

---

## 4. Entry points

- **CLI**: `python -m esper.scripts.train {ppo|heuristic}` → `scripts/train.py` (1,154 LOC).
- **MCP server**: `python -m esper.karn.mcp` → DuckDB telemetry SQL access (`esper-karn` server).
- **Training core**: `simic/training/vectorized_trainer.py::VectorizedPPOTrainer.run` (3,033 LOC file) — the orchestration spine.
- **Host model**: `kasmina/host.py::MorphogeneticModel`.
- **Decision**: `tamiyo/heuristic.py::HeuristicTamiyo.decide` or learned policy via `tamiyo/policy/factory.py::create_policy`.

---

## 5. Subsystem identification (8 active + support)

Confirmed cohesive groups for per-domain review: **kasmina, tamiyo, simic, tolaria, nissa, karn, leyline**, plus a **support cluster** (runtime + utils + scripts) reviewed together as the "platform glue."

The control-flow spine (the system's load-bearing seam):
```
scripts/train.py
  └─ simic VectorizedPPOTrainer.run  (outer PPO loop)
       ├─ tolaria environment + Governor  (host training, safety rollback)
       │    └─ kasmina MorphogeneticModel + SeedSlot  (topology growth)
       ├─ tamiyo policy.decide          (action selection per step)
       ├─ simic rewards (contribution/rent/churn)  (credit assignment)
       ├─ nissa hub  (telemetry emission)
       └─ karn store/sanctum/overwatch  (persistence + observability)
  shared contracts ← leyline (everyone imports; imports almost nothing)
```

---

## 6. Architectural signals (measured, pre-review)

**Positive:**
- **0 import cycles** across 27,976 edges — layering discipline is real.
- **Leyline as clean apex** (fan-in 197 / fan-out 2) — single source of truth for contracts, exactly as designed (Commandment-adjacent: contracts-first).
- Domain-oriented packaging with consistent `__init__.py` public surfaces.

**Concerns to investigate in review:**
- **God-files**: ≥15 files >1,000 LOC. Worst: `vectorized_trainer.py` (3,033), `slot.py` (2,831), `leyline/telemetry.py` (2,524), `sanctum/aggregator.py` (2,277), `sanctum/schema.py` (1,717), `ppo_agent.py` (1,696), `contribution.py` (1,514), `vectorized.py` (1,490), `factored_lstm.py` (1,394).
- **Two mega-domains** (simic + karn = 64% of LOC). Karn being nearly as large as the entire RL core suggests heavy UI/analytics surface area.
- **Coexisting `vectorized.py` (1,490) and `vectorized_trainer.py` (3,033)** in simic/training — possible responsibility overlap to clarify.

---

## 7. Constitutional claims to test (from ROADMAP "Nine Commandments")

Each is a *falsifiable* architectural assertion; per-domain review will confirm/refute with evidence:
1. **Sensors match capabilities** — every Body feature has a Nissa sensor.
2. **Complexity pays rent** — reward = accuracy − rent (param-ratio penalty wired in).
3. **GPU-first / inverted control flow** — Tolaria never blocks Simic; device-resident buffers, no Python queues/locks on the hot path.
4. **Train Anything** — Kasmina never imports host-specific `torch.nn`; relies on `HostProtocol`.
5. **One morphogenetic plane** — single Kasmina plane, many slots, one coordinate system.
6. **Governor un-disableable** — `TolariaGovernor` cannot be turned off by the policy it supervises.
7. **No legacy / no defensive programming** — CLAUDE.md hard rules; check for shims, `getattr` bug-hiding.
8. **Telemetry as a contract** — typed payloads, schema validation, no silent fallback.

---

## 8. Limitations of this discovery pass
- Sizing/coupling are measured; **behavioral correctness, RL soundness, and intent-vs-code fidelity are deferred** to per-domain review + specialist (drl/pytorch) passes.
- Loomweave index is one commit old (`e259169` vs working tree); structural facts are stable but exact line numbers may drift slightly.
- Tests (19 test dirs) are out of scope for the catalog but referenced by the quality assessment.
