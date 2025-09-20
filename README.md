# Esper-Lite Project Scaffold

Esper-Lite is a streamlined morphogenetic control stack centred on a PyTorch 2.8 runtime. This scaffold establishes the shared tooling, package layout, and subsystem contracts needed to begin feature development across Tolaria, Kasmina, Tamiyo, Simic, and the blueprint and infrastructure services.

## Getting Started

1. **Create a virtual environment**
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -e .[dev]
   ```
2. **Run quality gates**
   ```bash
   pytest tests
   pylint --rcfile .codacy/tools-configs/pylint.rc src/esper
   ruff check src tests
   mypy src/esper
   ```
3. **Launch local infrastructure (placeholder)**
   - Docker compose definitions will live under `infra/` (see backlog TKT-003).
   - Redis, Prometheus, and Elasticsearch endpoints are documented in `docs/project/implementation_plan.md`.

## Repository Layout

- `src/esper/` — Python packages for Esper subsystems, organised by lifecycle phase.
- `tests/` — Pytest suites mirroring the source tree (`tests/tolaria`, `tests/kasmina`, etc.).
- `docs/` — Design references, implementation plans, and backlog materials.
- `.codacy/` — Shared linting and CI bootstrap configuration.

## Subsystem Overview

Each subsystem module exposes a narrow public API under `src/esper/<subsystem>/__init__.py` and includes placeholder classes that cite the authoritative detailed design section:

- **Tolaria** (`docs/design/detailed_design/01-tolaria.md`) — Training orchestrator and epoch control loop.
- **Kasmina** (`docs/design/detailed_design/02-kasmina.md`) — Seed lifecycle manager and kernel executor.
- **Tamiyo** (`docs/design/detailed_design/03-tamiyo.md`) — Strategic controller, risk governance, and telemetry hub.
- **Simic** (`docs/design/detailed_design/04-simic.md`) — Offline policy trainer using PPO + LoRA on PyTorch 2.8.
- **Karn, Tezzeret, Urza** (`docs/design/detailed_design/05-karn.md`, `06-tezzeret.md`, `08-urza.md`) — Blueprint catalog, compiler, and artifact library.
- **Leyline, Oona, Nissa** (`docs/design/detailed_design/00-leyline.md`, `09-oona.md`, `10-nissa.md`) — Contracts, messaging fabric, and observability stack.

## Next Steps

The backlog in `docs/project/backlog.md` decomposes the first implementation sprints. Slice 0 focuses on CI/tooling and Leyline contracts. Slice 1 establishes the control loop with Tolaria, Kasmina, and Tamiyo, followed by blueprint management, observability, and offline learning in subsequent slices.

Refer to `docs/project/implementation_plan.md` for full sequencing and ownership recommendations.

