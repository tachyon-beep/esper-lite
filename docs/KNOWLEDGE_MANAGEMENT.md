# Knowledge Management Environment Service

This document defines the developer-facing knowledge management stack that supports Esper-Lite engineers and coding agents. It is an *environment* add-on and does **not** modify or extend the production subsystems (Tolaria, Tamiyo, Kasmina, etc.) described in `docs/architecture_summary.md`. Use this specification as the contract for building and operating the knowledge tooling container.

## 1. Objectives

- Provide rich project comprehension for humans and coding agents via retrieval-augmented generation (RAG) and graph reasoning over repository assets.
- Maintain a strict separation between the Esper-Lite runtime and auxiliary tools. Nothing in this stack becomes a hard dependency of Weatherlight or Leyline contracts; all interaction occurs through external APIs.
- Offer deterministic, reproducible indices of documentation, source code, and telemetry schemas so development agents can answer questions without directly crawling the repository on every request.

## 2. Core Components

The tool pack consists of three cooperative services plus automation:

1. **Qdrant Vector Store**
   - Role: persist embeddings of docs, source files, tests, protobuf specs, and design notes.
   - Access Pattern: similarity search with payload filtering (subsystem, contract, git metadata).
   - Persistence: volume-mounted snapshot directory to retain indexes across container restarts.

2. **Neo4j Graph Database**
   - Role: express relationships among subsystems, Leyline messages, telemetry channels, file artifacts, and design documents.
   - Access Pattern: Cypher queries for dependency analysis (e.g., subsystem → contract → telemetry endpoint) that mirror the architecture summary responsibilities.
   - Persistence: volume-mounted `data/` and `logs/` directories.

3. **Knowledge Gateway Service** (custom container)
   - Written in Python 3.11+.
   - Responsibilities:
     - Ingest/normalize repository artifacts.
     - Generate embeddings via a chosen model (default: `sentence-transformers/all-MiniLM-L6-v2`, configurable).
     - Upsert vectors into Qdrant with rich payload metadata.
     - Maintain Neo4j nodes/edges reflecting the contract-first architecture.
     - Expose a REST API for combined vector + graph queries.
     - Provide CLI/admin endpoints for re-indexing, health checks, and schema migrations.

4. **Ingestion Scheduler**
   - Lightweight task runner (APScheduler or Celery beat) embedded in the gateway.
   - Ensures periodic re-scans (default: every 30 minutes) and supports ad-hoc re-index triggers.

## 3. Data Sources & Coverage

The ingestion job must cover:

- `docs/` tree (design specs, prototype deltas, architecture summary).
- `src/esper/` source files (including Leyline bindings, subsystem modules, telemetry helpers).
- `tests/` mirror (unit and integration tests provide behavioral examples).
- Generated protobuf descriptors under `src/esper/leyline/_generated/` (include schema metadata but skip large binary outputs).
- Optional: CI configuration, tooling configs under `.codacy/` for pipeline context.

Each ingested artifact must capture:

- Git metadata: commit SHA, relative path, last modified timestamp.
- Subsystem classification inferred from path (e.g., `src/esper/kasmina/` → `Kasmina`).
- Artifact type: `doc`, `code`, `test`, `proto`, `config`.
- Relevant Leyline message or telemetry channel tags if detected via regex/pattern matching.

## 4. Vector Index Schema (Qdrant)

- Collection name: `esper_knowledge_v1` (suffix version allows future migrations).
- Vector size: determined by embedding model; store in collection metadata.
- Payload fields (all indexed):
  - `path` (string)
  - `artifact_type` (enum string)
  - `subsystem` (string)
  - `leyline_entities` (array of strings)
  - `telemetry_signals` (array of strings)
  - `git_commit` (string)
  - `git_timestamp` (int epoch seconds)
  - `content_digest` (SHA256 for dedupe)
  - `chunk_id` (string combining path + chunk counter)
- Chunking strategy: 1,000-character window with 200-character overlap for textual artifacts; configurable per artifact type.
- Deduplication: skip chunks whose digest already exists with identical metadata.

## 5. Graph Schema (Neo4j)

### Node Labels

- `Subsystem` (`name`, `description`)
- `LeylineMessage` (`name`, `proto_path`)
- `TelemetryChannel` (`name`, `source_subsystem`)
- `SourceFile` (`path`, `artifact_type`, `git_commit`, `git_timestamp`)
- `DesignDoc` (`path`, `title`)
- `TestCase` (`path`, `focus_subsystem`)

### Relationships

- `(:Subsystem)-[:IMPLEMENTS]->(:LeylineMessage)`
- `(:Subsystem)-[:EMITS]->(:TelemetryChannel)`
- `(:Subsystem)-[:DEPENDS_ON]->(:Subsystem)` (derived from import graph and architecture summary)
- `(:SourceFile)-[:BELONGS_TO]->(:Subsystem)`
- `(:DesignDoc)-[:DESCRIBES]->(:Subsystem)`
- `(:TestCase)-[:VALIDATES]->(:Subsystem)`
- `(:SourceFile)-[:DECLARES]->(:LeylineMessage)`
- Cross-links to connect vector chunks: store the `chunk_id` as a node property and create `(:SourceFile)-[:HAS_CHUNK]->(:Chunk)` nodes when needed for traceability.

### Constraints & Indexes

- Uniqueness constraints on primary identifiers (`Subsystem.name`, `LeylineMessage.name`, `SourceFile.path`).
- Relationship merge operations must be idempotent to allow repeated ingestion runs.

## 6. Knowledge Gateway API

Expose an HTTP API (FastAPI recommended) with the following endpoints:

- `POST /ingest/run` — trigger full re-index (requires auth token).
- `POST /ingest/path` — re-index a specific path.
- `GET /search/vector` — parameters: `query`, optional filters (`subsystem`, `artifact_type`, `leyline`); returns ranked chunks with metadata.
- `GET /search/graph/dependencies` — parameters: `subsystem` or `leyline`; runs Cypher to return relationship chains.
- `POST /search/hybrid` — combines vector search with graph expansion; accepts `query` and optional `mode` (`downstream`, `upstream`).
- `GET /health/live`, `GET /health/ready` — container health.
- `GET /metrics` — Prometheus-compatible metrics (index counts, ingestion durations).

Authentication: simple bearer token from environment variable (`GATEWAY_API_TOKEN`).

## 7. Ingestion Pipeline

1. **File Discovery**: walk configured roots, apply ignores from `.gitignore` and custom blocklist (`.venv`, `var/`, generated caches).
2. **Parsing & Metadata Extraction**:
   - Use tree-sitter or pygments for code summary (optional, nice-to-have).
   - Extract doc titles from Markdown H1 headings.
   - Identify Leyline messages via regex on `AdaptationCommand`, `TelemetryPacket`, etc.
3. **Chunking**: split text/code per schema above.
4. **Embedding**: run batches through sentence-transformers; support GPU acceleration when available but fall back to CPU.
5. **Vector Upsert**: send `upsert_points` to Qdrant with metadata payload.
6. **Graph Update**: upsert nodes/relationships via Neo4j `MERGE` statements; maintain a transaction log for debugging.
7. **Audit Log**: store ingestion runs and stats (start/end time, files touched, chunk count) in a lightweight SQLite file within the gateway container.

## 8. Deployment Blueprint

Provide a `docker-compose.yml` with services:

- `qdrant`
  - Image: `qdrant/qdrant:latest`
  - Ports: `6333` (REST), `6334` (gRPC)
  - Volumes: `./storage/qdrant:/qdrant/storage`

- `neo4j`
  - Image: `neo4j:5-enterprise` (or community if license constraints)
  - Ports: `7474` (HTTP), `7687` (Bolt)
  - Environment: `NEO4J_AUTH=neo4j/<password>`
  - Volumes: `./storage/neo4j/data:/data`, `./storage/neo4j/logs:/logs`

- `knowledge-gateway`
  - Build context: `./gateway`
  - Depends on: `qdrant`, `neo4j`
  - Environment:
    - `GATEWAY_API_TOKEN`
    - `EMBEDDING_MODEL_NAME`
    - `QDRANT_URL=http://qdrant:6333`
    - `NEO4J_BOLT_URL=bolt://neo4j:7687`
    - `NEO4J_USER`, `NEO4J_PASSWORD`
    - `REPO_ROOT=/workspace/esper-lite`
    - Scheduling knobs (`INGEST_INTERVAL_SECONDS`)
  - Volumes:
    - Mount host repository read-only at `/workspace/esper-lite`
    - Optional cache volume for embeddings (`./storage/cache:/app/.cache`)

### Build Context (`gateway/` directory)

- `pyproject.toml` with dependencies: `fastapi`, `uvicorn`, `qdrant-client`, `neo4j`, `pydantic`, `sentence-transformers`, `apscheduler`, `orjson`, `loguru` (or `structlog`).
- `Dockerfile` (python:3.11-slim) installing system deps (git, build tools for numpy/torch lite if required by embedding model).
- `app/` module containing API, ingestion logic, and schema definitions.
- `tests/` for pipeline unit tests (e.g., verifying metadata extraction and graph merge idempotency).

## 9. Telemetry & Observability

- Log ingestion results with structured JSON (fields: run_id, status, files_processed, duration_seconds) routed to stdout.
- Expose Prometheus metrics: `knowledge_ingest_duration_seconds`, `knowledge_chunks_total`, `knowledge_graph_nodes_total`.
- Add optional OpenTelemetry tracing (start spans for ingestion phases) for future expansion.

## 10. Security & Access Controls

- Keep Qdrant and Neo4j networks internal to the compose stack; expose only the gateway API publicly.
- Protect Neo4j with auth password and, if feasible, restrict Bolt access to the gateway container.
- Use API token auth for ingestion triggers to prevent accidental re-index.
- Document credentials in a `.env` file (never commit secrets), align with prototype strict dependency policy (fail fast when credentials missing).

## 11. Operational Runbook (for coding AI)

1. **Bootstrap**
   - Clone repository or mount it read-only.
   - Build gateway container (`docker compose build`).
   - Start services (`docker compose up -d`).
   - Wait for health endpoints to return `200`.

2. **Initial Index**
   - Call `POST /ingest/run` with API token.
   - Verify logs and metrics (expect non-zero chunk count, Neo4j node creation).

3. **Routine Use**
   - Use `GET /search/vector?query=...` for contextual retrieval.
   - Use `GET /search/graph/dependencies?subsystem=Kasmina` to trace relationships.
   - Combine via `POST /search/hybrid` for RAG prompts.

4. **Maintenance**
   - Rotate Neo4j password via environment variable and redeploy.
   - Prune old Qdrant snapshots when storage exceeds threshold.
   - Review ingestion logs for failures; rerun partial ingestion if needed.

5. **Updates**
   - When repository schema changes (new subsystems, new Leyline messages), update mapping configuration (`gateway/config/subsystem_map.yaml`).
   - Bump collection version to `esper_knowledge_v2` when altering chunk schema.

## 12. Acceptance Criteria For Implementation

A coding AI tasked with building this container must complete the following:

- Produce the docker-compose stack with persistent volumes and configured environment separation.
- Implement the gateway service with documented endpoints, automated ingestion, and combined search logic.
- Populate Qdrant and Neo4j schemas exactly as described; demonstrate sample queries returning meaningful results from the mounted repo.
- Provide automated tests (unit or integration) for metadata extraction, vector upserts, and graph merges.
- Deliver README/operator notes summarizing setup, credentials, and troubleshooting.

## 13. Future Extensions (Optional)

- Add a lightweight UI (Streamlit/Gradio) atop the gateway API for manual browsing.
- Integrate chat-style RAG agent that orchestrates vector + graph calls for conversational assistance.
- Incorporate change detection to push notifications when critical documents (e.g., prototype deltas) change.
- Support incremental ingestion via git diff scanning rather than full tree walks.

## 14. Knowledge Lifecycle Management

Design the gateway with first-class routines for keeping the knowledge base accurate and compact. These jobs run entirely inside the environment toolchain and never mutate the Esper-Lite repository.

### 14.1 Ingestion Cadence & Scheduling
- **Full Rescan**: run once every 24 hours to rebuild vector and graph state from scratch. Expose cron-style settings (`FULL_INGEST_CRON`) so operators can shift schedules without code changes.
- **Incremental Sweep**: default to every 30 minutes. Compute `git diff` against the last successful commit snapshot; only reprocess touched files. Skip chunks whose content digests and metadata are unchanged to save embedding work.
- **Ad-hoc Trigger**: `/ingest/path` endpoint forces an immediate refresh for targeted files; emit audit entries noting the caller and affected paths.

### 14.2 Staleness Detection & Pruning
- Persist per-chunk metadata (`git_commit`, `git_timestamp`, `content_digest`). The gateway tracks a `last_seen_at` timestamp each time an artifact is reprocessed.
- During ingestion, mark artifacts missing from the filesystem as `candidate_for_removal`. A nightly pruning job (`/maintenance/prune`) permanently deletes Qdrant points and Neo4j nodes/relationships whose `candidate_for_removal` flag has been set for two consecutive runs.
- Record removal operations in the audit log with structured reasons (`removed_file`, `renamed_file`, `manual_request`).

### 14.3 Versioning & Rollback Safety
- Maintain versioned collection names (`esper_knowledge_v1`, `esper_knowledge_v2`, ...). Promote to a new version when altering payload schemas or switching embedding models.
- Enable Qdrant snapshotting (`qdrant snapshot create`) post-ingestion; keep the last three snapshots and prune older ones.
- For Neo4j, schedule weekly `neo4j-admin dump` backups. Store snapshot/backup paths in a gateway-maintained manifest for quick rollback.

### 14.4 Drift Monitoring
- Metrics to emit via `/metrics`:
  - `knowledge_chunks_total` (labeled by `artifact_type` and `subsystem`).
  - `knowledge_chunks_removed_total` with reason labels.
  - `knowledge_ingest_duration_seconds` histogram for full vs incremental runs.
  - `knowledge_ingest_failures_total` with cause labels (`git_error`, `embedding_error`, `neo4j_unavailable`, etc.).
- Surface alerts (can wire to Prometheus rules) when:
  - Removal rate exceeds configurable threshold (default: >10% of chunks removed in 24 hours).
  - Incremental sweeps fail twice consecutively.
  - The latest successful ingest timestamp drifts beyond the SLA (e.g., +2 hours).

### 14.5 Manual Maintenance Operations
- Provide authenticated endpoints/CLI commands:
  - `POST /maintenance/rebuild` — drop current collection/graph and perform a clean bootstrapped ingest (requires explicit confirmation flag).
  - `POST /maintenance/prune` — execute pruning logic immediately; useful after large refactors or branch merges.
  - `POST /maintenance/mark-stale` — accept file paths from operators to mark as stale when git metadata is insufficient (e.g., external deletions).
- Document operator runbook steps for rotating credentials, restoring from snapshots, and validating post-restore by comparing expected chunk counts against baseline metrics.

### 14.6 Metadata Hygiene & Overrides
- Store subsystem classification rules in `gateway/config/subsystem_map.yaml`. Include glob patterns and optional regex overrides for special cases (e.g., shared utilities).
- Log any artifacts that fail classification to a dedicated stream (`classification_warnings.log`) and expose them via `/maintenance/unclassified` for human review.
- Support manual annotations (YAML) that tag specific files with additional metadata (telemetry channels, Leyline messages). Merge these annotations during ingestion, overriding automated guesses to keep the knowledge graph authoritative.

Implementing these lifecycle loops ensures the knowledge stack remains trustworthy, compact, and aligned with Esper-Lite’s contract-first philosophy without burdening the core system.

## 15. Data Management Enhancements (Director Requests)

This section captures additional guidance and stretch goals from the knowledge/data management perspective. Treat these as backlog items for the coding agent once the core system is operational.

### 15.1 Data Quality & Validation
- **Schema Guardrails**: define Pydantic models for ingestion payloads (vector chunks, graph nodes). Reject or quarantine records that fail validation; expose a `GET /maintenance/quarantine` endpoint for inspection.
- **Content Sanity Checks**: prior to embedding, enforce heuristics (non-empty text, max token length, ASCII-only for code unless file specifies otherwise). Log discarded chunks with reasons.
- **Duplicate Detection**: beyond `content_digest`, run periodic similarity clustering to surface near-duplicates across docs/tests. Provide a report so documentation owners can consolidate redundant material.

### 15.2 Provenance & Auditability
- Attach a `provenance` block to every chunk containing: embedding model version, ingestion job ID, operator (if manual), and source branch if applicable.
- Maintain an append-only audit ledger (e.g., SQLite or lightweight event log) capturing each ingestion/prune/rebuild operation with timestamp, actor, and outcome.
- Offer `GET /audit/history` endpoint with pagination for traceability.

### 15.3 Embedding Model Lifecycle
- Externalize embedding model config to `gateway/config/embedding.yaml` including model name, dimensionality, normalization, and stopwords.
- Support hot-swapping models: stage new collection (`esper_knowledge_vN+1`) and run A/B validation by comparing retrieval metrics (precision@k using curated question/answer fixtures).
- If GPU is available, allow the gateway to auto-detect and leverage it; otherwise default to CPU with batched inference to keep ingestion latency predictable.

### 15.4 Knowledge Coverage & Gaps
- Ship a coverage report job that compares repo file list against indexed artifacts; flag anything intentionally skipped (e.g., large binaries) versus unexpected omissions. Provide `coverage_report.json` for dashboarding.
- Track `last_indexed_commit` and alert when repository HEAD moves ahead by more than configurable commit count (default: 20) without ingestion catching up.
- Encourage engineers to register critical documents/tests in a `gateway/config/critical_paths.yaml`; ingestion should fail fast if these files are missing or not indexed.

### 15.5 Multi-Environment Strategy
- Support environment labels (`dev`, `staging`, `prod`) in payload metadata to allow side-by-side indexes for different branches or prototype variants.
- Provide tooling to snapshot knowledge base state per environment and promote from dev → staging → prod once validation passes.
- In docker-compose, allow optional additional repos to be mounted (e.g., `../esper-docs`) and define ingestion profiles per mount (toggle per compose override).

### 15.6 Security & Access Control Deepening
- Integrate with an external secret manager (Vault or AWS Secrets Manager) for Neo4j/Qdrant credentials when running outside local dev environments.
- Implement per-endpoint scopes: e.g., ingestion endpoints require `role=maintainer`, search endpoints allow `role=reader`. Bearer token should encode scope claims.
- Add rate limiting (`fastapi-limiter` or custom middleware) on search endpoints to prevent accidental overload from automated agents.

### 15.7 Observability Upgrades
- Emit OpenTelemetry traces for major ingestion phases (discover → embed → upsert → graph merge). Ship a collector configuration so traces can be forwarded to existing observability stacks (Nissa/Prometheus integrations).
- Provide structured logging enrichers that add `ingest_run_id`, `subsystem`, and `artifact_type` to every log line for easier correlation.
- Publish a weekly summary report (email/Slack webhook) containing ingestion stats, top new documents, and any classification warnings.

### 15.8 Data Retention & Compliance
- Define retention policies: default to keeping all indexed chunks unless upstream documents are deleted. Allow operators to configure TTL for volatile directories (e.g., scratch design notes) via `retention_policies.yaml`.
- Ensure snapshots/backups are encrypted at rest (enable AES on Qdrant snapshots, use Neo4j backup encryption options) when running in shared environments.
- Document procedures for right-to-forget requests—even if unlikely, note how to purge specific paths and verify removal across both stores.

### 15.9 Extensibility Hooks
- Architect the gateway ingestion pipeline with plugin hooks (Python entry points) so future data sources—Jira tickets, CI logs, telemetry exports—can be added without rewriting core logic.
- Provide a command-line scaffold (`knowledge-gateway scaffold-plugin --name foo`) that generates boilerplate for new source adapters.
- Reserve namespace in Neo4j for upcoming entity types (`Incident`, `Runbook`) to avoid schema churn later.

### 15.10 Documentation Expectations
- Maintain living documentation under `gateway/docs/` covering lifecycle jobs, configuration files, and troubleshooting scenarios. Update it whenever new management features land.
- Include diagrams (Mermaid or PlantUML) mapping ingestion flows and data relationships to help future maintainers reason about the system.
- Ensure README highlights the separation between runtime system and knowledge tooling, reiterating that repos remain the single source of truth.

---
This specification is intentionally detailed so that an autonomous coding agent can implement the knowledge management container without further clarification while preserving the system/environment boundary mandated by the prototype guidelines.
