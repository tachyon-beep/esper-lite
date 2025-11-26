#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"

echo "Stopping Redis, Prometheus, Grafana, and Elasticsearch..."
docker compose -f infra/docker-compose.observability.yml down
docker compose -f infra/docker-compose.redis.yml down

echo "Done."

