#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "$0")/.." && pwd)
cd "$here"

echo "Starting Redis (docker compose) for integration tests..."
docker compose -f infra/docker-compose.redis.yml up -d

echo "Running integration tests..."
REDIS_URL=${REDIS_URL:-redis://localhost:6379/0} \
pytest -m integration

echo "Stopping Redis..."
docker compose -f infra/docker-compose.redis.yml down

echo "Done."

