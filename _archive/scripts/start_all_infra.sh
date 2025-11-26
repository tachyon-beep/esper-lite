#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"

echo "Bringing up Redis, Prometheus, Grafana, and Elasticsearch..."
docker compose -f infra/docker-compose.redis.yml up -d
docker compose -f infra/docker-compose.observability.yml up -d

echo "Waiting briefly for health checks..."
sleep 3

echo "Active containers:"
docker ps --filter name=infra_ --format '{{.Names}}\t{{.Status}}'

echo "All infra started. Set REDIS_URL if using a non-default port."

