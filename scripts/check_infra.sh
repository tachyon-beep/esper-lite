#!/usr/bin/env bash
set -euo pipefail

echo "Checking local infra health..."

REDIS_CONTAINER=${REDIS_CONTAINER:-infra-redis-1}

fail=false

echo "- Redis: ping via docker exec"
if docker ps --format '{{.Names}}' | grep -q "^${REDIS_CONTAINER}$"; then
  if ! docker exec -i "$REDIS_CONTAINER" redis-cli ping >/dev/null 2>&1; then
    echo "  Redis ping failed"
    fail=true
  else
    echo "  Redis OK"
  fi
else
  echo "  Redis container ${REDIS_CONTAINER} not found"
  fail=true
fi

echo "- Prometheus: http://localhost:9090/-/ready"
if ! curl -fsS --max-time 2 http://localhost:9090/-/ready >/dev/null; then
  echo "  Prometheus not ready"
  fail=true
else
  echo "  Prometheus OK"
fi

echo "- Grafana: http://localhost:3000/api/health"
if ! curl -fsS --max-time 2 http://localhost:3000/api/health >/dev/null; then
  echo "  Grafana not healthy"
  fail=true
else
  echo "  Grafana OK"
fi

echo "- Elasticsearch: http://localhost:9200/_cluster/health"
if ! curl -fsS --max-time 2 http://localhost:9200/_cluster/health >/dev/null; then
  echo "  Elasticsearch not healthy"
  fail=true
else
  echo "  Elasticsearch OK"
fi

if [ "$fail" = true ]; then
  echo "One or more infra checks failed" >&2
  exit 1
fi

echo "All infra checks passed."

