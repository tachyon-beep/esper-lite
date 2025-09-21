#!/usr/bin/env bash
# Weatherlight launcher (prototype-diff draft)
#
# Purpose: Start core infra (Redis, optional observability) and run
#           Weatherlight supervisor + Nissa service side-by-side.
#
# Notes:
# - This draft lives under docs/ and is not wired into live code.
# - Weatherlight service runner is pending (python -m esper.weatherlight.service_runner).
#   Replace with `esper-weatherlight-service` once the console entrypoint lands.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../.. && pwd)"
cd "$ROOT_DIR"

OBSERVABILITY=0
RUN_NISSA=1

usage() {
  cat <<USAGE
Usage: $(basename "$0") [--with-observability] [--no-nissa]

Options:
  --with-observability   Also start Prometheus/Grafana/Elasticsearch compose stack.
  --no-nissa             Do not run Nissa service locally (run it elsewhere).

Environment:
  REDIS_URL               (default: redis://localhost:6379/0)
  ESPER_LEYLINE_SECRET    REQUIRED for bus HMAC signing/verification

This is a prototype-diff script: it does not modify source code.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-observability)
      OBSERVABILITY=1; shift ;;
    --no-nissa)
      RUN_NISSA=0; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

log()   { echo "[start_all_infra] $*"; }
error() { echo "[start_all_infra][ERROR] $*" >&2; }
die()   { error "$*"; exit 1; }

# Load .env if present to populate defaults
if [[ -f .env ]]; then
  set -a; # export
  # shellcheck disable=SC1091
  source .env || true
  set +a
fi

export REDIS_URL="${REDIS_URL:-redis://localhost:6379/0}"

[[ -n "${ESPER_LEYLINE_SECRET:-}" ]] || die "ESPER_LEYLINE_SECRET not set; export a non-empty value to enable HMAC."

log "Bringing up Redis via compose (infra/docker-compose.redis.yml)"
docker compose -f infra/docker-compose.redis.yml up -d

if (( OBSERVABILITY == 1 )); then
  log "Bringing up observability stack (infra/docker-compose.observability.yml)"
  docker compose -f infra/docker-compose.observability.yml up -d
fi

# Wait for Redis to be healthy
log "Waiting for Redis to respond..."
for i in {1..30}; do
  if command -v redis-cli >/dev/null 2>&1; then
    if redis-cli -u "$REDIS_URL" ping >/dev/null 2>&1; then
      log "Redis is up (via $REDIS_URL)"; break
    fi
  else
    # Best-effort healthcheck against local default port
    if docker exec "$(docker ps --format '{{.Names}}' | grep -E '^.+_redis_1$' | head -n1)" redis-cli ping >/dev/null 2>&1; then
      log "Redis is up (docker exec)"; break
    fi
  fi
  sleep 1
  [[ $i -eq 30 ]] && die "Redis did not become healthy in time"
done

mkdir -p var/log

# Start Weatherlight
WEATHERLIGHT_CMD=(esper-weatherlight-service)
NISSA_CMD=(esper-nissa-service)

log "Starting Weatherlight supervisor: ${WEATHERLIGHT_CMD[*]}"
(
  cd "$ROOT_DIR" || exit 1
  # Prefer existing venv if active; otherwise, assume environment is prepared
  stdbuf -oL -eL "${WEATHERLIGHT_CMD[@]}" \
    > var/log/weatherlight.out.log 2> var/log/weatherlight.err.log & echo $! > var/log/weatherlight.pid
) || die "Failed to start Weatherlight"

if (( RUN_NISSA == 1 )); then
  log "Starting Nissa service: ${NISSA_CMD[*]}"
  (
    cd "$ROOT_DIR" || exit 1
    stdbuf -oL -eL "${NISSA_CMD[@]}" \
      > var/log/nissa.out.log 2> var/log/nissa.err.log & echo $! > var/log/nissa.pid
  ) || die "Failed to start Nissa"
else
  log "Skipping Nissa (per --no-nissa)"
fi

cleanup() {
  log "Shutting down services..."
  for name in weatherlight nissa; do
    pid_file="var/log/${name}.pid"
    if [[ -f "$pid_file" ]]; then
      pid="$(cat "$pid_file" || true)"
      if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
        log "Stopping $name (pid=$pid)"
        kill "$pid" || true
        wait "$pid" 2>/dev/null || true
      fi
      rm -f "$pid_file"
    fi
  done
  log "Stopping docker compose services"
  docker compose -f infra/docker-compose.redis.yml down || true
  if (( OBSERVABILITY == 1 )); then
    docker compose -f infra/docker-compose.observability.yml down || true
  fi
  log "Shutdown complete."
}

trap cleanup INT TERM

log "Weatherlight and Nissa started. Logs: var/log/*.out.log (stderr in *.err.log)"
log "Press Ctrl-C to stop."

# Wait on child processes
weatherlight_pid="$(cat var/log/weatherlight.pid)"
wait "$weatherlight_pid"
