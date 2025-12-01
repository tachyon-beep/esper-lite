#!/usr/bin/env bash
set -euo pipefail

uv run python -m esper.scripts.train ppo --task tinystories "$@"
