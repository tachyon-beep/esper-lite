#!/usr/bin/env bash
set -euo pipefail

# Fast developer test runner with timeouts and safe defaults.
# - Disables torch.compile to avoid inductor worker stalls on CPU hosts
# - Sets per-test timeout (overrides pyproject if provided)
# - Fails fast on first error

export TAMIYO_ENABLE_COMPILE=${TAMIYO_ENABLE_COMPILE:-false}
export TEZZERET_ENABLE_COMPILE=${TEZZERET_ENABLE_COMPILE:-false}
export TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE:-1}
export TORCH_COMPILE_DISABLE=${TORCH_COMPILE_DISABLE:-1}

timeout_flag="--timeout=60"
method_flag="--timeout-method=thread"
maxfail_flag="--maxfail=1"

pytest -q ${timeout_flag} ${method_flag} ${maxfail_flag} "$@"
