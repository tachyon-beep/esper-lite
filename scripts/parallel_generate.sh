#!/bin/bash
# Parallel diverse data generation across GPUs
#
# Usage: ./scripts/parallel_generate.sh [options]
#
# Options:
#   -n, --episodes N      Episodes per env/policy combo (default: 10)
#   -w, --workers N       Workers per GPU (default: 3)
#   -d, --data-dir DIR    Output directory (default: data/datagen_v3)
#   -h, --help            Show this help
#
# Examples:
#   ./scripts/parallel_generate.sh -n 10 -w 3
#   ./scripts/parallel_generate.sh --episodes 5 --data-dir data/datagen_test
#   ./scripts/parallel_generate.sh  # Uses defaults

set -e

# Defaults
EPISODES_PER_COMBO=10
WORKERS_PER_GPU=3
DATA_DIR="data/datagen_v3"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--episodes)
            EPISODES_PER_COMBO="$2"
            shift 2
            ;;
        -w|--workers)
            WORKERS_PER_GPU="$2"
            shift 2
            ;;
        -d|--data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        -h|--help)
            head -17 "$0" | tail -15
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage"
            exit 1
            ;;
    esac
done

# Use venv python
PYTHON=".venv/bin/python"

# Detect GPUs
NUM_GPUS=$($PYTHON -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
if [ "$NUM_GPUS" -eq "0" ]; then
    echo "No GPUs detected, exiting"
    exit 1
fi

# Environment IDs split across workers (13 total)
# GPU 0 workers: smaller/faster models
GPU0_ENVS=(
    "baseline fast-lr slow-lr wide deep"
    "adam small-batch large-batch"
    "resnet18"
)

# GPU 1 workers: larger ResNet models
GPU1_ENVS=(
    "resnet34 resnet34-adam"
    "resnet34-slow"
    "resnet34-small-batch"
)

TOTAL_WORKERS=$((NUM_GPUS * WORKERS_PER_GPU))

# Create output directory
mkdir -p "$DATA_DIR"

echo "========================================"
echo "Parallel Diverse Data Generation"
echo "========================================"
echo "GPUs: $NUM_GPUS"
echo "Workers per GPU: $WORKERS_PER_GPU"
echo "Total workers: $TOTAL_WORKERS"
echo "Episodes per combo: $EPISODES_PER_COMBO"
echo "Output: $DATA_DIR"
echo "========================================"

# Launch workers
PIDS=()
WORKER_ID=0

# GPU 0 workers
for ((w=0; w<WORKERS_PER_GPU && w<${#GPU0_ENVS[@]}; w++)); do
    ENVS="${GPU0_ENVS[$w]}"
    echo "Starting worker $WORKER_ID on cuda:0 (envs: $ENVS)"

    PYTHONPATH=src $PYTHON -m esper.datagen.generate \
        --episodes-per-combo $EPISODES_PER_COMBO \
        --output-dir "$DATA_DIR" \
        --device cuda:0 \
        --env-ids $ENVS \
        > "$DATA_DIR/worker_${WORKER_ID}.log" 2>&1 &

    PIDS+=($!)
    WORKER_ID=$((WORKER_ID + 1))
done

# GPU 1 workers (if available)
if [ "$NUM_GPUS" -ge 2 ]; then
    for ((w=0; w<WORKERS_PER_GPU && w<${#GPU1_ENVS[@]}; w++)); do
        ENVS="${GPU1_ENVS[$w]}"
        echo "Starting worker $WORKER_ID on cuda:1 (envs: $ENVS)"

        PYTHONPATH=src $PYTHON -m esper.datagen.generate \
            --episodes-per-combo $EPISODES_PER_COMBO \
            --output-dir "$DATA_DIR" \
            --device cuda:1 \
            --env-ids $ENVS \
            > "$DATA_DIR/worker_${WORKER_ID}.log" 2>&1 &

        PIDS+=($!)
        WORKER_ID=$((WORKER_ID + 1))
    done
fi

echo ""
echo "All workers started. PIDs: ${PIDS[*]}"
echo "Logs: $DATA_DIR/worker_*.log"
echo ""
echo "Monitoring progress..."

# Monitor progress
while true; do
    RUNNING=0
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            RUNNING=$((RUNNING + 1))
        fi
    done

    CURRENT=$(ls "$DATA_DIR"/*.json 2>/dev/null | grep -v progress | wc -l)

    echo -ne "\rWorkers: $RUNNING/${#PIDS[@]} | Episodes: $CURRENT"

    if [ "$RUNNING" -eq 0 ]; then
        echo ""
        echo "========================================"
        echo "All workers complete!"
        echo "Total episodes in $DATA_DIR: $CURRENT"
        echo ""
        echo "Running health check..."
        PYTHONPATH=src $PYTHON -m esper.datagen.generate --health-check --output-dir "$DATA_DIR"
        echo "========================================"
        break
    fi

    sleep 10
done
