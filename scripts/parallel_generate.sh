#!/bin/bash
# Parallel episode generation across GPUs
#
# Usage: ./scripts/parallel_generate.sh [options]
#
# Options:
#   -n, --episodes N      Episodes per worker (default: 35)
#   -w, --workers N       Workers per GPU (default: 3)
#   -p, --profile NAME    Telemetry profile (minimal, standard, diagnostic, research)
#   -d, --data-dir DIR    Output directory (default: data/simic_episodes)
#   -e, --max-epochs N    Max epochs per episode (default: 25)
#   -h, --help            Show this help
#
# Examples:
#   ./scripts/parallel_generate.sh -n 35 -w 3 -p diagnostic -d data/simic_v2_episodes
#   ./scripts/parallel_generate.sh --episodes 50 --profile research
#   ./scripts/parallel_generate.sh  # Uses defaults (35 eps x 6 workers = 210 total)

set -e

# Defaults
EPISODES_PER_WORKER=35
WORKERS_PER_GPU=3
PROFILE=""
DATA_DIR="data/simic_episodes"
MAX_EPOCHS=25

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--episodes)
            EPISODES_PER_WORKER="$2"
            shift 2
            ;;
        -w|--workers)
            WORKERS_PER_GPU="$2"
            shift 2
            ;;
        -p|--profile)
            PROFILE="$2"
            shift 2
            ;;
        -d|--data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        -e|--max-epochs)
            MAX_EPOCHS="$2"
            shift 2
            ;;
        -h|--help)
            head -20 "$0" | tail -18
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
    echo "No GPUs detected, running on CPU"
    NUM_GPUS=1
    WORKERS_PER_GPU=2
fi

TOTAL_WORKERS=$((NUM_GPUS * WORKERS_PER_GPU))
TOTAL_EPISODES=$((EPISODES_PER_WORKER * TOTAL_WORKERS))

# Create output directory
mkdir -p "$DATA_DIR"

echo "========================================"
echo "Parallel Episode Generation"
echo "========================================"
echo "GPUs: $NUM_GPUS"
echo "Workers per GPU: $WORKERS_PER_GPU"
echo "Total workers: $TOTAL_WORKERS"
echo "Episodes per worker: $EPISODES_PER_WORKER"
echo "Total episodes: $TOTAL_EPISODES"
echo "Max epochs: $MAX_EPOCHS"
echo "Output: $DATA_DIR"
if [ -n "$PROFILE" ]; then
    echo "Telemetry profile: $PROFILE"
else
    echo "Telemetry: disabled (v1 format)"
fi
echo "========================================"

# Find next available episode ID in target directory
EXISTING=$(ls "$DATA_DIR"/episode_*.json 2>/dev/null | wc -l)
echo "Existing episodes: $EXISTING"
echo "Starting from ID: $EXISTING"
echo "========================================"

# Launch workers
PIDS=()
WORKER_ID=0

for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    for ((w=0; w<WORKERS_PER_GPU; w++)); do
        START_ID=$((EXISTING + WORKER_ID * EPISODES_PER_WORKER))
        DEVICE="cuda:$gpu"

        echo "Starting worker $WORKER_ID on $DEVICE (episodes $START_ID-$((START_ID + EPISODES_PER_WORKER - 1)))"

        # Build optional args
        EXTRA_ARGS=""
        if [ -n "$PROFILE" ]; then
            EXTRA_ARGS="$EXTRA_ARGS --profile $PROFILE"
        fi

        PYTHONPATH=src $PYTHON src/esper/simic_overnight.py \
            --generate-only \
            --episodes $EPISODES_PER_WORKER \
            --start-id $START_ID \
            --device $DEVICE \
            --data-dir "$DATA_DIR" \
            --max-epochs $MAX_EPOCHS \
            $EXTRA_ARGS \
            > "$DATA_DIR/worker_${WORKER_ID}.log" 2>&1 &

        PIDS+=($!)
        WORKER_ID=$((WORKER_ID + 1))
    done
done

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

    CURRENT=$(ls "$DATA_DIR"/episode_*.json 2>/dev/null | wc -l)
    NEW=$((CURRENT - EXISTING))

    echo -ne "\rWorkers: $RUNNING/$TOTAL_WORKERS | Episodes: $NEW/$TOTAL_EPISODES"

    if [ "$RUNNING" -eq 0 ]; then
        echo ""
        echo "========================================"
        echo "All workers complete!"
        echo "Total episodes in $DATA_DIR: $CURRENT"
        echo "========================================"
        break
    fi

    sleep 5
done
