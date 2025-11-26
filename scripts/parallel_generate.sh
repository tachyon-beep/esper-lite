#!/bin/bash
# Parallel episode generation across GPUs
#
# Usage: ./scripts/parallel_generate.sh [episodes_per_worker] [workers_per_gpu] [profile]
#
# Example: ./scripts/parallel_generate.sh 35 3 diagnostic
#   - Runs 3 workers on GPU 0, 3 workers on GPU 1 (6 total)
#   - Each generates 35 episodes = 210 total
#   - Uses 'diagnostic' telemetry profile for rich data
#
# Profiles: minimal, standard, diagnostic, research
# Default: 35 episodes x 6 workers = 210 episodes (no profile = v1 format)

EPISODES_PER_WORKER=${1:-35}
WORKERS_PER_GPU=${2:-3}
PROFILE=${3:-""}

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

echo "========================================"
echo "Parallel Episode Generation"
echo "========================================"
echo "GPUs: $NUM_GPUS"
echo "Workers per GPU: $WORKERS_PER_GPU"
echo "Total workers: $TOTAL_WORKERS"
echo "Episodes per worker: $EPISODES_PER_WORKER"
echo "Total episodes: $TOTAL_EPISODES"
if [ -n "$PROFILE" ]; then
    echo "Telemetry profile: $PROFILE"
fi
echo "========================================"

# Find next available episode ID
EXISTING=$(ls data/simic_episodes/episode_*.json 2>/dev/null | wc -l)
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

        # Build command with optional profile
        PROFILE_ARG=""
        if [ -n "$PROFILE" ]; then
            PROFILE_ARG="--profile $PROFILE"
        fi

        PYTHONPATH=src $PYTHON src/esper/simic_overnight.py \
            --generate-only \
            --episodes $EPISODES_PER_WORKER \
            --start-id $START_ID \
            --device $DEVICE \
            $PROFILE_ARG \
            > "data/simic_episodes/worker_${WORKER_ID}.log" 2>&1 &

        PIDS+=($!)
        WORKER_ID=$((WORKER_ID + 1))
    done
done

echo ""
echo "All workers started. PIDs: ${PIDS[*]}"
echo "Logs: data/simic_episodes/worker_*.log"
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

    CURRENT=$(ls data/simic_episodes/episode_*.json 2>/dev/null | wc -l)
    NEW=$((CURRENT - EXISTING))

    echo -ne "\rWorkers: $RUNNING/$TOTAL_WORKERS | Episodes: $NEW/$TOTAL_EPISODES"

    if [ "$RUNNING" -eq 0 ]; then
        echo ""
        echo "========================================"
        echo "All workers complete!"
        echo "Total episodes: $CURRENT"
        echo "========================================"
        break
    fi

    sleep 5
done
