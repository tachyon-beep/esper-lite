#!/bin/bash
# PPO Training for Policy Tamiyo
#
# Usage: ./scripts/train_ppo.sh [options]
#
# Options:
#   -e, --episodes N      Total episodes to train (default: 100)
#   -n, --n-envs N        Parallel environments (default: 6)
#   -m, --max-epochs N    Max epochs per episode (default: 25)
#   --entropy COEF        Entropy coefficient (default: 0.1)
#   --lr RATE             Learning rate (default: 3e-4)
#   -o, --output FILE     Model save path (default: data/models/ppo_tamiyo.pt)
#   -v, --vectorized      Use vectorized training (default: on)
#   --single              Single environment mode (not vectorized)
#   -h, --help            Show this help
#
# Examples:
#   ./scripts/train_ppo.sh                           # Defaults: 6 envs, 2 GPUs
#   ./scripts/train_ppo.sh -e 200 -n 8               # More episodes, more envs
#   ./scripts/train_ppo.sh --entropy 0.05 --lr 1e-4  # Custom hyperparams
#   ./scripts/train_ppo.sh --single -e 50            # Single env mode

set -e

# Defaults
EPISODES=100
N_ENVS=6
MAX_EPOCHS=25
ENTROPY_COEF=0.1
LR="3e-4"
OUTPUT="data/models/ppo_tamiyo.pt"
VECTORIZED=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--episodes)
            EPISODES="$2"
            shift 2
            ;;
        -n|--n-envs)
            N_ENVS="$2"
            shift 2
            ;;
        -m|--max-epochs)
            MAX_EPOCHS="$2"
            shift 2
            ;;
        --entropy)
            ENTROPY_COEF="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT="$2"
            shift 2
            ;;
        -v|--vectorized)
            VECTORIZED=true
            shift
            ;;
        --single)
            VECTORIZED=false
            shift
            ;;
        -h|--help)
            head -22 "$0" | tail -20
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
    echo "No GPUs detected, falling back to CPU"
    DEVICE="cpu"
    DEVICES=""
else
    DEVICE="cuda:0"
    # Build device list for multi-GPU
    DEVICES=""
    for ((i=0; i<NUM_GPUS; i++)); do
        DEVICES="$DEVICES cuda:$i"
    done
fi

# Create output directory
mkdir -p "$(dirname "$OUTPUT")"

echo "========================================"
echo "PPO Training for Policy Tamiyo"
echo "========================================"
echo "Episodes: $EPISODES"
echo "Parallel envs: $N_ENVS"
echo "Max epochs/episode: $MAX_EPOCHS"
echo "Entropy coef: $ENTROPY_COEF"
echo "Learning rate: $LR"
echo "Output: $OUTPUT"
echo "GPUs: $NUM_GPUS ($DEVICES)"
echo "Vectorized: $VECTORIZED"
echo "========================================"
echo ""

# Build command
CMD="PYTHONPATH=src $PYTHON -m esper.simic.ppo"
CMD="$CMD --episodes $EPISODES"
CMD="$CMD --max-epochs $MAX_EPOCHS"
CMD="$CMD --entropy-coef $ENTROPY_COEF"
CMD="$CMD --lr $LR"
CMD="$CMD --save $OUTPUT"

if [ "$VECTORIZED" = true ]; then
    CMD="$CMD --vectorized"
    CMD="$CMD --n-envs $N_ENVS"
    if [ -n "$DEVICES" ]; then
        CMD="$CMD --devices $DEVICES"
    fi
else
    CMD="$CMD --device $DEVICE"
fi

echo "Running: $CMD"
echo ""

# Run training
eval $CMD

echo ""
echo "========================================"
echo "Training complete!"
echo "Model saved to: $OUTPUT"
echo "========================================"
