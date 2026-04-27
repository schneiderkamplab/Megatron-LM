#!/bin/bash
# Run a single benchmark: one model x one replicator x one hardware config.
#
# Usage:
#   ./run.sh --config configs/experiments/smoke_test.yaml --model-idx 0 --replicator-idx 1
#
# Environment variables override defaults:
#   NODES, GPUS_PER_NODE, MASTER_ADDR, MASTER_PORT

set -euo pipefail
cd "$(dirname "$0")"

NODES=${NODES:-2}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}

# Parse --config, --model-idx, --replicator-idx from args
CONFIG=""
MODEL_IDX=0
REPLICATOR_IDX=0
HARDWARE_IDX=0
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)         CONFIG="$2"; shift 2 ;;
        --model-idx)      MODEL_IDX="$2"; shift 2 ;;
        --replicator-idx) REPLICATOR_IDX="$2"; shift 2 ;;
        --hardware-idx)   HARDWARE_IDX="$2"; shift 2 ;;
        --output-dir)     OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$CONFIG" ]; then
    echo "Usage: $0 --config <yaml> [--model-idx N] [--replicator-idx N] [--hardware-idx N] [--output-dir DIR]"
    exit 1
fi

EXTRA_ARGS=""
if [ -n "$OUTPUT_DIR" ]; then
    EXTRA_ARGS="--output-dir $OUTPUT_DIR"
fi

echo "Launching benchmark: config=$CONFIG model=$MODEL_IDX replicator=$REPLICATOR_IDX hw=$HARDWARE_IDX"
echo "Nodes: $NODES, GPUs per node: $GPUS_PER_NODE"

torchrun \
    --nnodes=$NODES \
    --nproc-per-node=$GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    run_benchmark.py \
    --config "$CONFIG" \
    --model-idx $MODEL_IDX \
    --replicator-idx $REPLICATOR_IDX \
    --hardware-idx $HARDWARE_IDX \
    $EXTRA_ARGS


# NODES=1 GPUS_PER_NODE=2 ./run.sh --config configs/experiments/smoke_test.yaml