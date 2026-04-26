#!/bin/bash
# Run a full sweep: iterate over all model x replicator x hardware combinations.
#
# Usage:
#   ./run_sweep.sh configs/experiments/replicator_sweep.yaml
#
# Environment variables:
#   NODES, GPUS_PER_NODE, MASTER_ADDR, MASTER_PORT

set -euo pipefail
cd "$(dirname "$0")"

CONFIG=${1:?Usage: $0 <experiment.yaml>}
NODES=${NODES:-2}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}

# Use Python to extract the experiment dimensions from the YAML config
read -r NUM_MODELS NUM_REPLICATORS NUM_HARDWARE < <(python3 -c "
import yaml, sys
with open('$CONFIG') as f:
    c = yaml.safe_load(f)
print(len(c['models']), len(c['replicators']), len(c.get('hardware', [{}])))
")

EXPERIMENT_NAME=$(python3 -c "
import yaml
with open('$CONFIG') as f:
    print(yaml.safe_load(f)['name'])
")

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_BASE="results/${EXPERIMENT_NAME}_${TIMESTAMP}"

echo "=== DeToNATION Benchmark Sweep ==="
echo "Config: $CONFIG"
echo "Models: $NUM_MODELS | Replicators: $NUM_REPLICATORS | Hardware configs: $NUM_HARDWARE"
echo "Total runs: $((NUM_MODELS * NUM_REPLICATORS * NUM_HARDWARE))"
echo "Results dir: $RESULTS_BASE"
echo ""

RUN=0
TOTAL=$((NUM_MODELS * NUM_REPLICATORS * NUM_HARDWARE))

for M in $(seq 0 $((NUM_MODELS - 1))); do
    for R in $(seq 0 $((NUM_REPLICATORS - 1))); do
        for H in $(seq 0 $((NUM_HARDWARE - 1))); do
            RUN=$((RUN + 1))
            MODEL_NAME=$(python3 -c "
import yaml
with open('$CONFIG') as f:
    print(yaml.safe_load(f)['models'][$M]['name'])
")
            REPL_STRATEGY=$(python3 -c "
import yaml
with open('$CONFIG') as f:
    print(yaml.safe_load(f)['replicators'][$R]['strategy'])
")

            OUTPUT_DIR="${RESULTS_BASE}/${MODEL_NAME}/${REPL_STRATEGY}"
            mkdir -p "$OUTPUT_DIR"

            echo "--- [$RUN/$TOTAL] model=$MODEL_NAME replicator=$REPL_STRATEGY ---"

            torchrun \
                --nnodes=$NODES \
                --nproc-per-node=$GPUS_PER_NODE \
                --master_addr=$MASTER_ADDR \
                --master_port=$MASTER_PORT \
                run_benchmark.py \
                --config "$CONFIG" \
                --model-idx $M \
                --replicator-idx $R \
                --hardware-idx $H \
                --output-dir "$OUTPUT_DIR" \
                2>&1 | tee "${OUTPUT_DIR}/run.log" || echo "WARNING: Run $RUN failed, continuing..."

            echo ""
        done
    done
done

echo "=== Sweep complete. Results in $RESULTS_BASE ==="
