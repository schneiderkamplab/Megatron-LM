#!/bin/bash
# Run all 6 DeToNation replicator strategies with Qwen2.5-0.5B.
#
# Usage:
#   ./test_qwen05b_all_strategies.sh
#
# Uses GPUs 0 and 3 (the two with ~18GB free on this machine).
set -uo pipefail

cd /mnt/odinstorage/users/jnn/codes/Megatron-LM

export CUDA_VISIBLE_DEVICES=0,3
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false

CONFIG="tests/detonation_benchmarks/configs/experiments/qwen05b.yaml"
RESULTS_DIR="tests/detonation_benchmarks/results/qwen05b_full_test"
mkdir -p "$RESULTS_DIR"

# Strategy indices in qwen05b.yaml:
#   0: none      1: full      2: demo
#   3: slicing   4: striding  5: random
STRATEGIES=("none" "full" "demo" "slicing" "striding" "random")
NUM_STRATEGIES=${#STRATEGIES[@]}

PASS=0
FAIL=0
RESULTS_FILE="$RESULTS_DIR/summary.json"

echo '{"results": []}' > "$RESULTS_FILE"

for IDX in $(seq 0 $((NUM_STRATEGIES - 1))); do
    STRATEGY="${STRATEGIES[$IDX]}"
    OUTPUT_DIR="${RESULTS_DIR}/${STRATEGY}"
    mkdir -p "$OUTPUT_DIR"
    LOG_FILE="${OUTPUT_DIR}/run.log"

    echo ""
    echo "================================================================"
    echo "  [$((IDX + 1))/${NUM_STRATEGIES}] Strategy: ${STRATEGY}"
    echo "  Output: ${OUTPUT_DIR}"
    echo "  Log:    ${LOG_FILE}"
    echo "================================================================"
    echo ""

    torchrun \
        --nproc-per-node=2 \
        --master_addr=localhost \
        --master_port=$((29500 + IDX)) \
        tests/detonation_benchmarks/run_benchmark.py \
        --config "$CONFIG" \
        --model-idx 0 \
        --replicator-idx "$IDX" \
        --hardware-idx 0 \
        --output-dir "$OUTPUT_DIR" \
        2>&1 | tee "$LOG_FILE"

    EXIT_CODE=${PIPESTATUS[0]}

    if [ "$EXIT_CODE" -eq 0 ]; then
        echo "  -> ${STRATEGY}: PASS"
        PASS=$((PASS + 1))
        STATUS="PASS"
    else
        echo "  -> ${STRATEGY}: FAIL (exit code $EXIT_CODE)"
        FAIL=$((FAIL + 1))
        STATUS="FAIL"
    fi

    # Append to summary JSON.
    python3 -c "
import json
with open('$RESULTS_FILE') as f:
    data = json.load(f)
data['results'].append({'strategy': '$STRATEGY', 'status': '$STATUS', 'exit_code': $EXIT_CODE})
with open('$RESULTS_FILE', 'w') as f:
    json.dump(data, f, indent=2)
"

    echo ""
done

echo ""
echo "================================================================"
echo "  SUMMARY: ${PASS}/${NUM_STRATEGIES} PASSED, ${FAIL} FAILED"
echo "================================================================"
echo ""

# Print per-strategy results
for IDX in $(seq 0 $((NUM_STRATEGIES - 1))); do
    STRATEGY="${STRATEGIES[$IDX]}"
    METRICS_FILE="${RESULTS_DIR}/${STRATEGY}/metrics.json"
    if [ -f "$METRICS_FILE" ]; then
        echo "  $STRATEGY: $(cat "$METRICS_FILE" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(f\"PASS  time={d.get(\"total_time_sec\",\"?\")}s  throughput={d.get(\"throughput_iters_per_sec\",\"?\")} iters/s  final_loss={d.get(\"final_loss\",\"?\")}\")')"
    else
        echo "  $STRATEGY: FAIL (no metrics.json)"
    fi
done

echo ""
echo "Results dir: $RESULTS_DIR"
echo "Summary:     $RESULTS_FILE"
