#!/bin/bash
# Reproduce the DeToNATION paper's OLMo2 experiment at the paper's scale:
# OLMo2-1B, 2 nodes (dp_outer = 2), 10k steps, 4% warmup, stage-1 LR 4e-4,
# sign on for every replicator, demo at 1/32 (topk 1, chunk 32), random at
# 1/16. Each strategy runs synchronously (as in the paper), async (results
# that have arrived when the step's gradients are ready apply in that step, the
# rest later) and delayed (every result applied one step late).
#
#   tests/detonation_benchmarks/submit_paper_repro.sh
#
# Must be run from the repository root on a LUMI login node.

set -euo pipefail

NODES=${NODES:-2}
TIME=${TIME:-4:00:00}
PARTITION=${PARTITION:-standard-g}
GROUP=${GROUP:-olmo2-1b-2n-paper-repro-$(date +%Y%m%d)}
LOG_DIR=${LOG_DIR:-../logs/$GROUP}
WANDB_DIR=${WANDB_DIR:-$(cd .. && pwd)/runs/$GROUP}
MODEL=${MODEL:-allenai/OLMo-2-0425-1B}
DATA=${DATA:-/scratch/project_465002606/frommoge/data/dolma_v1_6_sample/dolma_v1_6_sample_olmo2.bin}
WANDB_ENTITY=${WANDB_ENTITY:-kuchenpferd-org}

BASE="--model $MODEL --data $DATA --train-iters 10000 --lr 4e-4 --seed 0"
BASE="$BASE --wandb-entity $WANDB_ENTITY --wandb-group $GROUP"

# name | strategy flags | extra tag
RUNS=(
    "none|--strategy none|baseline"
    "full-sync|--strategy full_sync --sign|sync"
    "full-async|--strategy full --sign --wait-ms 0|async"
    "full-delayed|--strategy full --sign --wait-ms -1|delayed"
    "demo-sync|--strategy demo --topk 1 --chunk 32 --max-delay 0 --sign|sync"
    "demo-async|--strategy demo --topk 1 --chunk 32 --sign --wait-ms 0|async"
    "demo-delayed|--strategy demo --topk 1 --chunk 32 --sign --wait-ms -1|delayed"
    "random-sync|--strategy random --rate 0.0625 --max-delay 0 --sign|sync"
    "random-async|--strategy random --rate 0.0625 --sign --wait-ms 0|async"
    "random-delayed|--strategy random --rate 0.0625 --sign --wait-ms -1|delayed"
)

mkdir -p "$LOG_DIR" "$WANDB_DIR"
for run in "${RUNS[@]}"; do
    IFS="|" read -r name flags tag <<< "$run"
    args="$BASE $flags --wandb-name $name --wandb-tags paper-repro olmo2-1b $tag"
    args="$args ${EXTRA_ARGS:-}"
    job=$(sbatch --parsable --partition="$PARTITION" --nodes="$NODES" --time="$TIME" \
        --job-name="repro-$name" --output="$LOG_DIR/%x-%j.out" \
        --export=ALL,WANDB_DIR="$WANDB_DIR",SCRIPT=tests/detonation_benchmarks/train_lm.py,ARGS="$args" \
        tests/detonation_benchmarks/lumi_torchrun.sbatch)
    echo "$name $job" | tee -a "$LOG_DIR/jobs.txt"
done
echo "wandb dir: $WANDB_DIR"
