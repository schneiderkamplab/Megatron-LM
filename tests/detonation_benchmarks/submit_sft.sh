#!/bin/bash
# SFT sweep: pretrained OLMo2-1B fine-tuned on the Tulu 3 OLMo-2 SFT mixture
# (prepare_sft.py output), loss on assistant tokens only. Same replication
# settings as submit_paper_repro.sh; each strategy runs synchronously, async
# (results that have arrived when the step's gradients are ready apply in that
# step, the rest later) and delayed (every result applied one step late).
#
#   tests/detonation_benchmarks/submit_sft.sh
#
# Must be run from the repository root on a LUMI login node.

set -euo pipefail

NODES=${NODES:-2}
TIME=${TIME:-1:30:00}
PARTITION=${PARTITION:-standard-g}
GROUP=${GROUP:-olmo2-1b-sft-${NODES}n-$(date +%Y%m%d)}
LOG_DIR=${LOG_DIR:-../logs/$GROUP}
WANDB_DIR=${WANDB_DIR:-$(cd .. && pwd)/runs/$GROUP}
MODEL=${MODEL:-allenai/OLMo-2-0425-1B}
DATA=${DATA:-/scratch/project_465002606/frommoge/data/tulu3_sft/tulu3_olmo2}
TRAIN_ITERS=${TRAIN_ITERS:-2000}
LR=${LR:-3e-5}
WANDB_ENTITY=${WANDB_ENTITY:-kuchenpferd-org}
# SIGN=0 drops the DeToNATION sign step; such runs get a "-nosign" name suffix.
SIGN=${SIGN:-1}
if [ "$SIGN" = "1" ]; then SIGN_FLAG="--sign"; SUFFIX=""; else SIGN_FLAG=""; SUFFIX="-nosign"; fi

VAL_TOKENS=$(python3 -c "import json; print(json.load(open('$DATA.json'))['val_tokens'])")
BASE="--model $MODEL --init-pretrained --data $DATA.bin --loss-mask $DATA.mask"
BASE="$BASE --val-tokens $VAL_TOKENS --train-iters $TRAIN_ITERS --lr $LR --warmup-frac 0.03"
BASE="$BASE --eval-interval 100 --seed 0 --wandb-entity $WANDB_ENTITY --wandb-group $GROUP"

DEMO="--strategy demo --topk 1 --chunk 32"
RAND="--strategy random --rate 0.0625"
# name | flags | tag
RUNS=(
    "none|--strategy none|baseline"
    "full-sync|--strategy full --max-delay 0 $SIGN_FLAG|sync"
    "full-async|--strategy full $SIGN_FLAG --wait-ms 0|async"
    "full-delayed|--strategy full $SIGN_FLAG --wait-ms -1|delayed"
    "demo-sync|$DEMO --max-delay 0 $SIGN_FLAG|sync"
    "demo-async|$DEMO $SIGN_FLAG --wait-ms 0|async"
    "demo-delayed|$DEMO $SIGN_FLAG --wait-ms -1|delayed"
    "random-sync|$RAND --max-delay 0 $SIGN_FLAG|sync"
    "random-async|$RAND $SIGN_FLAG --wait-ms 0|async"
    "random-delayed|$RAND $SIGN_FLAG --wait-ms -1|delayed"
)

mkdir -p "$LOG_DIR" "$WANDB_DIR"
for run in "${RUNS[@]}"; do
    IFS="|" read -r name flags tag <<< "$run"
    [[ "$name" =~ ${ONLY:-.} ]] || continue  # e.g. ONLY=^random to rerun a subset
    args="$BASE $flags --wandb-name $name$SUFFIX --wandb-tags sft olmo2-1b $tag ${SUFFIX:+nosign} ${EXTRA_ARGS:-}"
    job=$(sbatch --parsable --partition="$PARTITION" --nodes="$NODES" --time="$TIME" \
        --job-name="sft-$name$SUFFIX" --output="$LOG_DIR/%x-%j.out" \
        --export=ALL,WANDB_DIR="$WANDB_DIR",SCRIPT=tests/detonation_benchmarks/train_lm.py,ARGS="$args" \
        tests/detonation_benchmarks/lumi_torchrun.sbatch)
    echo "$name$SUFFIX $job" | tee -a "$LOG_DIR/jobs.txt"
done
echo "wandb dir: $WANDB_DIR"
echo "sync: setsid -f nohup tests/detonation_benchmarks/wandb_sync_loop.sh $LOG_DIR/jobs.txt $WANDB_DIR > $LOG_DIR/wandb_sync.log 2>&1 < /dev/null"
