#!/bin/bash
# Submit one LUMI job per replication strategy, all logging to one wandb group.
#
#   NODES=8 TIME=12:00:00 TRAIN_ITERS=10000 tests/detonation_benchmarks/submit_lumi_sweep.sh
#
# Must be run from the repository root on a LUMI login node. Every run uses the
# same seed (model init, data order, validation set). Replicator runs use the
# DeToNATION sign step unless SIGN=0; the 'none' baseline never does. Extra
# train_lm.py arguments can be appended via EXTRA_ARGS.

set -euo pipefail

NODES=${NODES:-8}
TIME=${TIME:-12:00:00}
PARTITION=${PARTITION:-standard-g}
STRATEGIES=${STRATEGIES:-"none full full_sync demo slicing striding random"}
GROUP=${GROUP:-olmo2-7b-${NODES}n-$(date +%Y%m%d-%H%M)}
LOG_DIR=${LOG_DIR:-../logs/$GROUP}
# One wandb directory per sweep, so wandb_sync_loop.sh uploads exactly these runs.
WANDB_DIR=${WANDB_DIR:-$(cd .. && pwd)/runs/$GROUP}
MODEL=${MODEL:-allenai/OLMo-2-1124-7B}
DATA=${DATA:-/scratch/project_465002606/frommoge/data/dolma_v1_6_sample/dolma_v1_6_sample_olmo2.bin}
TRAIN_ITERS=${TRAIN_ITERS:-10000}
SEED=${SEED:-0}
SIGN=${SIGN:-1}
WANDB_ENTITY=${WANDB_ENTITY:-kuchenpferd-org}

mkdir -p "$LOG_DIR" "$WANDB_DIR"
for strategy in $STRATEGIES; do
    args="--model $MODEL --data $DATA --strategy $strategy --train-iters $TRAIN_ITERS --seed $SEED"
    if [ "$SIGN" = "1" ] && [ "$strategy" != "none" ]; then
        args="$args --sign"
    fi
    args="$args --wandb-entity $WANDB_ENTITY --wandb-group $GROUP --wandb-name $strategy"
    args="$args ${EXTRA_ARGS:-}"
    job=$(sbatch --parsable --partition="$PARTITION" --nodes="$NODES" --time="$TIME" \
        --job-name="deto-$strategy" --output="$LOG_DIR/%x-%j.out" \
        --export=ALL,WANDB_DIR="$WANDB_DIR",SCRIPT=tests/detonation_benchmarks/train_lm.py,ARGS="$args" \
        tests/detonation_benchmarks/lumi_torchrun.sbatch)
    echo "$strategy $job" | tee -a "$LOG_DIR/jobs.txt"
done
echo "wandb dir: $WANDB_DIR"
