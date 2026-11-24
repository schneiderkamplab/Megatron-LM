#!/bin/bash
# OLMo2-7B versions of the 1B experiments, synchronous and async only:
#
#   EXPERIMENT=pretrain  from scratch on Dolma, 2 nodes, 10k steps (paper setting)
#   EXPERIMENT=sft       pretrained OLMo-2-1124-7B on the Tulu 3 mixture, 2 nodes, 2k steps
#   EXPERIMENT=socket    from scratch on Dolma, 8 nodes, inter-node traffic over TCP
#   EXPERIMENT=ablation  pretrain for 3k steps: sign step and optimizer variants
#
#   EXPERIMENT=pretrain tests/detonation_benchmarks/submit_7b.sh
#
# Every run saves its final model and queues a quick downstream evaluation
# (eval_lm.sbatch; SUITE=base for pretraining, SUITE=sft for SFT) that starts
# once training succeeds. Must be run from the repository root on a LUMI
# login node.

set -euo pipefail

EXPERIMENT=${EXPERIMENT:?EXPERIMENT must be pretrain, sft, socket or ablation}
PARTITION=${PARTITION:-standard-g}
MODEL=allenai/OLMo-2-1124-7B
WANDB_ENTITY=${WANDB_ENTITY:-kuchenpferd-org}
DOLMA=/scratch/project_465002606/frommoge/data/dolma_v1_6_sample/dolma_v1_6_sample_olmo2.bin
TULU=/scratch/project_465002606/frommoge/data/tulu3_sft/tulu3_olmo2
EXPORT_EXTRA=""

case "$EXPERIMENT" in
    pretrain)
        NODES=2 TIME=14:00:00 SUITE=base
        BASE="--model $MODEL --data $DOLMA --train-iters 10000 --lr 3e-4 --warmup-frac 0.04"
        ;;
    sft)
        NODES=2 TIME=4:00:00 SUITE=sft
        VAL_TOKENS=$(python3 -c "import json; print(json.load(open('$TULU.json'))['val_tokens'])")
        BASE="--model $MODEL --init-pretrained --data $TULU.bin --loss-mask $TULU.mask"
        BASE="$BASE --val-tokens $VAL_TOKENS --train-iters 2000 --lr ${SFT_LR:-1e-5}"
        BASE="$BASE --warmup-frac 0.03 --eval-interval 100"
        ;;
    socket)
        NODES=8 TIME=40:00:00 SUITE=base EXPORT_EXTRA=",NCCL_NET=Socket"
        BASE="--model $MODEL --data $DOLMA --train-iters 10000 --lr 3e-4 --warmup-frac 0.04"
        BASE="$BASE --local-metrics"
        ;;
    ablation)
        # Sign step and optimizer ablation: the pretrain setting shortened to 3k steps.
        NODES=2 TIME=5:00:00 SUITE=base
        BASE="--model $MODEL --data $DOLMA --train-iters 3000 --lr 3e-4 --warmup-frac 0.04"
        ;;
    *)
        echo "unknown EXPERIMENT=$EXPERIMENT" >&2
        exit 1
        ;;
esac

GROUP=${GROUP:-olmo2-7b-$EXPERIMENT-${NODES}n-$(date +%Y%m%d)}
LOG_DIR=${LOG_DIR:-../logs/$GROUP}
WANDB_DIR=${WANDB_DIR:-$(cd .. && pwd)/runs/$GROUP}
CKPT_ROOT=${CKPT_ROOT:-/scratch/project_465002606/frommoge/checkpoints/$GROUP}
BASE="$BASE --seed 0 --wandb-entity $WANDB_ENTITY --wandb-group $GROUP"

# Async: results that have arrived when the step's gradients are ready on the
# GPU apply in that step, the rest later. WAIT_MS optionally waits longer for
# late results (runs with a non-zero wait are named e.g. full-async-w200).
# Older runs named *-adaptive / *-adaptive-w1000 (tag adaptive-v1) used a
# superseded cutoff: up to 200 or 1000 ms of CPU-time polling.
WAIT_MS=${WAIT_MS:-0}
WSUFFIX=""
[ "$WAIT_MS" = "0" ] || WSUFFIX="-w$WAIT_MS"
DEMO="--strategy demo --topk 1 --chunk 32 --sign"
RAND="--strategy random --rate 0.0625 --sign"
FULL="--strategy full --sign"
# name | flags | tag
RUNS=(
    "none|--strategy none|baseline"
    "full-sync|$FULL --max-delay 0|sync"
    "full-async$WSUFFIX|$FULL --wait-ms $WAIT_MS|async"
    "demo-sync|$DEMO --max-delay 0|sync"
    "demo-async$WSUFFIX|$DEMO --wait-ms $WAIT_MS|async"
    "random-sync|$RAND --max-delay 0|sync"
    "random-async$WSUFFIX|$RAND --wait-ms $WAIT_MS|async"
)
if [ "$EXPERIMENT" = ablation ]; then
    # Synchronous only. -nosign: AdamW on the replicated gradient itself.
    # -sgd: plain SGD (no momentum) on the sign, the reference DeMo/DeToNATION
    # update, where the replicator's decaying residual is the only momentum.
    SGD="--optimizer sgd"
    RUNS=(
        "none|--strategy none|baseline"
        "full-sync|$FULL --max-delay 0|sync sign adamw"
        "full-sync-nosign|--strategy full --max-delay 0|sync nosign adamw sanity"
        "demo-sync|$DEMO --max-delay 0|sync sign adamw"
        "demo-sync-nosign|${DEMO% --sign} --max-delay 0|sync nosign adamw"
        "random-sync|$RAND --max-delay 0|sync sign adamw"
        "random-sync-nosign|${RAND% --sign} --max-delay 0|sync nosign adamw"
        "full-sync-sgd-lr3e-4|$FULL --max-delay 0 $SGD --lr 3e-4|sync sign sgd"
        "demo-sync-sgd-lr1e-4|$DEMO --max-delay 0 $SGD --lr 1e-4|sync sign sgd"
        "demo-sync-sgd-lr3e-4|$DEMO --max-delay 0 $SGD --lr 3e-4|sync sign sgd"
        "demo-sync-sgd-lr1e-3|$DEMO --max-delay 0 $SGD --lr 1e-3|sync sign sgd"
        "random-sync-sgd-lr3e-4|$RAND --max-delay 0 $SGD --lr 3e-4|sync sign sgd"
    )
fi

mkdir -p "$LOG_DIR" "$WANDB_DIR" "$CKPT_ROOT"
JOBS_FILE=${JOBS_FILE:-$LOG_DIR/jobs.txt}
for run in "${RUNS[@]}"; do
    IFS="|" read -r name flags tag <<< "$run"
    [[ "$name" =~ ${ONLY:-.} ]] || continue
    ckpt="$CKPT_ROOT/$name"
    args="$BASE $flags --save-dir $ckpt --wandb-name $name"
    args="$args --wandb-tags olmo2-7b $EXPERIMENT $tag ${EXTRA_ARGS:-}"
    train=$(sbatch --parsable --partition="$PARTITION" --nodes="$NODES" --time="$TIME" \
        --job-name="7b-$EXPERIMENT-$name" --output="$LOG_DIR/%x-%j.out" \
        --export="ALL,WANDB_DIR=$WANDB_DIR,SCRIPT=tests/detonation_benchmarks/train_lm.py,ARGS=$args$EXPORT_EXTRA" \
        tests/detonation_benchmarks/lumi_torchrun.sbatch)
    eval=$(sbatch --parsable --partition="$PARTITION" --dependency="afterok:$train" \
        --time=5:00:00 --job-name="7b-eval-$EXPERIMENT-$name" --output="$LOG_DIR/%x-%j.out" \
        --export="ALL,CKPT=$ckpt,SUITE=$SUITE" tests/detonation_benchmarks/eval_lm.sbatch)
    echo "$name $train" | tee -a "$JOBS_FILE"
    echo "eval-$name $eval" >> "$JOBS_FILE"
done
echo "wandb dir: $WANDB_DIR"
echo "sync: setsid -f nohup tests/detonation_benchmarks/wandb_sync_loop.sh $JOBS_FILE $WANDB_DIR > $LOG_DIR/wandb_sync.log 2>&1 < /dev/null"
