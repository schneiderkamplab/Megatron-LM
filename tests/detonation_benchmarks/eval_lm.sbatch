#!/bin/bash
# Quick downstream evaluation of a Hugging Face checkpoint with
# lm-evaluation-harness, data-parallel over the 8 GCDs of one LUMI-G node.
#
#   sbatch --export=ALL,CKPT=/path/to/ckpt,SUITE=base tests/detonation_benchmarks/eval_lm.sbatch
#
# SUITE=base  zero-shot multiple choice (OLMES core tasks), for pretrained models
# SUITE=sft   the same plus MMLU, GSM8K (8-shot CoT) and IFEval, with the chat
#             template of TOKENIZER (default: the OLMo 2 SFT tokenizer)
# Every task is subsampled (LIMIT, MMLU_LIMIT per MMLU subject, GEN_LIMIT for
# generation tasks): an indicator, not a full benchmark run. Results land in
# $CKPT/eval/ (override with OUT).
#
# Must be submitted from the repository root.

#SBATCH --account=project_465002606
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem-per-gpu=60G
#SBATCH --time=3:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

: "${CKPT:?CKPT must be set}"
SUITE=${SUITE:-base}
LIMIT=${LIMIT:-500}
MMLU_LIMIT=${MMLU_LIMIT:-20}
GEN_LIMIT=${GEN_LIMIT:-250}
# Fixed batch sizes: lm_eval's auto batch search relies on accelerate catching
# "CUDA out of memory", which never matches ROCm's "HIP out of memory".
BATCH=${BATCH:-32}
GEN_BATCH=${GEN_BATCH:-16}
OUT=${OUT:-$CKPT/eval}
TOKENIZER=${TOKENIZER:-allenai/OLMo-2-1124-7B-SFT}
SIF=${SIF:-/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif}
LM_EVAL_ENV=${LM_EVAL_ENV:-/scratch/project_465002606/frommoge/lm-eval-env}

export HF_HOME=${HF_HOME:-/scratch/project_465002606/hf_cache}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MIOPEN_USER_DB_PATH="/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
export TOKENIZERS_PARALLELISM=false
# apex otherwise imports aiter, which JIT-compiles kernels and fails on
# compute nodes (no C++ compiler in the container).
export USE_ROCM_AITER_ROPE_BACKEND=0
mkdir -p "$OUT" "$MIOPEN_USER_DB_PATH"

BASE_TASKS=arc_easy,arc_challenge,hellaswag,piqa,winogrande,sciq,boolq,openbookqa,lambada_openai

run() {  # run <name> <limit> <batch size> <tasks> [extra lm_eval args...]
    local name=$1 limit=$2 batch=$3 tasks=$4
    shift 4
    echo "== $name: $tasks (limit $limit)"
    srun singularity exec --bind /pfs,/scratch,/project --env PYTHONPATH="$LM_EVAL_ENV" "$SIF" \
        accelerate launch --multi_gpu --num_processes 8 --num_machines 1 --mixed_precision no \
        --dynamo_backend no -m lm_eval \
        --model hf --model_args "pretrained=$CKPT,dtype=bfloat16$MODEL_EXTRA" \
        --tasks "$tasks" --limit "$limit" --batch_size "$batch" --seed 1234 \
        --output_path "$OUT/$name" "$@"
}

if [ "$SUITE" = "base" ]; then
    MODEL_EXTRA=""
    run base "$LIMIT" "$BATCH" "$BASE_TASKS"
elif [ "$SUITE" = "sft" ]; then
    MODEL_EXTRA=",tokenizer=$TOKENIZER"
    run base "$LIMIT" "$BATCH" "$BASE_TASKS" --apply_chat_template
    run mmlu "$MMLU_LIMIT" "$BATCH" mmlu --apply_chat_template
    run gsm8k "$GEN_LIMIT" "$GEN_BATCH" gsm8k_cot --apply_chat_template --fewshot_as_multiturn \
        --gen_kwargs max_gen_toks=512
    run ifeval "$GEN_LIMIT" "$GEN_BATCH" ifeval --apply_chat_template --gen_kwargs max_gen_toks=1024
else
    echo "unknown SUITE=$SUITE" >&2
    exit 1
fi
echo "done: results in $OUT"
