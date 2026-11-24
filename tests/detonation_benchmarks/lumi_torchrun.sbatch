#!/bin/bash
# Run a Python script on every GPU of a LUMI allocation (one rank per GCD).
#
#   sbatch --nodes=8 --time=12:00:00 --job-name=deto-random \
#       --export=ALL,SCRIPT=tests/detonation_benchmarks/train_lm.py,ARGS="--strategy random ..." \
#       tests/detonation_benchmarks/lumi_torchrun.sbatch
#
# With ARGS_FILE instead of ARGS, the script runs once per line of that file,
# one after another in the same allocation.
#
# Must be submitted from the repository root.

#SBATCH --account=project_465002606
#SBATCH --partition=standard-g
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --output=%x-%j.out

set -euo pipefail

: "${SCRIPT:?SCRIPT must be set}"
if [ -z "${ARGS_FILE:-}" ] && [ -z "${ARGS+x}" ]; then
    ARGS=""
fi
REPO_ROOT=${REPO_ROOT:-$SLURM_SUBMIT_DIR}
SIF=${SIF:-/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif}
cd "$REPO_ROOT"

export HF_HOME=${HF_HOME:-/scratch/project_465002606/hf_cache}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DIR=${WANDB_DIR:-$REPO_ROOT/../runs}
# Offline by default: LUMI compute nodes can't be relied on to reach wandb.
# wandb_sync_loop.sh uploads the runs from a login node while they train.
export WANDB_MODE=${WANDB_MODE:-offline}
export MIOPEN_USER_DB_PATH="/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH" "$WANDB_DIR"

export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_DEBUG=WARN
export PYTHONPATH="$REPO_ROOT"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# apex otherwise imports aiter, which JIT-compiles kernels and fails on
# compute nodes (no C++ compiler in the container).
export USE_ROCM_AITER_ROPE_BACKEND=0
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT="1${SLURM_JOB_ID:0-4}"
export WORLD_SIZE=$SLURM_NPROCS

# One CPU core group per GCD, matching LUMI-G's GCD-to-NUMA topology.
CPU_BIND="mask_cpu:0x00fe000000000000,0xfe00000000000000"
CPU_BIND="${CPU_BIND},0x0000000000fe0000,0x00000000fe000000"
CPU_BIND="${CPU_BIND},0x00000000000000fe,0x000000000000fe00"
CPU_BIND="${CPU_BIND},0x000000fe00000000,0x0000fe0000000000"

run() {
    echo "job ${SLURM_JOB_ID}: ${SLURM_NNODES} nodes, ${SCRIPT} $1"
    srun --cpu-bind=${CPU_BIND} \
        singularity exec --bind /pfs,/scratch,/project "$SIF" bash -c "
            export RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID
            python ${SCRIPT} $1
        "
}

if [ -n "${ARGS_FILE:-}" ]; then
    status=0
    while IFS= read -r line <&3; do
        [ -n "$line" ] || continue
        run "$line" || status=1  # Keep going so one failure doesn't lose the rest.
    done 3< "$ARGS_FILE"
    exit $status
fi
run "${ARGS:-}"
