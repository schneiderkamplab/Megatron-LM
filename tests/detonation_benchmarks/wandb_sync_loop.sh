#!/bin/bash
# Periodically upload offline wandb runs from a LUMI login node while jobs train.
#
#   nohup tests/detonation_benchmarks/wandb_sync_loop.sh ../logs/<group>/jobs.txt \
#       ../runs/<group> > ../logs/<group>/wandb_sync.log 2>&1 &
#
# Syncs every offline run under WANDB_DIR every INTERVAL seconds, and exits
# after a final sync once none of the jobs in JOBS_FILE ("<strategy> <job id>"
# per line) is queued or running.

set -uo pipefail

JOBS_FILE=${1:?usage: wandb_sync_loop.sh JOBS_FILE WANDB_DIR}
WANDB_DIR=${2:?usage: wandb_sync_loop.sh JOBS_FILE WANDB_DIR}
INTERVAL=${INTERVAL:-600}
SIF=${SIF:-/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif}
# The container's wandb 0.21 can no longer sync (the wandb.ai GraphQL schema
# changed in Oct 2026 and its upsert_run introspection crashes silently), so a
# current wandb is pip-installed to this directory and shadows it.
WANDB_NEW=${WANDB_NEW:-/scratch/project_465002606/frommoge/wandb-new}

job_ids=$(awk '{print $2}' "$JOBS_FILE" | paste -sd,)

sync_runs() {
    for run_dir in "$WANDB_DIR"/wandb/offline-run-*; do
        [ -d "$run_dir" ] || continue
        # Skip runs unchanged since this loop last uploaded them: a re-upload
        # resets tags and notes edited in the wandb UI or API.
        marker="$run_dir/.sync_loop_done"
        if [ -e "$marker" ] && [ -z "$(find "$run_dir" -maxdepth 1 -name '*.wandb' -newer "$marker")" ]; then
            continue
        fi
        touch "$marker.new"
        # Drop the marker, or the new client skips the run and live runs
        # never upload their later steps. Re-syncs are idempotent server-side.
        rm -f "$run_dir"/*.wandb.synced
        echo "  $(basename "$run_dir"): $(singularity exec --bind /pfs,/scratch \
            --env PYTHONPATH="$WANDB_NEW" "$SIF" \
            python -m wandb sync "$run_dir" 2>&1 | grep -E 'Finished|error|Error' | tail -1)"
        mv "$marker.new" "$marker"
    done
}

echo "$(date -Is) syncing $WANDB_DIR every ${INTERVAL}s for jobs $job_ids on $(hostname)"
while squeue -h -j "$job_ids" 2>/dev/null | grep -q .; do
    sync_runs
    echo "$(date -Is) synced"
    sleep "$INTERVAL"
done
sync_runs
echo "$(date -Is) all jobs finished; final sync done"
