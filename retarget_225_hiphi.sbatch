#!/bin/bash
#SBATCH --job-name=hiphi_rt
#SBATCH --account=oz430
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=hiphi_rt_%A_%a.out
#SBATCH --error=hiphi_rt_%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
HIPHI_ROOT="${HIPHI_ROOT:-/fred/oz430/tliu/data/HiPHI}"
RETARGET_PYTHON="${RETARGET_PYTHON:-$HOME/venvs/hsretargeting/bin/python}"
TASK_LIST="${TASK_LIST:-$REPO_ROOT/hiphi_223_tasks.txt}"

INPUT="$HIPHI_ROOT/validation_smokes"
OUT="$HIPHI_ROOT/retarget_225_hiphi"

cd "$REPO_ROOT"
mkdir -p "$OUT"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

name=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASK_LIST")

if [[ -z "$name" ]]; then
    echo "No task for array index $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "Running: $name"

"$RETARGET_PYTHON" \
    src/holosoma_retargeting/holosoma_retargeting/examples/robot_retarget.py \
    --robot g1 \
    --task-type object_interaction \
    --data-format hiphi \
    --data-path "$INPUT" \
    --task-name "$name" \
    --save-dir "$OUT"
