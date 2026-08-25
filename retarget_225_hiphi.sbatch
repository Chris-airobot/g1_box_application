#!/bin/bash
#SBATCH --job-name=hiphi_rt
#SBATCH --account=oz430
#SBATCH --array=0-222%32
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/fred/oz430/tliu/data/HiPHI/retarget_225_hiphi_logs/slurm_%A_%a.out
#SBATCH --error=/fred/oz430/tliu/data/HiPHI/retarget_225_hiphi_logs/slurm_%A_%a.err

cd /fred/oz430/tliu/code/g1_box_application
source ~/venvs/hsretargeting/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

INPUT=/fred/oz430/tliu/data/HiPHI/validation_smokes
OUT=/fred/oz430/tliu/data/HiPHI/retarget_225_hiphi

name=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" hiphi_223_tasks.txt)

echo "Running: $name"

python src/holosoma_retargeting/holosoma_retargeting/examples/robot_retarget.py \
    --robot g1 \
    --task-type object_interaction \
    --data-format hiphi \
    --data-path "$INPUT" \
    --task-name "$name" \
    --save-dir "$OUT"
