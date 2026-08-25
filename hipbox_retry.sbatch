#!/bin/bash
#SBATCH --job-name=hipbox_retry
#SBATCH --account=oz430
#SBATCH --array=0-15%16
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/fred/oz430/tliu/data/HiPHI/hipbox_retry_logs/slurm_%A_%a.out
#SBATCH --error=/fred/oz430/tliu/data/HiPHI/hipbox_retry_logs/slurm_%A_%a.err

source ~/venvs/hsretargeting/bin/activate
cd /fred/oz430/tliu/code/g1_box_application

mkdir -p /fred/oz430/tliu/data/HiPHI/hipbox_retry_logs
mkdir -p /fred/oz430/tliu/data/HiPHI/retarget_hipbox_retry

TASK=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" hipbox_retry_tasks.txt)

echo "TASK=$TASK"

python src/holosoma_retargeting/holosoma_retargeting/examples/robot_retarget.py \
    --robot g1 \
    --task-type object_interaction \
    --data-format hiphi \
    --data-path /fred/oz430/tliu/data/HiPHI/validation_smokes \
    --task-name "$TASK" \
    --save-dir /fred/oz430/tliu/data/HiPHI/retarget_hipbox_retry
