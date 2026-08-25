#!/bin/bash
#SBATCH --job-name=hiphi_g1_smoke
#SBATCH --output=/fred/oz430/tliu/data/HiPHI/retargeted_test/hiphi_g1_smoke_%j.out
#SBATCH --error=/fred/oz430/tliu/data/HiPHI/retargeted_test/hiphi_g1_smoke_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -e

module load gcccore/13.2.0
module load python/3.11.5

source ~/venvs/hsretargeting/bin/activate

cd /fred/oz430/tliu/code/g1_box_application

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python src/holosoma_retargeting/holosoma_retargeting/examples/robot_retarget.py \
    --task-type object_interaction \
    --robot g1 \
    --data-format hiphi \
    --task-name Bringing-carry_0113_pickup_smoke \
    --data-path /fred/oz430/tliu/data/HiPHI/test_motion/Bringing-carry_0113 \
    --save-dir /fred/oz430/tliu/data/HiPHI/retargeted_test
