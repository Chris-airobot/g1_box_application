#!/bin/bash
#SBATCH --job-name=hiphi_retarget_setup
#SBATCH --account=oz430
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=setup_retargeting_%j.out
#SBATCH --error=setup_retargeting_%j.err

cd /fred/oz430/tliu/code/g1_box_application

echo "Host: $(hostname)"
echo "Start: $(date)"

bash scripts/setup_retargeting.sh

echo "End: $(date)"
