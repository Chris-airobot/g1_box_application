#!/bin/bash
#SBATCH --job-name=c2a30_bench32768
#SBATCH --partition=milan-gpu
#SBATCH --account=oz430
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/fred/oz430/tliu/logs/c2a30_bench32768_%j.out
#SBATCH --error=/fred/oz430/tliu/logs/c2a30_bench32768_%j.err

set -euo pipefail
module load apptainer

cd /fred/oz430/tliu/code/g1_box_application

export OMNI_KIT_ACCEPT_EULA=1
export PRIVACY_CONSENT=Y
export WANDB_MODE=disabled

apptainer exec --nv --writable-tmpfs \
--bind /fred/oz430/tliu/data/HiPHI:/fred/oz430/tliu/data/HiPHI \
--bind $HOME/c2a_patch/isaaclab.python.kit:/workspace/isaaclab/apps/isaaclab.python.kit \
--bind $HOME/c2a_patch/urdf_converter.py:/workspace/isaaclab/source/isaaclab/isaaclab/sim/converters/urdf_converter.py \
/fred/oz430/tliu/containers/isaac-lab-bookshelf.sif \
bash -lc '
export PYTHONPATH=$HOME/c2a_pydeps:/fred/oz430/tliu/code/g1_box_application/src/holosoma:$PYTHONPATH

nvidia-smi

/isaac-sim/python.sh \
src/holosoma/holosoma/train_agent.py \
exp:g1-29dof-wbt-w-object \
--training.num_envs=32768 \
--training.headless=True \
--algo.config.num_learning_iterations=20
'
