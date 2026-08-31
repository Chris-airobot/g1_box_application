#!/usr/bin/env bash
set -e

cd /home/samsung/Chris/g1_box_application
source scripts/source_isaacsim_setup.sh

python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-w-object \
  --training.name=mentor_0205_box02289_pilot \
  --training.num_envs=2048 \
  --command.setup_terms.motion_command.params.motion_config.motion_dir=/home/samsung/Chris/g1_box_application/data/mentor_processed/wbt_0205 \
  --command.setup_terms.motion_command.params.motion_config.motion_glob="*_mj_w_obj.npz" \
  --robot.object.object_urdf_path=/home/samsung/Chris/g1_box_application/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/objects_box02289.urdf \
  2>&1 | tee /home/samsung/Chris/g1_box_application/logs/mentor_0205_box02289_train.log
