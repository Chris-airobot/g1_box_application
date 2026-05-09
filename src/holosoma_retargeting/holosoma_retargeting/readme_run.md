# Retarget Run Commands
#
# All commands are intended to run from the repo root.
#
# Python executable:
# /home/xty/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python

## Stickman (object_interaction, g1)
```bash
/home/xty/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python -u \
  src/holosoma_retargeting/holosoma_retargeting/examples/robot_retarget.py \
  --task-type object_interaction \
  --data_format stickman \
  --data_path actions \
  --task-name HumanoidCarry \
  --robot g1 \
  --save_dir src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking \
  --retargeter.visualize \
  --retargeter.debug
```

/home/xty/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python -u \
  src/holosoma_retargeting/holosoma_retargeting/viser_player.py \
  --qpos-npz src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/HumanoidCarry_original.npz \
  --robot-urdf src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof.urdf \
  --object-urdf src/holosoma_retargeting/holosoma_retargeting/models/box_0p2500_0p2500_0p2500/box_0p2500_0p2500_0p2500.urdf

## OMOMO (smplh, object_interaction, built-in)
```bash
/home/xty/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python -u \
  src/holosoma_retargeting/holosoma_retargeting/examples/robot_retarget.py \
  --data_path src/holosoma_retargeting/holosoma_retargeting/demo_data/OMOMO_new \
  --task-type object_interaction \
  --task-name sub3_largebox_003 \
  --data_format smplh \
  --retargeter.debug \
  --retargeter.visualize
```

## OMOMO (smplh, robot_only, built-in)
```bash
/home/xty/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python -u \
  src/holosoma_retargeting/holosoma_retargeting/examples/robot_retarget.py \
  --data_path src/holosoma_retargeting/holosoma_retargeting/demo_data/OMOMO_new \
  --task-type robot_only \
  --task-name sub3_largebox_003 \
  --data_format smplh \
  --retargeter.debug \
  --retargeter.visualize
```

## Climbing (mocap, built-in)
```bash
/home/xty/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python -u \
  src/holosoma_retargeting/holosoma_retargeting/examples/robot_retarget.py \
  --data_path src/holosoma_retargeting/holosoma_retargeting/demo_data/climb \
  --task-type climbing \
  --task-name mocap_climb_seq_0 \
  --data_format mocap \
  --robot-config.robot-urdf-file src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof_spherehand.urdf \
  --retargeter.debug \
  --retargeter.visualize
```
## convert

/home/xty/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python -u \
  src/holosoma_retargeting/holosoma_retargeting/data_conversion/convert_data_format_mj.py \
  --input_file src/holosoma_retargeting/holosoma_retargeting/demo_results/g1/object_interaction/omomo/sub3_largebox_004_original.npz \
  --output_name src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_004_mj_w_obj.npz \
  --data_format smplh \
  --has_dynamic_object \
  --object_name largebox \
  --output_fps 50 \
  --once

