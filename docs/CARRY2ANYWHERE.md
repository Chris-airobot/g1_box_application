# Module 1 — Carry2Anywhere

[← Back to main README](../README.md)

This module contains the humanoid box-carrying learning stack used in this repository. It is based on the original Carry2Anywhere codebase and is kept separate from the HiPHI data-retargeting pipeline.

## Purpose

Carry2Anywhere trains a Unitree G1 whole-body policy for carrying a box to target locations. The learning stack contains two main stages:

1. **Teacher policy** — PPO with whole-body tracking (WBT) and privileged reference information.
2. **Student policy** — policy distillation using a deployable observation set without motion-reference inputs.

The retargeted motions produced by Module 2 can be used as additional reference motions for the teacher.

## Changes in this repository

### Multi-motion training

The WBT motion loader supports multiple motion clips through either:

- `motion_files`, or
- `motion_dir + motion_glob`.

The clips are loaded into a single motion buffer so training can sample across a larger reference-motion set.

Relevant implementation:

`src/holosoma/holosoma/managers/command/terms/wbt.py`

### G1 box configuration

The current experiments use the Unitree G1 29-DoF model and the box geometry associated with the selected motion dataset. For the first HiPHI retraining stage, the selected subset uses an exact `0.30 × 0.30 × 0.30 m` box.

### Reference-motion format

Carry2Anywhere training motions use the converted WBT NPZ format containing robot joint states, body poses and velocities, and object states.

The HiPHI conversion step in Module 2 converts retargeted motions from 30 Hz to the 50 Hz reference format used here.

## Teacher training

A typical teacher run uses the G1 + object WBT experiment and a motion directory:

```bash
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-w-object \
  --command.setup_terms.motion_command.params.motion_config.motion_dir=<motion_dir> \
  --command.setup_terms.motion_command.params.motion_config.motion_glob="*_w_obj.npz" \
  --training.headless=True
```

The number of parallel environments should be chosen according to available GPU memory.

## Student distillation

The student is distilled from the teacher with a deployment-oriented observation set. The student does not require motion-reference inputs at inference time; it uses the configured proprioceptive/object/task observations and history instead.

The distillation and observation configurations remain under the HoloSoma / Carry2Anywhere training code in `src/holosoma/holosoma/`.

## Evaluation

Teacher and student checkpoints can be evaluated through:

`src/holosoma/holosoma/eval_agent.py`

Both GUI and headless evaluation are supported by the existing simulator configuration.

## Relationship to Module 2

Module 1 is the **learning and control module**.

Module 2 is the **data preparation and retargeting module**.

The interface between them is the converted motion-reference NPZ dataset:

```text
HiPHI human-object motion
        ↓
Module 2: retarget + QC + conversion
        ↓
G1 + box WBT motion NPZ
        ↓
Module 1: teacher training / distillation / evaluation
```
