# Thesis retargeting evaluation pipeline (OMOMO box-carrying)

End-to-end recipe for the 46 box-carrying motions kept in
`src/holosoma/holosoma/motions/*_mj_w_obj.npz`. We rebuild the original
OMOMO inputs, re-run retargeting, and run the quantitative evaluation.

All commands are run from the repo root unless noted.

## 0) One-time setup (paths)

* OMOMO release (already on disk): `/home/limx/code/omomo_release/data/`
  * `train_diffusion_manip_seq_joints24.p` (5 280 sequences -- all 46 thesis sequences live here)
  * `test_diffusion_manip_seq_joints24.p`  (602 sequences)
  * `smpl_all_models/smplh_amass/{neutral,male,female}/model.npz`
* Retargeter assets: the G1 XML scenes in
  `src/holosoma_retargeting/holosoma_retargeting/models/g1/*.xml` reference
  `assets/*.obj` meshes. The repo only ships the `meshes/*.STL` originals,
  so we converted them once with trimesh into
  `models/g1/assets/*.obj` (36 files). This is now committed-able.

## 1) Prepare per-task SMPL-H `.pt` files and `height_dict.pkl`

The retargeter's `smplh` loader expects `<task>.pt` files in InterMimic
layout (a torch tensor of shape `(T, 325)` -- 156 joint positions in
SMPLH_DEMO order at columns `[162:318]` and object pose
`[tx, ty, tz, qx, qy, qz, qw]` at columns `[318:325]`). The eval script
additionally needs a per-subject `height_dict.pkl` for SMPL <-> robot
scaling.

```bash
source /home/limx/.holosoma_deps/miniconda3/bin/activate hsretargeting
python -u src/holosoma_retargeting/holosoma_retargeting/data_utils/prep_omomo_for_thesis.py
```

Defaults already point at:
* OMOMO pickles in `/home/limx/code/omomo_release/data/`
* SMPL-H neutral model in `.../smpl_all_models/smplh_amass/neutral/model.npz`
* Task list = base names of `src/holosoma/holosoma/motions/*_mj_w_obj.npz`
* Outputs written to:
  * `src/holosoma_retargeting/holosoma_retargeting/demo_data/OMOMO_thesis/<task>.pt`
  * `src/holosoma_retargeting/holosoma_retargeting/demo_data/height_dict.pkl`

Verified on 46 sequences: every per-task `.pt` has correct shape and the
quaternion has unit norm. Heights computed in T-pose match the original
OmniRetarget convention (`max_y - min_y` of vertices):

```
sub3:  1.5477 m
sub7:  1.7445 m
sub8:  1.6940 m
sub10: 1.6852 m
sub12: 1.7073 m
```

## 2) Run retargeting on the 46 sequences

```bash
source /home/limx/.holosoma_deps/miniconda3/bin/activate hsretargeting
cd src/holosoma_retargeting
python -u holosoma_retargeting/examples/parallel_robot_retarget.py \
  --task_type object_interaction \
  --robot g1 \
  --data_format smplh \
  --data_dir holosoma_retargeting/demo_data/OMOMO_thesis \
  --save_dir demo_results_thesis/g1/object_interaction/omomo \
  --max_workers 12
```

* Each sequence writes `<task>_original.npz` with keys
  `qpos`, `human_joints`, `fps`, `cost`.
* On a 28-core box this completes in ~7 minutes.

### Known retargeting failures (default hyper-parameters)

11 of the 46 sequences trip the QP solver with `CVXPY solve failed:
infeasible`. They are all subjects where the foot/object configuration
violates the slack budget. List them in the thesis as a hard-failure
bucket:

```
sub10_largebox_083, sub10_largebox_089
sub12_largebox_042, sub12_largebox_045, sub12_largebox_048,
sub12_largebox_050, sub12_largebox_077
sub7_largebox_043,  sub7_largebox_045, sub7_largebox_047
sub8_largebox_043
```

If you want to recover them, try increasing `--retargeter.<...>` slack
weights or relaxing initial-pose constraints; this is a useful
"robustness" ablation for the thesis.

## 3) Run the quantitative evaluation

```bash
source /home/limx/.holosoma_deps/miniconda3/bin/activate hsretargeting
cd src/holosoma_retargeting/holosoma_retargeting
python -u evaluation/eval_retargeting.py \
  --res_dir ../demo_results_thesis/g1/object_interaction/omomo \
  --data_dir demo_data/OMOMO_thesis \
  --data_type robot_object \
  --robot g1 \
  --object_name largebox \
  --max_workers 8
```

The eval script must be launched from inside
`src/holosoma_retargeting/holosoma_retargeting/` (relative `models/...`
paths in MuJoCo XML compiler).

### Reported metrics on the 35 successful retargets (G1, largebox)

| Metric | Mean | Std |
|---|---:|---:|
| Penetration duration (fraction of frames) | **0.0015** | 0.0058 |
| Penetration max depth (m) | **0.0162** | 0.0055 |
| Foot-sliding duration (fraction of contact frames) | **0.0617** | 0.0825 |
| Max toe sliding velocity (m/s) | **0.0125** | 0.0029 |
| Contact preservation | **0.9696** | 0.0649 |
| Optimization cost (final IK loss) | **0.2924** | 0.1352 |

These four physical metrics (penetration, sliding) and the contact
preservation rate are exactly the numbers reported in OmniRetarget's
Table 2 -- you can reuse the same column layout in the thesis.

## 4) Suggested ablations / experiments to put in the thesis

Once the pipeline above runs cleanly, you can add columns to the table
by re-running steps 2 + 3 with different settings.

| Variant | Step 2 changes | What it tests |
|---|---|---|
| **Naive joint-copy (no IK)** | replace retarget with direct joint copy + global rescale | upper bound on errors |
| **Pure IK (no interaction mesh)** | turn off the interaction-mesh costs, keep only joint matching | importance of interaction-mesh term |
| **Ablate foot-sticking** | disable `foot_sticking_sequences` | sliding metric ablation |
| **Ablate object-contact term** | zero out the object-distance cost weight | contact preservation drop |
| **Cross-robot transfer (T1)** | `--robot t1` and run on the same 46 PTs | generalisation |
| **Cross-object size** | `--task_config.object_name=box_0p3500_0p3500_0p3500` (etc.) | rescaling generalisation |
| **Augmentation on/off** | `--augmentation true` | data-augmentation utility |
| **Per-frame timing** | already printed by retarget tqdm; aggregate | wall-clock complexity |

## 5) Where things live

```
src/holosoma_retargeting/holosoma_retargeting/
  data_utils/prep_omomo_for_thesis.py     <- this PR
  demo_data/OMOMO_thesis/                  <- 46 *.pt files (generated)
  demo_data/height_dict.pkl                <- per-subject heights (generated)
  models/g1/assets/*.obj                   <- 36 OBJ meshes (converted from STL once)
  examples/robot_retarget.py               <- single-sequence retargeter
  examples/parallel_robot_retarget.py      <- parallel batch driver
  evaluation/eval_retargeting.py           <- the metric script
src/holosoma_retargeting/
  demo_results_thesis/g1/object_interaction/omomo/*_original.npz <- 35 results
```
