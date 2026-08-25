<div align="center">

# G1 Box Application

### Unitree G1 Box-Carrying Research Repository

**English (default)** · [中文](#中文)

</div>

---

## Overview

This repository is organized as **two separate but connected modules**:

### Module 1 — Carry2Anywhere
The Carry2Anywhere training / distillation / evaluation stack for Unitree G1 box carrying.

### Module 2 — HiPHI → G1 Data Retargeting
Our motion-data extension pipeline that converts HiPHI human box-carrying demonstrations into Unitree G1 + box reference motions for Module 1.

The two modules are intentionally separated: **Module 1 can run with an existing motion dataset, while Module 2 produces additional motion data for retraining Module 1.**

---

# Module 1 — Carry2Anywhere

## Purpose

Carry2Anywhere trains a Unitree G1 humanoid to carry a box using whole-body tracking and teacher–student learning.

The training side contains:

1. **Teacher policy — PPO + Whole-Body Tracking (WBT)**
   - tracks G1 + object reference motions;
   - uses privileged motion-reference observations during training.

2. **Student policy — behavior-cloning / DAgger distillation**
   - removes motion-reference inputs;
   - uses deployable observations such as proprioception, object pose and target position;
   - supports history-conditioned observations.

3. **Evaluation / playback**
   - teacher or student checkpoints can be evaluated in Isaac Sim;
   - the motion loader supports both legacy single-motion loading and multi-motion datasets.

## Our changes to the Carry2Anywhere side

### 1. Updated Unitree G1 hand model

The current G1 model uses the **rubber hand geometry**:

- `left_rubber_hand_link`
- `right_rubber_hand_link`

instead of using the older sphere-hand endpoint representation as the main hand model.

Fixed auxiliary contact links are also included:

- `left_thumb_link`, `right_thumb_link`
- `left_pinky_link`, `right_pinky_link`

These links are used as simple contact / grasp geometry and are not actuated fingers.

### 2. Multi-motion WBT loading

The motion loader now supports:

- `motion_file`
- `motion_files`
- `motion_dir + motion_glob`

Multiple clips are loaded into one flattened motion buffer so training can sample across a motion dataset instead of being restricted to one reference sequence.

## Training

After activating the Isaac Sim / Isaac Lab environment, teacher training can use a motion directory:

```bash
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-w-object \
  --command.setup_terms.motion_command.params.motion_config.motion_dir=/path/to/motions \
  --command.setup_terms.motion_command.params.motion_config.motion_glob="*_w_obj.npz" \
  --training.headless=True
```

The original Carry2Anywhere motion set can still be used directly. Motions produced by Module 2 can also be supplied through the same interface.

---

# Module 2 — HiPHI → G1 Data Retargeting

## Purpose

Module 2 expands the motion dataset by converting **human box-carrying demonstrations from HiPHI** into Unitree G1 object-interaction trajectories.

It is not the policy-training module. Its output is a set of G1 + box motion references that can later be used by Module 1.

## Pipeline summary

```text
Raw HiPHI archives
        ↓
225 original non-mirrored Bringing-carry + box motions
        ↓
BVH + object preprocessing
        ↓
223 valid retarget inputs
        ↓
HiPHI → Unitree G1 interaction-mesh retargeting
        ↓
212 motions in the main retarget pool
        ↓
source-relative hand-contact QC
        ↓
full-body box-penetration QC
        ↓
168 final usable motions
        ↓
124 exact 0.30 × 0.30 × 0.30 m box motions
        ↓
30 Hz → 50 Hz Carry2Anywhere-format NPZ
```

The frozen manifests and final QC decisions are stored under:

```text
pipelines/hiphi_to_g1/manifests/
```

## 1. HiPHI motion selection

We selected HiPHI recordings satisfying:

- action: `Bringing-carry`
- object category: `box`
- `mirrored == false`

This produced **225 original recordings**. Artificially mirrored copies are not included in the baseline dataset.

The source archives are not duplicated into the repository. Instead, the exact source list and archive mapping are preserved in:

- `raw_box_225.txt`
- `raw_box_225_archive_map.csv`

## 2. BVH and object preprocessing

Each selected recording is converted into a compact retargeting NPZ containing the human joint trajectory and the box trajectory.

Important preprocessing details:

### Coordinate system conversion

HiPHI uses a Y-up coordinate convention and centimeter-scale translations. We convert to the robotics Z-up convention:

```text
X_robot =  X_hiphi
Y_robot = -Z_hiphi
Z_robot =  Y_hiphi
```

and convert centimeters to meters.

### BVH translation handling

For joints with BVH translation channels, the animated translation **replaces** the static BVH offset for that frame instead of being added to it.

### Box representation

The object mesh / trajectory is recentered so the stored object pose corresponds to the centered box representation used by the retargeter.

The preprocessing stage also records a local pickup frame used later by contact QC.

From the 225 selected recordings, **223** produced valid preprocessing outputs. Two recordings were excluded during source preprocessing.

## 3. Human-to-G1 joint mapping

The main HiPHI → G1 mapping includes pelvis, legs, feet, arms and hands.

The hand mapping is:

```text
LeftHandMiddle3  → left_rubber_hand_link
RightHandMiddle3 → right_rubber_hand_link
```

This mapping is important because the previous sphere-hand link names were not valid for the current G1 model and could cause incorrect hand/object interaction geometry.

## 4. Interaction-mesh retargeting

The retargeter is based on the HoloSoma interaction-mesh formulation.

For each frame, it preserves the spatial relationship between mapped human joints and object points using Laplacian coordinates in the object frame.

The optimization includes:

- interaction-mesh Laplacian deformation cost;
- joint-limit constraints;
- foot-sticking constraints;
- robot/object non-penetration constraints;
- trust-region / step-size constraints;
- temporal smoothness;
- nominal-pose tracking.

The solver is run frame-by-frame with the previous solution used as the initialization for the next frame.

## 5. Source-contact-aware hand weighting

A fixed strong hand weight was not suitable for every motion because many source demonstrations do not maintain two-hand box contact throughout the carry.

We therefore compute hand weights directly from the **source human hand-to-box distance** in the box local frame.

For each hand:

```text
distance ≤ 1 cm  → 10× tracking weight
distance ≥ 2 cm  → 1× tracking weight
1–2 cm           → smooth interpolation from 10× to 1×
```

This means the retargeter strongly preserves hand/object interaction only when that interaction actually exists in the source motion.

## 6. Source-relative hand-contact QC

After retargeting, contact quality is evaluated over the **two-second interval after pickup**.

For each source hand independently:

- calculate the fraction of frames with hand-to-box distance < 2 cm;
- if that fraction is at least 80%, that hand is considered a required source contact;
- the corresponding G1 hand must also remain within 2 cm for at least 80% of the same interval.

The resulting statuses are:

- `CONTACT_MATCH`
- `CONTACT_FAIL`
- `SOURCE_QC`

`SOURCE_QC` means that neither source hand provides a meaningful contact target under this rule, so the source demonstration itself is not suitable for this contact-preservation test.

## 7. Full-body box-penetration QC

A second QC stage checks whether non-grasp robot bodies penetrate the box.

Intended grasp-contact bodies are excluded from this audit:

- `rubber_hand`
- `thumb`
- `pinky`

The audit records the worst body, frame and penetration depth. Final reviewed decisions are preserved in:

```text
pipelines/hiphi_to_g1/manifests/final_validation.csv
```

Final reviewed result:

| Stage | Count |
|---|---:|
| Selected source motions | 225 |
| Valid preprocessing outputs | 223 |
| Main retarget pool | 212 |
| CONTACT_MATCH | 178 |
| CONTACT_FAIL | 21 |
| SOURCE_QC | 13 |
| BODY_PENETRATION_FAIL | 10 |
| Final usable | **168** |

## 8. 30 cm training subset

The final usable motions contain several box geometries. For the first retraining experiment we use only motions whose box is exactly:

```text
0.30 × 0.30 × 0.30 m
```

This gives **124 motions**.

The exact list is stored in:

```text
pipelines/hiphi_to_g1/manifests/usable_30cm_124.txt
```

## 9. Convert to Carry2Anywhere format

Retargeted HiPHI trajectories are 30 Hz. Carry2Anywhere reference motions are converted to 50 Hz.

Retargeted qpos layout:

```text
[0:3]   root position
[3:7]   root quaternion
[7:36]  29 G1 joints
[36:39] object position
[39:43] object quaternion
```

The converter must use:

```text
--has-dynamic-object
--no-use-omniretarget-data
```

The resulting NPZ files contain the G1 joint, body and object states required by the WBT motion loader in Module 1.

## Reproducing Module 2

The documented wrapper is:

```bash
./pipelines/hiphi_to_g1/run_pipeline.sh prepare
./pipelines/hiphi_to_g1/run_pipeline.sh retarget
./pipelines/hiphi_to_g1/run_pipeline.sh contact-qc
./pipelines/hiphi_to_g1/run_pipeline.sh body-qc
./pipelines/hiphi_to_g1/run_pipeline.sh convert-30cm
```

More implementation details are available in:

[`pipelines/hiphi_to_g1/README.md`](pipelines/hiphi_to_g1/README.md)

---

## Repository structure

```text
g1_box_application/
├── src/holosoma/                 # Module 1: training / evaluation / WBT
├── src/holosoma_retargeting/     # Module 2: human → G1 retargeting
├── pipelines/hiphi_to_g1/        # HiPHI pipeline wrapper + manifests
├── scripts/                      # preprocessing and QC utilities
├── retarget_225_hiphi.sbatch     # SLURM retargeting job
└── README.md
```

## Acknowledgements

This repository builds on the original Carry2Anywhere codebase and the HoloSoma motion-retargeting framework, together with Isaac Sim / Isaac Lab, HiPHI, and the Unitree G1 model. Please follow the licenses and terms of the corresponding upstream projects and datasets.

---

<a id="中文"></a>

# 中文

## 项目结构

本仓库现在明确分成 **两个独立但相互连接的模块**：

### 模块 1 — Carry2Anywhere
负责 Unitree G1 搬箱任务的策略训练、教师–学生蒸馏和评估。

### 模块 2 — HiPHI → G1 数据重定向
负责把 HiPHI 中的人类搬箱动作转换成 Unitree G1 + 箱子的参考动作，并把生成的数据提供给模块 1 重新训练。

也就是说：**模块 1 是策略学习系统；模块 2 是数据生成系统。** 两者不再混在一起描述。

---

# 模块 1 — Carry2Anywhere

Carry2Anywhere 部分保留原有的整体训练框架：

1. PPO + Whole-Body Tracking 教师策略；
2. 基于 BC / DAgger 的学生策略蒸馏；
3. Isaac Sim 中的评估和可视化；
4. 单动作以及多动作数据加载。

### 我们对该模块的主要修改

**G1 手部模型**现在使用：

```text
left_rubber_hand_link
right_rubber_hand_link
```

而不是把旧的 sphere-hand 末端表示作为主要手部模型。

同时保留固定的：

```text
thumb_link
pinky_link
```

作为简单的抓取 / 接触几何。

另外，WBT motion loader 已支持：

```text
motion_file
motion_files
motion_dir + motion_glob
```

因此可以直接加载由模块 2 生成的大规模动作集合进行训练。

---

# 模块 2 — HiPHI → G1 数据重定向

## 数据流程

```text
HiPHI 原始数据
→ 225 条非镜像 Bringing-carry + box 动作
→ BVH / 箱子预处理
→ 223 条有效 retarget 输入
→ HiPHI → Unitree G1 retargeting
→ 212 条主要 retarget 结果
→ 源动作相对手部接触 QC
→ 全身–箱子穿透 QC
→ 168 条最终可用动作
→ 124 条 30 cm 立方体动作
→ 30 Hz → 50 Hz Carry2Anywhere NPZ
```

## 1. 数据筛选

筛选规则：

- `frame_lu == Bringing-carry`
- `object_categories == box`
- `mirrored == false`

最终得到 **225 条原始非镜像动作**。

## 2. BVH 与物体预处理

HiPHI 的坐标和单位会转换为机器人使用的 Z-up 米制坐标：

```text
X_robot =  X_hiphi
Y_robot = -Z_hiphi
Z_robot =  Y_hiphi
```

BVH 中存在平移通道时，动画平移值会替代该帧的静态 OFFSET，而不是简单相加。

箱子轨迹也会重新居中，使其与 retargeting 中使用的中心化箱子几何一致。

225 条动作中有 **223 条**通过预处理。

## 3. HiPHI → G1 手部映射

关键手部映射为：

```text
LeftHandMiddle3  → left_rubber_hand_link
RightHandMiddle3 → right_rubber_hand_link
```

这也是当前版本相比旧 sphere-hand 表示的重要修正。

## 4. Interaction-Mesh Retargeting

我们沿用 HoloSoma 的 interaction-mesh 方法，在物体坐标系下构造人体关键点与箱子点之间的 Laplacian 关系，并逐帧优化 G1 姿态。

优化同时考虑：

- interaction-mesh Laplacian 误差；
- 关节限制；
- 脚部 sticking；
- 机器人与箱子非穿透；
- step-size / trust-region；
- 时间平滑；
- nominal pose tracking。

## 5. 接触感知手部权重

我们没有继续使用固定的强手部权重，而是根据**源人体手部到箱子的距离**动态决定权重：

```text
≤ 1 cm  → 10×
≥ 2 cm  → 1×
1–2 cm  → 10× 到 1× 平滑插值
```

因此只有源动作确实存在手–箱接触时，retargeter 才会强制保留该交互关系。

## 6. 手部接触 QC

在 pickup 后 2 秒内，对左右手分别检查：

- 如果源人体某只手有至少 80% 的帧距离箱子 < 2 cm，则该手被视为必须保持的接触；
- 对应 G1 手也必须达到相同标准。

结果分为：

```text
CONTACT_MATCH
CONTACT_FAIL
SOURCE_QC
```

## 7. 全身穿透 QC

随后检查非抓取身体部位是否穿入箱子。

抓取相关几何会从该检查中排除：

```text
rubber_hand
thumb
pinky
```

最终审核结果保存在：

```text
pipelines/hiphi_to_g1/manifests/final_validation.csv
```

最终得到 **168 条可用动作**。

## 8. 30 cm 子集与格式转换

第一轮重新训练只使用箱子尺寸严格为：

```text
0.30 × 0.30 × 0.30 m
```

的 **124 条动作**。

这些动作从 30 Hz 转换到 Carry2Anywhere 使用的 50 Hz NPZ 格式，然后交给模块 1 的 WBT motion loader。

## 复现命令

```bash
./pipelines/hiphi_to_g1/run_pipeline.sh prepare
./pipelines/hiphi_to_g1/run_pipeline.sh retarget
./pipelines/hiphi_to_g1/run_pipeline.sh contact-qc
./pipelines/hiphi_to_g1/run_pipeline.sh body-qc
./pipelines/hiphi_to_g1/run_pipeline.sh convert-30cm
```

详细实现见：

[`pipelines/hiphi_to_g1/README.md`](pipelines/hiphi_to_g1/README.md)
