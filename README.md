<div align="center">

# G1 Box Application

### HiPHI → Unitree G1 Retargeting + Carry2Anywhere Training
### HiPHI → 宇树 G1 动作重定向 + Carry2Anywhere 训练

**Unitree G1 · 29 DoF · Rubber Hands · Human–Object Motion Retargeting · Whole-Body Tracking**

</div>

---

## Overview / 项目简介

| English | 中文 |
|---|---|
| This repository is our research version of the Carry2Anywhere box-carrying pipeline. The main extension is a new **HiPHI → Unitree G1 human–object retargeting pipeline**, together with a modified G1 hand model, contact-aware retargeting, quality-control stages, multi-motion loading, and retraining support. | 本仓库是我们基于 Carry2Anywhere 继续开发的箱子搬运研究版本。核心扩展是新的 **HiPHI → 宇树 G1 人–物交互动作重定向流程**，同时包含修改后的 G1 手部模型、接触感知 retargeting、质量检查、多动作加载以及重新训练支持。 |
| The original Carry2Anywhere motion set is kept as a baseline/reference. Our current dataset extension starts from HiPHI human box-carrying recordings and converts them into G1 + box reference trajectories for whole-body-tracking policy training. | 原始 Carry2Anywhere 动作集保留作为 baseline / 参考。我们当前的数据扩展从 HiPHI 人类搬箱动作开始，将其转换为 G1 + 箱子的参考轨迹，用于全身跟踪策略训练。 |

## What changed / 我们修改了什么

| English | 中文 |
|---|---|
| **Robot hand model:** the G1 model now uses fixed **rubber hand meshes** (`left_rubber_hand_link`, `right_rubber_hand_link`) instead of relying on the old sphere-hand endpoint representation. | **机器人手部模型：** 当前 G1 使用固定的 **rubber hand 网格模型**（`left_rubber_hand_link`, `right_rubber_hand_link`），不再依赖旧的 sphere-hand 末端表示。 |
| **HiPHI hand mapping:** `LeftHandMiddle3 → left_rubber_hand_link` and `RightHandMiddle3 → right_rubber_hand_link`. The fixed `thumb_link` and `pinky_link` bodies are retained as auxiliary grasp/contact geometry rather than actuated fingers. | **HiPHI 手部映射：** `LeftHandMiddle3 → left_rubber_hand_link`，`RightHandMiddle3 → right_rubber_hand_link`。固定的 `thumb_link` 和 `pinky_link` 作为辅助抓取 / 接触几何保留，而不是可驱动手指。 |
| **Contact-aware retargeting:** hand tracking is strengthened only when the source human hand is actually close to the box. | **接触感知 retargeting：** 只有当源人体手部真正接近箱子时，才增强机器人手部的跟踪权重。 |
| **HiPHI preprocessing + QC:** raw HiPHI data are selected, converted, retargeted, checked for hand-contact preservation, audited for body–box penetration, and then converted to the Carry2Anywhere motion format. | **HiPHI 预处理 + QC：** 对原始 HiPHI 数据进行筛选、坐标转换、retargeting、手部接触保持检查、身体–箱子穿透检查，最后转换成 Carry2Anywhere 训练格式。 |
| **Multi-motion training:** the WBT motion loader supports `motion_files` or `motion_dir + motion_glob`, loads multiple clips, and builds one flattened GPU buffer for training. | **多动作训练：** WBT motion loader 支持 `motion_files` 或 `motion_dir + motion_glob`，可同时加载多条动作，并构建统一的 GPU motion buffer 用于训练。 |

---

## Dataset pipeline / 数据流程

```text
Raw HiPHI archives
        ↓
225 original non-mirrored Bringing-carry + box motions
        ↓
BVH + object preprocessing
        ↓
223 retarget inputs
        ↓
HiPHI → Unitree G1 retargeting
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
30 Hz → 50 Hz Carry2Anywhere conversion
        ↓
Whole-body-tracking teacher training
```

| Stage | Result | 阶段 | 结果 |
|---|---:|---|---:|
| Selected raw HiPHI box carries | 225 | 筛选出的 HiPHI 搬箱动作 | 225 |
| Valid preprocessing outputs | 223 | 有效预处理输出 | 223 |
| Main retarget pool | 212 | 主要 retarget 结果池 | 212 |
| Contact match | 178 | 接触保持成功 | 178 |
| Contact fail | 21 | 接触保持失败 | 21 |
| Source QC | 13 | 源动作接触质量不足 | 13 |
| Body penetration fail | 10 | 身体–箱子穿透失败 | 10 |
| Final usable | **168** | 最终可用 | **168** |
| Exact 30 cm cube subset | **124** | 30 cm 立方体子集 | **124** |

The exact selected motions and frozen QC results are preserved under [`pipelines/hiphi_to_g1/manifests/`](pipelines/hiphi_to_g1/manifests/).

精确的动作列表以及最终冻结的 QC 结果保存在 [`pipelines/hiphi_to_g1/manifests/`](pipelines/hiphi_to_g1/manifests/) 中。

---

# HiPHI → G1 retargeting method / HiPHI → G1 动作重定向方法

## 1. Source selection / 源数据筛选

| English | 中文 |
|---|---|
| We select HiPHI motions whose action is `Bringing-carry`, whose object category is exactly `box`, and whose `mirrored` flag is false. This gives **225 original recordings** without artificially mirrored copies. | 我们筛选 HiPHI 中动作类别为 `Bringing-carry`、物体类别严格为 `box`、并且 `mirrored=false` 的动作，共得到 **225 条原始录制动作**，不包含人工左右镜像数据。 |
| `raw_box_225_archive_map.csv` records which raw HiPHI archive contains each selected motion, so the large raw archive does not need to be duplicated into this repository. | `raw_box_225_archive_map.csv` 保存每条动作所在的原始 HiPHI 压缩包，因此无需将庞大的原始数据复制进本仓库。 |

## 2. BVH + object preprocessing / BVH + 物体预处理

| English | 中文 |
|---|---|
| HiPHI BVH global joint positions are reconstructed using BVH forward kinematics. Translation channels **replace** the BVH joint offset rather than being added on top of it. | 通过 BVH forward kinematics 重建 HiPHI 的全局关节位置。带 translation channel 的关节中，translation **替代** BVH joint offset，而不是与 offset 相加。 |
| HiPHI source coordinates are Y-up and stored in centimetres. They are converted to robotics Z-up coordinates and metres using `X' = X`, `Y' = -Z`, `Z' = Y`, followed by cm → m scaling. | HiPHI 原始坐标系为 Y-up，长度单位为厘米。转换到机器人常用的 Z-up 米制坐标：`X' = X`、`Y' = -Z`、`Z' = Y`，并执行 cm → m。 |
| The object trajectory origin is not assumed to be the geometric box centre. The object mesh is analysed, re-centred, and its true `box_size` is stored with the motion. | 不假设 HiPHI 物体轨迹原点就是箱子几何中心。预处理会分析物体网格、重新中心化，并将真实 `box_size` 与动作一起保存。 |
| Pickup is detected during preprocessing and stored as `local_pickup_frame`, which is then reused by later QC instead of re-detecting pickup from the retargeted motion. | 预处理阶段检测 pickup，并保存为 `local_pickup_frame`。后续 QC 直接使用这一帧，而不是在 retarget 结果中重新猜测 pickup 时刻。 |

Run:

```bash
export HIPHI_ROOT=/path/to/HiPHI
./pipelines/hiphi_to_g1/run_pipeline.sh prepare
```

On the Swinburne HPC used for the validated dataset:

```bash
export HIPHI_ROOT=/fred/oz430/tliu/data/HiPHI
export PYTHON_BIN=$HOME/venvs/hsretargeting/bin/python
```

## 3. Human-to-G1 joint mapping / 人体到 G1 关节映射

The HiPHI retargeter uses a reduced set of body landmarks rather than trying to reproduce every human finger joint.

HiPHI retargeting 使用精简的人体关键点，而不是尝试让 G1 重现所有人体手指关节。

| HiPHI landmark | Unitree G1 link |
|---|---|
| `Spine1` | `pelvis_contour_link` |
| `LeftUpLeg` | `left_hip_pitch_link` |
| `LeftLeg` | `left_knee_link` |
| `LeftFoot` | `left_ankle_intermediate_1_link` |
| `LeftToeBase` | `left_ankle_roll_sphere_5_link` |
| `RightUpLeg` | `right_hip_pitch_link` |
| `RightLeg` | `right_knee_link` |
| `RightFoot` | `right_ankle_intermediate_1_link` |
| `RightToeBase` | `right_ankle_roll_sphere_5_link` |
| `LeftArm` | `left_shoulder_roll_link` |
| `LeftForeArm` | `left_elbow_link` |
| **`LeftHandMiddle3`** | **`left_rubber_hand_link`** |
| `RightArm` | `right_shoulder_roll_link` |
| `RightForeArm` | `right_elbow_link` |
| **`RightHandMiddle3`** | **`right_rubber_hand_link`** |

The hand mapping is important: an earlier sphere-hand endpoint is not used for the HiPHI pipeline. The source middle-finger endpoint is mapped directly to the current rubber-hand body.

手部映射非常关键：HiPHI 流程不再使用旧的 sphere-hand endpoint，而是将人体中指末端直接映射到当前 rubber-hand body。

## 4. Interaction-mesh retargeting / Interaction Mesh 动作重定向

| English | 中文 |
|---|---|
| The implementation builds an **interaction mesh** from mapped human landmarks and sampled object points. The mesh is expressed relative to the object so that the optimization preserves human–object spatial relationships, not only absolute joint positions. | 实现会使用人体映射关键点和物体采样点构建 **interaction mesh**。该 mesh 在物体坐标系中表达，使优化目标保持人体–物体之间的空间关系，而不仅仅是绝对关节位置。 |
| Laplacian coordinates of this interaction mesh provide the main retargeting objective. This helps preserve how the torso, arms, hands and feet are arranged relative to the carried box. | 使用 interaction mesh 的 Laplacian coordinates 作为主要 retargeting 目标，从而保持躯干、手臂、手部、脚部相对于箱子的几何关系。 |
| Each frame is solved with a differential-IK / SQP-style optimization. | 每一帧通过 differential-IK / SQP 风格的优化求解。 |

The optimization includes:

- Laplacian interaction-mesh matching cost
- foot-sticking constraints
- robot–object non-penetration constraints
- robot joint-limit constraints
- SOC trust-region / step-size constraint
- temporal smoothness cost
- nominal-pose tracking cost

优化中包含：

- Laplacian interaction-mesh 匹配代价
- 脚部固定约束
- 机器人–物体 non-penetration 约束
- 机器人关节限位
- SOC trust-region / step-size 约束
- 时间平滑代价
- nominal pose 跟踪代价

## 5. Source-contact-aware hand weighting / 基于源动作接触的手部权重

A fixed large hand weight was found to be too aggressive because it forces the robot to chase the box even during source frames where the human hand is not actually in contact.

固定的大手部权重过于激进，因为即使源人体在某些帧并没有真正接触箱子，它也会强迫机器人手部追踪箱子。

For each frame we compute the source fingertip-to-box distance in the object-local frame and apply:

每一帧都在物体局部坐标系中计算源人体手部到箱子表面的距离，并设置：

| Source hand distance / 源手部距离 | Retarget weight / Retarget 权重 |
|---|---:|
| `≤ 1 cm` | `10×` |
| `1–2 cm` | smooth linear interpolation / 平滑线性插值 |
| `≥ 2 cm` | `1×` |

This is applied independently to `LeftHandMiddle3` and `RightHandMiddle3` before the Laplacian objective is solved.

该权重分别独立应用于 `LeftHandMiddle3` 和 `RightHandMiddle3`，然后进入 Laplacian 优化目标。

## 6. Retarget execution / 执行 retarget

The SLURM wrapper automatically builds an array from the available preprocessed smoke clips:

SLURM wrapper 会根据已经生成的 smoke clips 自动建立 array job：

```bash
./pipelines/hiphi_to_g1/run_pipeline.sh retarget
```

Equivalent retarget configuration:

```text
robot       = g1
task type   = object_interaction
data format = hiphi
output fps  = 30 Hz
```

The validated run produced 223 retarget inputs; 212 motions entered the final main retarget pool after solver failures/recovery were resolved.

最终验证流程中共有 223 个 retarget 输入；在处理 solver failure / recovery 后，212 条动作进入最终主要 retarget pool。

---

# Quality control / 质量检查

## 7. Source-relative hand-contact QC / 相对源动作的手部接触 QC

| English | 中文 |
|---|---|
| QC is evaluated over the **two seconds after the stored pickup frame**. | QC 在保存的 pickup frame 之后 **2 秒**的区间内进行。 |
| For each source hand, compute the fraction of frames whose distance to the box is below 2 cm. If that fraction is at least 80%, that hand is considered a required source contact. | 对源人体左右手分别计算距离箱子小于 2 cm 的帧比例。如果比例 ≥ 80%，则该手被认为是源动作中需要保持的接触手。 |
| Only hands required by the source motion are required from G1. This avoids incorrectly rejecting legitimate one-handed carrying motions. | 只有源动作中实际需要接触的手，才要求 G1 同样保持接触，因此不会错误拒绝合理的单手搬运动作。 |
| The corresponding G1 rubber hand must also remain within 2 cm for at least 80% of the interval. | 对应的 G1 rubber hand 也必须在至少 80% 的区间内保持距离箱子小于 2 cm。 |

Statuses:

```text
CONTACT_MATCH   source-required contact pattern is preserved
CONTACT_FAIL    required source contact is not sufficiently preserved
SOURCE_QC       neither source hand provides a meaningful contact target
```

Validated counts in the 212-motion pool:

```text
CONTACT_MATCH = 178
CONTACT_FAIL  = 21
SOURCE_QC     = 13
```

Run:

```bash
./pipelines/hiphi_to_g1/run_pipeline.sh contact-qc
```

## 8. Full-body box-penetration audit / 全身箱子穿透检查

| English | 中文 |
|---|---|
| A separate MuJoCo audit computes the worst robot-body-to-box penetration across the trajectory. | 单独的 MuJoCo audit 会计算整条轨迹中机器人身体与箱子之间最严重的穿透。 |
| Intended grasp-contact bodies are excluded from this audit: `rubber_hand`, `thumb`, and `pinky`. | 预期用于抓取接触的 body 不计入该 audit：`rubber_hand`、`thumb`、`pinky`。 |
| Torso/shoulder/wrist contacts are reviewed separately rather than applying one universal threshold to every body. | 对躯干 / 肩部 / 手腕等接触进行单独检查，而不是对所有 body 使用一个统一阈值。 |

Run:

```bash
./pipelines/hiphi_to_g1/run_pipeline.sh body-qc
```

The frozen final decision for each motion is stored in:

```text
pipelines/hiphi_to_g1/manifests/final_validation.csv
```

Final result: **168 usable retargeted motions**.

最终结果：**168 条可用 retarget 动作**。

---

# Carry2Anywhere conversion / Carry2Anywhere 数据转换

## 9. Exact 30 cm subset / 30 cm 箱子子集

Our first retraining set uses only motions whose original box geometry is exactly:

第一版重新训练数据只使用原始箱子尺寸严格为：

```text
0.30 × 0.30 × 0.30 m
```

This gives **124 motions**. We do not replace the box geometry after retargeting; each motion remains paired with the geometry used during retargeting.

共得到 **124 条动作**。我们不会在 retarget 完成后随意替换箱子尺寸；每条动作始终与 retarget 时使用的物体几何保持配对。

Manifest:

```text
pipelines/hiphi_to_g1/manifests/usable_30cm_124.txt
```

## 10. 30 Hz → 50 Hz conversion / 30 Hz → 50 Hz 转换

Retargeted trajectories use the following qpos layout:

```text
[0:3]   root position
[3:7]   root quaternion
[7:36]  29 G1 joints
[36:39] object position
[39:43] object quaternion
```

They are converted from 30 Hz to Carry2Anywhere's 50 Hz reference format using:

```bash
./pipelines/hiphi_to_g1/run_pipeline.sh convert-30cm
```

Critical converter settings:

```text
--robot g1
--object-name box_0p3000_0p3000_0p3000
--input-fps 30
--output-fps 50
--has-dynamic-object
--no-use-omniretarget-data
--once
```

The resulting NPZ files contain joint, body and object pose/velocity trajectories expected by the WBT environment.

最终 NPZ 包含 WBT 环境需要的机器人 joint/body 以及 object pose / velocity 轨迹。

---

# Multi-motion WBT training / 多动作 WBT 训练

The modified motion loader accepts, in priority order:

修改后的 motion loader 按以下优先级加载：

```text
motion_files
    ↓
motion_dir + motion_glob
    ↓
legacy motion_file
```

Multiple motion clips are loaded and concatenated into a flattened `MotionBuffer`, while per-motion boundaries and lengths are preserved for indexing and reset sampling.

多条 motion clip 被加载并拼接到统一的 `MotionBuffer` 中，同时保留每条动作的长度和边界信息，用于索引和 reset sampling。

Example teacher training command:

```bash
source scripts/source_isaacsim_setup.sh

python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-w-object \
  --command.setup_terms.motion_command.params.motion_config.motion_dir=/path/to/converted_30cm_124 \
  --command.setup_terms.motion_command.params.motion_config.motion_glob="*_w_obj.npz" \
  --training.num_envs=4096 \
  --training.headless=True
```

Adjust `--training.num_envs` for the available GPU memory.

根据 GPU 显存调整 `--training.num_envs`。

---

# Pipeline commands / 流程命令

```bash
# 1. HiPHI preprocessing / HiPHI 预处理
./pipelines/hiphi_to_g1/run_pipeline.sh prepare

# 2. G1 retargeting on SLURM / 在 SLURM 上执行 G1 retargeting
./pipelines/hiphi_to_g1/run_pipeline.sh retarget

# 3. source-relative contact QC / 源动作相对接触 QC
./pipelines/hiphi_to_g1/run_pipeline.sh contact-qc

# 4. full-body penetration audit / 全身穿透检查
./pipelines/hiphi_to_g1/run_pipeline.sh body-qc

# 5. convert 124 exact-30cm motions / 转换 124 条 30cm 箱子动作
./pipelines/hiphi_to_g1/run_pipeline.sh convert-30cm

# 6. show generated-data counts / 查看当前生成数据数量
./pipelines/hiphi_to_g1/run_pipeline.sh status
```

For the focused implementation notes, see:

更聚焦的实现说明见：

**[`pipelines/hiphi_to_g1/README.md`](pipelines/hiphi_to_g1/README.md)**

---

# Repository layout / 仓库结构

```text
g1_box_application/
├── pipelines/
│   └── hiphi_to_g1/
│       ├── README.md
│       ├── run_pipeline.sh
│       └── manifests/
├── scripts/
│   ├── prepare_hiphi_validation_hpc.py
│   ├── eval_hiphi_smokes.py
│   └── audit_fullbody_box_penetration.py
├── retarget_225_hiphi.sbatch
├── src/
│   ├── holosoma/
│   │   └── holosoma/
│   │       ├── motions/                 # original Carry2Anywhere references
│   │       └── managers/command/terms/  # multi-motion WBT loader
│   └── holosoma_retargeting/
│       └── holosoma_retargeting/
│           ├── config_types/            # HiPHI joint mapping
│           ├── data_conversion/         # 30 → 50 Hz conversion
│           ├── models/g1/               # modified G1 + rubber hands
│           └── src/                     # interaction-mesh retargeter
├── SETUP.md
└── README.md
```

---

# Key implementation files / 关键实现文件

| File | Purpose / 用途 |
|---|---|
| `src/holosoma_retargeting/holosoma_retargeting/config_types/data_type.py` | HiPHI joint list and HiPHI → G1 link mapping / HiPHI 关节定义与 G1 映射 |
| `src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof.urdf` | Current G1 model with rubber hands / 当前 rubber-hand G1 模型 |
| `src/holosoma_retargeting/holosoma_retargeting/src/interaction_mesh_retargeter.py` | Interaction-mesh optimization and source-contact-aware hand weighting / interaction mesh 优化与接触感知手部权重 |
| `scripts/prepare_hiphi_validation_hpc.py` | HiPHI extraction, BVH FK, coordinate conversion, object processing, pickup detection / HiPHI 解压、BVH FK、坐标转换、物体处理、pickup 检测 |
| `scripts/eval_hiphi_smokes.py` | Source-relative hand-contact QC / 源动作相对手部接触 QC |
| `scripts/audit_fullbody_box_penetration.py` | Full-body box penetration audit / 全身箱子穿透检查 |
| `src/holosoma_retargeting/holosoma_retargeting/data_conversion/convert_data_format_mj.py` | 30 Hz retarget → 50 Hz Carry2Anywhere NPZ / 30 Hz retarget 转 50 Hz Carry2Anywhere NPZ |
| `src/holosoma/holosoma/managers/command/terms/wbt.py` | Multi-motion loading and flattened training buffer / 多动作加载与训练 buffer |

---

# Setup / 环境配置

The existing environment scripts and [`SETUP.md`](SETUP.md) are retained for the HoloSoma / Isaac Sim stack.

现有环境脚本以及 [`SETUP.md`](SETUP.md) 继续用于 HoloSoma / Isaac Sim 环境配置。

Two logical environments are used:

| Environment | Purpose / 用途 |
|---|---|
| `hsretargeting` | HiPHI preprocessing, MuJoCo, retargeting and QC / HiPHI 预处理、MuJoCo、retargeting、QC |
| `hssim` | Isaac Sim / Isaac Lab WBT training and evaluation / Isaac Sim / Isaac Lab 全身跟踪训练与评估 |

---

# Acknowledgements / 致谢

| English | 中文 |
|---|---|
| This repository started from the **Carry2Anywhere** codebase and keeps its teacher/student box-carrying framework as the baseline. Our work extends it with the HiPHI retargeting and dataset pipeline described above. | 本仓库最初基于 **Carry2Anywhere** 代码，并保留其 teacher/student 箱子搬运框架作为 baseline。我们的工作在其基础上增加了本文档所描述的 HiPHI retargeting 与数据扩展流程。 |
| The retargeting implementation builds on **HoloSoma** and its interaction-mesh retargeting framework. | Retargeting 实现基于 **HoloSoma** 及其 interaction-mesh retargeting 框架。 |
| **HiPHI** provides the human–object interaction recordings used for the new box-carrying dataset extension. | **HiPHI** 提供了本项目新增箱子搬运数据所使用的人–物交互动作。 |
| **Isaac Sim** and **Isaac Lab** provide the simulation and reinforcement-learning infrastructure. | **Isaac Sim** 与 **Isaac Lab** 提供仿真与强化学习基础设施。 |
| **Unitree G1** is the target humanoid platform. | **宇树 Unitree G1** 是本项目的目标人形机器人平台。 |

Useful upstream links:

- [HoloSoma](https://github.com/amazon-far/holosoma)
- [Isaac Sim](https://developer.nvidia.com/isaac/sim)
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
- [Unitree G1](https://www.unitree.com/g1)

---

<div align="center">

**Current focus / 当前重点:** expanding robust G1 box-carrying references and retraining the Carry2Anywhere teacher on the HiPHI-derived dataset.

</div>
