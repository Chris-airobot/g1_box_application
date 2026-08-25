# G1 Box Application

[English](README.md) | **中文**

本仓库包含两个相互独立但可以连接使用的模块：

1. **Carry2Anywhere 学习与控制模块**
2. **HiPHI → Unitree G1 数据重定向模块**

我们将两部分分开维护：Module 1 负责策略训练、蒸馏和评估；Module 2 负责把人体搬箱动作转换成可用于 G1 全身跟踪训练的参考动作。

## Module 1 — Carry2Anywhere

Carry2Anywhere 是本仓库中的策略学习模块，主要用于 Unitree G1 的箱子搬运任务。

该模块包含：

- 基于 PPO 的 whole-body-tracking teacher；
- student policy distillation；
- 多 motion reference 加载；
- G1 + box 的训练与评估环境。

Module 2 生成的 G1 参考动作可以直接作为 Module 1 的新增训练数据。

**[查看 Module 1 详细说明 →](docs/CARRY2ANYWHERE.md)**

## Module 2 — HiPHI → G1 Data Retargeting

这一模块用于扩展 Carry2Anywhere 的动作数据。

高层流程是：从 HiPHI 中筛选人类搬箱动作，对人体和箱子轨迹进行统一预处理，然后把人体–箱子的空间关系重定向到 Unitree G1。当前 G1 模型使用 rubber-hand geometry，并在人体手部真正接近箱子时增强手部交互保持。Retargeting 完成后，再通过手–箱接触保持和身体–箱子穿透检查筛除质量较差的动作，最后转换成 Carry2Anywhere 使用的 50 Hz motion-reference NPZ。

当前验证流程从 225 条原始 non-mirrored box-carry motions 开始，最终得到 168 条可用动作，其中 124 条对应精确的 30 cm 立方体箱子，并作为第一阶段重新训练的数据集。

**[查看 Module 2 / HiPHI retargeting 详细说明 →](pipelines/hiphi_to_g1/README.md)**

## 两个模块之间的关系

```text
HiPHI human box-carrying data
        ↓
Module 2: preprocessing + retargeting + QC
        ↓
G1 + box reference motions
        ↓
Module 1: Carry2Anywhere training / distillation / evaluation
```

## Repository structure

```text
g1_box_application/
├── src/holosoma/                 # Module 1: training and evaluation
├── src/holosoma_retargeting/     # Module 2: retargeting implementation
├── pipelines/hiphi_to_g1/        # HiPHI pipeline, manifests and documentation
├── scripts/                      # preprocessing / QC / utility scripts
├── docs/CARRY2ANYWHERE.md        # Module 1 documentation
├── README.md                     # English default README
└── README_zh.md                  # 中文说明
```
