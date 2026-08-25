<div align="center">

# G1 Box Application

### Unitree G1 Box-Carrying Research Stack

**English** | [中文](README_zh.md)

</div>

---

## Overview

This repository contains two related but separate modules:

1. **Carry2Anywhere** — policy training, distillation and evaluation for Unitree G1 box carrying.
2. **HiPHI → G1 Data Retargeting** — a data pipeline that converts human box-carrying demonstrations into G1 + box reference motions.

The modules can be used independently. When used together, Module 2 provides additional reference motions for Module 1.

---

## Module 1 — Carry2Anywhere

Carry2Anywhere is the learning and control module in this repository. It contains the whole-body-tracking teacher, student distillation, evaluation tools, and our multi-motion training support for Unitree G1.

Our current version also uses the updated G1 box-carrying robot model with rubber-hand geometry.

**[Read the Carry2Anywhere module documentation →](docs/CARRY2ANYWHERE.md)**

---

## Module 2 — HiPHI → G1 Data Retargeting

This module expands the motion dataset used for G1 box-carrying training.

At a high level, it takes human box-carrying demonstrations from HiPHI, preprocesses the human and object trajectories, retargets the human–object interaction to the Unitree G1, checks whether important hand–box interaction is preserved, filters poor retargets, and converts the accepted motions into the reference format used by Carry2Anywhere.

The retargeting uses the G1 rubber-hand model and source-contact-aware interaction preservation rather than simply forcing both robot hands toward the box throughout the entire motion.

The validated pipeline starts from 225 original non-mirrored HiPHI box-carry motions and produces 168 usable G1 motions. The first retraining subset contains 124 motions using an exact 30 cm cube.

**[Read the HiPHI → G1 retargeting documentation →](pipelines/hiphi_to_g1/README.md)**

---

## How the modules connect

```text
HiPHI human box-carrying demonstrations
                ↓
Module 2: preprocessing + retargeting + QC
                ↓
G1 + box reference motions
                ↓
Module 1: Carry2Anywhere training / distillation / evaluation
```

---

## Repository layout

```text
g1_box_application/
├── src/holosoma/                 # Module 1: Carry2Anywhere training / evaluation
├── src/holosoma_retargeting/     # Module 2: motion retargeting implementation
├── pipelines/hiphi_to_g1/        # HiPHI pipeline, manifests and detailed README
├── scripts/                      # preprocessing, QC and utility scripts
├── docs/CARRY2ANYWHERE.md        # Module 1 documentation
├── README.md                     # English default README
└── README_zh.md                  # Chinese README
```

## Acknowledgements

This repository builds on Carry2Anywhere, HoloSoma, HiPHI, Isaac Sim / Isaac Lab, and the Unitree G1 platform. Please refer to the corresponding upstream projects and datasets for their licenses and usage terms.
