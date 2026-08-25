# Module 2 — HiPHI → Unitree G1 Data Retargeting

[← Back to main README](../../README.md)

This module converts HiPHI human box-carrying demonstrations into Unitree G1 + box reference motions that can be consumed by the Carry2Anywhere whole-body-tracking pipeline.

The goal is not to copy human joint angles directly. Instead, the pipeline preserves the important **human–object spatial relationships** while respecting the Unitree G1 kinematic structure, joint limits, foot constraints and object non-penetration constraints.

## Pipeline overview

```text
Raw HiPHI archives
        ↓
select original non-mirrored Bringing-carry + box motions
        ↓
225 source motions
        ↓
BVH + object preprocessing
        ↓
223 retarget inputs
        ↓
HiPHI → G1 interaction-mesh retargeting
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
```

## 1. Source-motion selection

The source dataset is HiPHI. We select motions satisfying all three conditions:

- `frame_lu == "Bringing-carry"`
- `object_categories == "box"`
- `mirrored == false`

This produces **225 original box-carrying demonstrations**.

The exact selected motion IDs and their source archive locations are frozen in:

- `manifests/raw_box_225.txt`
- `manifests/raw_box_225_archive_map.csv`

The raw HiPHI archives are intentionally kept outside the repository.

## 2. HiPHI preprocessing

Preprocessing is implemented in:

`scripts/prepare_hiphi_validation_hpc.py`

The stage extracts the required motion files and builds one NPZ per motion containing the human joint trajectory, object pose trajectory, object size and pickup metadata.

### BVH handling

HiPHI motions are represented with BVH skeletons. The preprocessing code parses the hierarchy and performs forward kinematics to recover global joint positions.

Translation channels **replace** the BVH joint offset on translated joints rather than being added to it.

### Coordinate-system conversion

HiPHI uses a Y-up coordinate convention and centimeter-scale data. The retargeting stack expects robotics-style Z-up coordinates in meters.

The conversion is:

```text
X_robot =  X_hiphi
Y_robot = -Z_hiphi
Z_robot =  Y_hiphi
```

and positions are converted from centimeters to meters.

### Object geometry

The box trajectory origin in the raw data is not assumed to be the geometric center of the mesh. The preprocessing stage recovers the mesh center and size, recenters the object representation, and stores the resulting box dimensions.

### Pickup frame

A pickup frame is detected during preprocessing and stored as `local_pickup_frame`. The later contact-QC stage uses this stored frame directly rather than redetecting pickup independently.

Two of the selected 225 recordings are excluded before retargeting:

- one recording with incompatible BVH joint ordering;
- one recording with a static object trajectory.

This leaves **223 retarget inputs**.

## 3. Human → G1 correspondence

The HiPHI skeleton-to-G1 mapping is defined in:

`src/holosoma_retargeting/holosoma_retargeting/config_types/data_type.py`

Important mappings include:

```text
Spine1            → pelvis_contour_link
LeftUpLeg         → left_hip_pitch_link
LeftLeg           → left_knee_link
LeftFoot          → left_ankle_intermediate_1_link
LeftToeBase       → left_ankle_roll_sphere_5_link
RightUpLeg        → right_hip_pitch_link
RightLeg          → right_knee_link
RightFoot         → right_ankle_intermediate_1_link
RightToeBase      → right_ankle_roll_sphere_5_link
LeftArm           → left_shoulder_roll_link
LeftForeArm       → left_elbow_link
RightArm          → right_shoulder_roll_link
RightForeArm      → right_elbow_link
LeftHandMiddle3   → left_rubber_hand_link
RightHandMiddle3  → right_rubber_hand_link
```

The hand mapping is important: the current G1 model uses the actual fixed **rubber-hand geometry** rather than the old sphere-hand endpoint representation.

The fixed `thumb_link` and `pinky_link` bodies are retained as auxiliary grasp/contact geometry; they are not actuated fingers.

## 4. Interaction-mesh retargeting

The core solver is implemented in:

`src/holosoma_retargeting/holosoma_retargeting/src/interaction_mesh_retargeter.py`

For each frame, the method builds an interaction representation containing mapped human/robot body points together with object points. The optimization minimizes deformation of the interaction structure in the object frame while solving for the G1 configuration.

The solver also includes:

- joint-limit constraints;
- foot-sticking constraints;
- robot/object non-penetration constraints;
- a trust-region / step-size constraint;
- temporal smoothness;
- nominal-pose tracking where configured.

This means the retargeting is driven by the **relative geometry between the body and the carried object**, rather than by independent end-effector IK alone.

## 5. Source-contact-aware hand weighting

A major modification in this repository is that hand tracking is strengthened only when the **source human hand is actually close to the box**.

For each source hand, the distance to the box surface is computed in the demonstration object's local frame.

The interaction weight is:

```text
source hand distance ≥ 2 cm  → 1×
source hand distance ≤ 1 cm  → 10×
1–2 cm                       → smooth interpolation from 1× to 10×
```

This avoids forcing both robot hands toward the box throughout the full motion. During approach or naturally one-handed segments, ordinary tracking is used; when the source demonstrates real contact, the hand–object relationship receives much stronger weight.

The weighting is applied to:

- `LeftHandMiddle3 → left_rubber_hand_link`
- `RightHandMiddle3 → right_rubber_hand_link`

## 6. Retarget execution

The pipeline wrapper is:

`pipelines/hiphi_to_g1/run_pipeline.sh`

On SLURM:

```bash
export HIPHI_ROOT=/fred/oz430/tliu/data/HiPHI
./pipelines/hiphi_to_g1/run_pipeline.sh retarget
```

The wrapper creates a task list from the available preprocessed smoke clips and submits the retargeting job array through `retarget_225_hiphi.sbatch`.

The retargeter is invoked with:

```text
robot       = g1
task type   = object_interaction
data format = hiphi
```

The validated main retarget pool contains **212 motions**.

## 7. Source-relative hand-contact QC

Contact validation is implemented in:

`scripts/eval_hiphi_smokes.py`

The check evaluates the interval from pickup to two seconds after pickup.

For each source hand independently:

```text
required source contact = hand is within 2 cm of box
                          for at least 80% of the interval
```

The corresponding G1 hand must satisfy the same 2 cm / 80% requirement.

This produces three contact statuses:

- `CONTACT_MATCH` — every source-required hand contact is preserved by the G1 motion;
- `CONTACT_FAIL` — at least one required source contact is not sufficiently preserved;
- `SOURCE_QC` — neither source hand provides a meaningful contact target under this rule.

For the validated 212-motion pool:

```text
CONTACT_MATCH : 178
CONTACT_FAIL  : 21
SOURCE_QC     : 13
```

This source-relative rule is important because many valid HiPHI demonstrations are naturally one-handed for part or all of the carry interval. We therefore do not require both G1 hands to contact the box when the source motion does not demonstrate both-hand contact.

Run with:

```bash
./pipelines/hiphi_to_g1/run_pipeline.sh contact-qc
```

## 8. Full-body box-penetration QC

Retargeted motions are additionally audited for unintended robot-body penetration into the box.

Implementation:

`scripts/audit_fullbody_box_penetration.py`

Intended grasp-contact bodies are excluded from this audit:

```text
rubber_hand
thumb
pinky
```

The remaining robot bodies are checked against the box geometry over the full motion.

The final reviewed decisions are frozen in:

`manifests/final_validation.csv`

The final validation contains:

```text
USABLE                 : 168
CONTACT_FAIL           : 21
SOURCE_QC              : 13
BODY_PENETRATION_FAIL  : 10
```

The body-penetration stage is a reviewed QC stage rather than a single universal penetration threshold, because some non-hand contacts can be intentional or geometrically acceptable while torso/hip penetration through the box is not.

Run the audit with:

```bash
./pipelines/hiphi_to_g1/run_pipeline.sh body-qc
```

## 9. 30 cm training subset

The 168 usable motions contain several source box sizes. The first retraining experiment deliberately uses only the largest consistent geometry group:

```text
0.30 × 0.30 × 0.30 m
```

This gives **124 motions**.

The exact list is stored in:

`manifests/usable_30cm_124.txt`

Using a fixed geometry avoids changing the box dimensions after retargeting, which would invalidate the original hand–object relationship.

## 10. Carry2Anywhere format conversion

The retargeter stores a 30 Hz G1 + object trajectory. Before teacher training, the selected motions are converted to the 50 Hz WBT reference format expected by Carry2Anywhere.

Run:

```bash
./pipelines/hiphi_to_g1/run_pipeline.sh convert-30cm
```

The converter uses:

```text
robot       = g1
object      = box_0p3000_0p3000_0p3000
input fps   = 30
output fps  = 50
```

with:

```text
--has-dynamic-object
--no-use-omniretarget-data
```

The second flag is critical because the HiPHI retargeted qpos layout is:

```text
[0:3]    root position
[3:7]    root quaternion
[7:36]   29 G1 joints
[36:39]  object position
[39:43]  object quaternion
```

The converted NPZ contains the robot joint states, body poses/velocities and object states required by the whole-body-tracking motion loader.

## Commands

From the repository root:

```bash
# 1. Extract and preprocess selected HiPHI motions
./pipelines/hiphi_to_g1/run_pipeline.sh prepare

# 2. Retarget on SLURM
./pipelines/hiphi_to_g1/run_pipeline.sh retarget

# 3. Source-relative hand-contact QC
./pipelines/hiphi_to_g1/run_pipeline.sh contact-qc

# 4. Full-body box-penetration audit
./pipelines/hiphi_to_g1/run_pipeline.sh body-qc

# 5. Convert final 30 cm subset to Carry2Anywhere format
./pipelines/hiphi_to_g1/run_pipeline.sh convert-30cm

# Check generated-data counts
./pipelines/hiphi_to_g1/run_pipeline.sh status
```

## Preserved manifests

The reproducibility-critical selections and final decisions are kept in Git:

```text
pipelines/hiphi_to_g1/manifests/
├── raw_box_225.txt
├── raw_box_225_archive_map.csv
├── final_validation.csv
└── usable_30cm_124.txt
```

The large raw HiPHI archives and generated motion files remain outside the repository.
