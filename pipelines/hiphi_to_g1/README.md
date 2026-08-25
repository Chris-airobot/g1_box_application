# HiPHI → Unitree G1 Box-Carrying Pipeline

This folder documents the reproducible HiPHI-to-G1 data pipeline used for Carry2Anywhere.

## Flow

Raw HiPHI archives
→ 225 original non-mirrored box-carry motions
→ BVH/object preprocessing
→ 223 retarget inputs
→ G1 retargeting
→ source-relative contact QC
→ full-body penetration QC
→ 168 usable motions
→ 124 exact 30 cm cube motions
→ 30 Hz to 50 Hz Carry2Anywhere NPZ

## Manifests

- `manifests/raw_box_225.txt`: selected 225 source motions.
- `manifests/raw_box_225_archive_map.csv`: archive mapping for those motions.
- `manifests/final_validation.csv`: canonical reviewed QC result.
- `manifests/usable_30cm_124.txt`: final 30 cm training subset.

## Data root

Set `HIPHI_ROOT` to the HiPHI data directory.

HPC example:

    export HIPHI_ROOT=/fred/oz430/tliu/data/HiPHI

Raw archives remain under `$HIPHI_ROOT/data/` and are not duplicated into this repository.

## Commands

Preprocess:

    ./pipelines/hiphi_to_g1/run_pipeline.sh prepare

Retarget on SLURM:

    ./pipelines/hiphi_to_g1/run_pipeline.sh retarget

Contact QC:

    ./pipelines/hiphi_to_g1/run_pipeline.sh contact-qc

Full-body penetration QC:

    ./pipelines/hiphi_to_g1/run_pipeline.sh body-qc

Convert the 124 exact 30 cm motions from 30 Hz to 50 Hz:

    ./pipelines/hiphi_to_g1/run_pipeline.sh convert-30cm

Check progress:

    ./pipelines/hiphi_to_g1/run_pipeline.sh status

## Contact QC

For the two-second carry interval after pickup, left and right source-hand contact are evaluated independently.

A source hand is required when it stays within 2 cm of the box for at least 80% of the interval. The corresponding G1 hand must satisfy the same rule.

Statuses are `CONTACT_MATCH`, `CONTACT_FAIL`, and `SOURCE_QC`.

## Body penetration QC

Intended grasp-contact bodies (`rubber_hand`, `thumb`, `pinky`) are excluded from the body-box penetration audit.

The canonical final reviewed outcome is stored in `manifests/final_validation.csv`.

## Conversion

The final 30 cm subset is listed in `manifests/usable_30cm_124.txt`.

The converter uses:

- robot: G1
- object: `box_0p3000_0p3000_0p3000`
- input: 30 Hz
- output: 50 Hz
- `--has-dynamic-object`
- `--no-use-omniretarget-data`

Retargeted qpos layout:

- `[0:3]`: root position
- `[3:7]`: root quaternion
- `[7:36]`: 29 G1 joints
- `[36:39]`: object position
- `[39:43]`: object quaternion

Validated reference counts: 225 selected, 223 retarget inputs, 168 usable, 124 exact 30 cm motions.
