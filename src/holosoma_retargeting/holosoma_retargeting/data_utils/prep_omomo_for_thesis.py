"""Prep OMOMO data for retargeting + evaluation experiments (thesis chapter).

For each task name in the input list (default = the 46 sequences in
`src/holosoma/holosoma/motions/*_mj_w_obj.npz`), this script:

1. Looks up the sequence in OMOMO's train/test pickle
   (`{train,test}_diffusion_manip_seq_joints24.p`) by `seq_name`.
2. Runs SMPL-H forward kinematics with the AMASS-style neutral SMPL-H model
   (`smpl_all_models/smplh_amass/neutral/model.npz`) to recover 52 world-frame
   joint positions per frame. Joints are reordered into `SMPLH_DEMO_JOINTS`
   layout (Pelvis, L_Hip, L_Knee, L_Ankle, L_Toe, R_Hip, ...).
3. Converts `obj_rot` (3x3) -> quaternion in [qx, qy, qz, qw] order.
4. Saves `<task>.pt` (a torch.Tensor of shape (T, 325)) under
   `--output_dir`. Layout (must match what `load_intermimic_data` decodes;
   note that loader applies the permutation [6, 3, 4, 5, 0, 1, 2] to map back
   to [qw, qx, qy, qz, x, y, z], hence the source layout below):
       [:, 0:162]    -> zeros (unused by the loader; reserved for axis-angle
                                pose if you ever need it)
       [:, 162:318]  -> 52 joints (in SMPLH_DEMO order), flattened (T, 156)
       [:, 318:321]  -> object translation (tx, ty, tz)
       [:, 321:325]  -> object quaternion (qx, qy, qz, qw)

5. Computes a per-subject `height_dict.pkl` (max - min vertical span of the
   SMPL-H mesh in T-pose with subject-specific betas) and writes it to
   `holosoma_retargeting/demo_data/height_dict.pkl` -- this is what
   `calculate_scale_factor` in `src/utils.py` reads. Heights are taken from
   the `y` axis since SMPL-H's canonical pose is upright along y. (Empirically
   verified: in T-pose, the head is around y=-1.16 and feet around y=0.56,
   so |max_y - min_y| is the body height.)

Usage:

    source scripts/source_retargeting_setup.sh
    python -u src/holosoma_retargeting/holosoma_retargeting/data_utils/prep_omomo_for_thesis.py \
        --omomo_train_pkl /home/limx/code/omomo_release/data/train_diffusion_manip_seq_joints24.p \
        --omomo_test_pkl  /home/limx/code/omomo_release/data/test_diffusion_manip_seq_joints24.p \
        --smplh_neutral_npz /home/limx/code/omomo_release/data/smpl_all_models/smplh_amass/neutral/model.npz \
        --motions_dir src/holosoma/holosoma/motions \
        --output_dir  src/holosoma_retargeting/holosoma_retargeting/demo_data/OMOMO_thesis

Output:
    <output_dir>/sub3_largebox_004.pt
    <output_dir>/sub3_largebox_005.pt
    ...
    src/holosoma_retargeting/holosoma_retargeting/demo_data/height_dict.pkl
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R  # noqa: N817
from smplx.lbs import lbs

# 52-element permutation: SMPLH_DEMO_JOINTS index -> SMPL-H canonical kintree index.
# Body block (0..21) follows demo order [Pelvis, L_Hip, L_Knee, L_Ankle, L_Toe,
# R_Hip, R_Knee, R_Ankle, R_Toe, Torso, Spine, Chest, Neck, Head, L_Thorax,
# L_Shoulder, L_Elbow, L_Wrist]. Hand block (18..32 left, 37..51 right) lines up 1:1.
DEMO_TO_CANONICAL = [
    0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20,  # body 0..17
    22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,    # left hand 18..32
    14, 17, 19, 21,                                                # R_Thorax/Shoulder/Elbow/Wrist 33..36
    37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,    # right hand 37..51
]
assert len(DEMO_TO_CANONICAL) == 52
assert sorted(DEMO_TO_CANONICAL) == list(range(52))


def load_smplh_arrays(npz_path: str | Path):
    """Load AMASS-style SMPL-H neutral model arrays needed by smplx.lbs."""
    m = np.load(str(npz_path), allow_pickle=True)
    v_template = torch.from_numpy(m["v_template"]).float()
    shapedirs = torch.from_numpy(m["shapedirs"]).float()  # (V, 3, 16)
    posedirs_npz = m["posedirs"]  # (V, 3, P=459)
    P = posedirs_npz.shape[-1]
    posedirs = torch.from_numpy(
        posedirs_npz.transpose(2, 0, 1).reshape(P, -1)
    ).float()  # (P, V*3)
    j_regressor = torch.from_numpy(np.asarray(m["J_regressor"])).float()  # (52, V)
    weights = torch.from_numpy(m["weights"]).float()  # (V, 52)
    parents_np = m["kintree_table"][0].astype(np.int64).copy()
    parents_np[0] = -1  # root has no parent
    parents = torch.from_numpy(parents_np).long()  # (52,)
    return {
        "v_template": v_template,
        "shapedirs": shapedirs,
        "posedirs": posedirs,
        "j_regressor": j_regressor,
        "weights": weights,
        "parents": parents,
    }


def smplh_fk(
    model_arrays: dict,
    betas: np.ndarray,         # (1, 16) or (16,)
    root_orient: np.ndarray,   # (T, 3)  axis-angle for root
    pose_body: np.ndarray,     # (T, 63) axis-angle for 21 body joints
    trans: np.ndarray,         # (T, 3)
    *,
    device: torch.device | None = None,
    chunk: int = 256,
) -> torch.Tensor:
    """Run SMPL-H FK and return joint positions (T, 52, 3) in canonical order.

    Hands are kept at zero pose (flat hand). Joints are returned in the SMPL-H
    canonical kintree order; reorder later via `DEMO_TO_CANONICAL` if needed.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    betas = np.asarray(betas).reshape(1, 16)
    T = root_orient.shape[0]

    # Move arrays to device once.
    arrays = {k: v.to(device) for k, v in model_arrays.items()}
    betas_t = torch.from_numpy(betas).float().to(device).expand(T, 16).contiguous()
    root_orient_t = torch.from_numpy(root_orient).float().to(device)
    pose_body_t = torch.from_numpy(pose_body).float().to(device)
    trans_t = torch.from_numpy(trans).float().to(device)

    n_total_pose = 52 * 3
    hand_pad = torch.zeros(T, 30 * 3, dtype=torch.float32, device=device)
    full_pose = torch.cat([root_orient_t, pose_body_t, hand_pad], dim=1)  # (T, 156)
    assert full_pose.shape[1] == n_total_pose

    out_joints = torch.empty(T, 52, 3, dtype=torch.float32)
    for start in range(0, T, chunk):
        end = min(T, start + chunk)
        _, joints = lbs(
            betas_t[start:end],
            full_pose[start:end],
            arrays["v_template"],
            arrays["shapedirs"],
            arrays["posedirs"],
            arrays["j_regressor"],
            arrays["parents"],
            arrays["weights"],
            pose2rot=True,
        )
        joints = joints + trans_t[start:end, None, :]
        out_joints[start:end] = joints.detach().cpu()
    return out_joints


def compute_height(model_arrays: dict, betas: np.ndarray, device: torch.device | None = None) -> float:
    """Height as max_y - min_y of the SMPL-H **vertices** in T-pose with given betas.

    Matches the convention used by `prep_amass_smplx_for_rt.py`.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    betas_t = torch.from_numpy(np.asarray(betas).reshape(1, 16)).float().to(device)
    full_pose = torch.zeros(1, 52 * 3, dtype=torch.float32, device=device)
    arrays = {k: v.to(device) for k, v in model_arrays.items()}

    verts, _ = lbs(
        betas_t,
        full_pose,
        arrays["v_template"],
        arrays["shapedirs"],
        arrays["posedirs"],
        arrays["j_regressor"],
        arrays["parents"],
        arrays["weights"],
        pose2rot=True,
    )
    verts = verts.detach().cpu().numpy()[0]  # (V, 3)
    return float(verts[:, 1].max() - verts[:, 1].min())


def collect_task_names(motions_dir: Path | None, explicit: list[str] | None) -> list[str]:
    if explicit:
        return list(explicit)
    if motions_dir is None:
        raise ValueError("Either --motions_dir or --task_names must be provided")
    suffix = "_mj_w_obj.npz"
    names = []
    for p in sorted(Path(motions_dir).glob(f"*{suffix}")):
        names.append(p.name[: -len(suffix)])
    if not names:
        raise FileNotFoundError(f"No *{suffix} files under {motions_dir}")
    return names


def index_omomo_pickles(*pkl_paths: str | Path) -> dict[str, dict]:
    """Build {seq_name: entry} index across one or more OMOMO pickles."""
    by_name: dict[str, dict] = {}
    for p in pkl_paths:
        if not p:
            continue
        path = Path(p)
        if not path.exists():
            print(f"[warn] skipping non-existent pickle: {path}")
            continue
        print(f"Loading OMOMO pickle: {path}")
        d = joblib.load(str(path))
        if isinstance(d, dict):
            for entry in d.values():
                seq = str(entry["seq_name"])
                if seq not in by_name:
                    by_name[seq] = entry
        else:
            for entry in d:
                seq = str(entry["seq_name"])
                if seq not in by_name:
                    by_name[seq] = entry
    return by_name


def build_intermimic_tensor(
    joints_demo_order: torch.Tensor,  # (T, 52, 3)
    obj_rot: np.ndarray,              # (T, 3, 3)
    obj_trans: np.ndarray,            # (T, 3, 1) or (T, 3)
) -> torch.Tensor:
    """Pack joints + object pose into the (T, 325) layout that
    `load_intermimic_data` expects.

    Layout:
        [0:162]   zeros (axis-angle pose -- unused by the loader)
        [162:318] 52 joints flattened (T, 156)
        [318:321] (tx, ty, tz)
        [321:325] (qx, qy, qz, qw)
    The loader (`load_intermimic_data`) applies the permutation
    `[6, 3, 4, 5, 0, 1, 2]` to slots `318:325`, which yields
    `[qw, qx, qy, qz, x, y, z]` -- the canonical pose order used by the
    rest of the retargeting/eval code.
    """
    T = joints_demo_order.shape[0]
    arr = np.zeros((T, 325), dtype=np.float32)
    arr[:, 162:318] = joints_demo_order.reshape(T, -1).cpu().numpy()

    quat_xyzw = R.from_matrix(obj_rot).as_quat()  # (T, 4) [qx, qy, qz, qw]
    trans = np.asarray(obj_trans).reshape(T, 3)
    arr[:, 318:321] = trans.astype(np.float32)
    arr[:, 321:325] = quat_xyzw.astype(np.float32)
    return torch.from_numpy(arr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--omomo_train_pkl",
        default="/home/limx/code/omomo_release/data/train_diffusion_manip_seq_joints24.p",
    )
    parser.add_argument(
        "--omomo_test_pkl",
        default="/home/limx/code/omomo_release/data/test_diffusion_manip_seq_joints24.p",
    )
    parser.add_argument(
        "--smplh_neutral_npz",
        default="/home/limx/code/omomo_release/data/smpl_all_models/smplh_amass/neutral/model.npz",
    )
    parser.add_argument(
        "--motions_dir",
        default="src/holosoma/holosoma/motions",
        help=(
            "Directory containing *_mj_w_obj.npz training files. The base names "
            "(stripped of the suffix) become the task list."
        ),
    )
    parser.add_argument(
        "--task_names",
        nargs="*",
        default=None,
        help="Override task list explicitly; otherwise derived from --motions_dir.",
    )
    parser.add_argument(
        "--output_dir",
        default="src/holosoma_retargeting/holosoma_retargeting/demo_data/OMOMO_thesis",
    )
    parser.add_argument(
        "--height_dict_path",
        default="src/holosoma_retargeting/holosoma_retargeting/demo_data/height_dict.pkl",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    height_dict_path = Path(args.height_dict_path)
    height_dict_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading SMPL-H neutral model: {args.smplh_neutral_npz}")
    model_arrays = load_smplh_arrays(args.smplh_neutral_npz)

    task_names = collect_task_names(
        Path(args.motions_dir) if args.motions_dir else None, args.task_names
    )
    print(f"Found {len(task_names)} task names")

    omomo = index_omomo_pickles(args.omomo_train_pkl, args.omomo_test_pkl)
    print(f"OMOMO index size: {len(omomo)}")

    height_dict: dict[str, float] = {}
    n_done, n_skip = 0, 0
    t0 = time.time()
    for i, task in enumerate(task_names, 1):
        out_pt = output_dir / f"{task}.pt"
        if out_pt.exists() and not args.overwrite:
            print(f"[{i}/{len(task_names)}] skip (exists): {task}")
            n_skip += 1
            continue
        if task not in omomo:
            print(f"[{i}/{len(task_names)}] MISSING in OMOMO pickles: {task}")
            continue
        entry = omomo[task]
        sub_name = task.split("_")[0]

        betas = np.asarray(entry["betas"]).reshape(1, 16)
        trans = np.asarray(entry["trans"])         # (T, 3)
        root_orient = np.asarray(entry["root_orient"])  # (T, 3)
        pose_body = np.asarray(entry["pose_body"])      # (T, 63)
        obj_rot = np.asarray(entry["obj_rot"])          # (T, 3, 3)
        obj_trans = np.asarray(entry["obj_trans"]).reshape(-1, 3)  # (T, 3)
        T = trans.shape[0]

        joints_canonical = smplh_fk(
            model_arrays, betas, root_orient, pose_body, trans, device=device
        )  # (T, 52, 3) canonical order
        joints_demo = joints_canonical[:, DEMO_TO_CANONICAL, :]  # reorder
        tensor = build_intermimic_tensor(joints_demo, obj_rot, obj_trans)
        torch.save(tensor, out_pt)
        n_done += 1

        # Sanity: pelvis z should be ~0.7-1.1m, toe z near 0.0..0.3
        pelvis_z = float(joints_demo[:, 0, 2].mean())
        l_toe_z_min = float(joints_demo[:, 4, 2].min())
        r_toe_z_min = float(joints_demo[:, 8, 2].min())
        print(
            f"[{i}/{len(task_names)}] {task}: T={T} pelvis_z={pelvis_z:.3f} "
            f"l_toe_z_min={l_toe_z_min:.3f} r_toe_z_min={r_toe_z_min:.3f}"
        )

        if sub_name not in height_dict:
            h = compute_height(model_arrays, betas, device=device)
            height_dict[sub_name] = h
            print(f"  -> height[{sub_name}] = {h:.4f} m")

    print(f"\nDone in {time.time() - t0:.1f}s | wrote {n_done} pt files | skipped {n_skip}")
    if height_dict:
        # Merge with existing height_dict.pkl if present
        if height_dict_path.exists():
            with open(height_dict_path, "rb") as f:
                existing = pickle.load(f)
            for k, v in height_dict.items():
                existing.setdefault(k, v)
            height_dict = existing
        with open(height_dict_path, "wb") as f:
            pickle.dump(height_dict, f)
        print(f"Wrote height_dict.pkl: {height_dict_path}")
        print(f"  contents: {height_dict}")


if __name__ == "__main__":
    main()
