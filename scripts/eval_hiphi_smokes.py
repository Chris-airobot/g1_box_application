from pathlib import Path
import os
import subprocess
import csv

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
HIPHI_ROOT = Path(
    os.environ.get(
        "HIPHI_ROOT",
        str(Path.home() / "Chris" / "HiPHI"),
    )
).expanduser()

DATA_DIR = HIPHI_ROOT / "validation_smokes"
RESULT_DIR = HIPHI_ROOT / "retarget_225_hiphi"
RETARGET = (
    ROOT
    / "src/holosoma_retargeting/holosoma_retargeting/examples/robot_retarget.py"
)
MODEL_DIR = (
    ROOT
    / "src/holosoma_retargeting/holosoma_retargeting/models/g1"
)

FPS = 90.0

# Current HiPHI 55-joint export.
LEFT_HAND_IDX = 21       # LeftHandMiddle3
RIGHT_HAND_IDX = 40      # RightHandMiddle3

CONTACT_THRESH = 0.01    # 1 cm
PICKUP_Z_THRESH = 0.02   # object rises 2 cm
CARRY_CONTACT_THRESH = 0.02  # 2 cm


def box_name(box_size):
    def fmt(x):
        return f"{float(x):.4f}".replace(".", "p")

    x, y, z = box_size
    return f"box_{fmt(x)}_{fmt(y)}_{fmt(z)}"


def get_model_path(box_size):
    name = box_name(box_size)
    path = MODEL_DIR / f"g1_29dof_w_{name}.xml"

    if not path.exists():
        raise FileNotFoundError(
            f"Expected generated MuJoCo model not found:\n{path}"
        )

    return path, name


def source_point_box_distance(point, pose, half):
    # HiPHI object pose:
    # [qw, qx, qy, qz, x, y, z]
    qw, qx, qy, qz = pose[:4]
    center = pose[4:7]

    R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()

    p_local = R.T @ (point - center)
    outside = np.maximum(np.abs(p_local) - half, 0.0)

    return float(np.linalg.norm(outside))


def first_both_contact(left, right, start, end, threshold=CONTACT_THRESH):
    ids = np.where(
        (left[start:end] < threshold)
        & (right[start:end] < threshold)
    )[0]

    if len(ids) == 0:
        return None

    return start + int(ids[0])


def robot_distances(result_path, model_path, box_geom_name):
    q = np.load(result_path)["qpos"]

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    lg = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "left_rubber_hand_link",
    )
    rg = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "right_rubber_hand_link",
    )
    bg = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        box_geom_name,
    )

    if min(lg, rg, bg) < 0:
        raise RuntimeError(
            f"Missing geom: left={lg}, right={rg}, box={bg}"
        )

    left = []
    right = []

    buf = np.zeros(6)

    for qi in q:
        data.qpos[:] = qi
        mujoco.mj_forward(model, data)

        left.append(
            mujoco.mj_geomDistance(
                model, data, lg, bg, 1.0, buf
            )
        )
        right.append(
            mujoco.mj_geomDistance(
                model, data, rg, bg, 1.0, buf
            )
        )

    return np.asarray(left), np.asarray(right)


def evaluate(task, source_path, result_path):
    src = np.load(source_path)

    human = src["global_joint_positions"]
    obj = src["object_poses"]
    box_size = np.asarray(src["box_size"]).reshape(3)
    half = box_size / 2.0

    source_L = np.array([
        source_point_box_distance(
            human[i, LEFT_HAND_IDX], obj[i], half
        )
        for i in range(len(human))
    ])

    source_R = np.array([
        source_point_box_distance(
            human[i, RIGHT_HAND_IDX], obj[i], half
        )
        for i in range(len(human))
    ])

    model_path, object_name = get_model_path(box_size)

    robot_L, robot_R = robot_distances(
        result_path,
        model_path,
        object_name,
    )

    pickup = int(src["local_pickup_frame"])
    fps = float(src["fps"])

    end = min(
        len(obj),
        len(robot_L),
        len(robot_R),
        pickup + int(2.0 * fps),
    )

    sL = source_L[pickup:end]
    sR = source_R[pickup:end]
    rL = robot_L[pickup:end]
    rR = robot_R[pickup:end]

    sfL = float(np.mean(sL < 0.02))
    sfR = float(np.mean(sR < 0.02))
    rfL = float(np.mean(rL < 0.02))
    rfR = float(np.mean(rR < 0.02))

    needL = sfL >= 0.80
    needR = sfR >= 0.80

    if needL and needR:
        source_mode = "BOTH"
    elif needL:
        source_mode = "LEFT_ONLY"
    elif needR:
        source_mode = "RIGHT_ONLY"
    else:
        source_mode = "NEITHER"

    if not needL and not needR:
        status = "SOURCE_QC"
    else:
        contact_match = (
            (not needL or rfL >= 0.80)
            and
            (not needR or rfR >= 0.80)
        )

        status = (
            "CONTACT_MATCH"
            if contact_match
            else "CONTACT_FAIL"
        )

    return {
        "task": task,
        "frames": len(obj),
        "pickup_frame": pickup,
        "source_contact_mode": source_mode,
        "source_left_lt_2cm": sfL,
        "source_right_lt_2cm": sfR,
        "robot_left_lt_2cm": rfL,
        "robot_right_lt_2cm": rfR,
        "contact_status": status,
        "PASS": status == "CONTACT_MATCH",
    }

def run_retarget(task):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    expected = RESULT_DIR / f"{task}_original.npz"

    if expected.exists():
        print(f"[SKIP] {task}: result already exists")
        return expected

    cmd = [
        "python",
        str(RETARGET),
        "--task-type", "object_interaction",
        "--robot", "g1",
        "--data-format", "hiphi",
        "--task-name", task,
        "--data-path", str(DATA_DIR),
        "--save-dir", str(RESULT_DIR),
    ]

    print()
    print("=" * 80)
    print(f"RETARGETING: {task}")
    print("=" * 80)

    subprocess.run(cmd, check=True)

    if not expected.exists():
        raise FileNotFoundError(
            f"Retarget finished but result missing: {expected}"
        )

    return expected


def main():
    from collections import Counter

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(DATA_DIR.glob("*.npz"))
    files = [
        f for f in files
        if "precontact" not in f.stem
        and (RESULT_DIR / f"{f.stem}_original.npz").exists()
    ]

    if not files:
        raise RuntimeError(
            f"No matched source/retarget files found in {DATA_DIR}"
        )

    rows = []

    for source_path in files:
        task = source_path.stem

        try:
            result_path = RESULT_DIR / f"{task}_original.npz"

            row = evaluate(
                task,
                source_path,
                result_path,
            )
            rows.append(row)

            print(
                f"[{row['contact_status']}] {task} "
                f"source={row['source_contact_mode']} "
                f"G1_L={100*row['robot_left_lt_2cm']:.1f}% "
                f"G1_R={100*row['robot_right_lt_2cm']:.1f}%"
            )

        except Exception as exc:
            print(f"[ERROR] {task}: {exc}")

            rows.append({
                "task": task,
                "frames": "",
                "pickup_frame": "",
                "source_contact_mode": "",
                "source_left_lt_2cm": "",
                "source_right_lt_2cm": "",
                "robot_left_lt_2cm": "",
                "robot_right_lt_2cm": "",
                "contact_status": "ERROR",
                "PASS": False,
            })

    csv_path = RESULT_DIR / "contact_validation.csv"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task",
                "frames",
                "pickup_frame",
                "source_contact_mode",
                "source_left_lt_2cm",
                "source_right_lt_2cm",
                "robot_left_lt_2cm",
                "robot_right_lt_2cm",
                "contact_status",
                "PASS",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(
        r["contact_status"]
        for r in rows
    )

    print()
    print("SUMMARY")
    for k, v in counts.items():
        print(f"{k}: {v}")

    print("CSV:", csv_path)


if __name__ == "__main__":
    main()
