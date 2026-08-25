from pathlib import Path
import subprocess
import csv

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path("/fred/oz430/tliu/data/HiPHI/validation_smokes")
RESULT_DIR = Path("/fred/oz430/tliu/data/HiPHI/retarget_225_hiphi")
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
            human[i, LEFT_HAND_IDX],
            obj[i],
            half,
        )
        for i in range(len(human))
    ])

    source_R = np.array([
        source_point_box_distance(
            human[i, RIGHT_HAND_IDX],
            obj[i],
            half,
        )
        for i in range(len(human))
    ])

    model_path, object_name = get_model_path(box_size)

    robot_L, robot_R = robot_distances(
        result_path,
        model_path,
        object_name,
    )

    # Use the pickup frame stored during HiPHI preprocessing.
    pickup = int(src["local_pickup_frame"])

    # Search two seconds before and after pickup.
    lo = max(0, pickup - int(2 * FPS))
    hi = min(len(obj), pickup + int(2 * FPS))

    src_contact = first_both_contact(
        source_L, source_R, lo, hi
    )
    robot_contact = first_both_contact(
        robot_L, robot_R, lo, hi
    )

    if src_contact is None:
        src_offset = np.nan
    else:
        src_offset = (src_contact - pickup) / FPS

    if robot_contact is None:
        robot_offset = np.nan
    else:
        robot_offset = (robot_contact - pickup) / FPS

    if src_contact is None or robot_contact is None:
        timing_error = np.nan
    else:
        timing_error = (
            robot_contact - src_contact
        ) / FPS

    # Carry interval:
    # pickup -> up to 2 seconds after pickup for smoke validation.
    carry_start = pickup
    carry_end = min(
        len(robot_L),
        pickup + int(2.0 * FPS),
    )

    carry_L = robot_L[carry_start:carry_end]
    carry_R = robot_R[carry_start:carry_end]

    both_under_2cm = np.mean(
        (carry_L < CARRY_CONTACT_THRESH)
        & (carry_R < CARRY_CONTACT_THRESH)
    )

    both_under_1cm = np.mean(
        (carry_L < CONTACT_THRESH)
        & (carry_R < CONTACT_THRESH)
    )

    # Negative distance = penetration in MuJoCo.
    min_distance = float(
        min(robot_L.min(), robot_R.min())
    )

    max_penetration_mm = max(
        0.0,
        -min_distance * 1000.0,
    )

    # Simple validation rule.
    timing_ok = (
        np.isfinite(timing_error)
        and abs(timing_error) <= 0.15
    )

    carry_ok = both_under_2cm >= 0.80
    penetration_ok = max_penetration_mm <= 2.0

    passed = timing_ok and carry_ok and penetration_ok

    return {
        "task": task,
        "frames": len(obj),
        "pickup_frame": pickup,
        "source_contact_offset_s": src_offset,
        "robot_contact_offset_s": robot_offset,
        "timing_error_s": timing_error,
        "carry_both_lt_1cm": both_under_1cm,
        "carry_both_lt_2cm": both_under_2cm,
        "max_penetration_mm": max_penetration_mm,
        "PASS": passed,
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
    files = sorted(DATA_DIR.glob("*.npz"))

    # Ignore helper/precontact files.
    files = [
        f for f in files
        if "precontact" not in f.stem
        and (RESULT_DIR / f"{f.stem}_original.npz").exists()
    ]

    if not files:
        raise RuntimeError(
            f"No NPZ files found in {DATA_DIR}"
        )

    print(f"Found {len(files)} validation motions")

    rows = []

    for source_path in files:
        task = source_path.stem

        try:
            result_path = run_retarget(task)

            row = evaluate(
                task,
                source_path,
                result_path,
            )

            rows.append(row)

            print()
            print(
                f"[{'PASS' if row['PASS'] else 'FAIL'}] "
                f"{task}"
            )
            print(
                f"  source contact : "
                f"{row['source_contact_offset_s']:+.3f} s"
            )
            print(
                f"  G1 contact     : "
                f"{row['robot_contact_offset_s']:+.3f} s"
            )
            print(
                f"  timing error   : "
                f"{row['timing_error_s']:+.3f} s"
            )
            print(
                f"  carry <1 cm    : "
                f"{100*row['carry_both_lt_1cm']:.1f}%"
            )
            print(
                f"  carry <2 cm    : "
                f"{100*row['carry_both_lt_2cm']:.1f}%"
            )
            print(
                f"  penetration    : "
                f"{row['max_penetration_mm']:.2f} mm"
            )

        except Exception as exc:
            print(f"[ERROR] {task}: {exc}")

            rows.append({
                "task": task,
                "frames": "",
                "pickup_frame": "",
                "source_contact_offset_s": "",
                "robot_contact_offset_s": "",
                "timing_error_s": "",
                "carry_both_lt_1cm": "",
                "carry_both_lt_2cm": "",
                "max_penetration_mm": "",
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
                "source_contact_offset_s",
                "robot_contact_offset_s",
                "timing_error_s",
                "carry_both_lt_1cm",
                "carry_both_lt_2cm",
                "max_penetration_mm",
                "PASS",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for r in rows:
        mark = "PASS" if r["PASS"] else "FAIL"

        if r["timing_error_s"] == "":
            print(f"{mark:4s}  {r['task']}")
        else:
            print(
                f"{mark:4s}  "
                f"{r['task']:<45s} "
                f"timing={r['timing_error_s']:+.3f}s  "
                f"<2cm={100*r['carry_both_lt_2cm']:5.1f}%  "
                f"pen={r['max_penetration_mm']:5.2f}mm"
            )

    passed = sum(bool(r["PASS"]) for r in rows)

    print()
    print(f"Passed: {passed}/{len(rows)}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
