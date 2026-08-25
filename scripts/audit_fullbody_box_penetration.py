import csv
import numpy as np
import mujoco
from pathlib import Path

from eval_hiphi_smokes import get_model_path

ROOT = Path("/fred/oz430/tliu/data/HiPHI")
SRC  = ROOT / "validation_smokes"
OUT  = ROOT / "retarget_225_hiphi"

# Intended grasp-contact geoms: don't count these as body penetration.
GRASP_ALLOWED = (
    "rubber_hand",
    "thumb",
    "pinky",
)

rows = list(csv.DictReader(open(OUT / "contact_validation.csv")))
results = []

for n, r in enumerate(rows, 1):
    task = r["task"]

    src = np.load(SRC / f"{task}.npz", allow_pickle=True)
    box_size = np.asarray(src["box_size"]).reshape(3)

    model_path, _ = get_model_path(box_size)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    ret = np.load(OUT / f"{task}_original.npz", allow_pickle=True)

    # Find qpos trajectory robustly.
    q = None
    for key in ["q", "qpos", "robot_qpos", "robot_q"]:
        if key in ret.files and ret[key].ndim == 2:
            q = ret[key]
            break

    if q is None:
        for key in ret.files:
            a = ret[key]
            if getattr(a, "ndim", 0) == 2 and a.shape[1] == model.nq:
                q = a
                break

    if q is None:
        raise RuntimeError(f"Cannot find q trajectory for {task}")

    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom_{i}"
        for i in range(model.ngeom)
    ]

    box_ids = [i for i, name in enumerate(names) if "box_" in name]
    if not box_ids:
        raise RuntimeError(f"No box geom found for {task}")

    body_names = [
        mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            model.geom_bodyid[i]
        ) or ""
        for i in range(model.ngeom)
    ]

    robot_ids = []
    for i, name in enumerate(names):
        body = body_names[i]

        if i in box_ids:
            continue
        if name == "ground" or body == "ground":
            continue
        if any(x in body for x in GRASP_ALLOWED):
            continue

        robot_ids.append(i)

    worst_dist = np.inf
    worst_robot = ""
    worst_frame = -1
    buf = np.zeros(6)

    for frame, qi in enumerate(q):
        data.qpos[:] = qi
        mujoco.mj_forward(model, data)

        for rg in robot_ids:
            for bg in box_ids:
                dist = mujoco.mj_geomDistance(
                    model, data, rg, bg, 1.0, buf
                )
                if dist < worst_dist:
                    worst_dist = dist
                    worst_robot = body_names[rg]
                    worst_frame = frame

    pen_mm = max(0.0, -float(worst_dist) * 1000.0)

    results.append({
        "task": task,
        "max_body_box_penetration_mm": pen_mm,
        "worst_geom": worst_robot,
        "worst_frame": worst_frame,
    })

    print(
        f"[{n:3d}/{len(rows)}] {task} "
        f"pen={pen_mm:.2f}mm "
        f"{worst_robot} frame={worst_frame}"
    )

csv_path = OUT / "fullbody_box_penetration.csv"

with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=results[0].keys())
    w.writeheader()
    w.writerows(results)

print("CSV:", csv_path)
