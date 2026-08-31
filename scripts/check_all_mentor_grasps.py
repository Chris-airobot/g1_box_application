from pathlib import Path
import csv
import json
import numpy as np
import mujoco

ROOT = Path("data/mentor_processed/motions")
MODEL_PATH = Path("clean_baseline_assets/g1/g1_29dof_w_largebox.xml")
OUT = Path("data/mentor_processed/grasp_consistency.csv")

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

def body_id(side):
    for name in [
        f"{side}_wrist_yaw_link",
        f"{side}_wrist_pitch_link",
        f"{side}_wrist_roll_link",
    ]:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            return bid
    raise RuntimeError(f"No {side} wrist body")

lid = body_id("left")
rid = body_id("right")

results = []

dirs = sorted(ROOT.glob("Bringing-carry_*"))

for k, root in enumerate(dirs, 1):
    mid = root.name.replace("Bringing-carry_", "")
    print(f"[{k}/{len(dirs)}] {mid}")

    try:
        npz_path = root / "motion_actor/motion_actor_retargeted.npz"
        box_path = root / "motion_actor/motion_actor_scaled_objects/Box_H_1_poses.json"

        motion = np.load(npz_path)
        joint_pos = motion["joint_pos"]
        base_pos = motion["base_pos_w"]
        base_quat = motion["base_quat_w"]

        with open(box_path) as f:
            box_data = json.load(f)

        box_pos = np.asarray([x["translation"] for x in box_data])
        box_rot = np.asarray([x["rotation_matrix"] for x in box_data])

        n = min(len(joint_pos), len(box_pos))
        left_rel = []
        right_rel = []

        for i in range(n):
            data.qpos[:] = model.qpos0
            data.qpos[0:3] = base_pos[i]
            data.qpos[3:7] = base_quat[i]
            data.qpos[7:36] = joint_pos[i]

            mujoco.mj_forward(model, data)

            b = box_pos[i]
            R = box_rot[i]

            left_rel.append(R.T @ (data.xpos[lid] - b))
            right_rel.append(R.T @ (data.xpos[rid] - b))

        left_rel = np.asarray(left_rel)
        right_rel = np.asarray(right_rel)

        # Find longest high-box segment = steady carry candidate
        z = box_pos[:n, 2]
        zmin, zmax = z.min(), z.max()
        threshold = zmin + 0.65 * (zmax - zmin)
        lifted = z > threshold

        segments = []
        start = None

        for i, flag in enumerate(lifted):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                segments.append((start, i - 1))
                start = None

        if start is not None:
            segments.append((start, n - 1))

        if not segments:
            raise RuntimeError("No lifted segment found")

        carry_start, carry_end = max(
            segments, key=lambda x: x[1] - x[0]
        )

        # Remove pickup/release edges
        trim = max(1, int(0.10 * (carry_end - carry_start + 1)))
        carry_start += trim
        carry_end -= trim

        if carry_end <= carry_start:
            raise RuntimeError("Carry segment too short")

        def metric(x):
            x = x[carry_start:carry_end + 1]
            ref = np.median(x, axis=0)
            dev = np.linalg.norm(x - ref, axis=1)

            return (
                np.median(dev) * 100,
                np.percentile(dev, 95) * 100,
                np.max(dev) * 100,
            )

        l_med, l_p95, l_max = metric(left_rel)
        r_med, r_p95, r_max = metric(right_rel)

        fps = float(motion["framerate"])
        frames = carry_end - carry_start + 1

        results.append({
            "motion_id": mid,
            "left_median_cm": l_med,
            "left_p95_cm": l_p95,
            "left_max_cm": l_max,
            "right_median_cm": r_med,
            "right_p95_cm": r_p95,
            "right_max_cm": r_max,
            "worst_p95_cm": max(l_p95, r_p95),
            "worst_max_cm": max(l_max, r_max),
            "carry_start": carry_start,
            "carry_end": carry_end,
            "carry_duration_s": frames / fps,
        })

    except Exception as e:
        print("  ERROR:", e)

results.sort(key=lambda x: x["worst_p95_cm"], reverse=True)

OUT.parent.mkdir(parents=True, exist_ok=True)

with open(OUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print()
print("Saved:", OUT)
print("Evaluated:", len(results))
print()
print("Worst 20:")
for x in results[:20]:
    print(
        x["motion_id"],
        f"P95={x['worst_p95_cm']:.2f} cm",
        f"MAX={x['worst_max_cm']:.2f} cm"
    )
