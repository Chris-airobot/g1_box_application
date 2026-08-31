from pathlib import Path
import json
import numpy as np
import mujoco

MOTION_ID = "0179"

root = Path("data/mentor_processed/motions") / f"Bringing-carry_{MOTION_ID}"
npz_path = root / "motion_actor/motion_actor_retargeted.npz"
box_path = root / "motion_actor/motion_actor_scaled_objects/Box_H_1_poses.json"

model_path = Path("clean_baseline_assets/g1/g1_29dof_w_largebox.xml")

motion = np.load(npz_path)
joint_pos = motion["joint_pos"]
base_pos = motion["base_pos_w"]
base_quat = motion["base_quat_w"]  # already MuJoCo wxyz

with open(box_path) as f:
    box_data = json.load(f)

box_pos = np.array([x["translation"] for x in box_data])
box_rot = np.array([x["rotation_matrix"] for x in box_data])

model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)

def find_body(side):
    candidates = [
        f"{side}_rubber_hand_link",
        f"{side}_wrist_yaw_link",
        f"{side}_wrist_pitch_link",
        f"{side}_wrist_roll_link",
    ]
    for name in candidates:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            return name, bid
    raise RuntimeError(f"Could not find {side} wrist body")

lname, lid = find_body("left")
rname, rid = find_body("right")

left_rel = []
right_rel = []

n = min(len(joint_pos), len(box_pos))

for i in range(n):
    data.qpos[:] = 0
    data.qpos[0:3] = base_pos[i]
    data.qpos[3:7] = base_quat[i]
    data.qpos[7:36] = joint_pos[i]

    mujoco.mj_forward(model, data)

    lp = data.xpos[lid].copy()
    rp = data.xpos[rid].copy()

    # world -> box coordinates
    R = box_rot[i]
    b = box_pos[i]

    left_rel.append(R.T @ (lp - b))
    right_rel.append(R.T @ (rp - b))

left_rel = np.asarray(left_rel)
right_rel = np.asarray(right_rel)

# Detect actual grasp/carry phase from wrist-box proximity.
# Rotation does not matter for distance.
left_dist = np.linalg.norm(left_rel, axis=1)
right_dist = np.linalg.norm(right_rel, axis=1)

# For the ~22.9 cm box, both wrists should be comfortably
# within 45 cm of the box centre while actually grasping it.
grasped = (left_dist < 0.45) & (right_dist < 0.45)

segments = []
start_idx = None

for i, flag in enumerate(grasped):
    if flag and start_idx is None:
        start_idx = i
    elif not flag and start_idx is not None:
        segments.append((start_idx, i - 1))
        start_idx = None

if start_idx is not None:
    segments.append((start_idx, n - 1))

if not segments:
    raise RuntimeError("No two-hand grasp segment found")

carry_start, carry_end = max(
    segments, key=lambda x: x[1] - x[0]
)

# Trim approach/release transition frames
trim = max(1, int(0.10 * (carry_end - carry_start + 1)))
carry_start += trim
carry_end -= trim

mask = np.zeros(n, dtype=bool)
mask[carry_start:carry_end + 1] = True

print("Steady carry frames:", carry_start, "->", carry_end)
print("Steady carry duration:", mask.sum() / float(motion["framerate"]), "sec")
print("Median wrist distances [cm]:",
      np.median(left_dist[mask]) * 100,
      np.median(right_dist[mask]) * 100)

for name, x in [("LEFT", left_rel), ("RIGHT", right_rel)]:
    x = x[mask]

    reference = np.median(x, axis=0)
    deviation = np.linalg.norm(x - reference, axis=1)

    print()
    print(name)
    print("median relative xyz [m]:", reference)
    print("median deviation [m]:", np.median(deviation))
    print("95th percentile [m]:", np.percentile(deviation, 95))
    print("max deviation [m]:", deviation.max())

print("\nLargest deviation frames:")
for name, x in [("LEFT", left_rel), ("RIGHT", right_rel)]:
    xx = x[mask]
    ref = np.median(xx, axis=0)
    dev = np.linalg.norm(xx - ref, axis=1)

    frame_ids = np.where(mask)[0]
    worst = np.argsort(dev)[-10:][::-1]

    print("\n", name)
    for j in worst:
        f = frame_ids[j]
        print(
            f"frame={f:4d}",
            f"dev={dev[j]*100:6.2f} cm",
            f"wrist_box_dist={np.linalg.norm(x[f])*100:6.2f} cm",
            f"box_z={box_pos[f,2]:.3f}"
        )
