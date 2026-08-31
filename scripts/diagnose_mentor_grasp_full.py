from pathlib import Path
import json
import csv
import numpy as np
import mujoco

IDS = ["0179", "0205"]

ROOT = Path("data/mentor_processed/motions")
MODEL_PATH = Path("clean_baseline_assets/g1/g1_29dof_w_largebox.xml")
OUT_DIR = Path("data/mentor_processed/grasp_diagnostics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

def bid(name):
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if i < 0:
        raise RuntimeError(f"Missing body: {name}")
    return i

BODY = {
    "left_wrist": bid("left_wrist_yaw_link"),
    "right_wrist": bid("right_wrist_yaw_link"),
    "left_hand": bid("left_rubber_hand_link"),
    "right_hand": bid("right_rubber_hand_link"),
}

def contiguous_segments(mask):
    out = []
    start = None
    for i, x in enumerate(mask):
        if x and start is None:
            start = i
        elif not x and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out

def read_obj_bounds(path):
    verts = []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                x, y, z = map(float, line.split()[1:4])
                verts.append([x, y, z])
    v = np.asarray(verts)
    return v.min(0), v.max(0)

def surface_distance(local_xyz, bmin, bmax):
    # Euclidean distance from point to box AABB.
    # 0 means the BODY ORIGIN itself lies inside the box volume.
    q = np.maximum(np.maximum(bmin - local_xyz, local_xyz - bmax), 0.0)
    return np.linalg.norm(q, axis=1)

def rotation_step_angle(R):
    if len(R) < 2:
        return np.zeros(0)
    out = []
    for i in range(len(R) - 1):
        D = R[i].T @ R[i + 1]
        c = np.clip((np.trace(D) - 1.0) / 2.0, -1.0, 1.0)
        out.append(np.arccos(c))
    return np.asarray(out)

def shift_score(handL, handR, box_pos, box_rot, start, end, shift):
    idx = np.arange(start, end + 1)
    j = idx + shift
    valid = (j >= 0) & (j < len(box_pos))
    idx = idx[valid]
    j = j[valid]

    if len(idx) < 30:
        return np.inf

    RL = np.transpose(box_rot[j], (0, 2, 1))

    llocal = np.einsum(
        "nij,nj->ni", RL, handL[idx] - box_pos[j]
    )
    rlocal = np.einsum(
        "nij,nj->ni", RL, handR[idx] - box_pos[j]
    )

    lm = np.median(llocal, axis=0)
    rm = np.median(rlocal, axis=0)

    ld = np.linalg.norm(llocal - lm, axis=1)
    rd = np.linalg.norm(rlocal - rm, axis=1)

    # robust combined score
    return (
        np.median(ld)
        + np.median(rd)
        + 0.25 * np.percentile(ld, 95)
        + 0.25 * np.percentile(rd, 95)
    )

for MID in IDS:

    print("\n")
    print("=" * 78)
    print("MOTION", MID)
    print("=" * 78)

    root = ROOT / f"Bringing-carry_{MID}" / "motion_actor"

    npz_path = root / "motion_actor_retargeted.npz"
    obj_dir = root / "motion_actor_scaled_objects"
    box_json_path = obj_dir / "Box_H_1_poses.json"
    box_obj_path = obj_dir / "Box_H_1.obj"

    motion = np.load(npz_path)

    joint_pos = motion["joint_pos"]
    base_pos = motion["base_pos_w"]
    base_quat = motion["base_quat_w"]
    fps = float(motion["framerate"])

    with open(box_json_path) as f:
        box = json.load(f)

    box_pos = np.asarray([x["translation"] for x in box])
    box_rot = np.asarray([x["rotation_matrix"] for x in box])
    box_frames = np.asarray([x["frame"] for x in box])

    n = min(len(joint_pos), len(box_pos))

    joint_pos = joint_pos[:n]
    base_pos = base_pos[:n]
    base_quat = base_quat[:n]
    box_pos = box_pos[:n]
    box_rot = box_rot[:n]

    bmin, bmax = read_obj_bounds(box_obj_path)
    bsize = bmax - bmin

    print("frames:", n)
    print("fps:", fps)
    print("duration:", n / fps, "sec")
    print("box entries:", len(box))
    print("box frames:", box_frames[0], "->", box_frames[-1])
    print("box sequential:",
          bool(np.all(np.diff(box_frames) == 1)))
    print("box mesh size [cm]:", bsize * 100)

    det = np.linalg.det(box_rot)
    ortho = np.linalg.norm(
        np.matmul(np.transpose(box_rot, (0,2,1)), box_rot)
        - np.eye(3),
        axis=(1,2)
    )

    print("rotation determinant min/max:",
          float(det.min()), float(det.max()))
    print("rotation orthogonality max error:",
          float(ortho.max()))

    # ------------------------------------------------------------
    # Forward kinematics for all candidate hand/wrist bodies
    # ------------------------------------------------------------

    pos = {name: np.zeros((n,3)) for name in BODY}

    for i in range(n):

        data.qpos[:] = model.qpos0

        data.qpos[0:3] = base_pos[i]
        data.qpos[3:7] = base_quat[i]
        data.qpos[7:36] = joint_pos[i]

        mujoco.mj_forward(model, data)

        for name, body_id in BODY.items():
            pos[name][i] = data.xpos[body_id]

    # ------------------------------------------------------------
    # Basic trajectory continuity
    # ------------------------------------------------------------

    box_step = np.linalg.norm(np.diff(box_pos, axis=0), axis=1)
    box_rot_step = rotation_step_angle(box_rot)

    lh_step = np.linalg.norm(
        np.diff(pos["left_hand"], axis=0), axis=1
    )
    rh_step = np.linalg.norm(
        np.diff(pos["right_hand"], axis=0), axis=1
    )

    print("\nTRAJECTORY JUMP CHECK")
    print("box max translation step [cm]:",
          box_step.max() * 100)
    print("box P99 translation step [cm]:",
          np.percentile(box_step,99) * 100)
    print("box max rotation step [deg]:",
          np.degrees(box_rot_step.max()))
    print("left hand max step [cm]:",
          lh_step.max() * 100)
    print("right hand max step [cm]:",
          rh_step.max() * 100)

    # ------------------------------------------------------------
    # Transform all hand/wrist origins into box frame
    # ------------------------------------------------------------

    local = {}

    Rt = np.transpose(box_rot, (0,2,1))

    for name in BODY:
        local[name] = np.einsum(
            "nij,nj->ni",
            Rt,
            pos[name] - box_pos
        )

    # center distance
    dist = {
        name: np.linalg.norm(local[name], axis=1)
        for name in BODY
    }

    # approximate distance of link origin to actual box surface
    surf = {
        name: surface_distance(local[name], bmin, bmax)
        for name in BODY
    }

    print("\nWHOLE-MOTION DISTANCE TO BOX CENTRE")
    for name in BODY:
        print(
            f"{name:12s}",
            f"median={np.median(dist[name])*100:6.2f} cm",
            f"min={dist[name].min()*100:6.2f} cm",
            f"P95={np.percentile(dist[name],95)*100:6.2f} cm"
        )

    # ------------------------------------------------------------
    # Grasp candidate based on actual rubber hands
    # ------------------------------------------------------------

    # Broad threshold intentionally used ONLY to identify candidate period.
    candidate = (
        (dist["left_hand"] < 0.45)
        & (dist["right_hand"] < 0.45)
    )

    segs = contiguous_segments(candidate)

    print("\nALL TWO-HAND PROXIMITY SEGMENTS (> 1 sec)")
    long_segs = [
        x for x in segs
        if (x[1] - x[0] + 1) / fps >= 1.0
    ]

    for a,b in long_segs:
        print(
            f"frames {a:4d}-{b:4d}",
            f"time {a/fps:6.2f}-{b/fps:6.2f}s",
            f"duration={(b-a+1)/fps:6.2f}s"
        )

    if not segs:
        print("ERROR: no candidate grasp segment")
        continue

    start,end = max(segs, key=lambda x:x[1]-x[0])

    # Don't trim yet: diagnose full candidate period
    idx = np.arange(start,end+1)

    print("\nMAIN CANDIDATE CARRY")
    print(
        f"frames {start}-{end}",
        f"time {start/fps:.2f}-{end/fps:.2f}s",
        f"duration={(end-start+1)/fps:.2f}s"
    )

    # ------------------------------------------------------------
    # Compare wrist origins vs rubber-hand origins
    # ------------------------------------------------------------

    print("\nLINK COMPARISON DURING MAIN SEGMENT")

    for name in BODY:

        x = local[name][idx]
        ref = np.median(x, axis=0)
        dev = np.linalg.norm(x-ref,axis=1)

        print(
            f"{name:12s}",
            f"center_med={np.median(dist[name][idx])*100:6.2f}cm",
            f"surface_med={np.median(surf[name][idx])*100:6.2f}cm",
            f"drift_med={np.median(dev)*100:6.2f}cm",
            f"drift_P95={np.percentile(dev,95)*100:6.2f}cm",
            f"drift_max={dev.max()*100:6.2f}cm"
        )

    # ------------------------------------------------------------
    # Main rubber-hand relative-position diagnostics
    # ------------------------------------------------------------

    L = local["left_hand"][idx]
    R = local["right_hand"][idx]

    Lref = np.median(L, axis=0)
    Rref = np.median(R, axis=0)

    Ldev = np.linalg.norm(L-Lref,axis=1)
    Rdev = np.linalg.norm(R-Rref,axis=1)
    worst = np.maximum(Ldev,Rdev)

    print("\nRUBBER-HAND REFERENCE POSITIONS")
    print("left median local xyz [cm]:", Lref * 100)
    print("right median local xyz [cm]:", Rref * 100)

    print("\nRUBBER-HAND DRIFT")
    print(
        "LEFT:",
        "median", np.median(Ldev)*100,
        "P95", np.percentile(Ldev,95)*100,
        "max", Ldev.max()*100,
        "cm"
    )
    print(
        "RIGHT:",
        "median", np.median(Rdev)*100,
        "P95", np.percentile(Rdev,95)*100,
        "max", Rdev.max()*100,
        "cm"
    )

    # ------------------------------------------------------------
    # Hand separation
    # ------------------------------------------------------------

    sep = np.linalg.norm(
        pos["left_hand"][idx] -
        pos["right_hand"][idx],
        axis=1
    )

    print("\nHAND SEPARATION")
    print("median [cm]:", np.median(sep)*100)
    print("P05/P95 [cm]:",
          np.percentile(sep,5)*100,
          np.percentile(sep,95)*100)
    print("min/max [cm]:",
          sep.min()*100,
          sep.max()*100)

    # ------------------------------------------------------------
    # Count meaningful drift
    # ------------------------------------------------------------

    print("\nDRIFT THRESHOLD COUNTS")

    for threshold_cm in [3,5,8,10,15,20,30]:

        count = int(np.sum(worst > threshold_cm/100))
        pct = 100 * count / len(worst)

        print(
            f">{threshold_cm:2d} cm:",
            f"{count:4d} frames",
            f"({pct:5.2f}%)",
            f"= {count/fps:.2f} sec"
        )

    # ------------------------------------------------------------
    # Find exact contiguous bad intervals
    # ------------------------------------------------------------

    bad_threshold = 0.08  # 8cm drift from typical grasp
    bad_local = worst > bad_threshold

    bad_segments = contiguous_segments(bad_local)

    print("\nBAD INTERVALS (>8 cm relative drift)")

    ranked = []

    for a,b in bad_segments:

        global_a = start+a
        global_b = start+b
        mx = worst[a:b+1].max()

        ranked.append(
            (mx, global_a, global_b)
        )

    ranked.sort(reverse=True)

    for mx,a,b in ranked[:15]:

        print(
            f"frames {a:4d}-{b:4d}",
            f"time {a/fps:6.2f}-{b/fps:6.2f}s",
            f"duration={(b-a+1)/fps:5.2f}s",
            f"max={mx*100:6.2f}cm"
        )

    if not ranked:
        print("none")

    # ------------------------------------------------------------
    # Show worst individual frames
    # ------------------------------------------------------------

    worst_order = np.argsort(worst)[-15:][::-1]

    print("\nWORST INDIVIDUAL FRAMES")

    for j in worst_order:

        f = start+j

        print(
            f"frame={f:4d}",
            f"time={f/fps:6.2f}s",
            f"Lcenter={dist['left_hand'][f]*100:6.2f}cm",
            f"Rcenter={dist['right_hand'][f]*100:6.2f}cm",
            f"Ldrift={Ldev[j]*100:6.2f}cm",
            f"Rdrift={Rdev[j]*100:6.2f}cm",
            f"box_z={box_pos[f,2]:.3f}"
        )

    # ------------------------------------------------------------
    # Test possible robot/object temporal misalignment
    # +/- 2 seconds
    # ------------------------------------------------------------

    max_shift = int(round(2.0 * fps))

    scores = []

    for shift in range(-max_shift,max_shift+1):
        score = shift_score(
            pos["left_hand"],
            pos["right_hand"],
            box_pos,
            box_rot,
            start,
            end,
            shift
        )
        scores.append((score,shift))

    scores.sort()

    best_score,best_shift = scores[0]

    zero_score = [
        x[0] for x in scores if x[1] == 0
    ][0]

    print("\nTIME-OFFSET TEST (robot vs box)")
    print(
        "zero-shift score:",
        zero_score*100,
        "cm"
    )
    print(
        "best shift:",
        best_shift,
        "frames =",
        best_shift/fps,
        "sec"
    )
    print(
        "best score:",
        best_score*100,
        "cm"
    )

    if zero_score > 0:
        print(
            "score improvement:",
            100*(zero_score-best_score)/zero_score,
            "%"
        )

    print("top 10 shifts:")
    for score,shift in scores[:10]:
        print(
            f" shift={shift:+4d}",
            f"({shift/fps:+6.3f}s)",
            f"score={score*100:7.3f}cm"
        )

    # ------------------------------------------------------------
    # Save detailed per-frame CSV
    # ------------------------------------------------------------

    csv_path = OUT_DIR / f"{MID}_grasp_diagnostic.csv"

    with open(csv_path,"w",newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "frame",
            "time_s",
            "box_x",
            "box_y",
            "box_z",
            "left_hand_center_cm",
            "right_hand_center_cm",
            "left_surface_cm",
            "right_surface_cm",
            "left_local_x_cm",
            "left_local_y_cm",
            "left_local_z_cm",
            "right_local_x_cm",
            "right_local_y_cm",
            "right_local_z_cm",
            "candidate_segment",
        ])

        for i in range(n):

            writer.writerow([
                i,
                i/fps,
                *box_pos[i],
                dist["left_hand"][i]*100,
                dist["right_hand"][i]*100,
                surf["left_hand"][i]*100,
                surf["right_hand"][i]*100,
                *(local["left_hand"][i]*100),
                *(local["right_hand"][i]*100),
                int(start <= i <= end),
            ])

    print("\nCSV saved:", csv_path)

print("\n")
print("="*78)
print("DONE")
print("="*78)
