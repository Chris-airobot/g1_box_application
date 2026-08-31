import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

parser = argparse.ArgumentParser()
parser.add_argument("--id", default="0205")
args = parser.parse_args()

root = Path("data/mentor_processed")
motion_dir = (
    root / "motions" / f"Bringing-carry_{args.id}" / "motion_actor"
)

npz_path = motion_dir / "motion_actor_retargeted.npz"
box_json = next(
    (motion_dir / "motion_actor_scaled_objects").glob("*_poses.json")
)

model_path = Path(
    "clean_baseline_assets/g1/g1_29dof_w_largebox.xml"
)

out = root / f"mentor_{args.id}.gif"

d = np.load(npz_path)
joint_pos = d["joint_pos"]
base_pos = d["base_pos_w"]
base_quat_xyzw = d["base_quat_w"]
fps = float(d["framerate"])

box = json.load(open(box_json))

n = min(
    len(joint_pos),
    len(base_pos),
    len(base_quat_xyzw),
    len(box),
)

model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)

print("frames:", n)
print("fps:", fps)
print("nq:", model.nq)

# Mentor cube ≈ 0.2289 m; our clean visualization cube ≈ 0.30 m.
# Scale only the cube mesh for visualization.
target_size = 0.22891

for mid in range(model.nmesh):
    adr = model.mesh_vertadr[mid]
    num = model.mesh_vertnum[mid]
    verts = model.mesh_vert[adr:adr+num]

    if len(verts) == 0:
        continue

    extent = np.ptp(verts, axis=0)

    # Identify our ~30 cm cube mesh.
    if np.all((extent > 0.28) & (extent < 0.32)):
        scale = target_size / extent
        model.mesh_vert[adr:adr+num] *= scale
        print("scaled box mesh:", mid, extent, "->", target_size)

renderer = mujoco.Renderer(model, height=480, width=640)

cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)
cam.distance = 3.0
cam.azimuth = 135
cam.elevation = -20

frames = []

for i in range(n):
    # G1 floating base
    data.qpos[0:3] = base_pos[i]

    # Mentor base quaternion is already MuJoCo wxyz
    data.qpos[3:7] = base_quat_xyzw[i]

    # 29 G1 joints
    data.qpos[7:36] = joint_pos[i]

    # box free joint
    pose = box[i]
    data.qpos[36:39] = pose["translation"]

    R = np.asarray(pose["rotation_matrix"])
    q_box_xyzw = Rotation.from_matrix(R).as_quat()
    data.qpos[39:43] = [
        q_box_xyzw[3],
        q_box_xyzw[0],
        q_box_xyzw[1],
        q_box_xyzw[2],
    ]

    mujoco.mj_forward(model, data)

    # camera follows robot
    cam.lookat[:] = data.qpos[0:3]
    cam.lookat[2] = 0.8

    renderer.update_scene(data, camera=cam)
    frames.append(renderer.render().copy())

renderer.close()

imageio.mimsave(
    out,
    frames,
    fps=fps,
    loop=0,
)

print("saved:", out)
