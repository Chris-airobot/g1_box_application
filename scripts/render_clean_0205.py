import numpy as np
import mujoco
import imageio.v2 as imageio

npz = "outputs/clean_omniretarget/0205_clean_original.npz"
xml = "clean_baseline_assets/g1/g1_29dof_w_largebox.xml"
out = "outputs/clean_omniretarget/0205_clean.gif"

qpos = np.load(npz)["qpos"]

model = mujoco.MjModel.from_xml_path(xml)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=480, width=640)

cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)
cam.lookat[:] = [0.4, 0.0, 0.7]
cam.distance = 2.8
cam.azimuth = 90
cam.elevation = -15

frames = []

for q in qpos:
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=cam)
    frames.append(renderer.render().copy())

imageio.mimsave(out, frames, fps=30, loop=0)
renderer.close()

print("Saved:", out)
