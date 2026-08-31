import runpy
import mujoco
import numpy as np

import holosoma_retargeting.src.interaction_mesh_retargeter as im
from holosoma_retargeting.src.interaction_mesh_retargeter import InteractionMeshRetargeter

# ---------------------------------------------------------
# Keep the MuJoCo enum compatibility fix
# ---------------------------------------------------------
_orig_T = InteractionMeshRetargeter._build_transform_qdot_to_qvel_fast

def fixed_T(self, use_world_omega=True):
    T = _orig_T(self, use_world_omega)
    for j in range(1, self.robot_model.njnt):
        jt = int(self.robot_model.jnt_type[j])
        if jt in (
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        ):
            qa = int(self.robot_model.jnt_qposadr[j])
            da = int(self.robot_model.jnt_dofadr[j])
            T[da, qa] = 1.0
    return T

InteractionMeshRetargeter._build_transform_qdot_to_qvel_fast = fixed_T

# ---------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------
topologies = {}
feet = {}
collision_counts = {}

# Interaction mesh: one call per frame
_orig_mesh = im.create_interaction_mesh
mesh_frame = 0

def diag_mesh(points):
    global mesh_frame
    vertices, tetra = _orig_mesh(points)

    # canonical tetrahedron representation
    topologies[mesh_frame] = {
        tuple(sorted(map(int, t))) for t in np.asarray(tetra)
    }

    mesh_frame += 1
    return vertices, tetra

im.create_interaction_mesh = diag_mesh

# Record which frame iterate() is processing
_orig_iterate = InteractionMeshRetargeter.iterate

def diag_iterate(self, *args, **kwargs):
    f = kwargs.get("frame_idx", 0)
    self._diag_frame = f

    if 54 <= f <= 61:
        feet[f] = dict(kwargs["foot_sticking"])

    return _orig_iterate(self, *args, **kwargs)

InteractionMeshRetargeter.iterate = diag_iterate

# Record active collision constraints during SQP iterations
_orig_collision = InteractionMeshRetargeter._update_jacobians_and_phis_from_q

def diag_collision(self, q):
    Js, phis = _orig_collision(self, q)

    f = getattr(self, "_diag_frame", -1)
    if 54 <= f <= 61:
        collision_counts.setdefault(f, []).append(len(phis))

    return Js, phis

InteractionMeshRetargeter._update_jacobians_and_phis_from_q = diag_collision

# ---------------------------------------------------------
# Run exactly the same clean adapter
# ---------------------------------------------------------
runpy.run_path(
    "scripts/run_clean_hiphi_retarget.py",
    run_name="__main__",
)

print("\n========== DIAGNOSTICS ==========")

print("\nInteraction mesh changes:")
for f in range(54, 61):
    A = topologies[f]
    B = topologies[f + 1]
    removed = len(A - B)
    added = len(B - A)

    print(
        f"{f}->{f+1}: "
        f"tetra {len(A)}->{len(B)}, "
        f"removed={removed}, added={added}"
    )

print("\nFoot sticking:")
for f in range(54, 62):
    print(f, feet.get(f))

print("\nActive collision constraints:")
for f in range(54, 62):
    vals = collision_counts.get(f, [])
    if vals:
        print(
            f,
            "min=", min(vals),
            "max=", max(vals),
            "sequence=", vals,
        )
    else:
        print(f, "none")
