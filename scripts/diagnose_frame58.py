import runpy
import mujoco
import numpy as np

from holosoma_retargeting.src.interaction_mesh_retargeter import InteractionMeshRetargeter

# ---------------------------------------------------------
# MuJoCo enum compatibility fix
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
# Track current frame
# ---------------------------------------------------------
_orig_iterate = InteractionMeshRetargeter.iterate

def diag_iterate(self, *args, **kwargs):
    self._diag_frame = kwargs.get("frame_idx", -1)
    self._diag_iter = 0
    return _orig_iterate(self, *args, **kwargs)

InteractionMeshRetargeter.iterate = diag_iterate

# ---------------------------------------------------------
# Record collision pairs
# ---------------------------------------------------------
_orig_col = InteractionMeshRetargeter._update_jacobians_and_phis_from_q

def diag_col(self, q):
    Js, phis = _orig_col(self, q)

    if getattr(self, "_diag_frame", -1) == 58:
        pairs = []
        for key in phis:
            if len(key) == 2:
                g1, g2 = key
                n1 = mujoco.mj_id2name(
                    self.robot_model, mujoco.mjtObj.mjOBJ_GEOM, g1
                )
                n2 = mujoco.mj_id2name(
                    self.robot_model, mujoco.mjtObj.mjOBJ_GEOM, g2
                )
                pairs.append((n1, n2))

        print(
            f"[frame58 iter {getattr(self,'_diag_iter',-1)}] "
            f"collisions={len(pairs)}"
        )

        if hasattr(self, "_last_pairs"):
            new = set(pairs) - self._last_pairs
            gone = self._last_pairs - set(pairs)

            if new:
                print("  NEW:", new)
            if gone:
                print("  GONE:", gone)

        self._last_pairs = set(pairs)

    return Js, phis

InteractionMeshRetargeter._update_jacobians_and_phis_from_q = diag_col

# ---------------------------------------------------------
# Record each SQP update
# ---------------------------------------------------------
_orig_solve = InteractionMeshRetargeter.solve_single_iteration

def diag_solve(self, *args, **kwargs):
    q_before = kwargs["q_a_n_last"].copy()

    q_after, cost = _orig_solve(self, *args, **kwargs)

    if getattr(self, "_diag_frame", -1) == 58:
        # right elbow qpos address = 32
        old = np.degrees(q_before[32])
        new = np.degrees(q_after[32])

        print(
            f"  SQP {self._diag_iter}: "
            f"right_elbow {old:.2f} -> {new:.2f} "
            f"(delta={new-old:+.2f} deg), cost={cost:.4f}"
        )

        self._diag_iter += 1

    return q_after, cost

InteractionMeshRetargeter.solve_single_iteration = diag_solve

runpy.run_path(
    "scripts/run_clean_hiphi_retarget.py",
    run_name="__main__",
)
