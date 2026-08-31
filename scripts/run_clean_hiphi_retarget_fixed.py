import runpy
import mujoco

from holosoma_retargeting.src.interaction_mesh_retargeter import InteractionMeshRetargeter

_original = InteractionMeshRetargeter._build_transform_qdot_to_qvel_fast

def _fixed_build_transform(self, use_world_omega=True):
    T = _original(self, use_world_omega)

    # Restore the intended identity mapping for hinge/slide joints.
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

InteractionMeshRetargeter._build_transform_qdot_to_qvel_fast = _fixed_build_transform

runpy.run_path(
    "scripts/run_clean_hiphi_retarget.py",
    run_name="__main__",
)
