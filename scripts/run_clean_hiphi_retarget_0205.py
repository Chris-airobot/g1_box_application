from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from holosoma_retargeting.config_types.data_type import JOINTS_MAPPINGS
from holosoma_retargeting.config_types.robot import RobotConfig
from holosoma_retargeting.config_types.retargeter import RetargeterConfig
from holosoma_retargeting.config_types.task import TaskConfig
from holosoma_retargeting.examples.robot_retarget import (
    build_retargeter_kwargs_from_config,
    initialize_robot_pose,
)
from holosoma_retargeting.src.interaction_mesh_retargeter import InteractionMeshRetargeter
from holosoma_retargeting.src.utils import (
    load_object_data,
    preprocess_motion_data,
    extract_foot_sticking_sequence_velocity,
)

src = ROOT / "data/hiphi_test/Bringing-carry_0205_pickup_smoke.npz"
assets = ROOT / "clean_baseline_assets"
outdir = ROOT / "outputs/clean_omniretarget"
outdir.mkdir(parents=True, exist_ok=True)

d = np.load(src)
human = d["global_joint_positions"].copy()
obj = d["object_poses"].copy()
joint_names = [str(x) for x in d["joint_names"]]
height = float(np.asarray(d["height"]).reshape(-1)[0])
fps = float(np.asarray(d["fps"]).reshape(-1)[0])

# HiPHI ~90 Hz -> upstream 30 Hz.
stride = max(1, round(fps / 30.0))
human = human[::stride]
obj = obj[::stride]

# Use an existing upstream G1 mapping, not a modified solver mapping.
mapping = dict(JOINTS_MAPPINGS[("lafan", "g1")])
missing = [x for x in mapping if x not in joint_names]
assert not missing, f"Missing mapped joints: {missing}"

robot = RobotConfig(
    robot_type="g1",
    robot_urdf_file=str(assets / "g1/g1_29dof.urdf"),
)

constants = SimpleNamespace(
    ROBOT_DOF=robot.ROBOT_DOF,
    ROBOT_HEIGHT=robot.ROBOT_HEIGHT,
    ROBOT_URDF_FILE=robot.ROBOT_URDF_FILE,
    FOOT_STICKING_LINKS=robot.FOOT_STICKING_LINKS,
    MANUAL_LB=robot.MANUAL_LB,
    MANUAL_UB=robot.MANUAL_UB,
    MANUAL_COST=robot.MANUAL_COST,
    NOMINAL_TRACKING_INDICES=robot.NOMINAL_TRACKING_INDICES,
    DEMO_JOINTS=joint_names,
    JOINTS_MAPPING=mapping,
    OBJECT_NAME="largebox",
)

scale = robot.ROBOT_HEIGHT / height

object_local_pts, object_local_pts_demo = load_object_data(
    str(assets / "largebox/largebox.obj"),
    smpl_scale=scale,
    sample_count=100,
)

cfg = RetargeterConfig()
kwargs = build_retargeter_kwargs_from_config(
    cfg,
    constants,
    str(assets / "largebox/largebox.urdf"),
    "object_interaction",
)
retargeter = InteractionMeshRetargeter(**kwargs)

toe_names = ["LeftToeBase", "RightToeBase"]

human, obj, _ = preprocess_motion_data(
    human,
    retargeter,
    toe_names,
    scale=scale,
    object_poses=obj,
)

q_init, q_nominal, obj_aug, human, obj = initialize_robot_pose(
    "object_interaction",
    "lafan",
    human,
    obj,
    constants,
    retargeter,
    TaskConfig(object_name="largebox"),
    False,
    outdir,
    "0205",
)

feet = extract_foot_sticking_sequence_velocity(
    human, retargeter.demo_joints, toe_names
)
feet[0][toe_names[0]] = False
feet[0][toe_names[1]] = False

dest = outdir / "0205_clean_original.npz"

print("source fps:", fps)
print("stride:", stride)
print("frames:", len(human))
print("scale:", scale)
print("output:", dest)

retargeter.retarget_motion(
    human_joint_motions=human,
    object_poses=obj,
    object_poses_augmented=obj_aug,
    object_points_local_demo=object_local_pts_demo,
    object_points_local=object_local_pts,
    foot_sticking_sequences=feet,
    q_a_init=q_init,
    q_nominal_list=q_nominal,
    original=True,
    dest_res_path=str(dest),
)
