"""Persistent-window HiPHI/G1 playlist replay implementation."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class MotionSpec:
    retargeted_motion: Path
    source_motion: Path
    metadata_file: Path
    source_slots: tuple[int, ...] = field(default_factory=tuple)
    robot_slots: tuple[int, ...] = field(default_factory=tuple)

    @property
    def motion_id(self) -> str:
        return self.retargeted_motion.parents[1].name


@dataclass
class SceneSlot:
    role: str
    mesh_key: str
    mesh: Any


@dataclass
class MotionBundle:
    source_positions: np.ndarray
    source_names: tuple[str, ...]
    source_parents: np.ndarray
    source_fps: float
    source_tracks: list[Any]
    robot_trajectory: np.ndarray
    robot_fps: float
    terrain: Any
    robot_tracks: list[Any]
    bone_pairs: tuple[tuple[int, int], ...]


@dataclass
class PlaybackState:
    current_index: int
    frame_idx: int = 0
    paused: bool = False
    step: int = 0
    reset: bool = False
    switch: int = 0

    def activate(self, playlist_index: int) -> None:
        """Select a motion and make its next rendered frame frame zero."""
        self.current_index = playlist_index
        self.frame_idx = 0
        self.step = 0
        self.reset = False
        self.switch = 0


class BundleCache:
    """Small lazy cache; loading a whole playlist can consume several GiB."""

    def __init__(self, loader, max_entries: int = 3):
        self._loader = loader
        self._max_entries = max(1, int(max_entries))
        self._entries: OrderedDict[int, MotionBundle] = OrderedDict()

    def put(self, index: int, bundle: MotionBundle) -> None:
        self._entries[index] = bundle
        self._entries.move_to_end(index)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def get(self, index: int) -> MotionBundle:
        bundle = self._entries.get(index)
        if bundle is None:
            bundle = self._loader(index)
            self.put(index, bundle)
        else:
            self._entries.move_to_end(index)
        return bundle


def parse_args(implementation) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--motion", required=True, type=Path, help="Exported HiPHI *_retargeted.npz"
    )
    parser.add_argument(
        "--source-motion",
        type=Path,
        help="Original motion_actor.bvh (inferred from --motion by default)",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Original sequence metadata.json (default: next to the BVH)",
    )
    parser.add_argument(
        "--hiphi-root",
        type=Path,
        default=implementation.DEFAULT_HIPHI_ROOT,
        help=f"HiPHI dataset root (default: {implementation.DEFAULT_HIPHI_ROOT})",
    )
    parser.add_argument(
        "--robot-model",
        type=Path,
        default=implementation.DEFAULT_G1_MODEL,
        help=f"G1 MJCF/URDF model (default: {implementation.DEFAULT_G1_MODEL})",
    )
    parser.add_argument(
        "--separation",
        type=float,
        default=2.2,
        help="Initial human/robot separation in meters (default: 2.2)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Viewer playback FPS override; source synchronization uses timestamps",
    )
    parser.add_argument(
        "--hide-thumb-ee",
        action="store_true",
        help="Hide the cyan/magenta G1 thumb EE proxy spheres",
    )
    parser.add_argument(
        "--thumb-ee-inset",
        type=float,
        default=implementation.DEFAULT_THUMB_EE_INSET,
        help="Thumb proxy inset toward the palm heel in meters (default: 0.0)",
    )
    return parser.parse_args()


def normalize_and_validate_args(args: argparse.Namespace) -> None:
    args.motion = args.motion.expanduser().resolve()
    args.hiphi_root = args.hiphi_root.expanduser().resolve()
    args.robot_model = args.robot_model.expanduser().resolve()
    if args.source_motion is not None:
        args.source_motion = args.source_motion.expanduser().resolve()
    if args.metadata is not None:
        args.metadata = args.metadata.expanduser().resolve()

    if args.separation <= 0:
        raise ValueError(f"--separation must be positive, got {args.separation}")
    if args.fps is not None and args.fps <= 0:
        raise ValueError(f"--fps must be positive, got {args.fps}")
    if args.thumb_ee_inset < 0:
        raise ValueError(
            f"--thumb-ee-inset must be nonnegative, got {args.thumb_ee_inset}"
        )
    for label, path in (
        ("Retargeted motion", args.motion),
        ("Robot model", args.robot_model),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")


def build_motion_specs(
    playlist: list[Path],
    initial_motion: Path,
    args: argparse.Namespace,
    implementation,
) -> list[MotionSpec]:
    specs = []
    for retargeted_motion in playlist:
        is_initial = retargeted_motion == initial_motion
        source_motion = (
            args.source_motion
            if is_initial and args.source_motion is not None
            else implementation._infer_source_motion(retargeted_motion, args.hiphi_root)
        )
        metadata_file = (
            args.metadata
            if is_initial and args.metadata is not None
            else source_motion.parent / "metadata.json"
        )
        for label, path in (
            ("Original motion", source_motion),
            ("Metadata", metadata_file),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{label} not found: {path}")
        specs.append(
            MotionSpec(
                retargeted_motion=retargeted_motion,
                source_motion=source_motion,
                metadata_file=metadata_file,
            )
        )
    return specs


def source_geom_capacities(specs: list[MotionSpec]) -> tuple[int, int]:
    """Return fixed user-scene capacities without loading BVH frame arrays."""
    joint_counts = []
    for spec in specs:
        joint_count = 0
        with spec.source_motion.open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                stripped = line.strip()
                if stripped == "MOTION":
                    break
                if stripped.startswith("ROOT ") or stripped.startswith("JOINT "):
                    joint_count += 1
        if joint_count <= 0:
            raise ValueError(f"No BVH joints found in {spec.source_motion}")
        joint_counts.append(joint_count)

    # A BVH hierarchy is a tree with one root, so it can draw at most N-1
    # parent-child connectors.  Individual bundles may omit zero-length bones.
    return max(joint_counts), max(count - 1 for count in joint_counts)


def mesh_digest(mesh: Any) -> str:
    vertices = np.ascontiguousarray(np.asarray(mesh.vertices, dtype="<f8"))
    faces = np.ascontiguousarray(np.asarray(mesh.faces, dtype="<i8"))
    digest = hashlib.sha256()
    digest.update(np.asarray(vertices.shape, dtype="<i8").tobytes())
    digest.update(vertices.tobytes())
    digest.update(np.asarray(faces.shape, dtype="<i8").tobytes())
    digest.update(faces.tobytes())
    return digest.hexdigest()


def source_meshes(spec: MotionSpec, hiphi_root: Path) -> list[Any]:
    from omniretargeting.data_sources.hiphi import _load_centered_object_mesh

    metadata = json.loads(spec.metadata_file.read_text(encoding="utf-8"))
    objects = metadata.get("objects", [])
    if len(objects) != 1:
        raise ValueError(
            f"HiPHI comparison requires exactly one object in "
            f"{spec.metadata_file}, got {len(objects)}"
        )
    mesh_path = Path(objects[0]["mesh_path"])
    if not mesh_path.is_absolute():
        mesh_path = hiphi_root / mesh_path
    if not mesh_path.is_file():
        raise FileNotFoundError(f"HiPHI object mesh not found: {mesh_path}")
    mesh, _ = _load_centered_object_mesh(mesh_path)
    return [mesh]


def robot_meshes(spec: MotionSpec) -> list[Any]:
    import trimesh

    object_dirs = sorted(
        path
        for path in spec.retargeted_motion.parent.glob("*_scaled_objects")
        if path.is_dir()
    )
    # Match load_replay_data: do not guess if adjacent object directories are
    # absent or ambiguous.
    if len(object_dirs) != 1:
        return []

    meshes = []
    for poses_path in sorted(object_dirs[0].glob("*_poses.json")):
        mesh_path = object_dirs[0] / (
            poses_path.name.removesuffix("_poses.json") + ".obj"
        )
        if not mesh_path.is_file():
            raise FileNotFoundError(
                f"Object mesh not found for {poses_path}: {mesh_path}"
            )
        meshes.append(trimesh.load(mesh_path, force="mesh"))
    return meshes


def build_scene_slots(
    specs: list[MotionSpec], hiphi_root: Path, implementation
) -> tuple[list[SceneSlot], list[Any]]:
    """Pre-register mesh variants while keeping the large motions lazy."""
    slots: list[SceneSlot] = []
    slot_by_key: dict[tuple[str, str], int] = {}

    def register(role: str, mesh: Any) -> int:
        mesh_key = mesh_digest(mesh)
        key = (role, mesh_key)
        slot_index = slot_by_key.get(key)
        if slot_index is None:
            slot_index = len(slots)
            slot_by_key[key] = slot_index
            slots.append(SceneSlot(role=role, mesh_key=mesh_key, mesh=mesh.copy()))
        return slot_index

    for spec in specs:
        spec.source_slots = tuple(
            register("source", mesh) for mesh in source_meshes(spec, hiphi_root)
        )
        spec.robot_slots = tuple(
            register("robot", mesh) for mesh in robot_meshes(spec)
        )

    identity = np.eye(4, dtype=float)
    scene_tracks = [
        implementation.ObjectTrack(
            name=f"persistent_{slot.role}_slot_{slot_index}",
            mesh=slot.mesh,
            transforms=[identity.copy()],
        )
        for slot_index, slot in enumerate(slots)
    ]
    return slots, scene_tracks


def load_bundle(
    spec: MotionSpec, args: argparse.Namespace, implementation
) -> MotionBundle:
    source = implementation.HiphiDataSource(
        motion_file=spec.source_motion,
        metadata_file=spec.metadata_file,
        data_root=args.hiphi_root,
    ).load()
    source_positions = np.asarray(source.positions, dtype=float).copy()
    source_names = tuple(source.target_names or [])
    source_parents = np.asarray(source.metadata["bone_parents"], dtype=int)
    source_fps = float(source.framerate or 30.0)
    source_tracks = implementation.build_object_tracks(source, 1.0, False) or []

    robot_trajectory, robot_fps, terrain, robot_tracks = (
        implementation.load_replay_data(spec.retargeted_motion)
    )
    robot_trajectory = np.asarray(robot_trajectory, dtype=float).copy()
    robot_fps = float(robot_fps)

    source_offset, robot_offset = implementation._comparison_offsets(
        source_positions[:, 0], robot_trajectory[:, :3], args.separation
    )
    source_positions += source_offset
    robot_trajectory[:, :3] += robot_offset
    source_tracks = implementation._offset_tracks(
        source_tracks, source_offset, "source"
    )
    robot_tracks = implementation._offset_tracks(
        robot_tracks, robot_offset, "retarget"
    )

    bone_pairs = tuple(
        (int(parent), child)
        for child, parent in enumerate(source_parents)
        if parent >= 0
        and np.linalg.norm(
            source_positions[0, child] - source_positions[0, parent]
        )
        > 1e-6
    )
    return MotionBundle(
        source_positions=source_positions,
        source_names=source_names,
        source_parents=source_parents,
        source_fps=source_fps,
        source_tracks=source_tracks,
        robot_trajectory=robot_trajectory,
        robot_fps=robot_fps,
        terrain=terrain,
        robot_tracks=robot_tracks,
        bone_pairs=bone_pairs,
    )


def terrain_signature(terrain: Any) -> tuple | None:
    if terrain is None:
        return None
    vertices = np.asarray(terrain.vertices, dtype=float)
    if len(vertices) and float(np.ptp(vertices[:, 2])) < 1e-6:
        return ("flat", round(float(np.mean(vertices[:, 2])), 7))
    return ("mesh", mesh_digest(terrain))


def validate_bundle(
    bundle: MotionBundle,
    reference: MotionBundle,
    spec: MotionSpec,
    slots: list[SceneSlot],
    source_joint_capacity: int,
    source_bone_capacity: int,
) -> None:
    if len(bundle.source_names) != bundle.source_positions.shape[1]:
        raise ValueError(
            f"{spec.motion_id}: source names/position width mismatch"
        )
    if len(bundle.source_names) > source_joint_capacity:
        raise ValueError(
            f"{spec.motion_id}: {len(bundle.source_names)} source joints exceed "
            f"the fixed viewer capacity {source_joint_capacity}"
        )
    if len(bundle.bone_pairs) > source_bone_capacity:
        raise ValueError(
            f"{spec.motion_id}: {len(bundle.bone_pairs)} source bones exceed "
            f"the fixed viewer capacity {source_bone_capacity}"
        )
    if any(
        parent < 0
        or child < 0
        or parent >= len(bundle.source_names)
        or child >= len(bundle.source_names)
        for parent, child in bundle.bone_pairs
    ):
        raise ValueError(f"{spec.motion_id}: source bone indices are out of range")
    if bundle.robot_trajectory.shape[1] != reference.robot_trajectory.shape[1]:
        raise ValueError(
            f"{spec.motion_id}: robot qpos width differs from the open scene"
        )
    if terrain_signature(bundle.terrain) != terrain_signature(reference.terrain):
        raise ValueError(
            f"{spec.motion_id}: terrain is incompatible with the open scene"
        )
    if len(bundle.source_tracks) != len(spec.source_slots):
        raise ValueError(
            f"{spec.motion_id}: source object-track count changed after scene prescan"
        )
    if len(bundle.robot_tracks) != len(spec.robot_slots):
        raise ValueError(
            f"{spec.motion_id}: robot object-track count changed after scene prescan"
        )
    for role, tracks, slot_indices in (
        ("source", bundle.source_tracks, spec.source_slots),
        ("robot", bundle.robot_tracks, spec.robot_slots),
    ):
        for track, slot_index in zip(tracks, slot_indices):
            slot = slots[slot_index]
            if slot.role != role or slot.mesh_key != mesh_digest(track.mesh):
                raise ValueError(
                    f"{spec.motion_id}: {role} object mesh does not match its "
                    "precompiled scene slot"
                )


def print_motion_info(
    spec: MotionSpec, bundle: MotionBundle, position: int, total: int
) -> None:
    source_duration = (len(bundle.source_positions) - 1) / bundle.source_fps
    robot_duration = (len(bundle.robot_trajectory) - 1) / bundle.robot_fps
    duration_error = abs(source_duration - robot_duration)
    frame_tolerance = max(1.0 / bundle.source_fps, 1.0 / bundle.robot_fps) * 1.5
    if duration_error > frame_tolerance:
        print(
            "Warning: source and retarget durations differ by "
            f"{duration_error:.3f}s; replay follows the retarget timeline."
        )
    print(f"Motion [{position}/{total}]: {spec.motion_id}")
    print(
        f"Original: {len(bundle.source_positions)} frames @ "
        f"{bundle.source_fps:.3f} fps | Retarget: "
        f"{len(bundle.robot_trajectory)} frames @ {bundle.robot_fps:.3f} fps"
    )


def body_geom_ids(model, body_name: str, mujoco) -> list[int]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Persistent object body not found: {body_name}")
    first = int(model.body_geomadr[body_id])
    count = int(model.body_geomnum[body_id])
    return list(range(first, first + count))


def park_object_body(model, data, body_name: str, slot_index: int, mujoco) -> None:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    joint_id = int(model.body_jntadr[body_id])
    qpos_adr = int(model.jnt_qposadr[joint_id])
    data.qpos[qpos_adr : qpos_adr + 7] = [
        0.0,
        0.0,
        -100.0 - slot_index,
        1.0,
        0.0,
        0.0,
        0.0,
    ]


def activate_object_slots(
    model,
    data,
    object_body_names: list[str],
    object_geom_ids: list[list[int]],
    spec: MotionSpec,
    mujoco,
) -> None:
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    for slot_index, body_name in enumerate(object_body_names):
        park_object_body(model, data, body_name, slot_index, mujoco)
        for geom_id in object_geom_ids[slot_index]:
            model.geom_rgba[geom_id, 3] = 0.0

    for role, slot_indices in (
        ("source", spec.source_slots),
        ("robot", spec.robot_slots),
    ):
        rgba = (
            np.array([0.95, 0.45, 0.12, 0.75])
            if role == "source"
            else np.array([0.58, 0.35, 0.12, 1.0])
        )
        for slot_index in slot_indices:
            for geom_id in object_geom_ids[slot_index]:
                model.geom_rgba[geom_id] = rgba


def set_active_object_poses(
    model,
    data,
    object_body_names: list[str],
    spec: MotionSpec,
    bundle: MotionBundle,
    source_frame_idx: int,
    robot_frame_idx: int,
    implementation,
    mujoco,
) -> None:
    for slot_index, track in zip(spec.source_slots, bundle.source_tracks):
        implementation._set_object_body_poses(
            model,
            data,
            [object_body_names[slot_index]],
            [track],
            source_frame_idx,
            mujoco,
        )
    for slot_index, track in zip(spec.robot_slots, bundle.robot_tracks):
        implementation._set_object_body_poses(
            model,
            data,
            [object_body_names[slot_index]],
            [track],
            robot_frame_idx,
            mujoco,
        )


def configure_source_geoms(
    scene,
    joint_base: int,
    bone_base: int,
    source_joint_capacity: int,
    source_bone_capacity: int,
    bundle: MotionBundle,
    implementation,
    mujoco,
) -> None:
    """Reconfigure fixed user-scene slots for the current BVH hierarchy."""
    hidden_rgba = np.zeros(4, dtype=float)
    parked = np.array([0.0, 0.0, -100.0], dtype=float)
    for joint_idx in range(source_joint_capacity):
        active = joint_idx < len(bundle.source_names)
        mujoco.mjv_initGeom(
            scene.geoms[joint_base + joint_idx],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([0.018, 0.0, 0.0]),
            pos=np.zeros(3) if active else parked,
            mat=implementation.IDENTITY_MAT,
            rgba=(
                implementation._bone_color(bundle.source_names[joint_idx])
                if active
                else hidden_rgba
            ),
        )

    for bone_idx in range(source_bone_capacity):
        active = bone_idx < len(bundle.bone_pairs)
        child = bundle.bone_pairs[bone_idx][1] if active else 0
        mujoco.mjv_initGeom(
            scene.geoms[bone_base + bone_idx],
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=np.array([0.012, 0.0, 0.0]),
            pos=np.zeros(3) if active else parked,
            mat=implementation.IDENTITY_MAT,
            rgba=(
                implementation._bone_color(bundle.source_names[child])
                if active
                else hidden_rgba
            ),
        )


def run_persistent_viewer(
    specs: list[MotionSpec],
    initial_index: int,
    slots: list[SceneSlot],
    scene_tracks: list[Any],
    source_joint_capacity: int,
    source_bone_capacity: int,
    args: argparse.Namespace,
    implementation,
) -> None:
    import mujoco
    import mujoco.viewer

    cache = BundleCache(
        lambda index: load_bundle(specs[index], args, implementation),
        max_entries=3,
    )
    reference = cache.get(initial_index)
    validate_bundle(
        reference,
        reference,
        specs[initial_index],
        slots,
        source_joint_capacity,
        source_bone_capacity,
    )

    print(f"Discovered {len(specs)} retargeted motion(s) in this batch.")
    print(
        f"Persistent scene contains {len(slots)} deduplicated object mesh slot(s); "
        "P/N keeps this MuJoCo window open."
    )
    print(
        f"Source skeleton capacity: {source_joint_capacity} joints, "
        f"{source_bone_capacity} bones."
    )
    print("Original human is on one side; retargeted G1 is on the other side.")
    controls = "Controls: Space pause/resume, [ and ] step, 0 reset"
    if len(specs) > 1:
        controls += ", P/N previous/next motion"
    print(controls)

    with implementation.temporary_visualization_scene(
        args.robot_model, reference.terrain, scene_tracks
    ) as composed:
        model = mujoco.MjModel.from_xml_path(composed.model_path)
        data = mujoco.MjData(model)
        implementation._configure_model_visuals(
            model, ambient=0.65, diffuse=0.7, specular=0.25
        )
        if len(composed.object_body_names) != len(slots):
            raise RuntimeError(
                "Persistent scene object-slot count changed during MJCF composition"
            )

        object_geom_ids = [
            body_geom_ids(model, body_name, mujoco)
            for body_name in composed.object_body_names
        ]
        for geom_ids in object_geom_ids:
            for geom_id in geom_ids:
                model.geom_contype[geom_id] = 0
                model.geom_conaffinity[geom_id] = 0

        thumb_ee_body_ids = []
        if not args.hide_thumb_ee:
            for link_name in implementation.THUMB_EE_LINKS:
                body_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, link_name
                )
                if body_id < 0:
                    raise ValueError(f"G1 thumb EE link not found: {link_name}")
                thumb_ee_body_ids.append(body_id)
            print(
                "G1 thumb EE markers: left=cyan, right=magenta "
                f"({', '.join(implementation.THUMB_EE_LINKS)}), "
                f"palm-heel inset={args.thumb_ee_inset:.3f}m"
            )

        playback = PlaybackState(current_index=initial_index)
        can_switch = len(specs) > 1

        def key_callback(keycode: int) -> None:
            if keycode == ord(" "):
                playback.paused = not playback.paused
            elif keycode == ord("["):
                playback.paused = True
                playback.step = -1
            elif keycode == ord("]"):
                playback.paused = True
                playback.step = 1
            elif keycode == ord("0"):
                playback.paused = True
                playback.reset = True
            elif can_switch and keycode == ord("P"):
                playback.switch = -1
            elif can_switch and keycode == ord("N"):
                playback.switch = 1

        activate_object_slots(
            model,
            data,
            composed.object_body_names,
            object_geom_ids,
            specs[initial_index],
            mujoco,
        )

        with mujoco.viewer.launch_passive(
            model, data, key_callback=key_callback
        ) as viewer:
            scene = implementation._viewer_scene(viewer)
            if scene is None:
                raise RuntimeError("MuJoCo viewer does not expose a user scene")
            implementation._configure_scene(scene, mujoco)
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = 0
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = 1

            required_geoms = (
                source_joint_capacity
                + source_bone_capacity
                + len(thumb_ee_body_ids)
            )
            if scene.ngeom + required_geoms > scene.maxgeom:
                raise RuntimeError(
                    f"Viewer needs {required_geoms} source-skeleton geoms, but only "
                    f"{scene.maxgeom - scene.ngeom} are available"
                )

            joint_base = scene.ngeom
            bone_base = joint_base + source_joint_capacity
            thumb_ee_base = bone_base + source_bone_capacity
            configure_source_geoms(
                scene,
                joint_base,
                bone_base,
                source_joint_capacity,
                source_bone_capacity,
                reference,
                implementation,
                mujoco,
            )
            for marker_idx, color in enumerate(
                implementation.THUMB_EE_COLORS[: len(thumb_ee_body_ids)]
            ):
                mujoco.mjv_initGeom(
                    scene.geoms[thumb_ee_base + marker_idx],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=np.array([0.03, 0.0, 0.0]),
                    pos=np.zeros(3),
                    mat=implementation.IDENTITY_MAT,
                    rgba=color,
                )
            scene.ngeom += required_geoms

            viewer.cam.distance = max(4.0, args.separation * 1.9)
            viewer.cam.azimuth = 120.0
            viewer.cam.elevation = -18.0
            bundle = reference
            print_motion_info(
                specs[initial_index], bundle, initial_index + 1, len(specs)
            )

            while viewer.is_running():
                start = time.perf_counter()

                switch = int(playback.switch)
                if switch:
                    playback.switch = 0
                    candidate_index = (playback.current_index + switch) % len(specs)
                    try:
                        candidate = cache.get(candidate_index)
                        validate_bundle(
                            candidate,
                            reference,
                            specs[candidate_index],
                            slots,
                            source_joint_capacity,
                            source_bone_capacity,
                        )
                    except Exception as exc:
                        print(
                            f"Cannot switch to {specs[candidate_index].motion_id}: "
                            f"{type(exc).__name__}: {exc}"
                        )
                    else:
                        bundle = candidate
                        playback.activate(candidate_index)
                        # Model RGBA and free-joint state are shared with the
                        # render thread; change the active slots atomically.
                        with viewer.lock():
                            activate_object_slots(
                                model,
                                data,
                                composed.object_body_names,
                                object_geom_ids,
                                specs[candidate_index],
                                mujoco,
                            )
                            configure_source_geoms(
                                scene,
                                joint_base,
                                bone_base,
                                source_joint_capacity,
                                source_bone_capacity,
                                candidate,
                                implementation,
                                mujoco,
                            )
                        print_motion_info(
                            specs[candidate_index],
                            bundle,
                            candidate_index + 1,
                            len(specs),
                        )
                        viewer.set_texts(
                            (
                                None,
                                None,
                                f"Motion {candidate_index + 1}/{len(specs)}",
                                specs[candidate_index].motion_id,
                            )
                        )

                if playback.reset:
                    playback.frame_idx = 0
                    playback.reset = False
                step = int(playback.step)
                if step:
                    playback.frame_idx = (
                        playback.frame_idx + step
                    ) % len(bundle.robot_trajectory)
                    playback.step = 0

                frame_idx = playback.frame_idx
                source_frame_idx = min(
                    int(round(frame_idx * bundle.source_fps / bundle.robot_fps)),
                    len(bundle.source_positions) - 1,
                )
                source_frame = bundle.source_positions[source_frame_idx]

                robot_qpos_dim = bundle.robot_trajectory.shape[1]
                data.qpos[:robot_qpos_dim] = bundle.robot_trajectory[frame_idx]
                set_active_object_poses(
                    model,
                    data,
                    composed.object_body_names,
                    specs[playback.current_index],
                    bundle,
                    source_frame_idx,
                    frame_idx,
                    implementation,
                    mujoco,
                )
                mujoco.mj_forward(model, data)

                for marker_idx, body_id in enumerate(thumb_ee_body_ids):
                    body_rotation = data.xmat[body_id].reshape(3, 3)
                    local_offset = np.array([-args.thumb_ee_inset, 0.0, 0.0])
                    scene.geoms[thumb_ee_base + marker_idx].pos = (
                        data.xpos[body_id] + body_rotation @ local_offset
                    )

                for joint_idx, point in enumerate(source_frame):
                    scene.geoms[joint_base + joint_idx].pos = point
                for bone_idx, (parent, child) in enumerate(bundle.bone_pairs):
                    mujoco.mjv_connector(
                        scene.geoms[bone_base + bone_idx],
                        mujoco.mjtGeom.mjGEOM_CAPSULE,
                        0.012,
                        source_frame[parent],
                        source_frame[child],
                    )

                viewer.cam.lookat[:] = 0.5 * (
                    source_frame[0] + bundle.robot_trajectory[frame_idx, :3]
                )
                if not playback.paused:
                    playback.frame_idx = (
                        playback.frame_idx + 1
                    ) % len(bundle.robot_trajectory)
                viewer.sync()

                display_fps = float(args.fps or bundle.robot_fps)
                remaining = 1.0 / display_fps - (time.perf_counter() - start)
                if remaining > 0:
                    time.sleep(remaining)


def main() -> None:
    import omniretargeting.hiphi_compare_replay as implementation

    args = parse_args(implementation)
    normalize_and_validate_args(args)
    playlist = implementation._discover_playlist(args.motion)
    initial_index = playlist.index(args.motion)
    specs = build_motion_specs(playlist, args.motion, args, implementation)
    print(f"Preparing persistent object slots for {len(specs)} motion(s)...")
    slots, scene_tracks = build_scene_slots(specs, args.hiphi_root, implementation)
    source_joint_capacity, source_bone_capacity = source_geom_capacities(specs)
    run_persistent_viewer(
        specs,
        initial_index,
        slots,
        scene_tracks,
        source_joint_capacity,
        source_bone_capacity,
        args,
        implementation,
    )


if __name__ == "__main__":
    main()
