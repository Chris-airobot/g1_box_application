#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


HIPHI = Path("/fred/oz430/tliu/data/HiPHI")
SELECTION = HIPHI / "validation_selection.csv"

ARCHIVE_DIR = HIPHI / "data"
MOTION_TO_PART = ARCHIVE_DIR / "motion_to_part.csv"

WORK = HIPHI / "validation_extract"
OUT = HIPHI / "validation_smokes"

PRE_SEC = 2.0
POST_SEC = 3.0

# Right-handed HiPHI Y-up -> right-handed robotics Z-up:
#
# X' = X
# Y' = -Z
# Z' = Y
C = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=float,
)


@dataclass
class Node:
    name: str
    parent: int | None
    offset: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )
    channels: list[str] = field(
        default_factory=list
    )
    channel_index: int | None = None
    is_end: bool = False


def rot_matrix(axis: str, degrees: float) -> np.ndarray:
    a = math.radians(degrees)
    c = math.cos(a)
    s = math.sin(a)

    if axis == "X":
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c],
        ], dtype=float)

    if axis == "Y":
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c],
        ], dtype=float)

    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1],
    ], dtype=float)


def parse_bvh(path: Path):
    lines = path.read_text(
        errors="ignore"
    ).splitlines()

    motion_idx = lines.index("MOTION")

    nodes: list[Node] = []
    stack: list[int] = []
    pending: int | None = None
    cursor = 0

    for line in lines[:motion_idx]:
        s = line.strip()

        if not s:
            continue

        parts = s.split()

        if parts[0] in {"ROOT", "JOINT"}:
            nodes.append(
                Node(
                    name=parts[1],
                    parent=stack[-1] if stack else None,
                )
            )
            pending = len(nodes) - 1

        elif s == "End Site":
            if not stack:
                continue

            parent = stack[-1]

            nodes.append(
                Node(
                    name=f"{nodes[parent].name}_End",
                    parent=parent,
                    is_end=True,
                )
            )
            pending = len(nodes) - 1

        elif s == "{":
            if pending is not None:
                stack.append(pending)
                pending = None

        elif s == "}":
            if stack:
                stack.pop()

        elif parts[0] == "OFFSET" and stack:
            nodes[stack[-1]].offset = np.array(
                [float(x) for x in parts[1:4]],
                dtype=float,
            )

        elif parts[0] == "CHANNELS" and stack:
            n = int(parts[1])

            nodes[stack[-1]].channels = parts[2:2+n]
            nodes[stack[-1]].channel_index = cursor

            cursor += n

    frames = int(
        lines[motion_idx + 1].split(":", 1)[1]
    )

    frame_time = float(
        lines[motion_idx + 2].split(":", 1)[1]
    )

    motion_lines = [
        x
        for x in lines[motion_idx + 3:]
        if x.strip()
    ]

    if len(motion_lines) < frames:
        raise RuntimeError(
            f"BVH says {frames} frames but "
            f"contains only {len(motion_lines)}"
        )

    articulated = [
        i
        for i, n in enumerate(nodes)
        if not n.is_end
    ]

    names = [
        nodes[i].name
        for i in articulated
    ]

    # Critical sanity check: must match the existing
    # HiPHI -> HoloSoma representation we validated.
    expected = {
        "LeftHandMiddle3": 21,
        "RightHandMiddle3": 40,
        "LeftToeBase": 50,
        "RightToeBase": 54,
    }

    for name, idx in expected.items():
        actual = names.index(name)

        if actual != idx:
            raise RuntimeError(
                f"Unexpected joint ordering: "
                f"{name}={actual}, expected={idx}"
            )

    if len(names) != 55:
        raise RuntimeError(
            f"Expected 55 articulated joints, got {len(names)}"
        )

    return (
        nodes,
        articulated,
        names,
        frames,
        frame_time,
        motion_lines,
    )


def fk(nodes: list[Node], values: np.ndarray):
    pos = []
    rot = []

    I = np.eye(3)

    for node in nodes:
        local_t = node.offset.copy()
        local_r = I.copy()

        if (
            node.channels
            and node.channel_index is not None
        ):
            vals = values[
                node.channel_index:
                node.channel_index + len(node.channels)
            ]

            channel_t = np.zeros(3)
            has_position = False

            for channel, value in zip(
                node.channels,
                vals,
            ):
                if channel.endswith("position"):
                    channel_t[
                        "XYZ".index(channel[0])
                    ] = value
                    has_position = True

                elif channel.endswith("rotation"):
                    local_r = (
                        local_r
                        @ rot_matrix(
                            channel[0],
                            value,
                        )
                    )

            # IMPORTANT HiPHI behavior:
            # translation channels REPLACE OFFSET.
            # Adding both double-counts limb lengths.
            if has_position:
                local_t = channel_t

        if node.parent is None:
            global_t = local_t
            global_r = local_r

        else:
            p = pos[node.parent]
            R = rot[node.parent]

            global_t = p + R @ local_t
            global_r = R @ local_r

        pos.append(global_t)
        rot.append(global_r)

    return np.asarray(pos)


def mesh_bounds_cm(path: Path):
    verts = []

    with path.open(
        "r",
        errors="ignore",
    ) as f:
        for line in f:
            if not line.startswith("v "):
                continue

            p = line.split()

            if len(p) >= 4:
                verts.append(
                    [
                        float(p[1]),
                        float(p[2]),
                        float(p[3]),
                    ]
                )

    if not verts:
        raise RuntimeError(
            f"No vertices in {path}"
        )

    V = np.asarray(verts)

    lo = V.min(axis=0)
    hi = V.max(axis=0)

    # HiPHI OBJ units = centimeters.
    center_m = (lo + hi) * 0.5 / 100.0
    size_m = (hi - lo) / 100.0

    return center_m, size_m


def recursive_dicts(obj):
    if isinstance(obj, dict):
        yield obj

        for value in obj.values():
            yield from recursive_dicts(value)

    elif isinstance(obj, list):
        for value in obj:
            yield from recursive_dicts(value)


def get_actor_height(meta):
    actor = meta.get("actor_metadata", {})

    if "height_cm" in actor:
        return float(actor["height_cm"]) / 100.0

    for d in recursive_dicts(meta):
        if "height_cm" in d:
            return float(d["height_cm"]) / 100.0

    raise RuntimeError(
        "Could not find actor height_cm in metadata.json"
    )


def get_box_entry(meta):
    entries = []

    for d in recursive_dicts(meta):
        if (
            "trajectory_path" in d
            and "mesh_path" in d
        ):
            entries.append(d)

    if not entries:
        raise RuntimeError(
            "No HOI object entry found in metadata.json"
        )

    box = [
        d for d in entries
        if "box" in (
            str(d.get("object_id", ""))
            + " "
            + str(d.get("mesh_id", ""))
        ).lower()
    ]

    if box:
        return box[0]

    if len(entries) == 1:
        return entries[0]

    raise RuntimeError(
        f"Multiple objects found and none identified as box: "
        f"{entries}"
    )


def resolve_mesh_path(raw: str):
    p = Path(raw)

    if p.is_absolute():
        return p

    if p.parts and p.parts[0] == "HiPHI":
        return HIPHI.parent / p

    return HIPHI / p


def resolve_trajectory_path(
    package: Path,
    raw: str,
):
    p = Path(raw)

    if p.is_absolute():
        return p

    candidate = package / p

    if candidate.exists():
        return candidate

    # fallback
    matches = list(
        package.glob("object_tracks/*.csv")
    )

    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(
        f"Could not resolve trajectory {raw}"
    )


def load_object_track(
    csv_path: Path,
    mesh_center_src_m: np.ndarray,
):
    rows = []

    with csv_path.open(newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    poses = []

    for row in rows:
        p_src = np.array(
            [
                float(row["px"]),
                float(row["py"]),
                float(row["pz"]),
            ]
        )

        q_xyzw = np.array(
            [
                float(row["qx"]),
                float(row["qy"]),
                float(row["qz"]),
                float(row["qw"]),
            ]
        )

        R_src = Rotation.from_quat(
            q_xyzw
        ).as_matrix()

        # Convert original mesh-local origin
        # to CENTERED box origin.
        p_center_src = (
            p_src
            + R_src @ mesh_center_src_m
        )

        # Basis transformation.
        R_new = (
            C
            @ R_src
            @ C.T
        )

        p_new = C @ p_center_src

        q_new = Rotation.from_matrix(
            R_new
        ).as_quat()  # xyzw

        poses.append(
            [
                q_new[3],
                q_new[0],
                q_new[1],
                q_new[2],
                p_new[0],
                p_new[1],
                p_new[2],
            ]
        )

    return np.asarray(
        poses,
        dtype=np.float32,
    )


def extract_group(
    archive: Path,
    motions: list[str],
):
    base = WORK / "HiPHI/data/Bringing/carry"

    motions = [
        m for m in motions
        if not (base / m / "motion_actor.bvh").exists()
    ]

    if not motions:
        print(f"Skipping {archive.name}: already extracted")
        return

    patterns = [
        f"HiPHI/data/Bringing/carry/{m}/*"
        for m in motions
    ]

    cmd = [
        "tar",
        "--zstd",
        "-xf",
        str(archive),
        "-C",
        str(WORK),
        "--wildcards",
        *patterns,
    ]

    print()
    print(
        f"Extracting {archive.name}: "
        f"{', '.join(motions)}"
    )

    try:
        subprocess.run(
            cmd,
            check=True,
        )

    except subprocess.CalledProcessError:
        # Fallback for tar builds without --zstd.
        cmd = [
            "tar",
            "--use-compress-program=unzstd",
            "-xf",
            str(archive),
            "-C",
            str(WORK),
            "--wildcards",
            *patterns,
        ]

        subprocess.run(
            cmd,
            check=True,
        )


def convert_motion(motion: str):
    package = (
        WORK
        / "HiPHI/data/Bringing/carry"
        / motion
    )

    bvh = package / "motion_actor.bvh"
    meta_path = package / "metadata.json"

    if not bvh.exists():
        raise FileNotFoundError(bvh)

    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    meta = json.loads(
        meta_path.read_text()
    )

    actor_height = get_actor_height(meta)
    obj_entry = get_box_entry(meta)

    mesh_path = resolve_mesh_path(
        obj_entry["mesh_path"]
    )

    traj_path = resolve_trajectory_path(
        package,
        obj_entry["trajectory_path"],
    )

    mesh_center_src, box_size_src = (
        mesh_bounds_cm(mesh_path)
    )

    # Size also has to follow the coordinate
    # basis transformation.
    box_size_target = (
        np.abs(C) @ box_size_src
    )

    object_poses = load_object_track(
        traj_path,
        mesh_center_src,
    )

    (
        nodes,
        articulated,
        joint_names,
        num_frames,
        frame_time,
        motion_lines,
    ) = parse_bvh(bvh)

    if len(object_poses) != num_frames:
        raise RuntimeError(
            f"{motion}: BVH frames={num_frames}, "
            f"object frames={len(object_poses)}"
        )

    fps = 1.0 / frame_time

    # ---------------------------------------------------------
    # Robust pickup/contact detection.
    #
    # Primary signal:
    #   both HiPHI fingertips enter the box-contact region.
    #
    # Fallbacks:
    #   object moves >2 cm in 3-D,
    #   then object rises >2 cm.
    #
    # This avoids assuming every carry begins with a clean
    # vertical lift relative to frame 0.
    # ---------------------------------------------------------

    left_idx = joint_names.index("LeftHandMiddle3")
    right_idx = joint_names.index("RightHandMiddle3")

    half = np.asarray(box_size_target) / 2.0

    def point_box_distance(point_world, pose):
        qw, qx, qy, qz = pose[:4]
        center = pose[4:7]

        R = Rotation.from_quat(
            [qx, qy, qz, qw]
        ).as_matrix()

        p_local = R.T @ (
            point_world - center
        )

        outside = np.maximum(
            np.abs(p_local) - half,
            0.0,
        )

        return float(
            np.linalg.norm(outside)
        )

    contact_mask = []

    for frame_idx in range(num_frames):
        values = np.fromstring(
            motion_lines[frame_idx],
            sep=" ",
            dtype=float,
        )

        xyz_cm_all = fk(
            nodes,
            values,
        )

        xyz_m = (
            xyz_cm_all[articulated]
            / 100.0
        )

        xyz_target = (
            C @ xyz_m.T
        ).T

        dl = point_box_distance(
            xyz_target[left_idx],
            object_poses[frame_idx],
        )

        dr = point_box_distance(
            xyz_target[right_idx],
            object_poses[frame_idx],
        )

        contact_mask.append(
            dl < 0.02 and dr < 0.02
        )

    # Require 5 consecutive contact frames to reject
    # isolated/noisy threshold crossings.
    contact_pickup = None
    sustain = 5

    for i in range(
        len(contact_mask) - sustain + 1
    ):
        if all(
            contact_mask[i:i+sustain]
        ):
            contact_pickup = i
            break

    pos = object_poses[:, 4:7]

    displacement = np.linalg.norm(
        pos - pos[0],
        axis=1,
    )

    move_ids = np.where(
        displacement > 0.02
    )[0]

    z = object_poses[:, 6]

    vertical_ids = np.where(
        z > z[0] + 0.02
    )[0]

    if contact_pickup is not None:
        pickup = int(contact_pickup)
        pickup_detector = "source_hand_contact"

    elif len(move_ids):
        pickup = int(move_ids[0])
        pickup_detector = "object_3d_motion"

    elif len(vertical_ids):
        pickup = int(vertical_ids[0])
        pickup_detector = "object_vertical_motion"

    else:
        raise RuntimeError(
            f"{motion}: no pickup/contact event detected"
        )

    pre = int(round(PRE_SEC * fps))
    post = int(round(POST_SEC * fps))

    start = max(
        0,
        pickup - pre,
    )

    end = min(
        num_frames,
        pickup + post + 1,
    )

    human = []

    for frame_idx in range(start, end):
        values = np.fromstring(
            motion_lines[frame_idx],
            sep=" ",
            dtype=float,
        )

        xyz_cm_all = fk(
            nodes,
            values,
        )

        xyz_cm = xyz_cm_all[
            articulated
        ]

        # cm -> m and Y-up -> Z-up
        xyz_m = (
            xyz_cm / 100.0
        )

        xyz_target = (
            C @ xyz_m.T
        ).T

        human.append(
            xyz_target
        )

    human = np.asarray(
        human,
        dtype=np.float32,
    )

    obj_clip = object_poses[
        start:end
    ]

    local_pickup = pickup - start

    task_name = (
        f"{motion}_pickup_smoke"
    )

    out_path = (
        OUT / f"{task_name}.npz"
    )

    np.savez(
        out_path,
        global_joint_positions=human,
        object_poses=obj_clip,
        height=np.float32(actor_height),
        box_size=np.asarray(
            box_size_target,
            dtype=np.float32,
        ),
        joint_names=np.asarray(
            joint_names,
        ),
        fps=np.float32(fps),
        source_start_frame=np.int32(start),
        source_pickup_frame=np.int32(pickup),
        local_pickup_frame=np.int32(local_pickup),
    )

    print()
    print(f"[OK] {motion}")
    print(
        f"  actor height : "
        f"{actor_height:.3f} m"
    )
    print(
        f"  box size     : "
        f"{box_size_target}"
    )
    print(
        f"  FPS          : "
        f"{fps:.4f}"
    )
    print(
        f"  pickup       : "
        f"source {pickup}, "
        f"local {local_pickup}"
    )
    print(
        f"  detector     : "
        f"{pickup_detector}"
    )
    print(
        f"  frames       : "
        f"{len(human)}"
    )
    print(
        f"  output       : "
        f"{out_path}"
    )


def main():
    WORK.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    with SELECTION.open(
        newline="",
        encoding="utf-8",
    ) as f:
        selected = [
            r["motion_id"].strip()
            for r in csv.DictReader(f)
        ]

    with MOTION_TO_PART.open(
        newline="",
        encoding="utf-8",
    ) as f:
        rows = list(
            csv.DictReader(f)
        )

    archive_for = {
        r["motion_id"]:
        r["archive_name"]
        for r in rows
    }

    missing = [
        m for m in selected
        if m not in archive_for
    ]

    if missing:
        raise RuntimeError(
            f"Missing archive mapping: {missing}"
        )

    groups = {}

    for motion in selected:
        archive = archive_for[motion]

        groups.setdefault(
            archive,
            [],
        ).append(motion)

    print(
        f"Preparing {len(selected)} motions "
        f"from {len(groups)} archives"
    )

    # Extract each archive only ONCE.
    for archive_name, motions in groups.items():
        archive = (
            ARCHIVE_DIR / archive_name
        )

        if not archive.exists():
            raise FileNotFoundError(
                archive
            )

        extract_group(
            archive,
            motions,
        )

    print()
    print("=" * 80)
    print("CONVERTING")
    print("=" * 80)

    for motion in selected:
        try:
            convert_motion(motion)
        except Exception as e:
            print(f"[FAIL] {motion}: {e}")
            continue

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)

    for p in sorted(
        OUT.glob("*_pickup_smoke.npz")
    ):
        print(p)


if __name__ == "__main__":
    main()
