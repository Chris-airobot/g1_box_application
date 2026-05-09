#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer


def _default_model_path() -> Path:
    module_root = Path(__file__).resolve().parents[1]
    return module_root / "models/g1/g1_29dof.xml"


def list_bodies(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    mujoco.mj_forward(model, data)
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if not name:
            continue
        pos = data.xpos[i]
        print(f"{i:03d} {name:32s} {pos[0]: .4f} {pos[1]: .4f} {pos[2]: .4f}")


def enable_body_labels(viewer) -> None:
    try:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_LABEL] = 1
    except Exception:
        pass
    try:
        viewer.opt.label = mujoco.mjtLabel.mjLABEL_BODY
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print MuJoCo body names/positions and optionally open a viewer."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=_default_model_path(),
        help="Path to a MuJoCo XML model.",
    )
    parser.add_argument("--no-viewer", action="store_true", help="Only print body names/positions.")
    parser.add_argument("--labels", action="store_true", help="Show body labels in the viewer if supported.")
    parser.add_argument("--fps", type=float, default=30.0, help="Viewer refresh rate.")
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")

    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)

    print(f"Model: {args.model}")
    list_bodies(model, data)

    if args.no_viewer:
        return

    with mujoco.viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        if args.labels:
            enable_body_labels(viewer)

        sleep_dt = 1.0 / args.fps if args.fps > 0 else 0.0
        while viewer.is_running():
            mujoco.mj_forward(model, data)
            viewer.sync()
            if sleep_dt:
                time.sleep(sleep_dt)


if __name__ == "__main__":
    main()
