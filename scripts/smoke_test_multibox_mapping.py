"""Company-PC Isaac smoke test for the combined-200 multi-box invariant.

This initializes a small evaluation environment and exercises full/partial environment resets.
It never constructs a PPO algorithm or starts training.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from typing import Sequence

from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt_combined_teacher_200
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment
from holosoma.utils.safe_torch_import import torch


GEOMETRY_ATOL_M = 1.0e-6


def _actual_spawned_collision_dimensions(env_id: int) -> tuple[float, float, float]:
    """Measure the authored collision geometry under an environment's object prim."""
    import omni.usd  # noqa: PLC0415
    from pxr import Usd, UsdGeom, UsdPhysics  # noqa: PLC0415

    stage = omni.usd.get_context().get_stage()
    object_path = f"/World/envs/env_{env_id}/Object"
    object_prim = stage.GetPrimAtPath(object_path)
    if not object_prim.IsValid():
        raise AssertionError(f"env {env_id}: spawned object prim does not exist at {object_path}")

    collision_prims = [
        prim for prim in Usd.PrimRange(object_prim) if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    if len(collision_prims) != 1:
        paths = [str(prim.GetPath()) for prim in collision_prims]
        raise AssertionError(
            f"env {env_id}: expected exactly one collision prim beneath {object_path}, found {paths}"
        )

    collision_prim = collision_prims[0]
    cube = UsdGeom.Cube(collision_prim)
    if not cube:
        raise AssertionError(
            f"env {env_id}: collision prim {collision_prim.GetPath()} is not an authored UsdGeom.Cube"
        )

    # Compute the collision prim's authored bound in Object-local coordinates. This reads the
    # spawned USD cube and its authored transforms directly; it does not use any HoloSoma size tensor.
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    local_bound = bbox_cache.ComputeRelativeBound(collision_prim, object_prim)
    dimensions = local_bound.ComputeAlignedRange().GetSize()
    return tuple(float(dimensions[index]) for index in range(3))


def _assert_dimensions_close(
    env_id: int,
    lhs_name: str,
    lhs: Sequence[float],
    rhs_name: str,
    rhs: Sequence[float],
) -> None:
    if len(lhs) != 3 or len(rhs) != 3 or any(
        not math.isclose(float(lhs[index]), float(rhs[index]), rel_tol=0.0, abs_tol=GEOMETRY_ATOL_M)
        for index in range(3)
    ):
        raise AssertionError(
            f"env {env_id}: {lhs_name}={tuple(float(value) for value in lhs)} != "
            f"{rhs_name}={tuple(float(value) for value in rhs)} "
            f"(absolute tolerance {GEOMETRY_ATOL_M} m)"
        )


def _assert_and_print(command, env_ids: torch.Tensor, *, print_rows: bool) -> None:
    asset_size_ids = command.env_asset_size_ids[env_ids]
    motion_ids = command.motion_ids[env_ids]
    motion_size_ids = command.motion_size_ids[motion_ids]
    if not torch.equal(asset_size_ids, motion_size_ids):
        mismatches = env_ids[asset_size_ids != motion_size_ids].detach().cpu().tolist()
        raise AssertionError(f"asset_size_id != motion_size_id for envs {mismatches}")

    corners = command.object_bbox_corners_local[env_ids]
    bbox_dimensions = corners.amax(dim=1) - corners.amin(dim=1)
    configured_dimensions_by_env = command._env.simulator.env_object_dimensions[env_ids]

    for local_index, env_id_tensor in enumerate(env_ids):
        env_id = int(env_id_tensor.item())
        motion_id = int(motion_ids[local_index].item())
        size_id = int(asset_size_ids[local_index].item())
        manifest_dimensions = command.motion_manifest_dimensions[motion_id]
        configured_dimensions = command._env.simulator.multibox_asset_dimensions[size_id]
        configured_env_dimensions = tuple(
            float(value) for value in configured_dimensions_by_env[local_index].tolist()
        )
        bbox_env_dimensions = tuple(float(value) for value in bbox_dimensions[local_index].tolist())
        actual_dimensions = _actual_spawned_collision_dimensions(env_id)

        _assert_dimensions_close(
            env_id,
            "actual spawned collision dimensions",
            actual_dimensions,
            "configured dimensions",
            configured_dimensions,
        )
        _assert_dimensions_close(
            env_id, "configured env dimensions", configured_env_dimensions, "configured dimensions", configured_dimensions
        )
        _assert_dimensions_close(
            env_id, "manifest dimensions", manifest_dimensions, "configured dimensions", configured_dimensions
        )
        _assert_dimensions_close(
            env_id, "bbox dimensions", bbox_env_dimensions, "configured dimensions", configured_dimensions
        )
        if print_rows:
            print(
                f"env_id={env_id} asset_size_id={size_id} "
                f"actual_spawned_collision_dimensions={actual_dimensions} "
                f"configured_dimensions={configured_dimensions} "
                f"motion_id={motion_id} motion_file={command.motion_file_names[motion_id]} "
                f"manifest_dimensions={manifest_dimensions} "
                f"bbox_dimensions={bbox_env_dimensions}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=18)
    parser.add_argument("--reset-rounds", type=int, default=50)
    args = parser.parse_args()
    if args.num_envs < 18:
        raise ValueError("--num-envs must be at least 18 so every size has two simultaneous environments")

    cfg = replace(
        g1_29dof_wbt_combined_teacher_200,
        training=replace(
            g1_29dof_wbt_combined_teacher_200.training,
            num_envs=args.num_envs,
            headless=True,
        ),
    )

    env = None
    simulation_app = None
    try:
        env, _, simulation_app = setup_simulation_environment(cfg)
        env.set_is_evaluating()
        command = env.command_manager.get_state("motion_command")
        if command is None or command.motion_size_ids is None:
            raise RuntimeError("Combined-teacher multi-box MotionCommand was not initialized")

        all_env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        env.reset_envs_idx(all_env_ids)
        _assert_and_print(command, all_env_ids, print_rows=True)

        for reset_round in range(args.reset_rounds):
            # Alternate a partial reset and a full reset to exercise both paths.
            env_ids = all_env_ids[reset_round % 2 :: 2] if reset_round % 2 == 0 else all_env_ids
            env.reset_envs_idx(env_ids)
            _assert_and_print(command, env_ids, print_rows=False)

        print(f"reset_rounds={args.reset_rounds} mismatches=0")
        print("MULTIBOX_MAPPING_SMOKE_TEST_PASS")
    finally:
        if env is not None and hasattr(env, "close"):
            env.close()
        close_simulation_app(simulation_app)


if __name__ == "__main__":
    main()
