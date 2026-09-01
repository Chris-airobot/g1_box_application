"""Focused Isaac diagnostic for one combined-200 multi-box teacher motion.

This script is intentionally evaluation-only. It creates nine fixed-size environments,
selects the one environment compatible with the requested manifest motion, and either
replays the reference state or runs a deterministic checkpoint policy while recording
tracking diagnostics. It does not train or record video.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
from typing import Any

from holosoma.config_types.logger import DisabledLoggerConfig
from holosoma.config_types.randomization import RandomizationManagerCfg
from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt_combined_teacher_200
from holosoma.utils.eval_utils import load_checkpoint
from holosoma.utils.helpers import get_class
from holosoma.utils.multibox import load_multibox_manifest
from holosoma.utils.path import resolve_data_file_path
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment


DEFAULT_MOTION = "mentor_Bringing-carry_0205_mj_w_obj.npz"
BAD_BODY_THRESHOLD_M = 0.25
REFERENCE_RELATION_TOLERANCE_M = 2.0e-3
MODES = ("reference", "policy_nominal", "policy_standard", "policy_start_sweep")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--motion", default=DEFAULT_MOTION, help="Exact manifest basename")
    parser.add_argument("--checkpoint", help="Required for policy modes; local path or wandb:// URI")
    parser.add_argument("--output-dir", default="outputs/single_motion_diagnostic")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap per start")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.mode != "reference" and not args.checkpoint:
        parser.error("--checkpoint is required for policy modes")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    return args


def _find_manifest_motion_id(motion_file: str) -> tuple[int, tuple[float, float, float]]:
    entries = load_multibox_manifest(resolve_data_file_path("data/combined_teacher_200_manifest.csv"))
    matches = [(index, entry) for index, entry in enumerate(entries) if entry.file == motion_file]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one manifest row for {motion_file!r}, found {len(matches)}")
    motion_id, entry = matches[0]
    return motion_id, entry.dimensions


def _make_config(args: argparse.Namespace, motion_id: int):
    base = g1_29dof_wbt_combined_teacher_200.get_eval_config()
    motion_term = base.command.setup_terms["motion_command"]
    motion_cfg = replace(
        motion_term.params["motion_config"],
        eval_motion_id=motion_id,
        eval_start_timestep=0,
        eval_start_timestep_range=None,
        eval_start_time_s=None,
        eval_start_time_range_s=None,
    )
    if args.mode in ("reference", "policy_nominal"):
        motion_cfg = replace(
            motion_cfg,
            noise_to_initial_pose=replace(motion_cfg.noise_to_initial_pose, overall_noise_scale=0.0),
            freeze_at_timestep_zero_prob=0.0,
        )

    command_cfg = replace(
        base.command,
        setup_terms={
            **base.command.setup_terms,
            "motion_command": replace(motion_term, params={"motion_config": motion_cfg}),
        },
    )
    output_dir = str(Path(args.output_dir).expanduser())
    return replace(
        base,
        training=replace(base.training, num_envs=9, headless=True, seed=args.seed, export_onnx=False),
        command=command_cfg,
        # An empty manager keeps the environment lifecycle intact while disabling
        # pushes and all domain randomizers for reference/nominal diagnostics only.
        randomization=(
            RandomizationManagerCfg()
            if args.mode in ("reference", "policy_nominal")
            else base.randomization
        ),
        logger=DisabledLoggerConfig(base_dir=output_dir),
    )


def _reset_without_control_step(env, command, start_timestep: int) -> dict[str, torch.Tensor]:
    command.motion_cfg = replace(
        command.motion_cfg,
        eval_start_timestep=int(start_timestep),
        eval_start_timestep_range=None,
        eval_start_time_s=None,
        eval_start_time_range_s=None,
    )
    command.init_pose_cfg = command.motion_cfg.noise_to_initial_pose
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    env.reset_envs_idx(env_ids)
    env._refresh_envs_after_reset(env_ids)
    # Refresh articulation forward kinematics without taking a physics/control step
    # (and therefore without advancing MotionCommand away from the requested start).
    env.simulator.scene.write_data_to_sim()
    env.simulator.sim.forward()
    env.simulator.scene.update(dt=1.0 / float(env.simulator.simulator_config.sim.fps))
    env._refresh_sim_tensors()
    env._pre_compute_observations_callback()
    env.reset_buf[env_ids] = 0
    env.time_out_buf[env_ids] = False
    env._compute_observations()
    env._post_compute_observations_callback()
    env._clip_observations()
    return env.obs_buf_dict


def _step_without_automatic_reset(env, actions: torch.Tensor):
    """Run BaseTask's normal step pipeline, stopping before automatic resets."""
    env._pre_physics_step(actions)
    env._physics_step()
    env._refresh_sim_tensors()
    env.episode_length_buf += 1
    env._update_counters_each_step()
    env._pre_compute_observations_callback()
    env._update_tasks_callback()
    env._check_termination()
    env._compute_reward()
    env._update_log_dict()
    env._compute_observations()
    env._post_compute_observations_callback()
    env._clip_observations()
    return env.obs_buf_dict, env.rew_buf, env.reset_buf


def _termination_reason(env, env_id: int) -> tuple[str, bool]:
    reasons = [
        name
        for name, result in env.termination_manager.last_term_results.items()
        if bool(result[env_id].item())
    ]
    parent_terms = {name.split("/", 1)[0] for name in reasons if "/" in name}
    if parent_terms:
        reasons = [name for name in reasons if name not in parent_terms]
    return "+".join(sorted(reasons)), bool(env.reset_buf[env_id].item())


def _body_index(command, name: str) -> int:
    try:
        return command.motion_cfg.body_names_to_track.index(name)
    except ValueError as exc:
        raise RuntimeError(f"Required diagnostic body {name!r} is not tracked") from exc


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().item())


def _collect_row(
    env,
    command,
    env_id: int,
    motion_id: int,
    start_timestep: int,
    first_bad_body: str,
    *,
    use_unaligned_reference: bool = False,
) -> tuple[dict[str, Any], str]:
    body_names = command.motion_cfg.body_names_to_track
    tracking_reference_body_pos = (
        command.body_pos_w if use_unaligned_reference else command.body_pos_relative_w
    )
    simulator_body_pos = command.robot_body_pos_w
    body_errors = torch.norm(
        tracking_reference_body_pos[env_id] - simulator_body_pos[env_id], dim=-1
    )

    bad_tracking_term = env.termination_manager._term_instances.get("bad_tracking")
    threshold_body_names = getattr(
        bad_tracking_term,
        "bad_motion_body_pos_body_names",
        body_names,
    )
    first_exceeding_this_step = ""
    for body_name in threshold_body_names:
        body_index = _body_index(command, body_name)
        if _scalar(body_errors[body_index]) > BAD_BODY_THRESHOLD_M:
            first_exceeding_this_step = body_name
            break
    if not first_bad_body and first_exceeding_this_step:
        first_bad_body = first_exceeding_this_step

    left_wrist = _body_index(command, "left_wrist_yaw_link")
    right_wrist = _body_index(command, "right_wrist_yaw_link")
    left_ankle = _body_index(command, "left_ankle_roll_link")
    right_ankle = _body_index(command, "right_ankle_roll_link")
    reference_object_pos = command.object_pos_w[env_id]
    simulator_object_pos = command.simulator_object_pos_w[env_id]

    # The reference relationship must use two points from the raw motion frame.
    # body_pos_relative_w is appropriate for policy tracking errors, but pairing
    # it with the unaligned reference object would corrupt this intrinsic distance.
    raw_reference_body_pos = command.body_pos_w
    reference_left_wrist_origin_to_box_center = torch.norm(
        raw_reference_body_pos[env_id, left_wrist] - reference_object_pos
    )
    reference_right_wrist_origin_to_box_center = torch.norm(
        raw_reference_body_pos[env_id, right_wrist] - reference_object_pos
    )
    simulator_left_wrist_origin_to_box_center = torch.norm(
        simulator_body_pos[env_id, left_wrist] - simulator_object_pos
    )
    simulator_right_wrist_origin_to_box_center = torch.norm(
        simulator_body_pos[env_id, right_wrist] - simulator_object_pos
    )
    reason, terminated = _termination_reason(env, env_id)
    timestep = int(command.time_steps[env_id].item())
    motion_length = int(command.motion_lengths[motion_id].item())
    denominator = max(motion_length - 1 - start_timestep, 1)
    progress = min(max((timestep - start_timestep) / denominator, 0.0), 1.0)

    row: dict[str, Any] = {
        "env_id": env_id,
        "motion_id": motion_id,
        "motion_file": command.motion_file_names[motion_id],
        "motion_timestep": timestep,
        "motion_time_s": timestep / float(command.motion_fps),
        "start_timestep": start_timestep,
        "episode_step": int(env.episode_length_buf[env_id].item()),
        "episode_progress": progress,
        "motion_length": motion_length,
        "root_position_error_m": _scalar(torch.norm(command.root_pos_w[env_id] - command.robot_root_pos_w[env_id])),
        "reference_body_position_error_m": _scalar(
            torch.norm(command.ref_pos_w[env_id] - command.robot_ref_pos_w[env_id])
        ),
        "left_wrist_position_error_m": _scalar(body_errors[left_wrist]),
        "right_wrist_position_error_m": _scalar(body_errors[right_wrist]),
        "left_ankle_position_error_m": _scalar(body_errors[left_ankle]),
        "right_ankle_position_error_m": _scalar(body_errors[right_ankle]),
        "reference_left_wrist_link_origin_to_box_center_m": _scalar(
            reference_left_wrist_origin_to_box_center
        ),
        "reference_right_wrist_link_origin_to_box_center_m": _scalar(
            reference_right_wrist_origin_to_box_center
        ),
        "reference_wrist_link_origin_to_box_center_min_m": _scalar(
            torch.minimum(
                reference_left_wrist_origin_to_box_center,
                reference_right_wrist_origin_to_box_center,
            )
        ),
        "simulator_left_wrist_link_origin_to_box_center_m": _scalar(
            simulator_left_wrist_origin_to_box_center
        ),
        "simulator_right_wrist_link_origin_to_box_center_m": _scalar(
            simulator_right_wrist_origin_to_box_center
        ),
        "simulator_wrist_link_origin_to_box_center_min_m": _scalar(
            torch.minimum(
                simulator_left_wrist_origin_to_box_center,
                simulator_right_wrist_origin_to_box_center,
            )
        ),
        "reference_object_vs_simulator_object_position_error_m": _scalar(
            torch.norm(reference_object_pos - simulator_object_pos)
        ),
        "terminated": int(terminated),
        "termination_reason": reason,
        "first_exceeding_body_this_step": first_exceeding_this_step,
        "first_body_to_exceed_0p25_m": first_bad_body,
    }
    return row, first_bad_body


def _print_step(row: dict[str, Any]) -> None:
    print(
        f"t={row['motion_timestep']:4d}/{row['motion_length']:4d} "
        f"time={row['motion_time_s']:7.3f}s progress={row['episode_progress']:.3f} "
        f"root/ref={row['root_position_error_m']:.3f}/{row['reference_body_position_error_m']:.3f}m "
        f"wrist L/R={row['left_wrist_position_error_m']:.3f}/{row['right_wrist_position_error_m']:.3f}m "
        f"ankle L/R={row['left_ankle_position_error_m']:.3f}/{row['right_ankle_position_error_m']:.3f}m "
        f"ref-wrist-origin/box-center L/R="
        f"{row['reference_left_wrist_link_origin_to_box_center_m']:.3f}/"
        f"{row['reference_right_wrist_link_origin_to_box_center_m']:.3f}m "
        f"sim-wrist-origin/box-center L/R="
        f"{row['simulator_left_wrist_link_origin_to_box_center_m']:.3f}/"
        f"{row['simulator_right_wrist_link_origin_to_box_center_m']:.3f}m "
        f"obj={row['reference_object_vs_simulator_object_position_error_m']:.3f}m "
        f"done={row['terminated']} reason={row['termination_reason'] or '-'} "
        f"first_bad={row['first_body_to_exceed_0p25_m'] or '-'}"
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("Diagnostic produced no rows")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _assert_selected_environment(command, motion_id: int) -> int:
    target_size_id = int(command.motion_size_ids[motion_id].item())
    compatible_env_ids = (command.env_asset_size_ids == target_size_id).nonzero(as_tuple=False).flatten()
    if compatible_env_ids.numel() != 1:
        raise RuntimeError(
            f"Expected exactly one env for size_id={target_size_id} with nine envs, got "
            f"{compatible_env_ids.detach().cpu().tolist()}"
        )
    env_id = int(compatible_env_ids[0].item())
    if int(command.motion_ids[env_id].item()) != motion_id:
        raise RuntimeError(f"Compatible env {env_id} did not receive requested motion {motion_id}")
    if int(command.motion_size_ids[command.motion_ids[env_id]].item()) != target_size_id:
        raise RuntimeError("physical-box/motion-size invariant failed in diagnostic environment")
    return env_id


def _set_reference_state(env, command, env_id: int, timestep: int) -> None:
    env_ids = torch.tensor([env_id], device=env.device, dtype=torch.long)
    command.time_steps[env_id] = int(timestep)
    env.simulator.dof_pos[env_id] = command.joint_pos[env_id]
    env.simulator.dof_vel[env_id] = command.joint_vel[env_id]
    env.simulator.robot_root_states[env_id, :3] = command.root_pos_w[env_id]
    env.simulator.robot_root_states[env_id, 3:7] = command.root_quat_w[env_id]
    env.simulator.robot_root_states[env_id, 7:10] = command.root_lin_vel_w[env_id]
    env.simulator.robot_root_states[env_id, 10:13] = command.root_ang_vel_w[env_id]
    env.simulator.set_actor_root_state_tensor_robots(env_ids, env.simulator.robot_root_states)
    env.simulator.set_dof_state_tensor_robots(env_ids, env.simulator.dof_state)
    object_state = torch.cat(
        [
            command.object_pos_w[env_ids],
            command.object_quat_w[env_ids],
            command.object_lin_vel_w[env_ids],
            command.object_ang_vel_w[env_ids],
        ],
        dim=-1,
    )
    env.simulator.set_actor_states([command.object_name], env_ids, object_state)
    env.simulator.scene.write_data_to_sim()
    env.simulator.sim.forward()
    # A positive data-update dt invalidates IsaacLab's timestamped articulation
    # caches so body poses are re-read after the authored joint/root state change.
    env.simulator.scene.update(dt=1.0 / float(env.simulator.simulator_config.sim.fps))
    env._refresh_sim_tensors()
    env._pre_compute_observations_callback()


def _run_reference(env, command, env_id: int, motion_id: int, output_dir: Path, max_steps: int | None) -> None:
    motion_length = int(command.motion_lengths[motion_id].item())
    stop = motion_length if max_steps is None else min(motion_length, max_steps)
    rows: list[dict[str, Any]] = []
    first_bad_body = ""
    max_relation_error = 0.0
    for timestep in range(stop):
        _set_reference_state(env, command, env_id, timestep)
        env.reset_buf[env_id] = 0
        row, first_bad_body = _collect_row(
            env,
            command,
            env_id,
            motion_id,
            0,
            first_bad_body,
            use_unaligned_reference=True,
        )
        # Termination terms use the policy-aligned reference buffers, which are
        # intentionally not advanced in reference replay. Do not report those
        # terms as meaningful reference-replay failures.
        row["terminated"] = 0
        row["termination_reason"] = "not_applicable_reference_replay"
        relation_error = max(
            abs(
                row["reference_left_wrist_link_origin_to_box_center_m"]
                - row["simulator_left_wrist_link_origin_to_box_center_m"]
            ),
            abs(
                row["reference_right_wrist_link_origin_to_box_center_m"]
                - row["simulator_right_wrist_link_origin_to_box_center_m"]
            ),
        )
        max_relation_error = max(max_relation_error, relation_error)
        rows.append(row)
        _print_step(row)

    csv_path = output_dir / "reference.csv"
    _write_csv(csv_path, rows)
    max_object_error = max(row["reference_object_vs_simulator_object_position_error_m"] for row in rows)
    if max_relation_error > REFERENCE_RELATION_TOLERANCE_M or max_object_error > REFERENCE_RELATION_TOLERANCE_M:
        raise AssertionError(
            "Reference replay did not preserve the hand-box relationship: "
            f"max wrist-box distance discrepancy={max_relation_error:.6g} m, "
            f"max object position error={max_object_error:.6g} m, "
            f"tolerance={REFERENCE_RELATION_TOLERANCE_M:.6g} m"
        )
    print(
        f"REFERENCE_RELATION_PASS rows={len(rows)} csv={csv_path} "
        f"max_relation_error={max_relation_error:.6g}m max_object_error={max_object_error:.6g}m"
    )


def _run_policy_start(
    env,
    command,
    algo,
    policy,
    motion_id: int,
    start_timestep: int,
    output_path: Path,
    max_steps: int | None,
) -> None:
    obs = _reset_without_control_step(env, command, start_timestep)
    env_id = _assert_selected_environment(command, motion_id)
    actual_start_timestep = int(command.time_steps[env_id].item())
    if actual_start_timestep != start_timestep:
        raise RuntimeError(
            f"Requested start timestep {start_timestep}, but compatible env {env_id} "
            f"started at {actual_start_timestep}"
        )
    if hasattr(algo.actor, "reset"):
        algo.actor.reset(torch.ones(env.num_envs, device=env.device, dtype=torch.bool))

    motion_length = int(command.motion_lengths[motion_id].item())
    available_steps = max(motion_length - start_timestep, 1)
    step_limit = available_steps if max_steps is None else min(available_steps, max_steps)
    rows: list[dict[str, Any]] = []
    first_bad_body = ""
    with torch.no_grad():
        for _ in range(step_limit):
            actor_obs = torch.cat([obs[key] for key in algo.actor_obs_keys], dim=1)
            actions = policy({"actor_obs": actor_obs})
            obs, _, dones = _step_without_automatic_reset(env, actions)
            row, first_bad_body = _collect_row(
                env, command, env_id, motion_id, start_timestep, first_bad_body
            )
            rows.append(row)
            _print_step(row)
            if bool(dones[env_id].item()):
                break

    _write_csv(output_path, rows)
    final = rows[-1]
    print(
        f"POLICY_DIAGNOSTIC_SUMMARY start={start_timestep} rows={len(rows)} "
        f"progress={final['episode_progress']:.3f} reason={final['termination_reason'] or 'step_limit'} "
        f"first_bad={final['first_body_to_exceed_0p25_m'] or '-'} csv={output_path}"
    )


def main() -> None:
    args = _parse_args()
    motion_id, manifest_dimensions = _find_manifest_motion_id(args.motion)
    cfg = _make_config(args, motion_id)
    output_dir = Path(args.output_dir).expanduser() / args.mode / Path(args.motion).stem
    output_dir.mkdir(parents=True, exist_ok=True)

    env = None
    simulation_app = None
    try:
        env, device, simulation_app = setup_simulation_environment(cfg)
        env.set_is_evaluating()
        command = env.command_manager.get_state("motion_command")
        if command is None or command.motion_size_ids is None:
            raise RuntimeError("Combined-200 multi-box MotionCommand was not initialized")
        if command.motion_file_names[motion_id] != args.motion:
            raise RuntimeError("Runtime motion ID does not match the manifest-selected filename")

        _reset_without_control_step(env, command, 0)
        env_id = _assert_selected_environment(command, motion_id)
        size_id = int(command.env_asset_size_ids[env_id].item())
        configured_dimensions = tuple(command._env.simulator.multibox_asset_dimensions[size_id])
        if configured_dimensions != manifest_dimensions:
            raise RuntimeError(
                f"Configured physical dimensions {configured_dimensions} != manifest {manifest_dimensions}"
            )
        print(
            f"DIAGNOSTIC_TARGET env_id={env_id} asset_size_id={size_id} motion_id={motion_id} "
            f"motion_file={args.motion} dimensions={manifest_dimensions}"
        )

        if args.mode == "reference":
            _run_reference(env, command, env_id, motion_id, output_dir, args.max_steps)
            return

        runtime_dir = output_dir / "runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        loaded_checkpoint = load_checkpoint(args.checkpoint, str(runtime_dir))
        algo_class = get_class(cfg.algo._target_)
        algo = algo_class(
            device=device,
            env=env,
            config=cfg.algo.config,
            log_dir=str(runtime_dir),
            multi_gpu_cfg=None,
        )
        algo.setup()
        algo.load(str(loaded_checkpoint))
        algo._eval_mode()
        policy = algo.get_inference_policy()

        motion_length = int(command.motion_lengths[motion_id].item())
        if args.mode == "policy_start_sweep":
            last_valid_start = max(motion_length - 2, 0)
            starts = sorted({int(round(fraction * last_valid_start)) for fraction in (0.0, 0.25, 0.5, 0.75)})
        else:
            starts = [0]

        for start_timestep in starts:
            _run_policy_start(
                env,
                command,
                algo,
                policy,
                motion_id,
                start_timestep,
                output_dir / f"start_{start_timestep:06d}.csv",
                args.max_steps,
            )
    finally:
        if env is not None and hasattr(env, "close"):
            env.close()
        close_simulation_app(simulation_app)


if __name__ == "__main__":
    main()
