"""Whole Body Tracking-specific termination terms."""

from __future__ import annotations

import math
from typing import Any, List

from holosoma.config_types.termination import TerminationTermCfg
from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager
from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.managers.observation.terms.wbt import gravity_vector
from holosoma.managers.termination.base import TerminationTermBase
from holosoma.utils.rotations import (
    quat_error_magnitude,
    quat_rotate_inverse,
)
from holosoma.utils.safe_torch_import import torch


#########################################################################################################
## Termination terms
#########################################################################################################
def motion_ends(env, **_) -> torch.Tensor:
    """Terminate if the motion ends."""
    motion_command = env.command_manager.get_state("motion_command")
    if hasattr(motion_command, "segment_end_steps"):
        return motion_command.time_steps >= motion_command.segment_end_steps - 2
    return motion_command.time_steps >= motion_command.motion.time_step_total - 2


class BadTracking(TerminationTermBase):
    """Terminate if the tracking is bad.

    - bad ref pos
    - bad ref ori
    - bad motion body pos
    if has object:
        - bad object pos
        - bad object ori

    When bad tracking is detected, the motion_commmand.AdaptiveTimestepsSampler will be updated.
    """

    def __init__(self, cfg: TerminationTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)

        self.bad_ref_pos_threshold = cfg.params["bad_ref_pos_threshold"]
        self.bad_ref_ori_threshold = cfg.params["bad_ref_ori_threshold"]

        self.bad_motion_body_pos_body_names = cfg.params["bad_motion_body_pos_body_names"]

        # NOTE: body_names_to_track is shared with command_manager
        self.body_names_to_track = cfg.params["body_names_to_track"]
        self.bad_motion_body_pos_threshold = cfg.params["bad_motion_body_pos_threshold"]
        self.bad_motion_body_pos_body_indexes = self._get_index_of_a_in_b(
            self.bad_motion_body_pos_body_names, self.body_names_to_track, self.env.device
        )

        self.bad_object_pos_threshold = cfg.params["bad_object_pos_threshold"]
        self.bad_object_ori_threshold = cfg.params["bad_object_ori_threshold"]
        self.disable_object_tracking_in_eval = cfg.params.get("disable_object_tracking_in_eval", True)
        self.last_subterm_results: dict[str, torch.Tensor] = {}

    def __call__(self, env: Any, **kwargs) -> torch.Tensor:
        motion_command = self.env.command_manager.get_state("motion_command")
        assert motion_command.motion_cfg.body_names_to_track == self.body_names_to_track, (
            "body_names_to_track in motion_command and termination.params are not the same"
            f"motion_command.motion_cfg.body_names_to_track: {motion_command.motion_cfg.body_names_to_track}"
            f"termination.params['body_names_to_track']: {self.body_names_to_track}"
        )

        bad_ref_pos = self.bad_ref_pos(motion_command)
        bad_ref_ori = self.bad_ref_ori(motion_command)
        bad_motion_body_pos = self.bad_motion_body_pos(motion_command)
        bad_tracking = bad_ref_pos | bad_ref_ori | bad_motion_body_pos

        bad_object_pos = torch.zeros_like(bad_ref_pos)
        bad_object_ori = torch.zeros_like(bad_ref_pos)
        has_object = getattr(motion_command, "motion_has_object", motion_command.motion.has_object)
        if has_object and not (self.env.is_evaluating and self.disable_object_tracking_in_eval):
            bad_object_pos = self.bad_object_pos(motion_command)
            bad_object_ori = self.bad_object_ori(motion_command)
            bad_tracking |= bad_object_pos | bad_object_ori

        if motion_command.motion_cfg.use_adaptive_timesteps_sampler and torch.any(bad_tracking):
            failed_at_time_step = motion_command.time_steps[bad_tracking]
            if hasattr(motion_command, "motion_ids") and hasattr(motion_command, "adaptive_timesteps_samplers"):
                failed_motion_ids = motion_command.motion_ids[bad_tracking]
                for motion_id in failed_motion_ids.unique().tolist():
                    mask = failed_motion_ids == motion_id
                    if not mask.any():
                        continue
                    sampler = motion_command.adaptive_timesteps_samplers[motion_id]
                    sampler.update_current_bin_failed_count(failed_at_time_step[mask])
            else:
                motion_command.adaptive_timesteps_sampler.update_current_bin_failed_count(failed_at_time_step)

        self.last_subterm_results = {
            "ref_pos": bad_ref_pos,
            "ref_ori": bad_ref_ori,
            "body_pos": bad_motion_body_pos,
            "object_pos": bad_object_pos,
            "object_ori": bad_object_ori,
        }
        return bad_tracking

    def bad_ref_pos(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the reference position is too far from the robot's position."""
        return torch.norm(motion_command.ref_pos_w - motion_command.robot_ref_pos_w, dim=1) > self.bad_ref_pos_threshold

    def bad_ref_ori(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the reference orientation is too far from the robot's orientation."""
        motion_projected_gravity_b = quat_rotate_inverse(
            motion_command.ref_quat_w, gravity_vector(self.env), w_last=True
        )
        robot_projected_gravity_b = quat_rotate_inverse(
            motion_command.robot_ref_quat_w, gravity_vector(self.env), w_last=True
        )
        return (
            torch.abs(motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]) > self.bad_ref_ori_threshold
        )

    def bad_motion_body_pos(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the motion body position is too far from the robot's body position."""
        body_idx = self.bad_motion_body_pos_body_indexes
        error = torch.norm(
            motion_command.body_pos_relative_w[:, body_idx] - motion_command.robot_body_pos_w[:, body_idx], dim=-1
        )
        return torch.any(error > self.bad_motion_body_pos_threshold, dim=-1)

    def bad_object_pos(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the object position is too far from the simulator's object position."""
        return (
            torch.norm(motion_command.object_pos_w - motion_command.simulator_object_pos_w, dim=-1)
            > self.bad_object_pos_threshold
        )

    def bad_object_ori(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the object orientation is too far from the simulator's object orientation."""
        return (
            quat_error_magnitude(motion_command.object_quat_w, motion_command.simulator_object_quat_w)
            > self.bad_object_ori_threshold
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset internal state for specified environments."""

    #########################################################################################################
    ## Internal Helper functions
    #########################################################################################################
    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)


#########################################################################################################
## Standalone (no motion clip required) termination terms
##
## These terms are useful for finetuning a distilled student where the motion
## reference / bad_tracking signals have been removed but you still want the
## episode to end on physical failure or task success rather than running out
## the full max_episode_length_s.
#########################################################################################################


def fall_over(
    env,
    min_pelvis_height: float = 0.4,
    max_tilt_rad: float = 1.0,
) -> torch.Tensor:
    """Terminate when the robot has clearly fallen.

    Two cheap checks combined with OR:

    1.  ``min_pelvis_height``: pelvis (root) world-z drops below this height
        in metres. G1 stands at ~0.76 m, so 0.4 m is a generous "on the
        ground" threshold.
    2.  ``max_tilt_rad``: angle between the body z-axis and the world z-axis
        exceeds this many radians. cos(max_tilt_rad) is computed once.
        Default 1.0 rad ≈ 57° -- the robot is well past balanced.

    Both checks read directly from ``simulator.robot_root_states`` and
    require no motion-reference state, so this is safe to use in any task.
    """
    root_states = env.simulator.robot_root_states[:]
    height = root_states[:, 2]
    quat = root_states[:, 3:7]  # xyzw
    qx = quat[:, 0]
    qy = quat[:, 1]
    upright_cos = 1.0 - 2.0 * (qx * qx + qy * qy)
    is_low = height < min_pelvis_height
    is_tilted = upright_cos < math.cos(max_tilt_rad)
    return is_low | is_tilted


class TaskSucceeded(TerminationTermBase):
    """Terminate after the box has stayed within ``success_threshold`` of the
    target for ``required_consecutive_steps`` consecutive control steps.

    Configured via ``params`` on its ``TerminationTermCfg``:

    - ``success_threshold``: world-frame distance (m) below which the box is
      "at the target". Default 0.10 to match the user-facing "0.1 m carried"
      success criterion.
    - ``required_consecutive_steps``: number of in-success steps needed
      before firing. A small dwell prevents premature termination from a
      single overshooting frame. Default 30 ≈ 0.6 s at 50 Hz.

    Should typically be configured with ``is_timeout=True`` so PPO bootstraps
    the value of the success state when computing returns.
    """

    def __init__(self, cfg: TerminationTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.success_threshold = float(cfg.params.get("success_threshold", 0.10))
        self.required_consecutive_steps = int(cfg.params.get("required_consecutive_steps", 30))
        self._counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._counter.zero_()
        else:
            ids = torch.as_tensor(env_ids, device=self._counter.device, dtype=torch.long)
            if ids.numel() > 0:
                self._counter[ids] = 0

    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:  # noqa: ARG002
        motion_command = self.env.command_manager.get_state("motion_command")
        if motion_command is None:
            return torch.zeros_like(self._counter, dtype=torch.bool)
        box_pos = motion_command.simulator_object_pos_w
        target_pos = motion_command.object_target_pos_w
        dist = torch.norm(box_pos - target_pos, dim=-1)
        in_success = dist < self.success_threshold
        self._counter = torch.where(in_success, self._counter + 1, torch.zeros_like(self._counter))
        return self._counter >= self.required_consecutive_steps


class Stagnation(TerminationTermBase):
    """Terminate when both the robot root and the box have been moving very
    slowly for ``required_steps`` consecutive control steps.

    Used to catch "frozen policy" failure modes where the robot just stands
    still and never picks up the box. Wastes no rollout budget on idle envs.

    ``params``:

    - ``robot_lin_vel_thresh``: m/s. Default 0.05.
    - ``box_lin_vel_thresh``: m/s. Default 0.05.
    - ``required_steps``: number of consecutive idle steps before firing.
      Default 100 = 2 s at 50 Hz.
    - ``grace_steps``: number of steps at the start of every episode during
      which stagnation is NOT counted (lets the policy ramp up). Default 50.

    Both thresholds are checked simultaneously: the term only fires when
    BOTH the robot AND the box are stationary, so a robot that walks past
    the box but never picks it up will still trigger this (because the box
    stays put), and a robot that lifts the box but stops walking after will
    NOT trigger this (because the box has motion).
    """

    def __init__(self, cfg: TerminationTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.robot_lin_vel_thresh = float(cfg.params.get("robot_lin_vel_thresh", 0.05))
        self.box_lin_vel_thresh = float(cfg.params.get("box_lin_vel_thresh", 0.05))
        self.required_steps = int(cfg.params.get("required_steps", 100))
        self.grace_steps = int(cfg.params.get("grace_steps", 50))
        self._counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._counter.zero_()
        else:
            ids = torch.as_tensor(env_ids, device=self._counter.device, dtype=torch.long)
            if ids.numel() > 0:
                self._counter[ids] = 0

    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:  # noqa: ARG002
        motion_command = self.env.command_manager.get_state("motion_command")
        if motion_command is None:
            return torch.zeros_like(self._counter, dtype=torch.bool)
        root_states = self.env.simulator.robot_root_states[:]
        robot_lin_vel = torch.norm(root_states[:, 7:10], dim=-1)
        box_lin_vel = torch.norm(motion_command.simulator_object_lin_vel_w, dim=-1)
        is_stagnant = (robot_lin_vel < self.robot_lin_vel_thresh) & (box_lin_vel < self.box_lin_vel_thresh)
        in_grace = self.env.episode_length_buf < self.grace_steps
        is_stagnant = is_stagnant & ~in_grace
        self._counter = torch.where(is_stagnant, self._counter + 1, torch.zeros_like(self._counter))
        return self._counter >= self.required_steps
