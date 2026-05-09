"""Reward terms for Whole Body Tracking tasks."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, List

import torch

from holosoma.config_types.reward import RewardTermCfg
from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.managers.reward.base import RewardTermBase
from holosoma.utils.rotations import quat_error_magnitude

if TYPE_CHECKING:
    from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager


def _get_motion_command_and_assert_type(env: WholeBodyTrackingManager) -> MotionCommand:
    motion_command = env.command_manager.get_state("motion_command")
    assert motion_command is not None, "motion_command not found in command manager"
    assert isinstance(motion_command, MotionCommand), f"Expected MotionCommand, got {type(motion_command)}"
    return motion_command


def _get_body_index(env: WholeBodyTrackingManager, body_name: str) -> int:
    cache = getattr(env, "_reward_body_index_cache", None)
    if cache is None:
        cache = {}
        setattr(env, "_reward_body_index_cache", cache)
    if body_name not in cache:
        cache[body_name] = env.simulator.body_names.index(body_name)  # type: ignore[attr-defined]
    return cache[body_name]


def _exp_reward(dist: torch.Tensor, sigma: float) -> torch.Tensor:
    return torch.exp(-(dist**2) / (sigma**2))


#########################################################################################################
## terms same to managers/reward/terms/locomotion.py
#########################################################################################################


def penalty_action_rate(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Penalize changes in actions between steps.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    actions = env.action_manager.action
    prev_actions = env.action_manager.prev_action
    return torch.sum(torch.square(prev_actions - actions), dim=1)


def limits_dof_pos(env: WholeBodyTrackingManager, soft_dof_pos_limit: float = 0.95) -> torch.Tensor:
    """Penalize joint positions too close to limits.

    Args:
        env: The environment instance
        soft_dof_pos_limit: Soft limit as fraction of hard limit

    Returns:
        Reward tensor [num_envs]
    """
    # Use soft limits as fraction of hard limits
    m = (env.simulator.hard_dof_pos_limits[:, 0] + env.simulator.hard_dof_pos_limits[:, 1]) / 2  # type: ignore[attr-defined]
    r = env.simulator.hard_dof_pos_limits[:, 1] - env.simulator.hard_dof_pos_limits[:, 0]  # type: ignore[attr-defined]
    lower_soft_limit = m - 0.5 * r * soft_dof_pos_limit
    upper_soft_limit = m + 0.5 * r * soft_dof_pos_limit

    out_of_limits = -(env.simulator.dof_pos - lower_soft_limit).clip(max=0.0)  # lower limit
    out_of_limits += (env.simulator.dof_pos - upper_soft_limit).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


#########################################################################################################
## terms specific to Whole Body Tracking
#########################################################################################################

# ================================================================================================
# Robot Tracking Rewards
# ================================================================================================


def motion_global_ref_position_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.ref_pos_w - motion_command.robot_ref_pos_w), dim=-1)
    return torch.exp(-error / sigma**2)


def motion_global_ref_orientation_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = quat_error_magnitude(motion_command.ref_quat_w, motion_command.robot_ref_quat_w) ** 2
    return torch.exp(-error / sigma**2)


def motion_relative_body_position_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.body_pos_relative_w - motion_command.robot_body_pos_w), dim=-1)
    return torch.exp(-error.mean(-1) / sigma**2)


def motion_relative_body_orientation_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = quat_error_magnitude(motion_command.body_quat_relative_w, motion_command.robot_body_quat_w) ** 2
    return torch.exp(-error.mean(-1) / sigma**2)


def motion_global_body_lin_vel(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.body_lin_vel_w - motion_command.robot_body_lin_vel_w), dim=-1)
    return torch.exp(-error.mean(-1) / sigma**2)


def motion_global_body_ang_vel(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.body_ang_vel_w - motion_command.robot_body_ang_vel_w), dim=-1)
    return torch.exp(-error.mean(-1) / sigma**2)


# ================================================================================================
# Object Tracking Rewards
# ================================================================================================


def object_global_ref_position_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.object_pos_w - motion_command.simulator_object_pos_w), dim=-1)
    return torch.exp(-error / sigma**2)


def object_global_ref_orientation_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = quat_error_magnitude(motion_command.object_quat_w, motion_command.simulator_object_quat_w) ** 2
    return torch.exp(-error / sigma**2)


# ================================================================================================
# Object Task Rewards (Carrying)
# ================================================================================================


def task_walk_to_object(
    env: WholeBodyTrackingManager,
    sigma: float,
    near_threshold: float,
) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    root_pos = motion_command.robot_root_pos_w
    obj_pos = motion_command.simulator_object_pos_w
    dist_xy = torch.norm(root_pos[:, :2] - obj_pos[:, :2], dim=-1)
    return torch.where(dist_xy < near_threshold, torch.ones_like(dist_xy), _exp_reward(dist_xy, sigma))


def task_pick_object(
    env: WholeBodyTrackingManager,
    sigma: float,
    activate_dist: float,
    left_hand_body: str,
    right_hand_body: str,
) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    obj_pos = motion_command.simulator_object_pos_w
    root_pos = motion_command.robot_root_pos_w
    dist_xy = torch.norm(root_pos[:, :2] - obj_pos[:, :2], dim=-1)
    activate = (dist_xy < activate_dist).to(obj_pos.dtype)

    left_idx = _get_body_index(env, left_hand_body)
    right_idx = _get_body_index(env, right_hand_body)
    left_pos = env.simulator._rigid_body_pos[:, left_idx, :]  # type: ignore[attr-defined]
    right_pos = env.simulator._rigid_body_pos[:, right_idx, :]  # type: ignore[attr-defined]
    hand_center = 0.5 * (left_pos + right_pos)

    dist = torch.norm(hand_center - obj_pos, dim=-1)
    return _exp_reward(dist, sigma) * activate


def task_carry_object(
    env: WholeBodyTrackingManager,
    sigma: float,
) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    obj_pos = motion_command.simulator_object_pos_w
    target_pos = motion_command.object_target_pos_w
    dist = torch.norm(obj_pos - target_pos, dim=-1)
    return _exp_reward(dist, sigma)


def task_put_object(
    env: WholeBodyTrackingManager,
    sigma: float,
    activate_dist: float,
) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    obj_pos = motion_command.simulator_object_pos_w
    target_pos = motion_command.object_target_pos_w
    dist_xy = torch.norm(obj_pos[:, :2] - target_pos[:, :2], dim=-1)
    activate = (dist_xy < activate_dist).to(obj_pos.dtype)
    height_error = torch.abs(obj_pos[:, 2] - target_pos[:, 2])
    return _exp_reward(height_error, sigma) * activate


# ================================================================================================
# Undesired Contacts Rewards
# ================================================================================================


class UndesiredContacts(RewardTermBase):
    def __init__(self, cfg: RewardTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.env = env
        undesired_contacts_body_names = [
            body_name
            for body_name in self.env.simulator.body_names  # type: ignore[attr-defined]
            if re.match(cfg.params.get("undesired_contacts_body_names", ""), body_name)
        ]
        self.undesired_contacts_body_indexes = self._get_index_of_a_in_b(
            undesired_contacts_body_names,
            self.env.simulator.body_names,  # type: ignore[attr-defined]
            self.env.device,
        )
        self.threshold = cfg.params.get("threshold", 1.0)

    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:
        # (num_envs, history_length, num_bodies, 3)
        net_contact_forces = self.env.simulator.contact_forces_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self.undesired_contacts_body_indexes], dim=-1), dim=1)[0]
            > self.threshold
        )
        return torch.sum(is_contact, dim=1)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        pass

    #########################################################################################################
    ## Internal Helper functions
    #########################################################################################################
    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)
