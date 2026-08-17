"""Task-phase command wrapper for Samsung box placement training.

This module deliberately keeps the original :class:`MotionCommand` interface
unchanged so released Carry2Anywhere teacher checkpoints remain compatible.
The phase is internal state for reward/curriculum/termination logic; it is not
added to the policy observation yet.
"""

from __future__ import annotations

import torch

from holosoma.managers.command.terms.wbt import MotionCommand


class PlacementPhase:
    """Integer ids for the box manipulation stages."""

    APPROACH = 0
    LIFT = 1
    CARRY = 2
    LOWER = 3
    RELEASE = 4
    RECOVER = 5

    NAMES = ("approach", "lift", "carry", "lower", "release", "recover")


class PlacementMotionCommand(MotionCommand):
    """MotionCommand plus an internal geometry-driven placement phase.

    The existing reference motion still provides the pickup/lift/carry prior.
    The new phase state gives later reward and termination terms a clean way to
    activate placement-specific behavior without changing the teacher network
    input shape.
    """

    # Conservative first-pass thresholds. These are intentionally broad and
    # should be tuned only after the first training/evaluation smoke tests.
    lift_start_height = 0.08
    carry_height = 0.20
    lower_xy_distance = 0.35
    release_xy_distance = 0.12
    release_height_error = 0.08
    release_max_object_speed = 0.25
    recover_hand_clearance = 0.25

    def setup(self) -> None:
        super().setup()
        self.task_phase = torch.full(
            (self.num_envs,),
            PlacementPhase.APPROACH,
            dtype=torch.long,
            device=self.device,
        )

        # Cache the wrist body indices once. These are the same wrist links
        # already used by the original WBT configuration.
        body_names = self._env.simulator.body_names
        self._left_wrist_idx = body_names.index("left_wrist_yaw_link")
        self._right_wrist_idx = body_names.index("right_wrist_yaw_link")

    def reset(self, env_ids: torch.Tensor | None) -> None:
        ids = self._ensure_index_tensor(env_ids)
        super().reset(ids)
        if ids.numel() > 0:
            self.task_phase[ids] = PlacementPhase.APPROACH

    def step(self) -> None:
        super().step()
        self._update_task_phase()

    def _update_task_phase(self) -> None:
        """Advance each environment using physical task progress.

        Phase transitions depend on the simulated box and target, not on a
        particular reference-frame number. This is important because the new
        placement behavior will eventually continue after the demonstration's
        useful carry portion.
        """
        if not self.motion_has_object:
            return

        obj_pos = self.simulator_object_pos_w
        target_pos = self.object_target_pos_w
        obj_speed = torch.norm(self.simulator_object_lin_vel_w, dim=-1)

        xy_error = torch.norm(obj_pos[:, :2] - target_pos[:, :2], dim=-1)
        height_error = torch.abs(obj_pos[:, 2] - target_pos[:, 2])
        lift_height = obj_pos[:, 2] - target_pos[:, 2]

        # Use sequential transitions so one simulation step cannot jump from
        # APPROACH all the way to RELEASE.
        phase_before = self.task_phase.clone()

        to_lift = (phase_before == PlacementPhase.APPROACH) & (lift_height > self.lift_start_height)
        self.task_phase[to_lift] = PlacementPhase.LIFT

        to_carry = (phase_before == PlacementPhase.LIFT) & (lift_height > self.carry_height)
        self.task_phase[to_carry] = PlacementPhase.CARRY

        to_lower = (phase_before == PlacementPhase.CARRY) & (xy_error < self.lower_xy_distance)
        self.task_phase[to_lower] = PlacementPhase.LOWER

        ready_to_release = (
            (xy_error < self.release_xy_distance)
            & (height_error < self.release_height_error)
            & (obj_speed < self.release_max_object_speed)
        )
        to_release = (phase_before == PlacementPhase.LOWER) & ready_to_release
        self.task_phase[to_release] = PlacementPhase.RELEASE

        # During RELEASE we only switch to RECOVER after both wrists have
        # clearly moved away from the box. This is a geometry check for now;
        # the later placement reward will additionally require stable support.
        rigid_body_pos = self._env.simulator._rigid_body_pos  # type: ignore[attr-defined]
        left_wrist_pos = rigid_body_pos[:, self._left_wrist_idx, :]
        right_wrist_pos = rigid_body_pos[:, self._right_wrist_idx, :]
        left_clear = torch.norm(left_wrist_pos - obj_pos, dim=-1) > self.recover_hand_clearance
        right_clear = torch.norm(right_wrist_pos - obj_pos, dim=-1) > self.recover_hand_clearance
        to_recover = (phase_before == PlacementPhase.RELEASE) & left_clear & right_clear
        self.task_phase[to_recover] = PlacementPhase.RECOVER

    @property
    def task_phase_name(self) -> tuple[str, ...]:
        """Human-readable phase names, mainly for debugging."""
        return PlacementPhase.NAMES
