from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, List

import numpy as np
import torch
from loguru import logger

from holosoma.config_types.command import MotionConfig, NoiseToInitialPoseConfig
from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager
from holosoma.managers.command.base import CommandTermBase
from holosoma.utils.file_cache import cached_open
from holosoma.utils.multibox import apply_multibox_eval_motion_id, load_multibox_manifest, map_manifest_sizes
from holosoma.utils.path import resolve_data_file_path
from holosoma.utils.rotations import (
    get_euler_xyz,
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inverse,
    quat_mul,
    slerp,
    yaw_quat,
)
from holosoma.utils.simulator_config import SimulatorType


#########################################################################################################
## MotionLoader and AdaptiveTimestepsSampler
#########################################################################################################
class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        robot_body_names: list[str],
        robot_joint_names: list[str],
        device: str = "cpu",
    ):
        # Resolve the motion file path using importlib.resources
        motion_file = resolve_data_file_path(motion_file)

        logger.info(f"Loading motion file: {motion_file}")
        body_names_in_motion_data, joint_names_in_motion_data = self._load_data_from_motion_npz(motion_file, device)
        body_indexes = self._get_index_of_a_in_b(robot_body_names, body_names_in_motion_data, device)
        joint_indexes = self._get_index_of_a_in_b(robot_joint_names, joint_names_in_motion_data, device)

        self._joint_indexes = joint_indexes
        self._body_indexes = body_indexes
        self.time_step_total = self._joint_pos.shape[0]

    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)

    def _load_data_from_motion_npz(self, motion_file: str, device: str) -> tuple[list[str], list[str]]:
        with cached_open(motion_file, "rb") as f, np.load(f) as data:
            self.fps = data["fps"]

            body_names = data["body_names"].tolist()
            joint_names = data["joint_names"].tolist()

            # The first 7 joints_pos are [xyz, wxyz] of the pelvis, omit them from the joint_pos
            # The first 6 joints_vel are [vel_xyz, vel_wxyz] of the pelvis, omit them from the joint_vel
            # We'll use the pelvis position and quaternion from body_pos_w[:, 0] and body_quat_w[:, 0] directly.
            self._joint_pos = torch.tensor(data["joint_pos"][:, 7:], dtype=torch.float32, device=device)
            self._joint_vel = torch.tensor(data["joint_vel"][:, 6:], dtype=torch.float32, device=device)
            assert len(joint_names) == self._joint_pos.shape[1], "Joint names in motion data does not match"

            self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
            assert len(body_names) == self._body_pos_w.shape[1], "Body names in motion data does not match"

            # NOTE: wxyz after loading from npz
            body_quat_w_wxyz = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)  # This is wxyz
            self._body_quat_w = body_quat_w_wxyz[:, :, [1, 2, 3, 0]]  # Change to xyzw

            self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
            self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)

            # add object pos and quat
            self.has_object = "object_pos_w" in data
            if self.has_object:
                # NOTE: wxyz after loading from npz
                self._object_pos_w = torch.tensor(data["object_pos_w"], dtype=torch.float32, device=device)
                object_quat_w = torch.tensor(data["object_quat_w"], dtype=torch.float32, device=device)
                self._object_quat_w = object_quat_w[:, [1, 2, 3, 0]]  # Change to xyzw
                self._object_lin_vel_w = torch.tensor(data["object_lin_vel_w"], dtype=torch.float32, device=device)
                if "object_ang_vel_w" in data:
                    self._object_ang_vel_w = torch.tensor(data["object_ang_vel_w"], dtype=torch.float32, device=device)
                else:
                    self._object_ang_vel_w = torch.zeros_like(self._object_lin_vel_w)
            else:
                self._object_pos_w = torch.zeros(0, 3, device=device)
                self._object_quat_w = torch.zeros(0, 4, device=device)
                self._object_lin_vel_w = torch.zeros(0, 3, device=device)
                self._object_ang_vel_w = torch.zeros(0, 3, device=device)
        return body_names, joint_names

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._joint_pos[:, self._joint_indexes]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._joint_vel[:, self._joint_indexes]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]

    @property
    def object_pos_w(self) -> torch.Tensor:
        return self._object_pos_w[:]

    @property
    def object_quat_w(self) -> torch.Tensor:
        return self._object_quat_w[:]

    @property
    def object_lin_vel_w(self) -> torch.Tensor:
        return self._object_lin_vel_w[:]

    @property
    def object_ang_vel_w(self) -> torch.Tensor:
        return self._object_ang_vel_w[:]

    def extend_with_segments(self, segments: dict[str, torch.Tensor], prepend: bool) -> MotionLoader:
        """Merge interpolated segments with motion data, mutating this MotionLoader."""
        concat_targets = [
            ("joint_pos", "_joint_pos"),
            ("joint_vel", "_joint_vel"),
            ("body_pos", "_body_pos_w"),
            ("body_quat", "_body_quat_w"),
            ("body_lin_vel", "_body_lin_vel_w"),
            ("body_ang_vel", "_body_ang_vel_w"),
        ]
        if self.has_object:
            concat_targets.extend(
                [
                    ("object_pos", "_object_pos_w"),
                    ("object_quat", "_object_quat_w"),
                    ("object_lin_vel", "_object_lin_vel_w"),
                    ("object_ang_vel", "_object_ang_vel_w"),
                ]
            )

        for seg_key, attr_name in concat_targets:
            existing = getattr(self, attr_name)
            tensors = (segments[seg_key], existing) if prepend else (existing, segments[seg_key])
            setattr(self, attr_name, torch.cat(tensors, dim=0))

        self.time_step_total = self._joint_pos.shape[0]
        return self


def _resolve_motion_files(cfg: MotionConfig) -> list[str]:
    """Resolve motion files from config (motion_files > motion_dir > motion_file)."""
    if cfg.motion_manifest:
        if not cfg.motion_dir:
            raise ValueError("motion_manifest requires motion_dir")
        manifest_path = resolve_data_file_path(cfg.motion_manifest)
        entries = load_multibox_manifest(manifest_path)
        resolved_dir = Path(resolve_data_file_path(cfg.motion_dir))
        if not resolved_dir.exists():
            raise FileNotFoundError(f"Motion directory not found: {resolved_dir}")
        manifest_files = [resolved_dir / entry.file for entry in entries]
        missing = [str(path) for path in manifest_files if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Manifest motion files are missing ({len(missing)}): {missing[:5]}")
        discovered_names = {path.name for path in resolved_dir.glob(cfg.motion_glob)}
        manifest_names = {entry.file for entry in entries}
        if discovered_names != manifest_names:
            missing_from_manifest = sorted(discovered_names - manifest_names)
            missing_from_directory = sorted(manifest_names - discovered_names)
            raise ValueError(
                "Manifest/directory motion filename mismatch: "
                f"unlisted={missing_from_manifest[:5]}, missing={missing_from_directory[:5]}"
            )
        return [str(path) for path in manifest_files]

    if cfg.motion_files:
        return [resolve_data_file_path(p) for p in cfg.motion_files]

    if cfg.motion_dir:
        resolved_dir = Path(resolve_data_file_path(cfg.motion_dir))
        if not resolved_dir.exists():
            raise FileNotFoundError(f"Motion directory not found: {resolved_dir}")
        files = sorted(resolved_dir.glob(cfg.motion_glob))
        if not files:
            raise FileNotFoundError(f"No motion files found in {resolved_dir} with pattern {cfg.motion_glob}")
        return [str(p) for p in files]

    if cfg.motion_file:
        return [resolve_data_file_path(cfg.motion_file)]

    raise ValueError("No motion_file/motion_files/motion_dir provided in MotionConfig.")


class MotionBuffer:
    """Flatten multiple motion clips into a single global buffer for fast GPU indexing."""

    def __init__(self, motions: list[MotionLoader], device: str):
        if not motions:
            raise ValueError("MotionBuffer requires at least one MotionLoader.")

        fps = motions[0].fps
        has_object = motions[0].has_object
        for motion in motions[1:]:
            if motion.fps != fps:
                raise ValueError("All motion clips must share the same fps.")
            if motion.has_object != has_object:
                raise ValueError("All motion clips must consistently include (or exclude) object data.")

        self.fps = fps
        self.has_object = has_object
        self.motions = motions

        lengths = torch.tensor([m.time_step_total for m in motions], device=device, dtype=torch.long)
        if torch.any(lengths < 2):
            raise ValueError("All motion clips must have at least 2 frames.")
        start = torch.zeros_like(lengths)
        if lengths.numel() > 1:
            start[1:] = torch.cumsum(lengths[:-1], dim=0)

        self.motion_lengths = lengths
        self.motion_start_indices = start

        # Global buffers (concatenated in motion order).
        self.joint_pos = torch.cat([m.joint_pos for m in motions], dim=0)
        self.joint_vel = torch.cat([m.joint_vel for m in motions], dim=0)
        self.body_pos_w = torch.cat([m.body_pos_w for m in motions], dim=0)
        self.body_quat_w = torch.cat([m.body_quat_w for m in motions], dim=0)
        self.body_lin_vel_w = torch.cat([m.body_lin_vel_w for m in motions], dim=0)
        self.body_ang_vel_w = torch.cat([m.body_ang_vel_w for m in motions], dim=0)

        if self.has_object:
            self.object_pos_w = torch.cat([m.object_pos_w for m in motions], dim=0)
            self.object_quat_w = torch.cat([m.object_quat_w for m in motions], dim=0)
            self.object_lin_vel_w = torch.cat([m.object_lin_vel_w for m in motions], dim=0)
            self.object_ang_vel_w = torch.cat([m.object_ang_vel_w for m in motions], dim=0)

            # Per-frame object target positions (repeat constants if needed).
            targets: list[torch.Tensor] = []
            for motion in motions:
                target = getattr(motion, "object_target_pos_w", None)
                if isinstance(target, torch.Tensor) and target.numel() > 0:
                    if target.ndim == 1:
                        target = target.unsqueeze(0)
                    if target.shape[0] == 1:
                        target = target.repeat(motion.time_step_total, 1)
                    elif target.shape[0] != motion.time_step_total:
                        target = target[:1].repeat(motion.time_step_total, 1)
                else:
                    target = motion.object_pos_w[-1:].repeat(motion.time_step_total, 1)
                targets.append(target)
            self.object_target_pos_w = torch.cat(targets, dim=0)

class AdaptiveTimestepsSampler:
    """Prioritizes training on motion segments where the robot fails most often."""

    def __init__(
        self,
        motion_time_step_total: int,
        device: str,
        env_fps: int,
        bin_size_s: float = 1.0,
        kernel_size: int = 3,
        decay_lambda: float = 0.001,
        kernel_lambda: float = 0.8,
    ):
        # TODO: think better about the decay_lambda, will 0.001 be too small?
        self.device = device
        # length of the motion in rl environment time steps
        self.motion_time_step_total = motion_time_step_total
        # fps of the rl environment
        self.env_fps = env_fps

        # size of the bin in seconds
        self.bin_size_s = bin_size_s
        # size of the kernel for smoothing the sampling probabilities
        self.kernel_size = kernel_size
        self.kernel_lambda = kernel_lambda
        # exponential decay when updating the failure counts over training steps.

        self.decay_lambda = decay_lambda

        # number of bins in the motion
        self.num_bins = math.ceil((self.motion_time_step_total / self.env_fps) / self.bin_size_s)

        # initialize exponential 1d decay kernel, used for smoothing the failure counts over time.
        assert self.kernel_size % 2 == 1, "Kernel size must be odd"
        self.kernel = torch.tensor(
            [self.kernel_lambda ** abs(i) for i in range((-self.kernel_size + 1) // 2, (self.kernel_size + 1) // 2)],
            device=self.device,
        )
        self.kernel = self.kernel / self.kernel.sum()

        # key data: failure counts
        self.init_buffers()
        # metrics
        self.metrics: dict[str, torch.Tensor] = {}

    def init_buffers(self):
        self.current_bin_failed_count = torch.zeros(self.num_bins, dtype=torch.float, device=self.device)
        self.bin_failed_count = torch.zeros(self.num_bins, dtype=torch.float, device=self.device)

    def update_current_bin_failed_count(self, failed_at_time_step: torch.Tensor):
        """Update the current bin failed count with terminated time steps."""
        failed_bin = torch.floor(failed_at_time_step / self.motion_time_step_total * self.num_bins).long()
        assert failed_bin.min() >= 0 and failed_bin.max() < self.num_bins, "Failed bin is out of range"
        self.current_bin_failed_count[:] = torch.bincount(failed_bin, minlength=self.num_bins)

    def update_bin_failed_count(self):
        """At every rl environment step, update the failed count with the current bin failed count."""
        self.bin_failed_count = (self.decay_lambda * self.current_bin_failed_count) + (
            1 - self.decay_lambda
        ) * self.bin_failed_count
        self.current_bin_failed_count.zero_()

    @property
    def sampling_probabilities(self) -> torch.Tensor:
        sampling_probabilities = self.bin_failed_count + 1e-6
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)
        sampling_probabilities += 0.01
        return sampling_probabilities / sampling_probabilities.sum()

    def sample(self, num_samples: int) -> torch.Tensor:
        sampled_bins = torch.multinomial(self.sampling_probabilities, num_samples, replacement=True)
        # inside of each bin, randomly sample a time step, ignoring the borders
        return (sampled_bins + torch.rand(num_samples, device=self.device)) / self.num_bins

    def get_stats(self):
        # Metrics
        prob = self.sampling_probabilities
        H = -(prob * (prob + 1e-12).log()).sum()
        H_norm = H / np.log(self.num_bins)
        pmax, imax = prob.max(dim=0)
        self.metrics["sampling_entropy"] = H_norm
        self.metrics["sampling_top1_prob"] = pmax
        self.metrics["sampling_top1_bin"] = imax.float() / self.num_bins


#########################################################################################################
## Helper functions
#########################################################################################################
FAKE_BODY_NAME_ALIASES: dict[str, str] = {
    # Fake foot contact bodies are authored in the URDF purely for height computation.
    # They do not exist in the motion-capture dataset, so we alias them back to the
    # closest real body when indexing into motion data. These are not actually used in training.
    "left_foot_contact_point": "left_ankle_roll_link",
    "right_foot_contact_point": "right_ankle_roll_link",
}


def get_filtered_body_names(body_list: List[str], pattern: str) -> List[str]:
    return [body_name for body_name in body_list if re.match(pattern, body_name)]


def _parse_box_size_from_name(name: str) -> list[float] | None:
    if not name.startswith("box_"):
        return None
    parts = name[len("box_") :].split("_")
    if len(parts) < 3:
        return None
    sizes: list[float] = []
    for part in parts[:3]:
        try:
            sizes.append(float(part.replace("p", ".")))
        except ValueError:
            return None
    return sizes


class MotionCommand(CommandTermBase):
    def __init__(self, cfg: Any, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)

        self._env = env
        # self.motion_cfg: MotionConfig = cfg.params["motion_config"]
        # TODO(jchen):temporary fix for motion_config being a dict after tyro.cli
        if isinstance(cfg.params["motion_config"], MotionConfig):
            self.motion_cfg = cfg.params["motion_config"]
        else:
            self.motion_cfg = MotionConfig(**cfg.params["motion_config"])
        self.init_pose_cfg: NoiseToInitialPoseConfig = self.motion_cfg.noise_to_initial_pose

    def setup(self) -> None:
        self.num_envs = self._env.num_envs
        self.device = self._env.device

        robot_body_names = self._env.simulator._body_list  # type: ignore[attr-defined]
        robot_body_names_alias = [FAKE_BODY_NAME_ALIASES.get(bn, bn) for bn in robot_body_names]

        robot_joint_names = self._env.simulator.dof_names  # type: ignore[attr-defined]

        # 1. load motion data (support multiple motions)
        motion_files = _resolve_motion_files(self.motion_cfg)
        self.motion_files = motion_files
        self.motion_file_names = [Path(path).name for path in motion_files]
        self._setup_multibox_mapping(motion_files)
        self.motions: list[MotionLoader] = []
        for motion_file in motion_files:
            motion = MotionLoader(
                motion_file,
                robot_body_names_alias,
                robot_joint_names,
                device=self.device,
            )
            # Store body and joint indexes for interpolation
            self.motion = motion
            self._body_indexes_in_motion = motion._body_indexes
            self._joint_indexes_in_motion = motion._joint_indexes

            # Maybe prepend/append interpolated transition from default pose
            self._maybe_add_default_pose_transition(prepend=True)
            self._maybe_add_default_pose_transition(prepend=False)

            self.motions.append(motion)

        # Build flattened buffers for fast indexing
        self.motion_buffer = MotionBuffer(self.motions, device=self.device)
        self.motion = self.motions[0]  # keep a representative motion for legacy usages
        self._body_indexes_in_motion = self.motion._body_indexes
        self._joint_indexes_in_motion = self.motion._joint_indexes

        self.motion_start_indices = self.motion_buffer.motion_start_indices
        self.motion_lengths = self.motion_buffer.motion_lengths
        self.motion_fps = self.motion_buffer.fps
        self.motion_has_object = self.motion_buffer.has_object

        self.global_joint_pos = self.motion_buffer.joint_pos
        self.global_joint_vel = self.motion_buffer.joint_vel
        self.global_body_pos_w = self.motion_buffer.body_pos_w
        self.global_body_quat_w = self.motion_buffer.body_quat_w
        self.global_body_lin_vel_w = self.motion_buffer.body_lin_vel_w
        self.global_body_ang_vel_w = self.motion_buffer.body_ang_vel_w

        self.global_object_pos_w = None
        self.global_object_quat_w = None
        self.global_object_lin_vel_w = None
        self.global_object_ang_vel_w = None
        self.global_object_target_pos_w = None
        if self.motion_has_object:
            self.global_object_pos_w = self.motion_buffer.object_pos_w
            self.global_object_quat_w = self.motion_buffer.object_quat_w
            self.global_object_lin_vel_w = self.motion_buffer.object_lin_vel_w
            self.global_object_ang_vel_w = self.motion_buffer.object_ang_vel_w
            self.global_object_target_pos_w = self.motion_buffer.object_target_pos_w

        # 2. get the indexes of the root link and the tracked links
        self.ref_body_index = robot_body_names.index(self.motion_cfg.body_name_ref[0])  # int
        self.tracked_body_indexes = self._get_index_of_a_in_b(
            self.motion_cfg.body_names_to_track, robot_body_names, self.device
        )

        # 3. get the name of the object, or indices of the object
        if self.motion_has_object:
            # cache the object_index_in_simulator
            self.object_name = "object"  # hardcoded object name
            self.object_indices_in_simulator = self._env.simulator.get_actor_indices(self.object_name, env_ids=None)

            assert self._env.simulator.get_simulator_type() == SimulatorType.ISAACSIM, (
                "Object is only supported in IsaacSim"
            )
            self._init_object_bbox_corners_local()

        # 4. get the adaptive timesteps sampler(s)
        self.adaptive_timesteps_samplers: list[AdaptiveTimestepsSampler] = []
        if self.motion_cfg.use_adaptive_timesteps_sampler:
            for motion in self.motions:
                self.adaptive_timesteps_samplers.append(
                    AdaptiveTimestepsSampler(motion.time_step_total, self.device, int(1 / (self._env.dt)))
                )
            self.adaptive_timesteps_sampler = self.adaptive_timesteps_samplers[0]

        # 5. metrics
        self.metrics: dict[str, torch.Tensor] = {}

        self.init_buffers()

        # Precompute segment length in steps (if requested)
        self.segment_length_steps: int | None = None
        if self.motion_cfg.segment_length_s is not None:
            self.segment_length_steps = max(2, int(round(self.motion_cfg.segment_length_s * float(self.motion_fps))))

        # 6. visualization markers for isaacsim
        # Only create interactive debug markers when a viewer exists.
        # Headless video recording must remain offline-safe because
        # FRAME_MARKER_CFG references NVIDIA Nucleus assets.
        if self._env.simulator.get_simulator_type() == SimulatorType.ISAACSIM:
            if self._env.viewer is not None:
                self._setup_visualization_markers_for_isaacsim()

    def reset(self, env_ids: torch.Tensor | None) -> None:
        """called per reset_idx, reset timesteps and robot/object poses."""
        env_ids = self._ensure_index_tensor(env_ids)
        if env_ids.numel() == 0:
            return

        num_envs = env_ids.numel()

        # 0. Sample motion ids. Multi-box environments always sample within
        # their immutable physical-size bucket, including during evaluation.
        if self.motion_size_ids is not None:
            env_size_ids = self.env_asset_size_ids[env_ids]
            eligible_counts = self.eligible_motion_counts[env_size_ids]
            random_columns = torch.floor(
                torch.rand(num_envs, device=self.device) * eligible_counts.to(dtype=torch.float32)
            ).long()
            motion_ids = self.eligible_motion_ids[env_size_ids, random_columns]
            if self._env.is_evaluating:
                motion_ids = apply_multibox_eval_motion_id(
                    motion_ids,
                    env_size_ids,
                    self.motion_size_ids,
                    getattr(self.motion_cfg, "eval_motion_id", -1),
                )
            selected_size_ids = self.motion_size_ids[motion_ids]
            if not torch.equal(selected_size_ids, env_size_ids):
                raise RuntimeError(
                    "Multi-box invariant violation before reset: selected motion size does not match env asset size"
                )
        elif self._env.is_evaluating:
            eval_motion_id = getattr(self.motion_cfg, "eval_motion_id", 0)
            if eval_motion_id is None or eval_motion_id < 0:
                motion_ids = torch.randint(0, len(self.motions), (num_envs,), device=self.device)
            else:
                motion_ids = torch.full((num_envs,), int(eval_motion_id), device=self.device, dtype=torch.long)
                motion_ids = torch.clamp(motion_ids, 0, len(self.motions) - 1)
        else:
            motion_ids = torch.randint(0, len(self.motions), (num_envs,), device=self.device)
        self.motion_ids[env_ids] = motion_ids

        motion_lengths = self.motion_lengths[motion_ids]

        # 1. Determine segment length (in steps)
        if self.segment_length_steps is not None:
            seg_len = torch.minimum(
                motion_lengths, torch.full_like(motion_lengths, self.segment_length_steps)
            )
        else:
            seg_len = motion_lengths
        seg_len = torch.clamp(seg_len, min=2)

        # 2. Sample start steps
        eval_start_steps = None
        if self._env.is_evaluating:
            eval_start_steps = self._get_eval_start_steps(num_envs, motion_lengths=motion_lengths)

        if self._env.is_evaluating and eval_start_steps is None:
            # Eval default: always start from the beginning of each selected motion.
            start_steps = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        elif eval_start_steps is not None:
            start_steps = eval_start_steps
        elif self.segment_length_steps is not None:
            # Uniform sampling within the allowed segment range
            max_start = motion_lengths - seg_len
            rand = torch.rand(num_envs, device=self.device)
            start_steps = torch.floor(rand * (max_start.to(dtype=rand.dtype) + 1.0)).long()
        elif self.motion_cfg.use_adaptive_timesteps_sampler:
            phase = torch.zeros(num_envs, device=self.device)
            for motion_id in motion_ids.unique().tolist():
                mask = motion_ids == motion_id
                if not mask.any():
                    continue
                sampler = self.adaptive_timesteps_samplers[motion_id]
                phase[mask] = sampler.sample(int(mask.sum().item()))
            start_steps = (phase * (motion_lengths - 1).to(dtype=phase.dtype)).long()
        else:
            phase = torch.rand(num_envs, device=self.device)
            start_steps = (phase * (motion_lengths - 1).to(dtype=phase.dtype)).long()

        # Handle start_at_timestep_zero_prob
        prob = self.motion_cfg.start_at_timestep_zero_prob
        if eval_start_steps is None:
            if prob >= 1.0:
                start_steps[:] = 0
            elif prob > 0.0:
                rand_vals = torch.rand_like(start_steps, dtype=torch.float32)
                start_steps = torch.where(rand_vals < prob, torch.zeros_like(start_steps), start_steps)

        # If the motion is at the last timestep, set it to the second last timestep;
        # Otherwise, update_tasks_callback will advance the timestep to the next timestep -> out of bounds error.
        last_step = motion_lengths - 1
        start_steps = torch.where(start_steps == last_step, motion_lengths - 2, start_steps)

        self.time_steps[env_ids] = start_steps
        self.segment_end_steps[env_ids] = torch.minimum(start_steps + seg_len, motion_lengths)

        # 1. Get the reference root/body poses
        root_pos = self.body_pos_w[env_ids, 0].clone()
        root_rot = self.body_quat_w[env_ids, 0].clone()  # xyzw
        root_lin_vel = self.body_lin_vel_w[env_ids, 0].clone()
        root_ang_vel = self.body_ang_vel_w[env_ids, 0].clone()

        dof_pos = self.joint_pos[env_ids].clone()
        dof_vel = self.joint_vel[env_ids].clone()

        # 2. Adding noise
        # 2.1 prepare the noise scale
        dof_pos_noise = self.init_pose_cfg.dof_pos * self.init_pose_cfg.overall_noise_scale  # float
        root_pos_noise = (
            torch.tensor(
                self.init_pose_cfg.root_pos,
                device=self.device,
            )
            * self.init_pose_cfg.overall_noise_scale
        )  # (3,)
        root_rot_noise_rpy = (
            torch.tensor(
                self.init_pose_cfg.root_rot,
                device=self.device,
            )
            * self.init_pose_cfg.overall_noise_scale
        )  # (3,)
        root_vel_noise = (
            torch.tensor(
                self.init_pose_cfg.root_lin_vel,
                device=self.device,
            )
            * self.init_pose_cfg.overall_noise_scale
        )  # (3,)
        root_ang_vel_noise_rpy = (
            torch.tensor(
                self.init_pose_cfg.root_ang_vel,
                device=self.device,
            )
            * self.init_pose_cfg.overall_noise_scale
        )  # (3,)

        # 2.2 Adding noise to dof_pos, root_pos, root_vel, root_ang_vel, root_rot
        # 1.2.1 dof_pos
        target_dof_pos = (
            dof_pos + (torch.rand(dof_pos.shape, device=self.device) - 0.5) * 2 * dof_pos_noise
        )  # (num_envs, num_dofs)
        soft_joint_pos_limits = self._env.simulator.dof_pos_limits  # type: ignore[attr-defined]  # (num_dofs, 2)
        target_dof_pos = torch.clip(target_dof_pos, soft_joint_pos_limits[:, 0], soft_joint_pos_limits[:, 1])

        # 1.2.2 dof_vel no noise
        target_dof_vel = dof_vel

        # 1.2.3 root_pos
        target_root_pos = root_pos + (
            torch.rand(root_pos.shape, device=self.device) - 0.5
        ) * 2 * root_pos_noise.unsqueeze(0)  # (num_envs, 3)

        # 1.2.4 root_rot
        rand_sample_rpy = (torch.rand((len(env_ids), 3), device=self.device) - 0.5) * 2 * root_rot_noise_rpy
        orientations_delta = quat_from_euler_xyz(
            rand_sample_rpy[:, 0], rand_sample_rpy[:, 1], rand_sample_rpy[:, 2]
        )  # (num_envs, 4), xyzw
        target_root_rot = quat_mul(orientations_delta, root_rot, w_last=True)  # (num_envs, 4), xyzw

        # 1.2.5 root_lin_vel
        target_root_lin_vel = root_lin_vel + (
            torch.rand(root_lin_vel.shape, device=self.device) - 0.5
        ) * 2 * root_vel_noise.unsqueeze(0)  # (num_envs, 3)

        # 1.2.6 root_ang_vel
        target_root_ang_vel = root_ang_vel + (
            torch.rand(root_ang_vel.shape, device=self.device) - 0.5
        ) * 2 * root_ang_vel_noise_rpy.unsqueeze(0)  # (num_envs, 3)

        # 3. Set the robot states in simulator
        self._env.simulator.dof_pos[env_ids] = target_dof_pos
        self._env.simulator.dof_vel[env_ids] = target_dof_vel

        self._env.simulator.robot_root_states[env_ids, :3] = target_root_pos
        self._env.simulator.robot_root_states[env_ids, 3:7] = target_root_rot
        self._env.simulator.robot_root_states[env_ids, 7:10] = target_root_lin_vel
        self._env.simulator.robot_root_states[env_ids, 10:13] = target_root_ang_vel

        # 4. Set the object states in simulator
        if self.motion.has_object:
            obj_pos = self.object_pos_w[env_ids]
            obj_ori = self.object_quat_w[env_ids]
            obj_lin_vel = self.object_lin_vel_w[env_ids]
            obj_ang_vel = self.object_ang_vel_w[env_ids]

            # 4.2 add noise to the object states
            obj_pos_noise = torch.tensor(
                [self.init_pose_cfg.object_pos],
                device=self.device,
            )
            obj_pos_noise = obj_pos_noise * self.init_pose_cfg.overall_noise_scale  # (3,)
            target_obj_pos = obj_pos + (torch.rand(obj_pos.shape, device=self.device) - 0.5) * 2 * obj_pos_noise

            object_states = torch.cat([target_obj_pos, obj_ori, obj_lin_vel, obj_ang_vel], dim=-1)  # (num_envs, 13)
            # 4.3 set the object states in simulator
            self._env.simulator.set_actor_states([self.object_name], env_ids, object_states)

    def step(self) -> None:
        """called in _update_tasks_callback of the environment. (after compute_reward, before compute_observations)"""
        # 0. update time steps, all motion joint/body poses are updated automatically with the time steps.
        advance_mask = torch.ones_like(self.time_steps, dtype=torch.bool)

        # Handle freeze_at_timestep_zero_prob: for envs at timestep 0, randomly decide whether to advance
        freeze_prob = self.motion_cfg.freeze_at_timestep_zero_prob
        if freeze_prob > 0.0:
            zero_mask = self.time_steps == 0
            if zero_mask.any():
                rand_vals = torch.rand(self.num_envs, device=self.device)
                freeze_mask = (rand_vals < freeze_prob) & zero_mask
                advance_mask = advance_mask & ~freeze_mask

        self.time_steps += advance_mask.long()
        max_steps = self.motion_lengths[self.motion_ids] - 1
        self.time_steps = torch.minimum(self.time_steps, max_steps)

        # 1. update body_pos_relative_w and body_quat_relative_w
        # definition of body_pos/quat_relative_w:
        # If I take this motion data and adapt it to where my robot currently is
        # (accounting for position(x, y) offset and yaw difference of a reference body),
        # what should each body part's target pose be?

        ## 1.0 get the reference body poses

        # Issue (This is a isaacgym only issue.):
        # ------------------------------------------------------------
        # In isaacgym, immediately after reset (self._env.episode_length_buf == 0), calling
        # simulator.set_actor_root_state_tensor and simulator.set_dof_state_tensor will reset
        # the robot_root_pos_w and robot_root_quat_w successfully.
        # However, the robot_body_pos_w and robot_body_quat_w are not updated successfully,
        # (since kinematic forward has not been applied yet).
        # Therefore, using robot_ref_pos_w and robot_ref_quat_w as reference body poses is not resetted correctly.

        # Solution:
        # ------------------------------------------------------------
        # if episode_length_buf == 0, use robot_root_pos_w and robot_root_quat_w as reference body.
        # else, use configured reference body as reference body.
        use_root = (self._env.episode_length_buf == 0).unsqueeze(1).float()

        ref_pos_w = self.root_pos_w * use_root + self.ref_pos_w * (1 - use_root)
        ref_quat_w = self.root_quat_w * use_root + self.ref_quat_w * (1 - use_root)
        robot_ref_pos_w = self.robot_root_pos_w * use_root + self.robot_ref_pos_w * (1 - use_root)
        robot_ref_quat_w = self.robot_root_quat_w * use_root + self.robot_ref_quat_w * (1 - use_root)

        ## 1.1 repeat to match the number of body parts
        ref_pos_w_repeat = ref_pos_w[:, None, :].repeat(1, len(self.motion_cfg.body_names_to_track), 1)  # type: ignore[arg-type]
        ref_quat_w_repeat = ref_quat_w[:, None, :].repeat(1, len(self.motion_cfg.body_names_to_track), 1)  # type: ignore[arg-type]
        robot_ref_pos_w_repeat = robot_ref_pos_w[:, None, :].repeat(1, len(self.motion_cfg.body_names_to_track), 1)  # type: ignore[arg-type]
        robot_ref_quat_w_repeat = robot_ref_quat_w[:, None, :].repeat(1, len(self.motion_cfg.body_names_to_track), 1)  # type: ignore[arg-type]

        ## 1.2 compute the relative body poses
        delta_quat_w = yaw_quat(
            quat_mul(robot_ref_quat_w_repeat, quat_inverse(ref_quat_w_repeat, w_last=True), w_last=True), w_last=True
        )
        ### 1.2.1 body_quat_relative_w
        self.body_quat_relative_w = quat_mul(delta_quat_w, self.body_quat_w, w_last=True)
        ### 1.2.2 body_pos_relative_w
        delta_pos_w_height = ref_pos_w_repeat - robot_ref_pos_w_repeat
        delta_pos_w_height[..., :2] = 0.0  # adjusting for height differences
        self.body_pos_relative_w = (
            robot_ref_pos_w_repeat
            + delta_pos_w_height
            + quat_apply(delta_quat_w, self.body_pos_w - ref_pos_w_repeat, w_last=True)
        )

        ### 1.3 update the adaptive timesteps sampler
        if self.motion_cfg.use_adaptive_timesteps_sampler:
            for sampler in self.adaptive_timesteps_samplers:
                sampler.update_bin_failed_count()

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    def _absolute_indices(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if env_ids is None:
            motion_ids = self.motion_ids
            time_steps = self.time_steps
        else:
            motion_ids = self.motion_ids[env_ids]
            time_steps = self.time_steps[env_ids]
        start = self.motion_start_indices[motion_ids]
        return start + time_steps

    #########################################################################################
    ## Robot from motion data
    #########################################################################################
    @property
    def joint_pos(self) -> torch.Tensor:
        absolute_idx = self._absolute_indices()
        return self.global_joint_pos[absolute_idx]

    @property
    def joint_vel(self) -> torch.Tensor:
        absolute_idx = self._absolute_indices()
        return self.global_joint_vel[absolute_idx]

    @property
    def body_pos_w(self) -> torch.Tensor:
        absolute_idx = self._absolute_indices()
        env_origins = self._env.simulator.scene.env_origins
        pos = self.global_body_pos_w[absolute_idx][:, self.tracked_body_indexes]
        return pos + env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        absolute_idx = self._absolute_indices()
        return self.global_body_quat_w[absolute_idx][:, self.tracked_body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        absolute_idx = self._absolute_indices()
        return self.global_body_lin_vel_w[absolute_idx][:, self.tracked_body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        absolute_idx = self._absolute_indices()
        return self.global_body_ang_vel_w[absolute_idx][:, self.tracked_body_indexes]

    @property
    def ref_pos_w(self) -> torch.Tensor:
        absolute_idx = self._absolute_indices()
        env_origins = self._env.simulator.scene.env_origins
        return self.global_body_pos_w[absolute_idx, self.ref_body_index] + env_origins

    @property
    def ref_quat_w(self) -> torch.Tensor:
        absolute_idx = self._absolute_indices()
        return self.global_body_quat_w[absolute_idx, self.ref_body_index]

    @property
    def root_pos_w(self) -> torch.Tensor:
        absolute_idx = self._absolute_indices()
        env_origins = self._env.simulator.scene.env_origins
        return self.global_body_pos_w[absolute_idx, 0] + env_origins

    @property
    def root_quat_w(self) -> torch.Tensor:
        absolute_idx = self._absolute_indices()
        return self.global_body_quat_w[absolute_idx, 0]

    @property
    def ref_lin_vel_w(self) -> torch.Tensor:
        absolute_idx = self._absolute_indices()
        return self.global_body_lin_vel_w[absolute_idx, self.ref_body_index]

    @property
    def ref_ang_vel_w(self) -> torch.Tensor:
        absolute_idx = self._absolute_indices()
        return self.global_body_ang_vel_w[absolute_idx, self.ref_body_index]

    #########################################################################################
    ## Robot from simulator
    #########################################################################################
    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self._env.simulator.dof_pos  # (num_envs, num_dofs)

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self._env.simulator.dof_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_pos[:, self.tracked_body_indexes, :]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_rot[:, self.tracked_body_indexes, :]  # xyzw

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_vel[:, self.tracked_body_indexes, :]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_ang_vel[:, self.tracked_body_indexes, :]

    @property
    def robot_root_pos_w(self) -> torch.Tensor:
        return self._env.simulator.robot_root_states[:, :3]  # type: ignore[attr-defined]

    @property
    def robot_root_quat_w(self) -> torch.Tensor:
        return self._env.simulator.robot_root_states[:, 3:7]  # type: ignore[attr-defined]

    @property
    def robot_root_lin_vel_w(self) -> torch.Tensor:
        return self._env.simulator.robot_root_states[:, 7:10]  # type: ignore[attr-defined]

    @property
    def robot_root_ang_vel_w(self) -> torch.Tensor:
        return self._env.simulator.robot_root_states[:, 10:13]  # type: ignore[attr-defined]

    @property
    def robot_ref_pos_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_pos[:, self.ref_body_index, :]

    @property
    def robot_ref_quat_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_rot[:, self.ref_body_index, :]  # xyzw

    @property
    def robot_ref_lin_vel_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_vel[:, self.ref_body_index, :]

    @property
    def robot_ref_ang_vel_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_ang_vel[:, self.ref_body_index, :]

    #########################################################################################
    ## Object from motion data
    #########################################################################################
    @property
    def object_pos_w(self) -> torch.Tensor:
        # Applies env origins, but ideally we should rely on the simulator
        if self.global_object_pos_w is None:
            return torch.zeros(self.num_envs, 3, device=self.device)
        absolute_idx = self._absolute_indices()
        env_origins = self._env.simulator.scene.env_origins
        return self.global_object_pos_w[absolute_idx] + env_origins

    @property
    def object_quat_w(self) -> torch.Tensor:
        if self.global_object_quat_w is None:
            return torch.zeros(self.num_envs, 4, device=self.device)
        absolute_idx = self._absolute_indices()
        return self.global_object_quat_w[absolute_idx]

    @property
    def object_lin_vel_w(self) -> torch.Tensor:
        if self.global_object_lin_vel_w is None:
            return torch.zeros(self.num_envs, 3, device=self.device)
        absolute_idx = self._absolute_indices()
        return self.global_object_lin_vel_w[absolute_idx]

    @property
    def object_ang_vel_w(self) -> torch.Tensor:
        if self.global_object_ang_vel_w is None:
            return torch.zeros(self.num_envs, 3, device=self.device)
        absolute_idx = self._absolute_indices()
        return self.global_object_ang_vel_w[absolute_idx]

    @property
    def object_target_pos_w(self) -> torch.Tensor:
        if self.global_object_target_pos_w is None:
            return torch.zeros(self.num_envs, 3, device=self.device)
        absolute_idx = self._absolute_indices()
        env_origins = self._env.simulator.scene.env_origins
        target = self.global_object_target_pos_w[absolute_idx]
        return target + env_origins

    @property
    def object_bbox_corners_local(self) -> torch.Tensor:
        if self._object_bbox_corners_local is None:
            return torch.zeros(0, 3, device=self.device)
        return self._object_bbox_corners_local

    #########################################################################################
    ## Object from simulator
    #########################################################################################
    @property
    def simulator_object_pos_w(self) -> torch.Tensor:
        return self._env.simulator.all_root_states[self.object_indices_in_simulator][:, :3]

    @property
    def simulator_object_quat_w(self) -> torch.Tensor:
        return self._env.simulator.all_root_states[self.object_indices_in_simulator][:, 3:7]

    @property
    def simulator_object_lin_vel_w(self) -> torch.Tensor:
        return self._env.simulator.all_root_states[self.object_indices_in_simulator][:, 7:10]

    @property
    def simulator_object_ang_vel_w(self) -> torch.Tensor:
        return self._env.simulator.all_root_states[self.object_indices_in_simulator][:, 10:13]

    #########################################################################################
    ## Methods that does not fit into setup/step/reset pattern
    #########################################################################################

    def init_buffers(self):
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.segment_end_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(
            self.num_envs, len(self.motion_cfg.body_names_to_track), 3, device=self.device
        )  # type: ignore[arg-type]
        self.body_quat_relative_w = torch.zeros(
            self.num_envs, len(self.motion_cfg.body_names_to_track), 4, device=self.device
        )  # type: ignore[arg-type]
        self.body_quat_relative_w[:, :, 0] = 1.0

        if self.motion_cfg.use_adaptive_timesteps_sampler:
            for sampler in self.adaptive_timesteps_samplers:
                sampler.init_buffers()

    def _init_object_target(self) -> None:
        target = getattr(self.motion, "object_target_pos_w", None)
        if isinstance(target, torch.Tensor) and target.numel() > 0:
            self._object_target_pos_w = target
        else:
            self._object_target_pos_w = self.motion.object_pos_w[-1:].clone()

    def _init_object_bbox_corners_local(self) -> None:
        self._object_bbox_corners_local = None
        env_dimensions = getattr(self._env.simulator, "env_object_dimensions", None)
        if env_dimensions is not None:
            if env_dimensions.shape != (self.num_envs, 3):
                raise RuntimeError(
                    f"Expected per-env object dimensions [{self.num_envs}, 3], got {tuple(env_dimensions.shape)}"
                )
            signs = torch.tensor(
                [
                    [-1.0, -1.0, -1.0],
                    [-1.0, -1.0, 1.0],
                    [-1.0, 1.0, -1.0],
                    [-1.0, 1.0, 1.0],
                    [1.0, -1.0, -1.0],
                    [1.0, -1.0, 1.0],
                    [1.0, 1.0, -1.0],
                    [1.0, 1.0, 1.0],
                ],
                device=self.device,
            )
            self._object_bbox_corners_local = signs.unsqueeze(0) * env_dimensions[:, None, :] * 0.5
            return
        half_extents = self._infer_object_half_extents()
        if half_extents is None:
            return
        signs = torch.tensor(
            [
                [-1.0, -1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, 1.0, 1.0],
                [1.0, -1.0, -1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, -1.0],
                [1.0, 1.0, 1.0],
            ],
            device=self.device,
        )
        self._object_bbox_corners_local = signs * half_extents

    def _infer_object_half_extents(self) -> torch.Tensor | None:
        object_cfg = getattr(self._env.robot_config, "object", None)
        if object_cfg is None or not object_cfg.object_urdf_path:
            return None
        urdf_path = Path(object_cfg.object_urdf_path)
        size = _parse_box_size_from_name(urdf_path.stem)
        if size is None:
            size = _parse_box_size_from_name(urdf_path.parent.name)
        if size is None:
            return None
        return torch.tensor(size, dtype=torch.float32, device=self.device) * 0.5

    def _setup_multibox_mapping(self, motion_files: list[str]) -> None:
        """Validate the manifest and prepare GPU bucket sampling tensors."""
        self.motion_size_ids: torch.Tensor | None = None
        self.eligible_motion_ids: torch.Tensor | None = None
        self.eligible_motion_counts: torch.Tensor | None = None
        self.env_asset_size_ids: torch.Tensor | None = None
        self.motion_manifest_dimensions: list[tuple[float, float, float]] | None = None

        configured_dimensions = getattr(self._env.simulator, "multibox_asset_dimensions", None)
        if configured_dimensions is None:
            if self.motion_cfg.motion_manifest:
                raise RuntimeError("A motion_manifest requires a simulator multi-box asset configuration")
            return
        if not self.motion_cfg.motion_manifest:
            raise RuntimeError("Multi-box assets require motion_config.motion_manifest")
        entries = load_multibox_manifest(resolve_data_file_path(self.motion_cfg.motion_manifest))
        manifest_names = [entry.file for entry in entries]
        loaded_names = [Path(path).name for path in motion_files]
        if manifest_names != loaded_names:
            raise ValueError("Loaded motion order does not exactly match manifest row order")

        configured_dimensions = [tuple(size) for size in configured_dimensions]
        motion_size_ids, motions_by_size = map_manifest_sizes(entries, configured_dimensions)
        self.motion_manifest_dimensions = [entry.dimensions for entry in entries]
        self.motion_size_ids = torch.tensor(motion_size_ids, dtype=torch.long, device=self.device)
        self.env_asset_size_ids = getattr(self._env.simulator, "env_asset_size_ids", None)
        if self.env_asset_size_ids is None or self.env_asset_size_ids.shape != (self.num_envs,):
            raise RuntimeError("Simulator did not provide one canonical multi-box asset size ID per environment")

        max_bucket_size = max(len(ids) for ids in motions_by_size)
        padded = torch.full(
            (len(motions_by_size), max_bucket_size), -1, dtype=torch.long, device=self.device
        )
        for size_id, motion_ids_for_size in enumerate(motions_by_size):
            padded[size_id, : len(motion_ids_for_size)] = torch.tensor(
                motion_ids_for_size, dtype=torch.long, device=self.device
            )
        self.eligible_motion_ids = padded
        self.eligible_motion_counts = torch.tensor(
            [len(ids) for ids in motions_by_size], dtype=torch.long, device=self.device
        )

    def update_metrics(self):
        """Update the metrics. After action, before step() is called."""
        self.metrics["motion/error_ref_pos"] = torch.norm(self.ref_pos_w - self.robot_ref_pos_w, dim=-1)
        self.metrics["motion/error_ref_rot"] = quat_error_magnitude(self.ref_quat_w, self.robot_ref_quat_w)
        self.metrics["motion/error_ref_lin_vel"] = torch.norm(self.ref_lin_vel_w - self.robot_ref_lin_vel_w, dim=-1)
        self.metrics["motion/error_ref_ang_vel"] = torch.norm(self.ref_ang_vel_w - self.robot_ref_ang_vel_w, dim=-1)

        self.metrics["motion/error_body_pos"] = torch.norm(
            self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
        ).mean(dim=-1)

        self.metrics["motion/error_body_rot"] = quat_error_magnitude(
            self.body_quat_relative_w, self.robot_body_quat_w
        ).mean(dim=-1)

        self.metrics["motion/error_body_lin_vel"] = torch.norm(
            self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
        ).mean(dim=-1)
        self.metrics["motion/error_body_ang_vel"] = torch.norm(
            self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
        ).mean(dim=-1)

        self.metrics["motion/error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["motion/error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

        if self.motion_cfg.use_adaptive_timesteps_sampler and self.adaptive_timesteps_samplers:
            entropies = []
            top1_probs = []
            top1_bins = []
            for sampler in self.adaptive_timesteps_samplers:
                sampler.get_stats()
                entropies.append(sampler.metrics["sampling_entropy"])
                top1_probs.append(sampler.metrics["sampling_top1_prob"])
                top1_bins.append(sampler.metrics["sampling_top1_bin"])
            self.metrics["motion/adaptive_timesteps_sampler_entropy"] = torch.stack(entropies).mean()
            self.metrics["motion/adaptive_timesteps_sampler_top1_prob"] = torch.stack(top1_probs).mean()
            self.metrics["motion/adaptive_timesteps_sampler_top1_bin"] = torch.stack(top1_bins).mean()

    #########################################################################################
    ## Internal helpers
    #########################################################################################
    def _get_eval_start_steps(
        self, batch_size: int, motion_lengths: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        cfg = self.motion_cfg
        if cfg.eval_start_time_range_s is not None:
            t_min, t_max = cfg.eval_start_time_range_s
            t_min = max(t_min, 0.0)
            t_max = max(t_max, t_min)
            times = torch.rand(batch_size, device=self.device) * (t_max - t_min) + t_min
            return self._seconds_to_timestep(times, motion_lengths=motion_lengths)

        if cfg.eval_start_time_s is not None:
            times = torch.full((batch_size,), cfg.eval_start_time_s, device=self.device)
            return self._seconds_to_timestep(times, motion_lengths=motion_lengths)

        if cfg.eval_start_timestep_range is not None:
            t_min, t_max = cfg.eval_start_timestep_range
            t_min = max(t_min, 0)
            t_max = max(t_max, t_min)
            steps = torch.randint(t_min, t_max + 1, (batch_size,), device=self.device)
            if motion_lengths is not None:
                max_steps = torch.clamp(motion_lengths - 2, min=0)
                steps = torch.minimum(steps, max_steps)
            return steps

        if cfg.eval_start_timestep is not None:
            steps = torch.full((batch_size,), int(cfg.eval_start_timestep), device=self.device, dtype=torch.long)
            if motion_lengths is not None:
                max_steps = torch.clamp(motion_lengths - 2, min=0)
                steps = torch.minimum(steps, max_steps)
            return steps

        return None

    def _seconds_to_timestep(
        self, times_s: torch.Tensor, motion_lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        fps = getattr(self, "motion_fps", 1.0)
        steps = torch.round(times_s * float(fps)).to(dtype=torch.long)
        if motion_lengths is None:
            max_step = max(int(self.motion_lengths[0].item()) - 2, 0) if hasattr(self, "motion_lengths") else 0
            return torch.clamp(steps, 0, max_step)
        max_steps = torch.clamp(motion_lengths - 2, min=0)
        return torch.minimum(steps, max_steps)

    def _maybe_add_default_pose_transition(self, *, prepend: bool) -> None:
        """Shared path for optionally inserting default-pose interpolation before/after the clip."""
        enabled = self.motion_cfg.enable_default_pose_prepend if prepend else self.motion_cfg.enable_default_pose_append
        if not enabled:
            return

        duration = (
            self.motion_cfg.default_pose_prepend_duration_s
            if prepend
            else self.motion_cfg.default_pose_append_duration_s
        )
        if duration <= 0.0:
            return

        num_steps = round(duration / self._env.dt)
        if num_steps <= 1:
            logger.warning(
                "Default pose {} duration {}s is too short for dt {}; skipping augmentation.",
                "prepend" if prepend else "append",
                duration,
                self._env.dt,
            )
            return

        default_state = self._build_default_pose_state(use_motion_end=not prepend)

        action = "prepend" if prepend else "append"
        log_str = f"{action} {num_steps} interpolated frames ({duration}s) from default pose to motion"
        try:
            self._add_transition_to_motion(default_state, num_steps, prepend=prepend)
            logger.info(log_str)
        except Exception as exc:
            logger.error(f"Failed to {action} default pose transition: {exc}")
            raise RuntimeError(
                f"Critical error during motion interpolation setup: {exc}\n"
                "This indicates a mismatch in tensor dimensions during interpolation. "
                "Please check that the motion file and robot configuration are compatible."
            ) from exc

    def _build_default_pose_state(self, use_motion_end: bool = False) -> dict[str, torch.Tensor]:
        """Build the state dict representing the robot's default standing pose.

        By default, anchor root pos/yaw to the motion start; when use_motion_end is True, anchor to motion end.
        """
        init_state = self._env.robot_config.init_state
        joint_pos = self._env.default_dof_pos_base.squeeze(0).to(self.device)
        joint_vel = torch.zeros_like(joint_pos)

        init_root_quat = torch.tensor(init_state.rot, dtype=torch.float32, device=self.device).unsqueeze(0)
        init_roll, init_pitch, _ = get_euler_xyz(init_root_quat, w_last=True)

        motion_idx = -1 if use_motion_end else 0

        # Assume the pelvis is the first in robot_body_names
        motion_root_pos = self.motion.body_pos_w[motion_idx, 0].to(self.device)
        motion_root_quat = self.motion.body_quat_w[motion_idx, 0].to(self.device).unsqueeze(0)
        _, _, motion_yaw = get_euler_xyz(motion_root_quat, w_last=True)

        # Keep z from init config but adopt the clip's x,y at the chosen anchor frame.
        default_root_pos = torch.tensor(
            [motion_root_pos[0], motion_root_pos[1], init_state.pos[2]],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        # Keep roll/pitch from init config but adopt the clip's yaw at the chosen anchor frame.
        default_root_quat = quat_from_euler_xyz(
            init_roll.squeeze(0),
            init_pitch.squeeze(0),
            motion_yaw.squeeze(0),
        )
        default_root_lin_vel = torch.tensor(init_state.lin_vel, dtype=torch.float32, device=self.device)
        default_root_ang_vel = torch.tensor(init_state.ang_vel, dtype=torch.float32, device=self.device)

        body_states = self._capture_body_states(
            joint_pos,
            joint_vel,
            default_root_pos,
            default_root_quat,
            default_root_lin_vel,
            default_root_ang_vel,
        )

        default_body_pos = self._map_robot_bodies_to_motion_order(body_states["pos"])
        default_body_quat = self._map_robot_bodies_to_motion_order(body_states["quat"])
        default_body_lin_vel = self._map_robot_bodies_to_motion_order(body_states["lin_vel"])
        default_body_ang_vel = self._map_robot_bodies_to_motion_order(body_states["ang_vel"])

        if self.motion.has_object:
            object_pos = self.motion._object_pos_w[motion_idx].to(self.device)
            object_quat = self.motion._object_quat_w[motion_idx].to(self.device)
            object_lin_vel = self.motion._object_lin_vel_w[motion_idx].to(self.device)
            object_ang_vel = self.motion._object_ang_vel_w[motion_idx].to(self.device)
        else:
            object_pos = torch.zeros(0, 3, device=self.device, dtype=torch.float32)
            object_quat = torch.zeros(0, 4, device=self.device, dtype=torch.float32)
            object_lin_vel = torch.zeros(0, 3, device=self.device, dtype=torch.float32)
            object_ang_vel = torch.zeros(0, 3, device=self.device, dtype=torch.float32)

        return {
            "joint_pos": joint_pos.clone(),
            "joint_vel": joint_vel,
            "root_pos": default_root_pos,
            "root_quat": default_root_quat,
            "root_lin_vel": default_root_lin_vel,
            "root_ang_vel": default_root_ang_vel,
            "body_pos": default_body_pos,
            "body_quat": default_body_quat,
            "body_lin_vel": default_body_lin_vel,
            "body_ang_vel": default_body_ang_vel,
            "object_pos": object_pos,
            "object_quat": object_quat,
            "object_lin_vel": object_lin_vel,
            "object_ang_vel": object_ang_vel,
        }

    def _add_transition_to_motion(self, default_state: dict[str, torch.Tensor], num_steps: int, prepend: bool) -> None:
        """Add interpolated frames either before or after the motion data."""
        assert self._body_indexes_in_motion is not None
        assert self._joint_indexes_in_motion is not None

        if num_steps <= 0:
            return

        device = self.device
        dtype = self.motion._joint_pos.dtype

        default_motion_state = self._default_motion_state(default_state, dtype=dtype, device=device)
        motion_state = self._motion_state(0 if prepend else -1, dtype=dtype, device=device)

        start_state = default_motion_state if prepend else motion_state
        target_state = motion_state if prepend else default_motion_state
        drop_first, drop_last = (False, True) if prepend else (True, False)

        self._build_and_apply_transition(
            start_state=start_state,
            target_state=target_state,
            num_steps=num_steps,
            prepend=prepend,
            drop_first=drop_first,
            drop_last=drop_last,
            dtype=dtype,
            device=device,
        )

    def _slerp_quat_sequence(self, start: torch.Tensor, end: torch.Tensor, alphas: torch.Tensor) -> torch.Tensor:
        """Spherically interpolate quaternions across multiple time steps."""
        if alphas.numel() == 0:
            return start.new_zeros((0,) + start.shape)

        num_steps = alphas.shape[0]
        start_expand = start.unsqueeze(0).expand(num_steps, -1, -1)
        end_expand = end.unsqueeze(0).expand(num_steps, -1, -1)
        alpha_flat = alphas.repeat_interleave(start.shape[0]).unsqueeze(-1)
        blended = slerp(
            start_expand.reshape(-1, 4),
            end_expand.reshape(-1, 4),
            alpha_flat,
        )
        return blended.view(num_steps, start.shape[0], 4)

    def _capture_body_states(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        root_lin_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Capture body states by temporarily setting the robot state in the simulator."""
        simulator = self._env.simulator
        assert simulator.get_simulator_type() == SimulatorType.ISAACSIM, (
            "Default-pose interpolation only supports IsaacSim; IsaacGym write_state_updates does not run FK."
        )
        env_id = 0
        env_origin = simulator.scene.env_origins[env_id].to(self.device)

        root_backup = simulator.robot_root_states[env_id].clone()
        dof_pos_backup = simulator.dof_pos[env_id].clone()
        dof_vel_backup = simulator.dof_vel[env_id].clone()

        try:
            simulator.robot_root_states[env_id, :3] = root_pos + env_origin
            simulator.robot_root_states[env_id, 3:7] = root_quat
            simulator.robot_root_states[env_id, 7:10] = root_lin_vel
            simulator.robot_root_states[env_id, 10:13] = root_ang_vel
            simulator.dof_pos[env_id] = joint_pos
            simulator.dof_vel[env_id] = joint_vel

            simulator.set_actor_root_state_tensor_robots()
            simulator.set_dof_state_tensor_robots()
            simulator.write_state_updates()
            simulator.refresh_sim_tensors()

            body_pos = (simulator._rigid_body_pos[env_id] - env_origin).clone()
            body_quat = simulator._rigid_body_rot[env_id].clone()
            body_lin_vel = simulator._rigid_body_vel[env_id].clone()
            body_ang_vel = simulator._rigid_body_ang_vel[env_id].clone()
        finally:
            simulator.robot_root_states[env_id] = root_backup
            simulator.dof_pos[env_id] = dof_pos_backup
            simulator.dof_vel[env_id] = dof_vel_backup
            simulator.set_actor_root_state_tensor_robots()
            simulator.set_dof_state_tensor_robots()
            simulator.write_state_updates()
            simulator.refresh_sim_tensors()

        return {
            "pos": body_pos,
            "quat": body_quat,
            "lin_vel": body_lin_vel,
            "ang_vel": body_ang_vel,
        }

    def _map_robot_bodies_to_motion_order(self, robot_tensor: torch.Tensor) -> torch.Tensor:
        """Map robot body tensor to motion data order using body indexes."""
        assert self._body_indexes_in_motion is not None
        num_motion_bodies = self.motion._body_pos_w.shape[1]
        motion_shape = (num_motion_bodies,) + robot_tensor.shape[1:]
        motion_tensor = torch.zeros(motion_shape, device=robot_tensor.device, dtype=robot_tensor.dtype)
        motion_tensor[self._body_indexes_in_motion] = robot_tensor
        return motion_tensor

    def _map_robot_joints_to_motion_order(
        self, robot_tensor: torch.Tensor, num_motion_joints: int | None = None
    ) -> torch.Tensor:
        """Map robot joint tensor to motion data order using joint indexes."""
        assert self._joint_indexes_in_motion is not None
        if num_motion_joints is None:
            num_motion_joints = self.motion._joint_pos.shape[1]
        motion_shape = robot_tensor.shape[:-1] + (num_motion_joints,)
        motion_tensor = torch.zeros(motion_shape, device=robot_tensor.device, dtype=robot_tensor.dtype)
        motion_tensor[..., self._joint_indexes_in_motion] = robot_tensor
        return motion_tensor

    def _motion_state(self, idx: int, dtype: torch.dtype, device: torch.device) -> dict[str, torch.Tensor]:
        """Slice motion tensors at a given index into a state dict."""
        state = {
            "joint_pos": self.motion._joint_pos[idx].to(device=device, dtype=dtype),
            "joint_vel": self.motion._joint_vel[idx].to(device=device, dtype=dtype),
            "body_pos": self.motion._body_pos_w[idx].to(device=device, dtype=dtype),
            "body_quat": self.motion._body_quat_w[idx].to(device=device, dtype=dtype),
            "body_lin_vel": self.motion._body_lin_vel_w[idx].to(device=device, dtype=dtype),
            "body_ang_vel": self.motion._body_ang_vel_w[idx].to(device=device, dtype=dtype),
        }
        if self.motion.has_object:
            state["object_pos"] = self.motion._object_pos_w[idx].to(device=device, dtype=dtype)
            state["object_quat"] = self.motion._object_quat_w[idx].to(device=device, dtype=dtype)
            state["object_lin_vel"] = self.motion._object_lin_vel_w[idx].to(device=device, dtype=dtype)
            state["object_ang_vel"] = self.motion._object_ang_vel_w[idx].to(device=device, dtype=dtype)
        return state

    def _default_motion_state(
        self, default_state: dict[str, torch.Tensor], dtype: torch.dtype, device: torch.device
    ) -> dict[str, torch.Tensor]:
        """Map default robot-state tensors into motion order for interpolation."""
        state = {
            "joint_pos": self._map_robot_joints_to_motion_order(
                default_state["joint_pos"].to(device=device, dtype=dtype),
                num_motion_joints=self.motion._joint_pos.shape[1],
            ),
            "joint_vel": self._map_robot_joints_to_motion_order(
                default_state["joint_vel"].to(device=device, dtype=dtype),
                num_motion_joints=self.motion._joint_vel.shape[1],
            ),
            "body_pos": default_state["body_pos"].to(device=device, dtype=dtype),
            "body_quat": default_state["body_quat"].to(device=device, dtype=dtype),
            "body_lin_vel": default_state["body_lin_vel"].to(device=device, dtype=dtype),
            "body_ang_vel": default_state["body_ang_vel"].to(device=device, dtype=dtype),
        }
        if self.motion.has_object:
            state["object_pos"] = default_state["object_pos"].to(device=device, dtype=dtype)
            state["object_quat"] = default_state["object_quat"].to(device=device, dtype=dtype)
            state["object_lin_vel"] = default_state["object_lin_vel"].to(device=device, dtype=dtype)
            state["object_ang_vel"] = default_state["object_ang_vel"].to(device=device, dtype=dtype)
        return state

    def _build_transition_segments(
        self,
        start: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        alphas: torch.Tensor,
        alphas_joint: torch.Tensor,
        alphas_body: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Linearly/spherically interpolate between start and target states."""

        def _lerp(a: torch.Tensor, b: torch.Tensor, view: torch.Tensor) -> torch.Tensor:
            return a.unsqueeze(0) + view * (b - a).unsqueeze(0)

        segments = {
            "joint_pos": _lerp(start["joint_pos"], target["joint_pos"], alphas_joint),
            "joint_vel": _lerp(start["joint_vel"], target["joint_vel"], alphas_joint),
            "body_pos": _lerp(start["body_pos"], target["body_pos"], alphas_body),
            "body_lin_vel": _lerp(start["body_lin_vel"], target["body_lin_vel"], alphas_body),
            "body_ang_vel": _lerp(start["body_ang_vel"], target["body_ang_vel"], alphas_body),
            "body_quat": self._slerp_quat_sequence(start["body_quat"], target["body_quat"], alphas),
        }

        if self.motion.has_object:
            segments["object_pos"] = _lerp(start["object_pos"], target["object_pos"], alphas_joint)
            segments["object_lin_vel"] = _lerp(start["object_lin_vel"], target["object_lin_vel"], alphas_joint)
            segments["object_ang_vel"] = _lerp(start["object_ang_vel"], target["object_ang_vel"], alphas_joint)
            segments["object_quat"] = self._slerp_quat_sequence(
                start["object_quat"].unsqueeze(0), target["object_quat"].unsqueeze(0), alphas
            ).squeeze(1)

        return segments

    def _apply_transition_segments(self, segments: dict[str, torch.Tensor], prepend: bool) -> None:
        """Splice interpolated segments into motion data, either prepending or appending."""
        self.motion = self.motion.extend_with_segments(segments, prepend=prepend)

    def _build_and_apply_transition(
        self,
        start_state: dict[str, torch.Tensor],
        target_state: dict[str, torch.Tensor],
        num_steps: int,
        prepend: bool,
        drop_first: bool,
        drop_last: bool,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Shared interpolation path for prepend/append transitions."""
        if num_steps <= 0:
            return

        alphas = torch.linspace(0.0, 1.0, steps=num_steps + 1, device=device, dtype=dtype)
        if drop_first:
            alphas = alphas[1:]
        if drop_last:
            alphas = alphas[:-1]
        if alphas.numel() == 0:
            return

        alphas_joint = alphas.view(num_steps, 1)
        alphas_body = alphas.view(num_steps, 1, 1)

        segments = self._build_transition_segments(start_state, target_state, alphas, alphas_joint, alphas_body)
        self._apply_transition_segments(segments, prepend=prepend)

    def _setup_visualization_markers_for_isaacsim(self):
        from isaaclab.markers import VisualizationMarkers
        from isaaclab.markers.config import FRAME_MARKER_CFG, RAY_CASTER_MARKER_CFG

        visualization_markers_cfg = FRAME_MARKER_CFG.replace(
            prim_path="/Visuals/Command/real_robot",
        )
        visualization_markers_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
        real_robot_visualizer = VisualizationMarkers(visualization_markers_cfg)

        visualization_markers_cfg = FRAME_MARKER_CFG.replace(
            prim_path="/Visuals/Command/motion_robot",
        )
        visualization_markers_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
        motion_robot_visualizer = VisualizationMarkers(visualization_markers_cfg)
        self.visualization_markers = {
            "real_robot": real_robot_visualizer,
            "motion_robot": motion_robot_visualizer,
        }

        for body_names in self.motion_cfg.body_names_to_track:
            visualization_markers_cfg = RAY_CASTER_MARKER_CFG.replace(
                prim_path=f"/Visuals/Command/motion_robot_body/motion_{body_names}",
            )
            visualization_markers_cfg.markers["hit"].radius = 0.03
            visualization_markers_cfg.markers["hit"].visual_material.diffuse_color = (0.0, 1.0, 0.0)
            self.visualization_markers[f"motion_{body_names}"] = VisualizationMarkers(visualization_markers_cfg)
            visualization_markers_cfg = RAY_CASTER_MARKER_CFG.replace(
                prim_path=f"/Visuals/Command/real_robot_body/real_{body_names}",
            )
            visualization_markers_cfg.markers["hit"].radius = 0.03
            visualization_markers_cfg.markers["hit"].visual_material.diffuse_color = (0.0, 0.6, 1.0)
            self.visualization_markers[f"real_{body_names}"] = VisualizationMarkers(visualization_markers_cfg)

        if self.motion.has_object:
            visualization_markers_cfg = FRAME_MARKER_CFG.replace(
                prim_path="/Visuals/Command/real_object",
            )
            visualization_markers_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
            real_object_visualizer = VisualizationMarkers(visualization_markers_cfg)

            visualization_markers_cfg = FRAME_MARKER_CFG.replace(
                prim_path="/Visuals/Command/motion_object",
            )
            visualization_markers_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
            motion_object_visualizer = VisualizationMarkers(visualization_markers_cfg)

            self.visualization_markers["real_object"] = real_object_visualizer
            self.visualization_markers["motion_object"] = motion_object_visualizer

    def _ensure_index_tensor(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)
