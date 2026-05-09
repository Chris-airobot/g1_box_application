"""Whole Body Tracking observation presets for the G1 robot."""

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg

actor_obs_shared = ObsGroupCfg(
    concatenate=True,
    enable_noise=True,
    history_length=1,
    terms={
        "motion_command": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:motion_command",
            scale=1.0,
            noise=0.0,
        ),
        "motion_ref_ori_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:motion_ref_ori_b",
            scale=1.0,
            noise=0.05,
        ),
        "base_ang_vel": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:base_ang_vel",
            scale=1.0,
            noise=0.2,
        ),
        "dof_pos": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:dof_pos",
            scale=1.0,
            noise=0.01,
        ),
        "dof_vel": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:dof_vel",
            scale=1.0,
            noise=0.5,
        ),
        "actions": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:actions",
            scale=1.0,
            noise=0.0,
        ),
    },
)

critic_obs_shared_terms = {
    "motion_command": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_command",
        scale=1.0,
        noise=0.0,
    ),
    "motion_ref_pos_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_ref_pos_b",
        scale=1.0,
        noise=0.25,
    ),
    "motion_ref_ori_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_ref_ori_b",
        scale=1.0,
        noise=0.05,
    ),
    "robot_body_pos_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:robot_body_pos_b",
        scale=1.0,
        noise=0.0,
    ),
    "robot_body_ori_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:robot_body_ori_b",
        scale=1.0,
        noise=0.0,
    ),
    "robot_body_lin_vel_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:robot_body_lin_vel_b",
        scale=1.0,
        noise=0.0,
    ),
    "robot_body_ang_vel_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:robot_body_ang_vel_b",
        scale=1.0,
        noise=0.0,
    ),
    "base_lin_vel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:base_lin_vel",
        scale=1.0,
        noise=0.0,
    ),
    "base_ang_vel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:base_ang_vel",
        scale=1.0,
        noise=0.2,
    ),
    "dof_pos": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:dof_pos",
        scale=1.0,
        noise=0.01,
    ),
    "dof_vel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:dof_vel",
        scale=1.0,
        noise=0.5,
    ),
    "actions": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:actions",
        scale=1.0,
        noise=0.0,
    ),
}

critic_obs_w_object_terms = critic_obs_shared_terms.copy()
critic_obs_w_object_terms.update(
    {
        "obj_target_pos_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_target_pos_b",
            scale=1.0,
            noise=0.0,
        ),
        "obj_pos_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_pos_b",
            scale=1.0,
            noise=0.0,
        ),
        "obj_ori_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_ori_b",
            scale=1.0,
            noise=0.0,
        ),
        "obj_lin_vel_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_lin_vel_b",
            scale=1.0,
            noise=0.0,
        ),
        "obj_ang_vel_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_ang_vel_b",
            scale=1.0,
            noise=0.0,
        ),
        "obj_bbox_corners_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_bbox_corners_b",
            scale=1.0,
            noise=0.0,
        ),
    }
)

actor_obs_w_object_terms = critic_obs_w_object_terms.copy()
actor_obs_w_object = ObsGroupCfg(
    concatenate=True,
    enable_noise=False,
    history_length=1,
    terms=actor_obs_w_object_terms,
)

student_actor_obs_terms = actor_obs_w_object_terms

student_actor_obs = ObsGroupCfg(
    concatenate=True,
    enable_noise=False,
    history_length=1,
    terms=student_actor_obs_terms,
)

teacher_obs = ObsGroupCfg(
    concatenate=True,
    enable_noise=False,
    history_length=1,
    terms=actor_obs_w_object_terms,
)

# Distillation preset without motion-reference inputs in student observations.
# Teacher keeps full privileged observations for imitation targets.
student_actor_obs_no_motion_terms = {
    "actions": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:actions",
        scale=1.0,
        noise=0.0,
    ),
    "base_ang_vel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:base_ang_vel",
        scale=1.0,
        noise=0.2,
    ),
    "base_lin_vel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:base_lin_vel",
        scale=1.0,
        noise=0.0,
    ),
    "dof_pos": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:dof_pos",
        scale=1.0,
        noise=0.01,
    ),
    "dof_vel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:dof_vel",
        scale=1.0,
        noise=0.5,
    ),
    "obj_ang_vel_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:obj_ang_vel_b",
        scale=1.0,
        noise=0.0,
    ),
    "obj_lin_vel_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:obj_lin_vel_b",
        scale=1.0,
        noise=0.0,
    ),
    "obj_pos_real_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:obj_pos_real_b",
        scale=1.0,
        noise=0.0,
    ),
    "obj_target_pos_real_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:obj_target_pos_real_b",
        scale=1.0,
        noise=0.0,
    ),
    "robot_body_ang_vel_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:robot_body_ang_vel_b",
        scale=1.0,
        noise=0.0,
    ),
    "robot_body_lin_vel_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:robot_body_lin_vel_b",
        scale=1.0,
        noise=0.0,
    ),
    "robot_body_ori_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:robot_body_ori_b",
        scale=1.0,
        noise=0.0,
    ),
    "robot_body_pos_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:robot_body_pos_b",
        scale=1.0,
        noise=0.0,
    ),
}

student_actor_obs_no_motion_h4 = ObsGroupCfg(
    concatenate=True,
    enable_noise=False,
    history_length=4,
    terms=student_actor_obs_no_motion_terms,
)

student_actor_obs_no_motion_h50 = ObsGroupCfg(
    concatenate=True,
    enable_noise=False,
    history_length=50,
    terms=student_actor_obs_no_motion_terms,
)

teacher_obs_full = ObsGroupCfg(
    concatenate=True,
    enable_noise=False,
    history_length=1,
    terms=actor_obs_w_object_terms.copy(),
)

g1_29dof_wbt_observation = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_shared,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_shared_terms,
        ),
    },
)

g1_29dof_wbt_observation_w_object = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_w_object,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=actor_obs_w_object_terms,
        ),
    },
)

g1_29dof_wbt_observation_distill = ObservationManagerCfg(
    groups={
        "actor_obs": student_actor_obs,
        "critic_obs": student_actor_obs,
        "teacher_obs": teacher_obs,
    },
)

g1_29dof_wbt_observation_distill_no_motion_h4 = ObservationManagerCfg(
    groups={
        "actor_obs": student_actor_obs_no_motion_h4,
        "critic_obs": student_actor_obs_no_motion_h4,
        "teacher_obs": teacher_obs_full,
    },
)

g1_29dof_wbt_observation_distill_no_motion_h50 = ObservationManagerCfg(
    groups={
        "actor_obs": student_actor_obs_no_motion_h50,
        "critic_obs": student_actor_obs_no_motion_h50,
        "teacher_obs": teacher_obs_full,
    },
)

__all__ = [
    "g1_29dof_wbt_observation",
    "g1_29dof_wbt_observation_w_object",
    "g1_29dof_wbt_observation_distill",
    "g1_29dof_wbt_observation_distill_no_motion_h4",
    "g1_29dof_wbt_observation_distill_no_motion_h50",
]
