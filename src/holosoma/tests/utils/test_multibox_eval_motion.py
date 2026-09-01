from pathlib import Path

import numpy as np
import pytest

from holosoma.utils.multibox import (
    COMBINED_TEACHER_200_BOX_DIMENSIONS,
    apply_multibox_eval_motion_id,
    load_multibox_manifest,
    map_manifest_sizes,
)


TARGET_MOTION = "mentor_Bringing-carry_0205_mj_w_obj.npz"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _mapping_arrays() -> tuple[np.ndarray, int]:
    entries = load_multibox_manifest(_repo_root() / "data/combined_teacher_200_manifest.csv")
    motion_size_ids, _ = map_manifest_sizes(entries, COMBINED_TEACHER_200_BOX_DIMENSIONS)
    target_ids = [motion_id for motion_id, entry in enumerate(entries) if entry.file == TARGET_MOTION]
    assert target_ids == [151]
    return np.asarray(motion_size_ids, dtype=np.int64), target_ids[0]


def test_fixed_eval_motion_only_overrides_compatible_multibox_envs() -> None:
    motion_size_ids, target_motion_id = _mapping_arrays()
    env_size_ids = np.arange(len(COMBINED_TEACHER_200_BOX_DIMENSIONS), dtype=np.int64)

    # One valid, pre-sampled motion per immutable environment size bucket.
    sampled_motion_ids = np.asarray(
        [int(np.flatnonzero(motion_size_ids == size_id)[0]) for size_id in env_size_ids],
        dtype=np.int64,
    )
    selected_motion_ids = apply_multibox_eval_motion_id(
        sampled_motion_ids,
        env_size_ids,
        motion_size_ids,
        target_motion_id,
    )

    compatible = env_size_ids == motion_size_ids[target_motion_id]
    assert compatible.sum() == 1
    assert np.all(selected_motion_ids[compatible] == target_motion_id)
    assert np.array_equal(selected_motion_ids[~compatible], sampled_motion_ids[~compatible])
    assert np.array_equal(motion_size_ids[selected_motion_ids], env_size_ids)


@pytest.mark.parametrize("eval_motion_id", [None, -1])
def test_unfixed_multibox_sampling_is_unchanged(eval_motion_id: int | None) -> None:
    motion_size_ids, _ = _mapping_arrays()
    env_size_ids = np.arange(len(COMBINED_TEACHER_200_BOX_DIMENSIONS), dtype=np.int64)
    sampled_motion_ids = np.asarray(
        [int(np.flatnonzero(motion_size_ids == size_id)[0]) for size_id in env_size_ids],
        dtype=np.int64,
    )

    selected_motion_ids = apply_multibox_eval_motion_id(
        sampled_motion_ids,
        env_size_ids,
        motion_size_ids,
        eval_motion_id,
    )

    assert selected_motion_ids is sampled_motion_ids
    assert np.array_equal(motion_size_ids[selected_motion_ids], env_size_ids)


def test_invalid_fixed_eval_motion_fails_closed() -> None:
    motion_size_ids, _ = _mapping_arrays()
    with pytest.raises(ValueError, match="outside the loaded motion range"):
        apply_multibox_eval_motion_id(
            np.asarray([0]),
            np.asarray([int(motion_size_ids[0])]),
            motion_size_ids,
            len(motion_size_ids),
        )
