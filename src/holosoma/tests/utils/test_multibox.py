from collections import Counter
from pathlib import Path

from holosoma.config_values.wbt.g1.command import combined_teacher_200_motion_config
from holosoma.utils.multibox import (
    COMBINED_TEACHER_200_BOX_DIMENSIONS,
    load_multibox_manifest,
    map_manifest_sizes,
)


EXPECTED_COUNTS = [7, 1, 12, 64, 12, 53, 2, 3, 46]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def test_combined_teacher_200_manifest_and_buckets() -> None:
    manifest = load_multibox_manifest(_repo_root() / "data/combined_teacher_200_manifest.csv")
    assert len(manifest) == 200
    assert Counter(entry.source for entry in manifest) == {"mentor": 154, "c2a": 46}

    motion_size_ids, motions_by_size = map_manifest_sizes(manifest, COMBINED_TEACHER_200_BOX_DIMENSIONS)
    assert len(motion_size_ids) == 200
    assert [len(ids) for ids in motions_by_size] == EXPECTED_COUNTS
    assert sorted(motion_id for ids in motions_by_size for motion_id in ids) == list(range(200))


def test_combined_teacher_config_uses_local_dataset_and_bucketed_eval() -> None:
    assert combined_teacher_200_motion_config.motion_dir == "data/combined_teacher_200"
    assert combined_teacher_200_motion_config.motion_manifest == "data/combined_teacher_200_manifest.csv"
    assert combined_teacher_200_motion_config.eval_motion_id == -1
