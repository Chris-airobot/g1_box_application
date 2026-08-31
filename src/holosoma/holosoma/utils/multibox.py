"""Pure-Python helpers for exact manifest-driven multi-box motion datasets."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


REQUIRED_MANIFEST_COLUMNS = ("file", "source", "motion", "box_x", "box_y", "box_z")
BOX_DIMENSION_DECIMAL_PLACES = 8

COMBINED_TEACHER_200_BOX_DIMENSIONS = [
    (0.21639568, 0.21640438, 0.21640646),
    (0.22247422, 0.22248315, 0.22248529),
    (0.22500234, 0.22501137, 0.22501354),
    (0.2289041, 0.2289133, 0.22891551),
    (0.23023494, 0.23024419, 0.23024641),
    (0.23158134, 0.23159065, 0.23159288),
    (0.24750256, 0.24751251, 0.24751489),
    (0.25063551, 0.25064558, 0.25064799),
    (0.30000002, 0.30000002, 0.30000002),
]


@dataclass(frozen=True)
class MultiBoxManifestEntry:
    file: str
    source: str
    motion: str
    dimensions: tuple[float, float, float]


def load_multibox_manifest(path: str | Path) -> list[MultiBoxManifestEntry]:
    """Load a manifest in row order and reject ambiguous filenames or malformed sizes."""
    manifest_path = Path(path)
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(REQUIRED_MANIFEST_COLUMNS):
            raise ValueError(
                f"Unexpected manifest columns in {manifest_path}: {reader.fieldnames}; "
                f"expected {list(REQUIRED_MANIFEST_COLUMNS)}"
            )
        entries = []
        seen_files: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            filename = row["file"]
            if not filename or Path(filename).name != filename:
                raise ValueError(f"Manifest row {row_number} has an invalid filename: {filename!r}")
            if filename in seen_files:
                raise ValueError(f"Manifest contains duplicate file entry: {filename}")
            seen_files.add(filename)
            try:
                # The tracked CSV contains a few serialization tails such as
                # 0.23024419000000002. The physical specifications are defined
                # to eight decimal places, so canonicalize only those tails.
                dimensions = tuple(
                    round(float(row[column]), BOX_DIMENSION_DECIMAL_PLACES)
                    for column in ("box_x", "box_y", "box_z")
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Manifest row {row_number} has invalid box dimensions") from exc
            if any(value <= 0.0 for value in dimensions):
                raise ValueError(f"Manifest row {row_number} has non-positive box dimensions: {dimensions}")
            entries.append(
                MultiBoxManifestEntry(
                    file=filename,
                    source=row["source"],
                    motion=row["motion"],
                    dimensions=dimensions,
                )
            )
    if not entries:
        raise ValueError(f"Manifest contains no motions: {manifest_path}")
    return entries


def map_manifest_sizes(
    entries: list[MultiBoxManifestEntry],
    configured_dimensions: list[tuple[float, float, float]],
) -> tuple[list[int], list[list[int]]]:
    """Build exact motion-to-size and size-to-motion mappings."""
    if len(set(configured_dimensions)) != len(configured_dimensions):
        raise ValueError("Configured multi-box dimensions must be unique")
    size_by_dimensions = {dimensions: size_id for size_id, dimensions in enumerate(configured_dimensions)}
    motion_size_ids: list[int] = []
    motions_by_size: list[list[int]] = [[] for _ in configured_dimensions]
    for motion_id, entry in enumerate(entries):
        if entry.dimensions not in size_by_dimensions:
            raise ValueError(
                f"Motion {entry.file} requires unconfigured box dimensions {entry.dimensions}"
            )
        size_id = size_by_dimensions[entry.dimensions]
        motion_size_ids.append(size_id)
        motions_by_size[size_id].append(motion_id)
    empty = [size_id for size_id, motion_ids in enumerate(motions_by_size) if not motion_ids]
    if empty:
        raise ValueError(f"Configured multi-box size IDs have no motions: {empty}")
    return motion_size_ids, motions_by_size
