"""PLY inspection and reusable geometry helpers."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import open3d as o3d

def header_has_faces(path: Path) -> bool:
    header_size = 0
    with path.open("rb") as ply_file:
        while header_size <= 1024 * 1024:
            line = ply_file.readline()
            if not line:
                raise ValueError("PLY başlığı tamamlanmamış; end_header satırı bulunamadı.")
            header_size += len(line)
            if line.strip() == b"end_header":
                return False
            match = re.fullmatch(rb"element\s+face\s+(\d+)", line.strip())
            if match and int(match.group(1)) > 0:
                return True
    raise ValueError("PLY başlığı beklenenden uzun (1 MB üzeri).")


def create_grid(geometry: o3d.geometry.Geometry) -> o3d.geometry.LineSet:
    bounds = geometry.get_axis_aligned_bounding_box()
    minimum, maximum = bounds.min_bound, bounds.max_bound
    center = (minimum + maximum) / 2
    radius = max(float(np.max(maximum - minimum)) * 0.65, 0.5)
    coordinates = np.linspace(-radius, radius, 21)
    points = []
    for coordinate in coordinates:
        points.extend(
            (
                (center[0] + coordinate, center[1] - radius, minimum[2]),
                (center[0] + coordinate, center[1] + radius, minimum[2]),
                (center[0] - radius, center[1] + coordinate, minimum[2]),
                (center[0] + radius, center[1] + coordinate, minimum[2]),
            )
        )
    lines = np.arange(len(points), dtype=np.int32).reshape(-1, 2)
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64)),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.paint_uniform_color((0.90, 0.91, 0.92))
    return line_set


def pick_visible_vertex(
    points: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    x: float,
    y: float,
    max_distance: float,
) -> np.ndarray | None:
    camera_points = points @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    visible = camera_points[:, 2] > 0
    if not np.any(visible):
        return None
    visible_points = camera_points[visible]
    projected_x = intrinsic[0, 0] * visible_points[:, 0] / visible_points[:, 2] + intrinsic[0, 2]
    projected_y = intrinsic[1, 1] * visible_points[:, 1] / visible_points[:, 2] + intrinsic[1, 2]
    distances_squared = (projected_x - x) ** 2 + (projected_y - y) ** 2
    nearest_index = int(np.argmin(distances_squared))
    if distances_squared[nearest_index] > max_distance**2:
        return None
    return points[np.flatnonzero(visible)[nearest_index]].copy()
