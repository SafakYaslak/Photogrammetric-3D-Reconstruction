"""Scene data and Open3D geometry builders for the four-camera viewer."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
    import open3d as o3d
    from PIL import Image
except ImportError as error:
    missing_package = {"open3d": "Open3D", "numpy": "NumPy", "PIL": "Pillow"}.get(error.name, error.name or "a required package")
    raise ImportError(f"{missing_package} is required; install the project dependencies with python -m pip install -r requirements.txt") from error

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
CAMERAS = (
    ("1", "ÖN", np.array([0.0, -1.0, 0.0])),
    ("2", "SAĞ", np.array([1.0, 0.0, 0.0])),
    ("3", "ARKA", np.array([0.0, 1.0, 0.0])),
    ("4", "SOL", np.array([-1.0, 0.0, 0.0])),
)
ORANGE = np.array([1.0, 0.34, 0.035])
RAY_ORANGE = np.array([1.0, 0.48, 0.12])
FRAME_COLOR = np.array([0.56, 0.61, 0.68])
LENS_COLOR = np.array([0.075, 0.105, 0.14])
COMMAND_TRANSLATION = str.maketrans(
    {
        "ı": "i", "İ": "i", "ş": "s", "Ş": "s", "ğ": "g", "Ğ": "g",
        "ü": "u", "Ü": "u", "ö": "o", "Ö": "o", "ç": "c", "Ç": "c",
    }
)


@dataclass(frozen=True)
class CameraImage:
    folder: str
    direction_name: str
    direction: np.ndarray
    path: Path
    pixels: np.ndarray


def load_camera_images(folder: Path) -> list[CameraImage]:
    root = folder.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Fotoğraf klasörü bulunamadı: {root}")

    images = []
    for folder_name, direction_name, direction in CAMERAS:
        image_folder = root / folder_name
        if not image_folder.is_dir():
            raise FileNotFoundError(
                f"{image_folder} klasörü bulunamadı. Ana klasörde 1, 2, 3 ve 4 klasörleri olmalı."
            )
        paths = sorted(
            (
                path for path in image_folder.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ),
            key=lambda path: path.name.casefold(),
        )
        if not paths:
            raise FileNotFoundError(
                f"{image_folder} içinde desteklenen bir resim bulunamadı."
            )
        if len(paths) > 1:
            raise ValueError(
                f"{image_folder} içinde birden fazla resim var. "
                "Her kamera klasöründe yalnızca kullanılacak tek resmi bırakın."
            )
        image_path = paths[0]
        try:
            with Image.open(image_path) as source:
                pixels = np.asarray(source.convert("RGB").copy())
        except OSError as error:
            raise ValueError(f"Resim okunamadı ({image_path}): {error}") from error
        if pixels.ndim != 3 or pixels.shape[0] < 1 or pixels.shape[1] < 1:
            raise ValueError(f"Resim boş veya geçersiz: {image_path}")
        images.append(
            CameraImage(
                folder=folder_name,
                direction_name=direction_name,
                direction=direction.copy(),
                path=image_path,
                pixels=pixels,
            )
        )
    return images


def create_image_plane(
    pixels: np.ndarray,
    center: np.ndarray,
    horizontal: np.ndarray,
    width: float,
) -> tuple[o3d.geometry.TriangleMesh, o3d.geometry.AxisAlignedBoundingBox, o3d.geometry.Image]:
    pixel_height, pixel_width = pixels.shape[:2]
    height = width * pixel_height / pixel_width
    up = np.array([0.0, 0.0, 1.0])
    half_horizontal = horizontal * (width * 0.5)
    half_vertical = up * (height * 0.5)
    vertices = np.asarray(
        (
            center - half_horizontal - half_vertical,
            center + half_horizontal - half_vertical,
            center + half_horizontal + half_vertical,
            center - half_horizontal + half_vertical,
        ),
        dtype=np.float64,
    )
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(
        np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32)
    )
    mesh.triangle_uvs = o3d.utility.Vector2dVector(
        np.asarray(
            ((0, 0), (1, 0), (1, 1), (0, 0), (1, 1), (0, 1)),
            dtype=np.float64,
        )
    )
    texture = o3d.geometry.Image(np.ascontiguousarray(pixels))
    mesh.textures = [texture]
    mesh.compute_triangle_normals()
    bounds = o3d.geometry.AxisAlignedBoundingBox.create_from_points(mesh.vertices)
    return mesh, bounds, texture


def create_camera_frustum(
    origin: np.ndarray,
    image_corners: np.ndarray,
    center: np.ndarray,
    ray_color: np.ndarray = RAY_ORANGE,
    frame_color: np.ndarray = FRAME_COLOR,
    target_color: np.ndarray = ORANGE,
) -> o3d.geometry.LineSet:
    points = np.vstack((origin, image_corners, center))
    camera_edges = ((0, 1), (0, 2), (0, 3), (0, 4))
    plane_edges = ((1, 2), (2, 3), (3, 4), (4, 1))
    target_ray = ((0, 5),)
    lines = np.asarray((*camera_edges, *plane_edges, *target_ray), dtype=np.int32)
    colors = np.asarray(
        (
            *([ray_color] * len(camera_edges)),
            *([frame_color] * len(plane_edges)),
            target_color,
        ),
        dtype=np.float64,
    )
    frustum = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    frustum.colors = o3d.utility.Vector3dVector(colors)
    return frustum


def create_camera_body(
    origin: np.ndarray,
    direction: np.ndarray,
    size: float,
) -> tuple[o3d.geometry.TriangleMesh, o3d.geometry.TriangleMesh]:
    forward = -direction
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    rotation = np.column_stack((right, up, direction))

    body = o3d.geometry.TriangleMesh.create_box(
        width=size * 1.25,
        height=size * 0.82,
        depth=size * 0.74,
    )
    body.translate(-np.array([size * 0.625, size * 0.41, size * 0.37]))
    body.rotate(rotation, center=np.zeros(3))
    body.translate(origin + direction * (size * 0.42))
    body.compute_vertex_normals()

    lens = o3d.geometry.TriangleMesh.create_cylinder(
        radius=size * 0.31,
        height=size * 0.68,
        resolution=24,
        split=3,
    )
    lens_rotation = rotation @ o3d.geometry.get_rotation_matrix_from_xyz(
        (math.pi / 2, 0.0, 0.0)
    )
    lens.rotate(lens_rotation, center=np.zeros(3))
    lens.translate(origin - direction * (size * 0.15))
    lens.compute_vertex_normals()
    return body, lens


def create_orbit_rings(radius: float, z: float = -0.12) -> o3d.geometry.LineSet:
    points: list[list[float]] = []
    lines: list[tuple[int, int]] = []
    colors: list[list[float]] = []
    segments = 128
    for ring_index, ring_radius in enumerate((radius * 0.36, radius * 0.54, radius * 0.72)):
        start = len(points)
        for segment in range(segments):
            angle = 2.0 * math.pi * segment / segments
            points.append(
                [
                    ring_radius * math.cos(angle),
                    ring_radius * math.sin(angle),
                    z,
                ]
            )
        color = FRAME_COLOR * (0.55 if ring_index == 1 else 0.37)
        for segment in range(segments):
            lines.append((start + segment, start + (segment + 1) % segments))
            colors.append(color.tolist())
    rings = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64)),
        lines=o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32)),
    )
    rings.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return rings


def rotation_matrix(axis: int, angle: float) -> np.ndarray:
    direction = np.zeros(3, dtype=np.float64)
    direction[axis] = 1.0
    cross = np.array(
        [
            [0.0, -direction[2], direction[1]],
            [direction[2], 0.0, -direction[0]],
            [-direction[1], direction[0], 0.0],
        ],
        dtype=np.float64,
    )
    return (
        np.eye(3) * math.cos(angle)
        + (1.0 - math.cos(angle)) * np.outer(direction, direction)
        + math.sin(angle) * cross
    )
