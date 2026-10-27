"""Color parsing, palettes, and geometry coloring utilities."""
from __future__ import annotations

import re

import numpy as np
import open3d as o3d

ORANGE = "#FF791F"
PALETTES: dict[str, tuple[tuple[float, tuple[int, int, int]], ...]] = {
    "spectrum": (
        (0.0, (68, 1, 84)),
        (0.25, (59, 82, 139)),
        (0.5, (33, 145, 140)),
        (0.75, (94, 201, 98)),
        (1.0, (253, 231, 37)),
    ),
    "ocean": (
        (0.0, (24, 57, 98)),
        (0.4, (25, 104, 150)),
        (0.75, (82, 204, 191)),
        (1.0, (226, 255, 233)),
    ),
    "sunset": (
        (0.0, (47, 31, 89)),
        (0.35, (166, 49, 103)),
        (0.7, (239, 114, 81)),
        (1.0, (255, 222, 128)),
    ),
    "cividis": (
        (0.0, (0, 32, 77)),
        (0.25, (66, 78, 108)),
        (0.5, (124, 123, 120)),
        (0.75, (187, 173, 111)),
        (1.0, (253, 234, 69)),
    ),
    "plasma": (
        (0.0, (13, 8, 135)),
        (0.25, (126, 3, 168)),
        (0.5, (204, 71, 120)),
        (0.75, (248, 149, 64)),
        (1.0, (240, 249, 33)),
    ),
    "inferno": (
        (0.0, (0, 0, 4)),
        (0.25, (87, 15, 109)),
        (0.5, (188, 55, 84)),
        (0.75, (249, 142, 9)),
        (1.0, (252, 255, 164)),
    ),
    "magma": (
        (0.0, (0, 0, 4)),
        (0.25, (81, 18, 124)),
        (0.5, (183, 55, 121)),
        (0.75, (251, 136, 97)),
        (1.0, (252, 253, 191)),
    ),
}
VISUAL_STYLES: dict[str, dict[str, object]] = {
    "makale": {
        "background": "#FFFFFF",
        "grid": (0.86, 0.88, 0.91),
        "monochrome": False,
        "grid_visible": False,
        "coordinate_frame_visible": False,
        "wireframe_visible": False,
    },
    "karanlik": {
        "background": "#111722",
        "grid": (0.24, 0.28, 0.34),
        "monochrome": False,
        "grid_visible": False,
        "coordinate_frame_visible": False,
        "wireframe_visible": False,
    },
    "monokrom": {
        "background": "#FFFFFF",
        "grid": (0.80, 0.82, 0.85),
        "monochrome": True,
        "grid_visible": False,
        "coordinate_frame_visible": False,
        "wireframe_visible": False,
    },
    "teknik": {
        "background": "#EAF0F6",
        "grid": (0.72, 0.77, 0.83),
        "monochrome": False,
        "grid_visible": True,
        "coordinate_frame_visible": True,
        "wireframe_visible": True,
    },
    "bilimsel": {
        "background": "#FFFFFF",
        "grid": (0.86, 0.88, 0.91),
        "monochrome": False,
        "grid_visible": False,
        "coordinate_frame_visible": False,
        "wireframe_visible": False,
        "color_mode": "height",
        "palette": "cividis",
    },
    "sunum": {
        "background": "#111722",
        "grid": (0.24, 0.28, 0.34),
        "monochrome": False,
        "grid_visible": False,
        "coordinate_frame_visible": False,
        "wireframe_visible": False,
        "color_mode": "height",
        "palette": "inferno",
    },
    "pastel": {
        "background": "#F7F4EC",
        "grid": (0.84, 0.82, 0.76),
        "monochrome": False,
        "grid_visible": False,
        "coordinate_frame_visible": False,
        "wireframe_visible": False,
        "color_mode": "height",
        "palette": "ocean",
    },
    "kontrast": {
        "background": "#10131A",
        "grid": (0.24, 0.28, 0.34),
        "monochrome": False,
        "grid_visible": False,
        "coordinate_frame_visible": False,
        "wireframe_visible": False,
        "color_mode": "height",
        "palette": "plasma",
    },
}
COLOR_NAMES = {
    "orange": ORANGE,
    "turuncu": ORANGE,
    "white": "#FFFFFF",
    "beyaz": "#FFFFFF",
    "black": "#000000",
    "siyah": "#000000",
}
TURKISH_TRANSLATION = str.maketrans(
    {
        "ı": "i", "İ": "i", "ş": "s", "Ş": "s", "ğ": "g", "Ğ": "g",
        "ü": "u", "Ü": "u", "ö": "o", "Ö": "o", "ç": "c", "Ç": "c",
    }
)


def parse_color(value: str) -> tuple[float, float, float]:
    color = COLOR_NAMES.get(value.lower(), value)
    if re.fullmatch(r"#[0-9a-fA-F]{6}", color):
        return tuple(int(color[offset:offset + 2], 16) / 255 for offset in (1, 3, 5))
    if re.fullmatch(r"#[0-9a-fA-F]{3}", color):
        return tuple(int(character * 2, 16) / 255 for character in color[1:])
    raise ValueError("Renk adı (turuncu, beyaz) veya #RRGGBB biçiminde renk girin.")


def vertex_colors(geometry: o3d.geometry.Geometry) -> np.ndarray | None:
    if isinstance(geometry, o3d.geometry.PointCloud) and geometry.has_colors():
        return np.asarray(geometry.colors).copy()
    if isinstance(geometry, o3d.geometry.TriangleMesh) and geometry.has_vertex_colors():
        return np.asarray(geometry.vertex_colors).copy()
    return None


def set_vertex_colors(geometry: o3d.geometry.Geometry, colors: np.ndarray) -> None:
    if isinstance(geometry, o3d.geometry.PointCloud):
        geometry.colors = o3d.utility.Vector3dVector(colors)
    elif isinstance(geometry, o3d.geometry.TriangleMesh):
        geometry.vertex_colors = o3d.utility.Vector3dVector(colors)


def height_colors(geometry: o3d.geometry.Geometry, palette: str) -> np.ndarray:
    points = np.asarray(geometry.points if isinstance(geometry, o3d.geometry.PointCloud) else geometry.vertices)
    if not len(points):
        raise ValueError("Modelde renklendirilecek nokta bulunamadı.")
    low, high = points[:, 2].min(), points[:, 2].max()
    heights = np.zeros(len(points)) if high == low else (points[:, 2] - low) / (high - low)
    stops = PALETTES[palette]
    result = np.empty((len(points), 3), dtype=np.float64)
    for channel in range(3):
        result[:, channel] = np.interp(
            heights,
            [stop[0] for stop in stops],
            [stop[1][channel] / 255 for stop in stops],
        )
    return result
