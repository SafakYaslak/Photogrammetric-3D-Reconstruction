"""PLY Studio application setup and model state."""
from __future__ import annotations

import argparse
import queue
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d

from ..recording.screen_recorder import ScreenRecorder
from .appearance import ORANGE, VISUAL_STYLES, height_colors, parse_color, set_vertex_colors, vertex_colors
from .commands import CommandMixin
from .geometry import create_grid, header_has_faces
from .interactions import InteractionMixin
from .runtime import RuntimeMixin


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="View PLY point clouds and triangle meshes with Open3D."
    )
    parser.add_argument(
        "file", nargs="?", type=Path,
        default=Path(__file__).resolve().parents[2] / "gala" / "Dense" / "new2.ply",
        help="PLY file to open (default: gala/Dense/new2.ply)",
    )
    parser.add_argument("--view", choices=("points", "mesh"), default="points", help="Initial view")
    parser.add_argument("--color", default=ORANGE, help="Initial color: original, height, or #RRGGBB")
    parser.add_argument("--palette", choices=("spectrum", "ocean", "sunset", "cividis", "plasma", "inferno", "magma"), default="spectrum", help="Palette for height coloring")
    parser.add_argument("--point-size", type=float, default=2.5, help="Point size (0.1-30)")
    parser.add_argument("--background", default="#FFFFFF", help="Background color in #RRGGBB format")
    parser.add_argument("--grid", action="store_true", help="Show the reference grid at startup")
    parser.add_argument("--window-width", type=int, default=1440, help="Window width")
    parser.add_argument("--window-height", type=int, default=900, help="Window height")
    return parser.parse_args()


class PLYViewer(InteractionMixin, CommandMixin, RuntimeMixin):
    def __init__(self, args: argparse.Namespace) -> None:
        self.path = args.file.expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"PLY dosyası bulunamadı: {self.path}")
        if self.path.suffix.lower() != ".ply":
            raise ValueError("Yalnızca .ply uzantılı dosyalar destekleniyor.")
        if not 0.1 <= args.point_size <= 30:
            raise ValueError("--point-size 0.1 ile 30 arasında olmalı.")
        if args.window_width < 640 or args.window_height < 480:
            raise ValueError("Pencere boyutu en az 640×480 piksel olmalı.")

        self.color_mode, self.solid_color = self._parse_initial_color(args.color)
        self.palette = args.palette
        self.background = parse_color(args.background)
        self.point_size = args.point_size
        self.view = args.view
        self.grid_visible = args.grid
        self.visual_style = "makale"
        self.coordinate_frame_visible = False
        self.wireframe_visible = False
        self.grid_color = VISUAL_STYLES["makale"]["grid"]
        self.window_width = args.window_width
        self.window_height = args.window_height
        self.orbit_center = np.zeros(3, dtype=np.float64)
        self.orbit_speed = 12.0
        self.orbit_direction = 1.0
        self.orbit_axis = 2
        self.orbiting = False
        self.orbit_last_time: float | None = None
        self.mouse_x = 0.0
        self.mouse_y = 0.0
        self.drag_button: int | None = None
        self.drag_origin = (0.0, 0.0)
        self.drag_last = (0.0, 0.0)
        self.drag_moved = False
        self.pending_rotation = np.zeros(2, dtype=np.float64)
        self.pending_translation = np.zeros(2, dtype=np.float64)
        self.mouse_smoothing_time = 0.06
        self.mouse_sensitivity = 0.35
        self.last_mouse_smoothing_time = time.perf_counter()
        self.last_recording_frame_time = 0.0
        self.commands: queue.Queue[str] = queue.Queue()
        self.recorder = ScreenRecorder(
            f"PLY Studio — {self.path.name}",
            Path(__file__).resolve().parents[2] / "recordings",
        )
        self.geometry: o3d.geometry.Geometry
        self.point_cloud: o3d.geometry.PointCloud
        self.original_colors: np.ndarray | None
        self.mesh: o3d.geometry.TriangleMesh | None = None
        self.grid: o3d.geometry.LineSet
        self.visualizer = o3d.visualization.VisualizerWithKeyCallback()
        self.current_geometry: o3d.geometry.Geometry | None = None
        self.current_grid: o3d.geometry.LineSet | None = None

    @staticmethod
    def _parse_initial_color(value: str) -> tuple[str, tuple[float, float, float]]:
        if value.lower() == "original":
            return "original", parse_color(ORANGE)
        if value.lower() == "height":
            return "height", parse_color(ORANGE)
        return "solid", parse_color(value)

    def load_model(self) -> None:
        print(f"\nPLY yükleniyor: {self.path}", flush=True)
        if header_has_faces(self.path):
            self.mesh = o3d.io.read_triangle_mesh(str(self.path))
            if self.mesh.is_empty() or not len(self.mesh.triangles):
                raise ValueError("PLY başlığında yüzey var, ancak üçgen ağı okunamadı.")
            if not self.mesh.has_vertex_normals():
                print("Yüzey normalleri hesaplanıyor…", flush=True)
                self.mesh.compute_vertex_normals()
            self.geometry = self.mesh
            self.point_cloud = o3d.geometry.PointCloud()
            self.point_cloud.points = o3d.utility.Vector3dVector(np.asarray(self.mesh.vertices))
            if self.mesh.has_vertex_colors():
                self.point_cloud.colors = o3d.utility.Vector3dVector(np.asarray(self.mesh.vertex_colors))
        else:
            self.point_cloud = o3d.io.read_point_cloud(str(self.path))
            if self.point_cloud.is_empty():
                raise ValueError("PLY dosyasında görüntülenecek nokta bulunamadı.")
            self.geometry = self.point_cloud

        self.original_colors = vertex_colors(self.geometry)
        self.grid = create_grid(self.geometry)
        self.orbit_center = self.geometry.get_axis_aligned_bounding_box().get_center()
        points = len(self.point_cloud.points)
        triangles = len(self.mesh.triangles) if self.mesh is not None else 0
        print(f"Yüklendi: {points:,} nokta, {triangles:,} üçgen.".replace(",", "."), flush=True)
        if self.view == "mesh" and self.mesh is None:
            print("Dosyada yüzey bulunmuyor; nokta bulutu görünümü kullanılacak.", flush=True)
            self.view = "points"

    def _active_geometry(self) -> o3d.geometry.Geometry:
        if self.view == "mesh" and self.mesh is not None:
            return self.mesh
        return self.point_cloud

    def _apply_colors(self) -> None:
        geometries = [self.point_cloud]
        if self.mesh is not None:
            geometries.append(self.mesh)
        for geometry in geometries:
            if self.color_mode == "original" and self.original_colors is not None:
                colors = self.original_colors
            elif self.color_mode == "height":
                colors = height_colors(geometry, self.palette)
            else:
                points = geometry.points if isinstance(geometry, o3d.geometry.PointCloud) else geometry.vertices
                colors = np.tile(self.solid_color, (len(points), 1))
            if bool(VISUAL_STYLES[self.visual_style]["monochrome"]):
                luminance = colors @ np.array([0.2126, 0.7152, 0.0722])
                colors = np.repeat(luminance[:, np.newaxis], 3, axis=1)
            set_vertex_colors(geometry, colors)

    def _update_geometry(self, reset_camera: bool = False) -> None:
        active = self._active_geometry()
        if self.current_geometry is not None:
            self.visualizer.remove_geometry(self.current_geometry, reset_bounding_box=False)
        self.visualizer.add_geometry(active, reset_bounding_box=reset_camera)
        self.current_geometry = active
        if self.current_grid is not None:
            self.visualizer.remove_geometry(self.current_grid, reset_bounding_box=False)
            self.current_grid = None
        if self.grid_visible:
            self.grid = create_grid(active)
            self.grid.paint_uniform_color(self.grid_color)
            self.visualizer.add_geometry(self.grid, reset_bounding_box=False)
            self.current_grid = self.grid
        self.visualizer.update_renderer()

    def _set_status(self, message: str) -> None:
        print(message, flush=True)

def main() -> int:
    args = parse_args()
    try:
        if args.color.lower() not in {"original", "height"}:
            parse_color(args.color)
        parse_color(args.background)
        return PLYViewer(args).run()
    except (OSError, RuntimeError, ValueError) as error:
        print(f"HATA: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nKullanıcı tarafından durduruldu.", file=sys.stderr)
        return 130

if __name__ == "__main__":
    raise SystemExit(main())
