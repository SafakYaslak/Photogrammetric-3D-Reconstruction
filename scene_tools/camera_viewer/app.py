"""Interactive Open3D application for the four-camera scene."""
from __future__ import annotations

import argparse
import math
import queue
import shlex
import sys
import threading
import time
from pathlib import Path

try:
    import numpy as np
    import open3d as o3d
except ImportError as error:
    missing_package = {"open3d": "Open3D", "numpy": "NumPy"}.get(error.name, error.name or "a required package")
    print(f"ERROR: {missing_package} is required. Install project dependencies with python -m pip install -r requirements.txt", file=sys.stderr)
    raise SystemExit(1) from error

from .scene import (
    COMMAND_TRANSLATION,
    CameraImage,
    create_camera_body,
    create_camera_frustum,
    create_image_plane,
    create_orbit_rings,
    load_camera_images,
    rotation_matrix,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="1, 2, 3 ve 4 klasörlerindeki fotoğrafları dört kameralı bir 3D sahnede gösterir."
    )
    parser.add_argument(
        "folder",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "camera_images",
        help="İçinde 1, 2, 3, 4 fotoğraf klasörleri bulunan ana klasör",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=3.0,
        help="Kameraların sahne merkezinden uzaklığı (varsayılan: 3)",
    )
    parser.add_argument(
        "--image-width",
        type=float,
        default=1.35,
        help="Sahnedeki fotoğraf düzlemlerinin genişliği (varsayılan: 1.35)",
    )
    return parser.parse_args()


class FourCameraViewer:
    def __init__(self, args: argparse.Namespace) -> None:
        if args.radius <= 0:
            raise ValueError("--radius sıfırdan büyük olmalı.")
        if args.image_width <= 0:
            raise ValueError("--image-width sıfırdan büyük olmalı.")
        self.images = load_camera_images(args.folder)
        self.radius = args.radius
        self.image_width = args.image_width
        self.application = o3d.visualization.gui.Application.instance
        self.window = self.application.create_window(
            "Dört Kamera — 3D Görüntüleyici",
            1440,
            900,
        )
        self.widget = o3d.visualization.gui.SceneWidget()
        self.widget.scene = o3d.visualization.rendering.Open3DScene(self.window.renderer)
        self.widget.scene.set_background([0.965, 0.975, 0.99, 1.0])
        self.widget.scene.set_lighting(
            o3d.visualization.rendering.Open3DScene.LightingProfile.SOFT_SHADOWS,
            [0.577, -0.577, -0.577],
        )
        self.widget.set_view_controls(
            o3d.visualization.gui.SceneWidget.Controls.ROTATE_CAMERA
        )
        self.window.add_child(self.widget)
        self.bounds = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=[-self.radius - self.image_width, -self.radius - self.image_width, -1.0],
            max_bound=[self.radius + self.image_width, self.radius + self.image_width, 1.0],
        )
        self.console_commands: queue.Queue[str] = queue.Queue()
        self.orbit_axis = 2
        self.orbit_direction = 1.0
        self.orbit_speed = 12.0
        self.orbit_running = False
        self.orbit_last_time: float | None = None
        self.orbit_eye: np.ndarray | None = None
        self.orbit_up: np.ndarray | None = None
        self._populate_scene()
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_tick_event(self._on_tick)

    def _add_textured_plane(self, image: CameraImage, center: np.ndarray, right: np.ndarray) -> np.ndarray:
        mesh, bounds, texture = create_image_plane(
            image.pixels,
            center,
            right,
            self.image_width,
        )
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.base_color = [1.0, 1.0, 1.0, 1.0]
        material.albedo_img = texture
        material.sRGB_color = True
        name = f"photo-{image.folder}-{image.path.stem}"
        self.widget.scene.add_geometry(name, mesh, material)
        self.bounds += bounds
        return np.asarray(mesh.vertices)

    def _populate_scene(self) -> None:
        for image in self.images:
            direction = image.direction
            camera_position = direction * self.radius
            camera_scale = self.image_width * 0.13
            camera_origin = camera_position - direction * (camera_scale * 0.52)
            forward = -direction
            right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
            right /= np.linalg.norm(right)
            plane_center = direction * (self.radius * 0.64)
            corners = self._add_textured_plane(image, plane_center, right)

            frame = create_camera_frustum(camera_origin, corners, np.zeros(3))
            frustum_name = f"frustum-{image.folder}-{image.path.stem}"
            self.widget.scene.add_geometry(
                frustum_name,
                frame,
                self._line_material(),
            )
            self.bounds += frame.get_axis_aligned_bounding_box()

            camera_body, camera_lens = create_camera_body(
                camera_origin,
                direction,
                camera_scale,
            )
            camera_body_name = f"camera-body-{image.folder}"
            camera_lens_name = f"camera-lens-{image.folder}"
            self.widget.scene.add_geometry(
                camera_body_name,
                camera_body,
                self._camera_body_material(),
            )
            self.widget.scene.add_geometry(
                camera_lens_name,
                camera_lens,
                self._camera_lens_material(),
            )
            self.bounds += camera_body.get_axis_aligned_bounding_box()
            self.bounds += camera_lens.get_axis_aligned_bounding_box()
            print(
                f"Kamera {image.folder} ({image.direction_name}): "
                f"{image.path.name} — {image.pixels.shape[1]}×{image.pixels.shape[0]}",
                flush=True,
            )

        rings = create_orbit_rings(self.radius)
        self.widget.scene.add_geometry("orbit-rings", rings, self._line_material())
        self.bounds += rings.get_axis_aligned_bounding_box()

    @staticmethod
    def _line_material() -> o3d.visualization.rendering.MaterialRecord:
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "unlitLine"
        material.line_width = 2.5
        return material

    @staticmethod
    def _camera_body_material() -> o3d.visualization.rendering.MaterialRecord:
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultLit"
        material.base_color = [1.0, 0.30, 0.025, 1.0]
        material.base_metallic = 0.16
        material.base_roughness = 0.32
        return material

    @staticmethod
    def _camera_lens_material() -> o3d.visualization.rendering.MaterialRecord:
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultLit"
        material.base_color = [0.065, 0.095, 0.135, 1.0]
        material.base_metallic = 0.34
        material.base_roughness = 0.2
        return material

    def _on_layout(self, _context: o3d.visualization.gui.LayoutContext) -> None:
        del _context
        self.widget.frame = self.window.content_rect
        if self.widget.frame.width > 0 and self.widget.frame.height > 0:
            self.widget.setup_camera(55.0, self.bounds, np.zeros(3, dtype=np.float32))

    def _on_tick(self) -> bool:
        changed = False
        while True:
            try:
                command = self.console_commands.get_nowait()
            except queue.Empty:
                break
            changed = self._handle_console_command(command) or changed

        now = time.perf_counter()
        previous_time = self.orbit_last_time
        self.orbit_last_time = now
        if (
            self.orbit_running
            and previous_time is not None
            and self.orbit_eye is not None
            and self.orbit_up is not None
        ):
            angle = math.radians(
                self.orbit_speed * self.orbit_direction * (now - previous_time)
            )
            rotation = rotation_matrix(self.orbit_axis, angle)
            self.orbit_eye = rotation @ self.orbit_eye
            self.orbit_up = rotation @ self.orbit_up
            self.widget.scene.camera.look_at(
                np.zeros(3, dtype=np.float32),
                self.orbit_eye.astype(np.float32),
                self.orbit_up.astype(np.float32),
            )
            changed = True
        return changed

    def _handle_console_command(self, line: str) -> bool:
        try:
            words = shlex.split(line)
        except ValueError as error:
            print(f"Komut okunamadı: {error}", flush=True)
            return False
        if not words:
            return False
        command = words[0].translate(COMMAND_TRANSLATION).casefold()
        arguments = [
            word.translate(COMMAND_TRANSLATION).casefold() for word in words[1:]
        ]

        if command in {"exit", "quit", "cikis"}:
            self.window.close()
            return False
        if command in {"yardim", "help"}:
            self._print_console_help()
            return False
        if command in {"eksen", "axis"} and len(arguments) == 1:
            if arguments[0] not in {"x", "y", "z"}:
                print("Kullanım: eksen x|y|z", flush=True)
                return False
            self.orbit_axis = {"x": 0, "y": 1, "z": 2}[arguments[0]]
            print(f"Otomatik dönüş ekseni: {arguments[0].upper()}", flush=True)
            return False
        if command in {"hiz", "speed"} and len(arguments) == 1:
            try:
                speed = float(arguments[0])
            except ValueError:
                print("Hız sayı olmalı; örnek: hiz 12", flush=True)
                return False
            if not math.isfinite(speed) or not 0 <= speed <= 180:
                print("Hız 0 ile 180 derece/saniye arasında olmalı.", flush=True)
                return False
            self.orbit_speed = speed
            print(f"Otomatik dönüş hızı: {speed:g}°/sn", flush=True)
            return False
        if command in {"yon", "direction"} and len(arguments) == 1:
            if arguments[0] in {"saat", "saat-yonu", "cw"}:
                self.orbit_direction = -1.0
                direction_name = "saat yönü"
            elif arguments[0] in {"ters", "saat-yonu-tersi", "ccw"}:
                self.orbit_direction = 1.0
                direction_name = "saat yönünün tersi"
            else:
                print("Kullanım: yon saat|ters", flush=True)
                return False
            print(f"Otomatik dönüş yönü: {direction_name}", flush=True)
            return False
        if command in {"donus", "orbit"} and len(arguments) == 1:
            if arguments[0] in {"basla", "start"}:
                if self.orbit_speed <= 0:
                    print("Dönüşü başlatmadan önce pozitif hız seçin: hiz 12", flush=True)
                    return False
                camera_matrix = np.asarray(self.widget.scene.camera.get_model_matrix())
                self.orbit_eye = camera_matrix[:3, 3].astype(np.float64)
                self.orbit_up = camera_matrix[:3, 1].astype(np.float64)
                self.orbit_running = True
                self.orbit_last_time = time.perf_counter()
                print(
                    f"Otomatik dönüş başladı: {self.orbit_speed:g}°/sn, "
                    f"{('XYZ'[self.orbit_axis])} ekseni.",
                    flush=True,
                )
                return True
            if arguments[0] in {"dur", "stop"}:
                self.orbit_running = False
                self.orbit_last_time = None
                print("Otomatik dönüş durduruldu.", flush=True)
                return False
            print("Kullanım: donus basla|dur", flush=True)
            return False

        print("Komut anlaşılmadı. Komutları görmek için 'yardim' yazın.", flush=True)
        return False

    @staticmethod
    def _print_console_help() -> None:
        print(
            "\nKonsol komutları:\n"
            "  eksen x|y|z       Dönüş eksenini seç\n"
            "  hiz 12            Hızı derece/saniye olarak ayarla (0–180)\n"
            "  yon saat|ters     Dönüş yönünü seç\n"
            "  donus basla       Otomatik kamera dönüşünü başlat\n"
            "  donus dur         Otomatik kamera dönüşünü durdur\n"
            "  yardim            Bu komutları göster\n"
            "  cikis             3D görüntüleyiciyi kapat\n",
            flush=True,
        )

    def _read_console(self) -> None:
        self._print_console_help()
        print("Komutları bu pencereyi başlattığınız PowerShell'e yazın.", flush=True)
        while True:
            try:
                line = input("> ")
            except EOFError:
                self.console_commands.put("cikis")
                return
            self.console_commands.put(line)
            if line.strip().casefold() in {"exit", "quit", "cikis"}:
                return

    def run(self) -> None:
        print(
            "\nSahne hazır. Fareyle sürükleyip yakınlaşıp uzaklaşabilirsiniz.",
            flush=True,
        )
        print(
            "Yazısız 3D sunum sahnesi hazır; dönüşü konsol komutlarıyla yönetin.",
            flush=True,
        )
        threading.Thread(
            target=self._read_console,
            daemon=True,
            name="four-camera-console",
        ).start()
        self.application.run()


def main() -> int:
    args = parse_args()
    try:
        o3d.visualization.gui.Application.instance.initialize()
        viewer = FourCameraViewer(args)
        viewer.run()
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"HATA: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nKullanıcı tarafından durduruldu.", file=sys.stderr)
        return 130

if __name__ == "__main__":
    raise SystemExit(main())
