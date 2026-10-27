"""Mouse interaction, picking, zoom, and orbit behavior for PLY Studio."""
from __future__ import annotations

import time

import numpy as np
import open3d as o3d

from .geometry import pick_visible_vertex


class InteractionMixin:
    def _set_orbit_center(self, center: np.ndarray) -> None:
        self.orbit_center = np.asarray(center, dtype=np.float64).copy()

    def _on_mouse_move(self, _visualizer: o3d.visualization.Visualizer, x: float, y: float) -> bool:
        del _visualizer
        previous_x, previous_y = self.mouse_x, self.mouse_y
        self.mouse_x, self.mouse_y = x, y
        if self.drag_button is None:
            return False

        if np.hypot(x - self.drag_origin[0], y - self.drag_origin[1]) > 3:
            self.drag_moved = True
        delta_x, delta_y = x - previous_x, y - previous_y
        if delta_x == 0 and delta_y == 0:
            return False
        if self.drag_button == 0:
            self.pending_rotation += (delta_x, delta_y)
            if self.orbiting:
                self.orbit_last_time = time.perf_counter()
        else:
            self.pending_translation += (delta_x, delta_y)
        self.drag_last = (x, y)
        return False

    def _apply_smoothed_mouse_motion(self, now: float) -> None:
        elapsed = now - self.last_mouse_smoothing_time
        self.last_mouse_smoothing_time = now
        amount = 1.0 - np.exp(-elapsed / self.mouse_smoothing_time)
        view_control = self.visualizer.get_view_control()
        for pending, apply_motion in (
            (self.pending_rotation, view_control.rotate),
            (self.pending_translation, view_control.translate),
        ):
            applied = pending * amount
            pending -= applied
            if np.max(np.abs(applied)) >= 0.01:
                apply_motion(
                    float(applied[0] * self.mouse_sensitivity),
                    float(applied[1] * self.mouse_sensitivity),
                )
        if np.max(np.abs(self.pending_rotation)) < 0.01:
            self.pending_rotation.fill(0)
        if np.max(np.abs(self.pending_translation)) < 0.01:
            self.pending_translation.fill(0)
        if self.orbiting and np.any(self.pending_rotation):
            self.orbit_last_time = now

    def _on_mouse_button(
        self,
        _visualizer: o3d.visualization.Visualizer,
        button: int,
        action: int,
        _mods: int,
    ) -> bool:
        del _visualizer, _mods
        if action == 1 and button in {0, 1, 2}:
            self.drag_button = button
            self.drag_origin = (self.mouse_x, self.mouse_y)
            self.drag_last = self.drag_origin
            self.drag_moved = False
        elif action == 0 and button == self.drag_button:
            was_click = button == 0 and not self.drag_moved
            self.drag_button = None
            if was_click:
                self._pick_orbit_center(self.mouse_x, self.mouse_y)
        return False

    def _on_mouse_scroll(
        self,
        _visualizer: o3d.visualization.Visualizer,
        _x: float,
        y: float,
    ) -> bool:
        del _visualizer, _x
        if not y:
            return False

        view_control = self.visualizer.get_view_control()
        camera = view_control.convert_to_pinhole_camera_parameters()
        camera_pose = np.linalg.inv(camera.extrinsic)
        eye = camera_pose[:3, 3]
        offset = eye - self.orbit_center
        distance = float(np.linalg.norm(offset))
        if distance <= 0:
            return False

        bounds = self.geometry.get_axis_aligned_bounding_box()
        model_radius = max(float(np.max(bounds.get_extent())) * 0.5, 1e-6)
        zoom_factor = 1.08**y
        target_distance = float(
            np.clip(distance / zoom_factor, model_radius * 0.001, model_radius * 1000)
        )
        camera_pose[:3, 3] = self.orbit_center + offset * (target_distance / distance)
        camera.extrinsic = np.linalg.inv(camera_pose)
        if not view_control.convert_from_pinhole_camera_parameters(camera, allow_arbitrary=True):
            self._set_status("Yakınlaştırma uygulanamadı; kameranın geçerli uzaklığını kontrol edin.")
        return False

    def _pick_orbit_center(self, x: float, y: float) -> None:
        camera = self.visualizer.get_view_control().convert_to_pinhole_camera_parameters()
        picked_point = pick_visible_vertex(
            np.asarray(self.point_cloud.points),
            camera.extrinsic,
            camera.intrinsic.intrinsic_matrix,
            x,
            y,
            max(6.0, self.point_size * 1.5),
        )
        if picked_point is None:
            self._set_status("Bu noktada seçilebilir model noktası bulunamadı; modelin üzerine tıklayın.")
            return
        self._set_orbit_center(picked_point)
        self._set_status(
            "Dönüş merkezi seçildi: "
            + ", ".join(f"{coordinate:.4f}" for coordinate in self.orbit_center)
            + ". Döndürmek için 'donus basla' yazın."
        )

    def _update_orbit(self, now: float) -> None:
        previous_time = self.orbit_last_time
        self.orbit_last_time = now
        if not self.orbiting or previous_time is None:
            return

        elapsed = now - previous_time
        angle = np.deg2rad(self.orbit_speed * self.orbit_direction * elapsed)
        cosine, sine = np.cos(angle), np.sin(angle)
        rotation = np.eye(3)
        first_axis, second_axis = (1, 2) if self.orbit_axis == 0 else (
            (2, 0) if self.orbit_axis == 1 else (0, 1)
        )
        rotation[first_axis, first_axis] = cosine
        rotation[second_axis, second_axis] = cosine
        rotation[first_axis, second_axis] = -sine
        rotation[second_axis, first_axis] = sine

        view_control = self.visualizer.get_view_control()
        camera_parameters = view_control.convert_to_pinhole_camera_parameters()
        camera_pose = np.linalg.inv(camera_parameters.extrinsic)
        orbit_transform = np.eye(4)
        orbit_transform[:3, :3] = rotation
        orbit_transform[:3, 3] = self.orbit_center - rotation @ self.orbit_center
        camera_parameters.extrinsic = np.linalg.inv(orbit_transform @ camera_pose)
        if not view_control.convert_from_pinhole_camera_parameters(
            camera_parameters, allow_arbitrary=True
        ):
            self.orbiting = False
            self._set_status("Kamera dönüşü durduruldu; orbit kamera konumu uygulanamadı.")
