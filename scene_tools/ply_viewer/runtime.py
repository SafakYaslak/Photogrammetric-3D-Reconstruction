"""Application lifecycle, terminal input, and video capture integration."""
from __future__ import annotations

import queue
import sys
import threading
import time

import numpy as np

from ..recording.screen_recorder import FPS


class RuntimeMixin:
    def _read_commands(self) -> None:
        while True:
            try:
                line = input("> ")
            except EOFError:
                self.commands.put("cikis")
                return
            self.commands.put(line)
            if line.strip().lower() in {"quit", "exit", "cikis", "q"}:
                return

    def run(self) -> int:
        self.load_model()
        if self.view == "mesh" and self.mesh is None:
            self.view = "points"
        self._apply_colors()

        self.visualizer.register_mouse_move_callback(self._on_mouse_move)
        self.visualizer.register_mouse_button_callback(self._on_mouse_button)
        self.visualizer.register_mouse_scroll_callback(self._on_mouse_scroll)
        if not self.visualizer.create_window(
            window_name=f"PLY Studio — {self.path.name}",
            width=self.window_width,
            height=self.window_height,
        ):
            raise RuntimeError("3D pencere oluşturulamadı. Grafik kartı sürücüsünü kontrol edin.")

        render_option = self.visualizer.get_render_option()
        render_option.background_color = np.asarray(self.background)
        render_option.point_size = self.point_size
        render_option.mesh_show_back_face = True
        render_option.light_on = True
        render_option.show_coordinate_frame = self.coordinate_frame_visible
        render_option.mesh_show_wireframe = self.wireframe_visible
        self._update_geometry(reset_camera=True)

        self._print_help()
        threading.Thread(target=self._read_commands, daemon=True, name="terminal-input").start()
        print("Model hazır. Komutları bu terminale yazın.", flush=True)
        try:
            while self.visualizer.poll_events():
                self._drain_recording_notifications()
                while True:
                    try:
                        command = self.commands.get_nowait()
                    except queue.Empty:
                        break
                    if not self._handle_command(command):
                        return 0
                if self.orbiting:
                    self._update_orbit(time.perf_counter())
                self.visualizer.update_renderer()
                self._apply_smoothed_mouse_motion(time.perf_counter())
                self._capture_recording_frame()
                time.sleep(0.01)
        finally:
            if self.recorder.is_running:
                try:
                    self.recorder.stop()
                    self.recorder.wait()
                    self._drain_recording_notifications()
                except RuntimeError as error:
                    print(f"HATA: Video kaydı kapatılırken sorun oluştu: {error}", file=sys.stderr)
            self.recorder.close_capture()
            self.visualizer.destroy_window()
        return 0

    def _drain_recording_notifications(self) -> None:
        while True:
            try:
                event, message, width, height = self.recorder.notifications.get_nowait()
            except queue.Empty:
                return
            if event == "started":
                self._set_status(
                    f"PLY Studio görüntüsü kaydediliyor: {message} "
                    f"({width} × {height} piksel, {FPS:g} FPS hedef). "
                    "Bitirmek için 'video durdur' yazın."
                )
            elif event == "stopped":
                self._set_status(f"Video kaydedildi: {message}")
            elif event == "error":
                self._set_status(f"HATA: Video kaydı başarısız: {message}")

    def _capture_recording_frame(self) -> None:
        if not self.recorder.is_running:
            return
        now = time.perf_counter()
        if now - self.last_recording_frame_time < 1.0 / FPS:
            return
        self.last_recording_frame_time = now
        try:
            frame = self.recorder.capture_window_frame()
            self.recorder.submit_frame(frame)
        except (OSError, RuntimeError, ValueError) as error:
            self._set_status(f"HATA: PLY Studio penceresi kaydedilemedi: {error}")
            if self.recorder.is_running:
                self.recorder.stop()
