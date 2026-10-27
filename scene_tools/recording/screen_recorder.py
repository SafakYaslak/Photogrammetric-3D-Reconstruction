"""Capture the PLY Studio window and encode its frames in the background."""

from __future__ import annotations

import queue
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import mss
import numpy as np


FPS = 30.0


class ScreenRecorder:
    def __init__(self, window_title: str, output_dir: Path) -> None:
        self.window_title = window_title
        self.output_dir = output_dir
        self.output_path: Path | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._frames: queue.Queue[np.ndarray] = queue.Queue(maxsize=2)
        self.notifications: queue.Queue[tuple[str, str, int | None, int | None]] = queue.Queue()
        self._window_api: Any = None
        self._window_handle: Any = None
        self._screen_capture: mss.mss | None = None
        self._capture_size: tuple[int, int] | None = None

    def _find_window(self) -> tuple[Any, Any]:
        import ctypes
        import os
        from ctypes import wintypes

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.EnumWindows.argtypes = (
            ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM),
            wintypes.LPARAM,
        )
        user32.EnumWindows.restype = wintypes.BOOL
        user32.IsWindowVisible.argtypes = (wintypes.HWND,)
        user32.IsWindowVisible.restype = wintypes.BOOL
        user32.GetWindowTextLengthW.argtypes = (wintypes.HWND,)
        user32.GetWindowTextLengthW.restype = ctypes.c_int
        user32.GetWindowTextW.argtypes = (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int)
        user32.GetWindowTextW.restype = ctypes.c_int
        user32.GetWindowThreadProcessId.argtypes = (
            wintypes.HWND,
            ctypes.POINTER(wintypes.DWORD),
        )
        user32.GetWindowThreadProcessId.restype = wintypes.DWORD
        matches: list[Any] = []

        def find_visible_window(hwnd: Any, lparam: Any) -> bool:
            del lparam
            if not user32.IsWindowVisible(hwnd):
                return True
            process_id = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
            if process_id.value != os.getpid():
                return True
            title = ctypes.create_unicode_buffer(
                user32.GetWindowTextLengthW(hwnd) + 1
            )
            user32.GetWindowTextW(hwnd, title, len(title))
            if title.value.startswith(self.window_title):
                matches.append(hwnd)
                return False
            return True

        callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        callback = callback_type(find_visible_window)
        user32.EnumWindows(callback, 0)
        if not matches:
            raise RuntimeError("PLY Studio penceresi bulunamadı.")
        return user32, matches[0]

    def capture_window_frame(self) -> np.ndarray:
        import ctypes
        from ctypes import wintypes

        if self._window_api is None or self._window_handle is None:
            self._window_api, self._window_handle = self._find_window()
            self._window_api.GetWindowRect.argtypes = (
                wintypes.HWND,
                ctypes.POINTER(wintypes.RECT),
            )
            self._window_api.GetWindowRect.restype = wintypes.BOOL
            self._window_api.IsIconic.argtypes = (wintypes.HWND,)
            self._window_api.IsIconic.restype = wintypes.BOOL
            self._screen_capture = mss.mss()

        if self._window_api.IsIconic(self._window_handle):
            raise RuntimeError("PLY Studio penceresi küçültülmüş; kayda devam etmek için geri yükleyin.")
        rect = wintypes.RECT()
        if not self._window_api.GetWindowRect(
            self._window_handle, ctypes.byref(rect)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        region = {
            "left": rect.left,
            "top": rect.top,
            "width": rect.right - rect.left,
            "height": rect.bottom - rect.top,
        }
        if region["width"] < 1 or region["height"] < 1:
            raise RuntimeError("PLY Studio penceresinin görünür boyutu yok.")
        size = (region["height"], region["width"])
        if self._capture_size is None:
            self._capture_size = size
        elif size != self._capture_size:
            raise RuntimeError(
                "Kayıt sırasında PLY Studio penceresinin boyutu değişti. "
                "Kaydı durdurup yeniden başlatın."
            )
        if self._screen_capture is None:
            raise RuntimeError("Windows pencere yakalayıcısı başlatılamadı.")
        return np.asarray(self._screen_capture.grab(region))[:, :, :3].copy()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> Path:
        if self.is_running:
            raise RuntimeError("Video kaydı zaten devam ediyor.")

        self._stop_event.clear()
        while not self._frames.empty():
            try:
                self._frames.get_nowait()
            except queue.Empty:
                break
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.output_path = self.output_dir / f"ply-studio-{timestamp}.avi"
        self._thread = threading.Thread(
            target=self._record,
            args=(self.output_path,),
            name="screen-recorder",
            daemon=True,
        )
        self._thread.start()
        return self.output_path

    def submit_frame(self, frame: np.ndarray) -> None:
        if not self.is_running or self._stop_event.is_set():
            return
        try:
            self._frames.put_nowait(frame)
        except queue.Full:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frames.put_nowait(frame)
            except queue.Full:
                pass

    def stop(self) -> Path:
        if not self.is_running:
            raise RuntimeError("Şu anda devam eden bir video kaydı yok.")
        self._stop_event.set()
        if self.output_path is None:
            raise RuntimeError("Video kayıt dosyası oluşturulamadı.")
        return self.output_path

    def wait(self) -> None:
        thread = self._thread
        if thread is not None:
            thread.join()

    def close_capture(self) -> None:
        if self._screen_capture is not None:
            self._screen_capture.close()
            self._screen_capture = None
        self._window_api = None
        self._window_handle = None
        self._capture_size = None

    def _record(self, output_path: Path) -> None:
        writer: cv2.VideoWriter | None = None
        frame_count = 0
        frame_size: tuple[int, int] | None = None
        error_message: str | None = None
        try:
            while not self._stop_event.is_set() or not self._frames.empty():
                try:
                    frame = self._frames.get(timeout=0.05)
                except queue.Empty:
                    continue
                height, width = frame.shape[:2]
                if writer is None:
                    frame_size = (height, width)
                    video_width = width + width % 2
                    video_height = height + height % 2
                    self.output_dir.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter_fourcc(*"MJPG"),
                        FPS,
                        (video_width, video_height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(
                            "MJPG video kodlayıcısı açılamadı; OpenCV/FFmpeg kurulumunu kontrol edin."
                        )
                    self.notifications.put(("started", str(output_path), width, height))
                elif frame.shape[:2] != frame_size:
                        raise RuntimeError(
                            "Kayıt sırasında PLY Studio görüntü çözünürlüğü değişti."
                        )
                if frame.shape[1] % 2 or frame.shape[0] % 2:
                        frame = cv2.copyMakeBorder(
                            frame,
                            0,
                            frame.shape[0] % 2,
                            0,
                            frame.shape[1] % 2,
                            cv2.BORDER_REPLICATE,
                        )
                writer.write(frame)
                frame_count += 1
        except Exception as error:
            error_message = str(error)
            self.notifications.put(
                ("error", f"{error} (kısmi dosya: {output_path})", None, None)
            )
        finally:
            if writer is not None:
                writer.release()
            if error_message is None and frame_count:
                self.notifications.put(("stopped", str(output_path), None, None))
            elif error_message is None:
                self.notifications.put(
                    (
                        "error",
                        "Kayıt durdurulmadan önce PLY Studio görüntüsü alınamadı.",
                        None,
                        None,
                    )
                )
