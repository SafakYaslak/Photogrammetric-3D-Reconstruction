"""Terminal command handling and visual-style application."""
from __future__ import annotations

import shlex
import time
from pathlib import Path

import numpy as np
import open3d as o3d

from .appearance import (
    ORANGE,
    PALETTES,
    TURKISH_TRANSLATION,
    VISUAL_STYLES,
    parse_color,
)


class CommandMixin:
    def _handle_command(self, line: str) -> bool:
        try:
            words = shlex.split(line.strip())
        except ValueError as error:
            self._set_status(f"Komut okunamadı: {error}")
            return True
        if not words:
            return True
        command = words[0].translate(TURKISH_TRANSLATION).lower()
        args = [word.translate(TURKISH_TRANSLATION).lower() for word in words[1:]]
        try:
            if command in {"quit", "exit", "cikis", "q"}:
                return False
            if command in {"help", "yardim", "?"}:
                self._print_help()
            elif command in {"mode", "gorunum"} and len(args) == 1:
                if args[0] in {"points", "point", "nokta", "noktalar"}:
                    self.view = "points"
                elif args[0] in {"mesh", "yuzey"}:
                    if self.mesh is None:
                        raise ValueError("Bu PLY dosyasında yüzey bulunmuyor.")
                    self.view = "mesh"
                else:
                    raise ValueError("Kullanım: gorunum nokta|yuzey")
                self._update_geometry()
                self._set_status(f"Görünüm: {'yüzey' if self.view == 'mesh' else 'nokta bulutu'}")
            elif command in {"color", "colour", "renk"} and len(args) == 1:
                value = args[0]
                if value in {"original", "orijinal"}:
                    self.color_mode = "original"
                    if self.original_colors is None:
                        self.color_mode = "solid"
                        self.solid_color = parse_color(ORANGE)
                        self._set_status("Dosyada renk yok; varsayılan turuncu renk kullanılıyor.")
                    else:
                        self._set_status("PLY dosyasındaki orijinal renkler gösteriliyor.")
                elif value in {"height", "yukseklik"}:
                    self.color_mode = "height"
                    self._set_status(f"Yüksekliğe göre renklendirme etkin ({self.palette}).")
                else:
                    self.solid_color = parse_color(value)
                    self.color_mode = "solid"
                    self._set_status(f"Model rengi: {value}")
                self._apply_colors()
                self._update_geometry()
            elif command == "palette" and len(args) == 1:
                if args[0] not in PALETTES:
                    raise ValueError(
                        "Paletler: spectrum, ocean, sunset, cividis, plasma, inferno, magma"
                    )
                self.palette = args[0]
                if self.color_mode == "height":
                    self._apply_colors()
                    self._update_geometry()
                self._set_status(f"Yükseklik paleti: {self.palette}")
            elif command in {"point-size", "point_size", "nokta-boyutu", "boyut"} and len(args) == 1:
                size = float(args[0])
                if not 0.1 <= size <= 30:
                    raise ValueError("Nokta boyutu 0.1 ile 30 arasında olmalı.")
                self.point_size = size
                self.visualizer.get_render_option().point_size = size
                self.visualizer.update_renderer()
                self._set_status(f"Nokta boyutu: {size:g}")
            elif command in {"stil", "style"} and len(args) == 1:
                if args[0] not in VISUAL_STYLES:
                    raise ValueError(
                        "Stiller: makale, karanlik, monokrom, teknik, "
                        "bilimsel, sunum, pastel, kontrast"
                    )
                self._set_visual_style(args[0])
                self._set_status(f"Görsel stil uygulandı: {args[0]}")
            elif command in {"background", "arka-plan", "fon"} and len(args) == 1:
                self.background = parse_color(args[0])
                self.visualizer.get_render_option().background_color = np.asarray(self.background)
                self.visualizer.update_renderer()
                self._set_status(f"Arka plan: {args[0]}")
            elif command in {"grid", "izgara"} and len(args) == 1:
                if args[0] not in {"on", "off", "ac", "kapat"}:
                    raise ValueError("Kullanım: izgara ac|kapat")
                self.grid_visible = args[0] in {"on", "ac"}
                self._update_geometry()
                self._set_status(f"Izgara {'açık' if self.grid_visible else 'kapalı'}.")
            elif command in {"katman", "layer"} and len(args) == 2:
                layer = args[0]
                if args[1] not in {"ac", "kapat", "on", "off"}:
                    raise ValueError("Kullanım: katman izgara|eksen|kafes ac|kapat")
                visible = args[1] in {"ac", "on"}
                render_option = self.visualizer.get_render_option()
                if layer in {"izgara", "grid"}:
                    self.grid_visible = visible
                    self._update_geometry()
                elif layer in {"eksen", "koordinat", "axes"}:
                    self.coordinate_frame_visible = visible
                    render_option.show_coordinate_frame = visible
                    self.visualizer.update_renderer()
                elif layer in {"kafes", "tel-kafes", "wireframe"}:
                    self.wireframe_visible = visible
                    render_option.mesh_show_wireframe = visible
                    self.visualizer.update_renderer()
                else:
                    raise ValueError("Katmanlar: izgara, eksen, kafes")
                self._set_status(f"{layer} katmanı {'açık' if visible else 'kapalı'}.")
            elif command in {"kaydet", "save"} and len(words) == 2:
                output_path = Path(words[1]).expanduser().resolve()
                if output_path.suffix.lower() != ".png":
                    raise ValueError("Şekil çıktısı .png uzantılı olmalı.")
                if not output_path.parent.is_dir():
                    raise FileNotFoundError(
                        f"Çıktı klasörü bulunamadı: {output_path.parent}"
                    )
                if output_path.exists():
                    raise FileExistsError(
                        f"Dosya zaten var; üzerine yazılmadı: {output_path}. "
                        "Farklı bir dosya adı seçin."
                    )
                screenshot = self.visualizer.capture_screen_float_buffer(do_render=True)
                if screenshot.is_empty():
                    raise RuntimeError(
                        "Görüntü alınamadı; 3D pencerenin açık olduğundan emin olun."
                    )
                if not o3d.io.write_image(str(output_path), screenshot, 9):
                    raise OSError(f"PNG dosyası kaydedilemedi: {output_path}")
                height, width = np.asarray(screenshot).shape[:2]
                self._set_status(
                    f"PNG şekil kaydedildi: {output_path} ({width} × {height} piksel)"
                )
            elif command in {"video", "record"} and len(args) == 1:
                if args[0] in {"kaydet", "basla", "start"}:
                    output_path = self.recorder.start()
                    self._set_status(f"Kayıt hazırlanıyor: {output_path}")
                elif args[0] in {"durdur", "dur", "stop"}:
                    output_path = self.recorder.stop()
                    self._set_status(f"Kayıt durduruluyor: {output_path}")
                else:
                    raise ValueError("Kullanım: video kaydet|durdur")
            elif command in {"merkez", "origin", "center", "orbit-center"}:
                if len(args) == 1 and args[0] in {"model", "varsayilan"}:
                    self._set_orbit_center(self.geometry.get_axis_aligned_bounding_box().get_center())
                elif len(args) == 3:
                    center = np.asarray([float(value) for value in args], dtype=np.float64)
                    if not np.isfinite(center).all():
                        raise ValueError("Merkez koordinatları sonlu sayılar olmalı.")
                    self._set_orbit_center(center)
                else:
                    raise ValueError("Kullanım: model üzerine tıklayın | merkez X Y Z | merkez model")
                self._set_status(
                    "Dönüş merkezi: "
                    + ", ".join(f"{coordinate:.4f}" for coordinate in self.orbit_center)
                )
            elif command in {"donus", "orbit"} and len(args) == 1:
                if args[0] in {"basla", "start", "ac"}:
                    if self.orbit_speed <= 0:
                        raise ValueError("Önce pozitif bir hız ayarlayın: hiz 12")
                    self.orbiting = True
                    self.orbit_last_time = time.perf_counter()
                    self._set_status(
                        f"Otomatik dönüş başladı: {self.orbit_speed:g}°/sn, "
                        f"eksen {('X', 'Y', 'Z')[self.orbit_axis]}."
                    )
                elif args[0] in {"dur", "stop", "kapat"}:
                    self.orbiting = False
                    self.orbit_last_time = None
                    self._set_status("Otomatik dönüş durduruldu.")
                else:
                    raise ValueError("Kullanım: donus basla|dur")
            elif command in {"hiz", "speed", "orbit-speed"} and len(args) == 1:
                speed = float(args[0])
                if not np.isfinite(speed) or not 0 <= speed <= 180:
                    raise ValueError("Dönüş hızı 0 ile 180 derece/sn arasında olmalı.")
                self.orbit_speed = speed
                self._set_status(f"Dönüş hızı: {speed:g}°/sn")
            elif command in {"yon", "direction", "orbit-direction"} and len(args) == 1:
                if args[0] in {"saat", "saat-yonu", "clockwise", "cw"}:
                    self.orbit_direction = -1.0
                    direction_name = "saat yönü"
                elif args[0] in {"ters", "saat-yonu-tersi", "counterclockwise", "ccw"}:
                    self.orbit_direction = 1.0
                    direction_name = "saat yönünün tersi"
                else:
                    raise ValueError("Kullanım: yon saat|ters")
                self._set_status(f"Dönüş yönü: {direction_name}")
            elif command in {"eksen", "axis", "orbit-axis"} and len(args) == 1:
                if args[0] not in {"x", "y", "z"}:
                    raise ValueError("Dönüş ekseni x, y veya z olmalı.")
                self.orbit_axis = {"x": 0, "y": 1, "z": 2}[args[0]]
                self._set_status(f"Dönüş ekseni: {args[0].upper()}")
            elif command in {"reset", "sifirla"} and not args:
                self.visualizer.reset_view_point(True)
                self.visualizer.update_renderer()
                self._set_status("Kamera sıfırlandı.")
            else:
                self._set_status("Komut anlaşılmadı. Komut listesini görmek için 'yardim' yazın.")
        except (ValueError, TypeError) as error:
            self._set_status(f"Hata: {error}")
        except (OSError, RuntimeError) as error:
            self._set_status(f"Dosya veya görüntü işlemi başarısız: {error}")
        return True

    def _set_visual_style(self, style: str) -> None:
        profile = VISUAL_STYLES[style]
        self.visual_style = style
        self.background = parse_color(str(profile["background"]))
        self.grid_color = profile["grid"]
        self.grid_visible = bool(profile["grid_visible"])
        self.coordinate_frame_visible = bool(profile["coordinate_frame_visible"])
        self.wireframe_visible = bool(profile["wireframe_visible"])
        if "color_mode" in profile:
            self.color_mode = str(profile["color_mode"])
        if "palette" in profile:
            self.palette = str(profile["palette"])
        render_option = self.visualizer.get_render_option()
        render_option.background_color = np.asarray(self.background)
        render_option.show_coordinate_frame = self.coordinate_frame_visible
        render_option.mesh_show_wireframe = self.wireframe_visible
        self._apply_colors()
        self._update_geometry()

    @staticmethod
    def _print_help() -> None:
        print(
            "\nKomutlar:\n"
            "  gorunum nokta|yuzey       Görünümü değiştir\n"
            "  renk turuncu|#RRGGBB     Tek renk seç\n"
            "  renk orijinal|yukseklik  Dosya renklerine veya Z yüksekliğine göre boya\n"
            "  palette spectrum|ocean|sunset|cividis|plasma|inferno|magma\n"
            "  stil makale|karanlik|monokrom|teknik|bilimsel|sunum|pastel|kontrast\n"
            "  boyut 2.5                Nokta boyutunu ayarla (0.1–30)\n"
            "  arka-plan #FFFFFF        Arka plan rengini ayarla\n"
            "  izgara ac|kapat          Referans ızgarasını göster/gizle\n"
            "  katman izgara|eksen|kafes ac|kapat\n"
            "  kaydet sekil.png         Görünümü PNG olarak kaydet\n"
            "  video kaydet             PLY Studio penceresini yüksek kaliteli videoya kaydet\n"
            "  video durdur             Video kaydını bitir\n"
            "  model üzerine sol tıkla  Tıklanan noktayı dönüş merkezi yap\n"
            "  merkez X Y Z             Otomatik dönüş merkezini elle ayarla\n"
            "  merkez model             Merkezi modelin ortasına al\n"
            "  donus basla|dur          Otomatik kamera dönüşünü başlat/durdur\n"
            "  hiz 12                   Dönüş hızını derece/sn olarak ayarla\n"
            "  yon saat|ters            Dönüş yönünü ayarla\n"
            "  eksen x|y|z              Dönüş eksenini ayarla (varsayılan: z)\n"
            "  sifirla                  Kamerayı modele sığdır\n"
            "  yardim                   Bu komutları göster\n"
            "  cikis                    Görüntüleyiciyi kapat\n"
            "Fare: sol tuşla döndür, tekerleği yukarı çevirerek yakınlaştır ve aşağı çevirerek uzaklaştır; sağ tuşla kaydır.\n",
            flush=True,
        )
