#!/usr/bin/env python3
"""Extract evenly spaced video frames for a COLMAP GUI project.

Example (Windows PowerShell):
    python extract_colmap_frames.py --video "D:\\Videos\\oda.mp4" --output "D:\\COLMAP\\oda\\images" --fps 2 --max-side 1600

Requires:
    python -m pip install opencv-python

The output directory should be empty to avoid mixing frames from old runs.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    print("HATA: OpenCV kurulu değil. Şunu çalıştırın: python -m pip install opencv-python", file=sys.stderr)
    raise SystemExit(1)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Videodan COLMAP için eşit aralıklı JPG görüntüleri çıkarır."
    )
    parser.add_argument("--video", required=True, help="Girdi video dosyasının yolu")
    parser.add_argument("--output", required=True, help="Çıkarılan görüntülerin kaydedileceği klasör")
    parser.add_argument("--fps", type=float, default=2.0,
                        help="Videodan saniyede kaç görüntü çıkarılsın (varsayılan: 2)")
    parser.add_argument("--max-side", type=int, default=1600,
                        help="Görüntünün en uzun kenarı için üst sınır; büyütme yapılmaz (varsayılan: 1600)")
    parser.add_argument("--jpg-quality", type=int, default=95,
                        help="JPG kalitesi 1-100 (varsayılan: 95)")
    parser.add_argument("--min-sharpness", type=float, default=0.0,
                        help="İsteğe bağlı bulanıklık filtresi; 0 kapalıdır (varsayılan: 0)")
    return parser.parse_args()


def validate_options(args: argparse.Namespace, video_path: Path) -> int | None:
    if not video_path.is_file():
        print(f"HATA: Video bulunamadı: {video_path}", file=sys.stderr)
        return 2
    if args.fps <= 0:
        print("HATA: --fps sıfırdan büyük olmalı.", file=sys.stderr)
        return 2
    if args.max_side < 1:
        print("HATA: --max-side en az 1 olmalı.", file=sys.stderr)
        return 2
    if not 1 <= args.jpg_quality <= 100:
        print("HATA: --jpg-quality 1 ile 100 arasında olmalı.", file=sys.stderr)
        return 2
    if args.min_sharpness < 0:
        print("HATA: --min-sharpness negatif olamaz.", file=sys.stderr)
        return 2
    return None


def _show_progress(read_count: int, total_frames: int, saved: int) -> None:
    if total_frames > 0:
        pct = min(100.0, 100.0 * read_count / total_frames)
        print(
            f"\rOkunan: {read_count}/{total_frames} (%{pct:5.1f}) | Kaydedilen: {saved}",
            end="",
            flush=True,
        )
    else:
        print(f"\rOkunan: {read_count} | Kaydedilen: {saved}", end="", flush=True)


def _resize_frame(frame: np.ndarray, max_side: int) -> tuple[np.ndarray, int, int]:
    height, width = frame.shape[:2]
    longest = max(width, height)
    if longest > max_side:
        scale = max_side / float(longest)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        frame = cv2.resize(
            frame,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA,
        )
        height, width = frame.shape[:2]
    return frame, width, height


def extract_frames(args: argparse.Namespace) -> int:
    video_path = Path(args.video).expanduser()
    output_dir = Path(args.output).expanduser()

    validation_error = validate_options(args, video_path)
    if validation_error is not None:
        return validation_error

    output_dir.mkdir(parents=True, exist_ok=True)
    existing_images = [p for p in output_dir.iterdir()
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    if existing_images:
        print(
            f"HATA: Çıktı klasöründe zaten {len(existing_images)} görüntü var:\n"
            f"{output_dir}\n"
            "COLMAP veri setlerinin karışmaması için boş bir klasör seçin.",
            file=sys.stderr,
        )
        return 2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(
            "HATA: Video açılamadı. Dosyanın sağlam olduğunu ve OpenCV'nin video codec'ini desteklediğini kontrol edin.",
            file=sys.stderr,
        )
        return 3

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if not math.isfinite(source_fps) or source_fps <= 0:
        cap.release()
        print("HATA: Videonun FPS bilgisi okunamadı.", file=sys.stderr)
        return 3

    # For a typical constant-frame-rate phone video, this gives approximately
    # the requested output FPS while reading the video only once.
    frame_step = max(1, int(round(source_fps / args.fps)))
    effective_fps = source_fps / frame_step
    duration = total_frames / source_fps if total_frames > 0 else 0.0

    print("COLMAP görüntü çıkarma başladı")
    print(f"Video       : {video_path}")
    print(f"Kaynak FPS  : {source_fps:.3f}")
    print(f"Tahmini süre: {duration / 60:.2f} dakika")
    print(f"Çıktı FPS   : yaklaşık {effective_fps:.3f}")
    print(f"Maks. boyut : {args.max_side} px")
    print(f"Çıktı klasörü: {output_dir}")
    if args.min_sharpness > 0:
        print(f"Bulanıklık filtresi: Laplacian varyansı < {args.min_sharpness:g} olan kareler atlanacak")
    print()

    manifest_path = output_dir / "frames_manifest.csv"
    saved = 0
    read_count = 0
    skipped_blurry = 0
    started = time.monotonic()
    last_report = started

    try:
        with manifest_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=["image", "source_frame", "timestamp_seconds", "sharpness", "width", "height"],
            )
            writer.writeheader()

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                source_frame = read_count
                read_count += 1

                if source_frame % frame_step != 0:
                    now = time.monotonic()
                    if now - last_report >= 2.0:
                        _show_progress(read_count, total_frames, saved)
                        last_report = now
                    continue

                sharpness = 0.0
                if args.min_sharpness > 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                    if sharpness < args.min_sharpness:
                        skipped_blurry += 1
                        continue

                frame, width, height = _resize_frame(frame, args.max_side)

                saved += 1
                filename = f"frame_{saved:06d}.jpg"
                image_path = output_dir / filename
                written = cv2.imwrite(
                    str(image_path), frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), args.jpg_quality],
                )
                if not written:
                    print(f"\nHATA: Görüntü yazılamadı: {image_path}", file=sys.stderr)
                    return 4

                timestamp = source_frame / source_fps
                writer.writerow({
                    "image": filename,
                    "source_frame": source_frame,
                    "timestamp_seconds": f"{timestamp:.3f}",
                    "sharpness": f"{sharpness:.3f}" if args.min_sharpness > 0 else "",
                    "width": width,
                    "height": height,
                })

                now = time.monotonic()
                if now - last_report >= 2.0:
                    _show_progress(read_count, total_frames, saved)
                    last_report = now

    finally:
        cap.release()

    elapsed = time.monotonic() - started
    print("\n\nTamamlandı.")
    print(f"Okunan kare  : {read_count}")
    print(f"Kaydedilen   : {saved}")
    print(f"Bulanık atlanan: {skipped_blurry}")
    print(f"Süre         : {elapsed:.1f} saniye")
    print(f"Görüntüler   : {output_dir}")
    print(f"Kayıt listesi: {manifest_path}")

    if saved == 0:
        print("UYARI: Hiç görüntü kaydedilmedi. --min-sharpness değerini düşürün veya videoyu kontrol edin.")
        return 5
    return 0


def main() -> int:
    return extract_frames(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
