import tempfile
import unittest
import argparse
import contextlib
import csv
import io
from pathlib import Path

import cv2
import numpy as np

from scene_tools.camera_viewer.scene import rotation_matrix
from scene_tools.colmap_frames import extract_frames
from scene_tools.ply_viewer.appearance import parse_color
from scene_tools.ply_viewer.geometry import header_has_faces


class ColorParsingTests(unittest.TestCase):
    def test_expands_short_hex_color(self) -> None:
        self.assertEqual(parse_color("#abc"), (170 / 255, 187 / 255, 204 / 255))

    def test_parses_named_color_case_insensitively(self) -> None:
        self.assertEqual(parse_color("WHITE"), (1.0, 1.0, 1.0))

    def test_rejects_invalid_color(self) -> None:
        with self.assertRaises(ValueError):
            parse_color("not-a-color")


class PLYHeaderTests(unittest.TestCase):
    def test_detects_non_empty_face_element(self) -> None:
        contents = b"ply\nformat ascii 1.0\nelement face 2\nend_header\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mesh.ply"
            path.write_bytes(contents)
            self.assertTrue(header_has_faces(path))

    def test_accepts_point_cloud_header_without_faces(self) -> None:
        contents = b"ply\nformat ascii 1.0\nelement vertex 3\nend_header\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cloud.ply"
            path.write_bytes(contents)
            self.assertFalse(header_has_faces(path))


class RotationMatrixTests(unittest.TestCase):
    def test_rotation_preserves_vector_length(self) -> None:
        vector = np.array([1.0, 2.0, 3.0])
        rotated = rotation_matrix(2, np.pi / 3) @ vector
        self.assertAlmostEqual(float(np.linalg.norm(rotated)), float(np.linalg.norm(vector)))


class FrameExtractionTests(unittest.TestCase):
    def test_extracts_sampled_and_resized_frames_with_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_path = root / "sample.avi"
            output_dir = root / "frames"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                4.0,
                (64, 48),
            )
            self.assertTrue(writer.isOpened())
            for value in (0, 60, 120, 180):
                writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
            writer.release()

            args = argparse.Namespace(
                video=str(video_path),
                output=str(output_dir),
                fps=2.0,
                max_side=32,
                jpg_quality=90,
                min_sharpness=0.0,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                result = extract_frames(args)

            self.assertEqual(result, 0)
            images = sorted(output_dir.glob("frame_*.jpg"))
            self.assertEqual(len(images), 2)
            image = cv2.imread(str(images[0]))
            self.assertEqual(image.shape[:2], (24, 32))
            with (output_dir / "frames_manifest.csv").open(
                encoding="utf-8-sig", newline=""
            ) as manifest_file:
                records = list(csv.DictReader(manifest_file))
            self.assertEqual(len(records), 2)
            self.assertEqual(records[1]["source_frame"], "2")


if __name__ == "__main__":
    unittest.main()
