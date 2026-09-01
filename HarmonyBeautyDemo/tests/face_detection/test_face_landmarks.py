#!/usr/bin/env python3
"""Offline regression and NPU fixture generator for the 68-point PFLD model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
MODEL_PATH = ROOT / "models" / "npu_landmark" / "landmarks_68_pfld_fixed.onnx"
RAWFILE_DIR = ROOT / "entry" / "src" / "main" / "resources" / "rawfile"
OUTPUT_DIR = ROOT / "artifacts" / "beauty-debug" / "landmark-offline"
MODEL_SIZE = 112
FACE_BOX = [0.263269, 0.169422, 0.567942, 0.577192]


def prepare_input(image: Image.Image, face_box: list[float]) -> tuple[np.ndarray, tuple[float, ...]]:
    width, height = image.size
    x, y, box_width, box_height = (
        face_box[0] * width,
        face_box[1] * height,
        face_box[2] * width,
        face_box[3] * height,
    )
    padding = int(max(box_width, box_height) * 0.1)
    crop_x = max(0, int(x) - padding)
    crop_y = max(0, int(y) - padding)
    crop_width = min(width - crop_x, int(box_width) + 2 * padding)
    crop_height = min(height - crop_y, int(box_height) + 2 * padding)
    crop = image.convert("RGB").crop((crop_x, crop_y, crop_x + crop_width, crop_y + crop_height))

    scale = min(MODEL_SIZE / crop_width, MODEL_SIZE / crop_height)
    scaled_width = max(1, round(crop_width * scale))
    scaled_height = max(1, round(crop_height * scale))
    resized = crop.resize((scaled_width, scaled_height), Image.Resampling.BILINEAR)
    pad_x = (MODEL_SIZE - scaled_width) // 2
    pad_y = (MODEL_SIZE - scaled_height) // 2
    canvas = Image.new("RGB", (MODEL_SIZE, MODEL_SIZE), (128, 128, 128))
    canvas.paste(resized, (pad_x, pad_y))
    rgb = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = np.transpose(rgb, (2, 0, 1))[None, ...]
    transform = (float(crop_x), float(crop_y), float(crop_width), float(crop_height),
                 float(pad_x), float(pad_y), float(scale))
    return tensor, transform


def map_landmarks(output: np.ndarray, transform: tuple[float, ...]) -> np.ndarray:
    crop_x, crop_y, _, _, pad_x, pad_y, scale = transform
    points = output.reshape(68, 2) * MODEL_SIZE
    points[:, 0] = crop_x + (points[:, 0] - pad_x) / scale
    points[:, 1] = crop_y + (points[:, 1] - pad_y) / scale
    return points


def main() -> int:
    fixture = json.loads((FIXTURE_DIR / "front_camera_face.json").read_text(encoding="utf-8"))
    screenshot = Image.open(FIXTURE_DIR / fixture["image"])
    preview = screenshot.crop(tuple(fixture["previewCrop"]))
    model_input, transform = prepare_input(preview, FACE_BOX)
    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    output = session.run(None, {session.get_inputs()[0].name: model_input})[0].astype(np.float32)
    if output.size != 136 or not np.isfinite(output).all():
        raise AssertionError(f"invalid landmark output: shape={output.shape}")
    points = map_landmarks(output, transform)

    RAWFILE_DIR.mkdir(parents=True, exist_ok=True)
    (RAWFILE_DIR / "front_camera_landmark_input.bin").write_bytes(model_input.astype("<f4").tobytes())
    (RAWFILE_DIR / "front_camera_landmark_expected.bin").write_bytes(output.astype("<f4").tobytes())

    annotated = preview.copy()
    draw = ImageDraw.Draw(annotated)
    for index, (x, y) in enumerate(points):
        radius = 4 if index in (36, 39, 42, 45, 30, 48, 54) else 3
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="lime")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "front_camera_68_points.jpeg"
    annotated.save(output_path, quality=95)
    print(f"input_shape={model_input.shape}")
    print(f"output_range=({output.min():.6f}, {output.max():.6f})")
    print(f"transform={transform}")
    print(f"annotated={output_path}")
    print("PASS: generated 68-point landmark fixture")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
