#!/usr/bin/env python3
"""Offline regression test for the phone face-detection pipeline."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
MODEL_PATH = ROOT / "models" / "npu_face" / "version-slim-320-clean.onnx"
OUTPUT_DIR = ROOT / "artifacts" / "face-detection-regression"
NPU_FIXTURE_PATH = ROOT / "entry" / "src" / "main" / "resources" / "rawfile" / "front_camera_face_input.bin"
INPUT_WIDTH = 320
INPUT_HEIGHT = 240
FEATURE_SIZES = ((40, 30), (20, 15), (10, 8), (5, 4))
MIN_BOXES = ((10, 16, 24), (32, 48), (64, 96), (128, 192, 256))


def make_priors() -> np.ndarray:
    priors: list[list[float]] = []
    for (feature_width, feature_height), min_boxes in zip(FEATURE_SIZES, MIN_BOXES):
        for y in range(feature_height):
            for x in range(feature_width):
                for box_size in min_boxes:
                    priors.append([
                        (x + 0.5) / feature_width,
                        (y + 0.5) / feature_height,
                        box_size / INPUT_WIDTH,
                        box_size / INPUT_HEIGHT,
                    ])
    return np.asarray(priors, dtype=np.float32)


def prepare_input(preview: Image.Image) -> tuple[np.ndarray, float, float]:
    preview_aspect = preview.width / preview.height
    input_aspect = INPUT_WIDTH / INPUT_HEIGHT
    x_scale = min(1.0, preview_aspect / input_aspect)
    y_scale = min(1.0, input_aspect / preview_aspect)
    content_width = max(1, round(INPUT_WIDTH * x_scale))
    content_height = max(1, round(INPUT_HEIGHT * y_scale))
    resized = preview.convert("RGB").resize((content_width, content_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (INPUT_WIDTH, INPUT_HEIGHT))
    canvas.paste(resized, ((INPUT_WIDTH - content_width) // 2, (INPUT_HEIGHT - content_height) // 2))
    rgb = np.asarray(canvas, dtype=np.float32)
    nchw = np.transpose((rgb - 127.0) / 128.0, (2, 0, 1))[None, ...]
    return nchw, x_scale, y_scale


def decode_best(scores: np.ndarray, boxes: np.ndarray,
                priors: np.ndarray) -> tuple[float, list[float], int, list[float]]:
    best_index = int(np.argmax(scores[0, :, 1]))
    confidence = float(scores[0, best_index, 1])
    location = boxes[0, best_index]
    prior = priors[best_index]
    center_x = float(location[0] * 0.1 * prior[2] + prior[0])
    center_y = float(location[1] * 0.1 * prior[3] + prior[1])
    width = float(math.exp(float(location[2]) * 0.2) * prior[2])
    height = float(math.exp(float(location[3]) * 0.2) * prior[3])
    return (
        confidence,
        [center_x - width / 2, center_y - height / 2, width, height],
        best_index,
        [float(value) for value in location],
    )


def map_from_letterbox(box: list[float], x_scale: float, y_scale: float) -> list[float]:
    content_left = (1.0 - x_scale) / 2.0
    content_top = (1.0 - y_scale) / 2.0
    left = (box[0] - content_left) / x_scale
    top = (box[1] - content_top) / y_scale
    return [left, top, box[2] / x_scale, box[3] / y_scale]


def map_to_letterbox(box: list[float], x_scale: float, y_scale: float) -> list[float]:
    return [
        (1.0 - x_scale) / 2.0 + box[0] * x_scale,
        (1.0 - y_scale) / 2.0 + box[1] * y_scale,
        box[2] * x_scale,
        box[3] * y_scale,
    ]


def iou(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[0] + first[2], second[0] + second[2])
    bottom = min(first[1] + first[3], second[1] + second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = first[2] * first[3] + second[2] * second[3] - intersection
    return intersection / union if union > 0 else 0.0


def draw_box(draw: ImageDraw.ImageDraw, box: list[float], size: tuple[int, int], color: str) -> None:
    width, height = size
    draw.rectangle(
        (
            round(box[0] * width),
            round(box[1] * height),
            round((box[0] + box[2]) * width),
            round((box[1] + box[3]) * height),
        ),
        outline=color,
        width=7,
    )


def assert_mapping_round_trip() -> None:
    for box in ([0.1, 0.2, 0.3, 0.4], [0.55, 0.05, 0.4, 0.8]):
        mapped = map_from_letterbox(map_to_letterbox(list(box), 0.5625, 1.0), 0.5625, 1.0)
        if not np.allclose(mapped, box, atol=1e-6):
            raise AssertionError(f"letterbox round trip failed: expected={box}, actual={mapped}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--export-npu-fixture",
        action="store_true",
        help="write the deterministic float32 tensor packaged by the HarmonyOS NPU self-test",
    )
    args = parser.parse_args()
    fixture_path = FIXTURE_DIR / "front_camera_face.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    screenshot = Image.open(FIXTURE_DIR / fixture["image"])
    preview = screenshot.crop(tuple(fixture["previewCrop"]))
    model_input, x_scale, y_scale = prepare_input(preview)
    if args.export_npu_fixture:
        NPU_FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        NPU_FIXTURE_PATH.write_bytes(model_input.astype("<f4", copy=False).tobytes())
        print(f"npu_fixture={NPU_FIXTURE_PATH}")

    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    scores, boxes = session.run(None, {session.get_inputs()[0].name: model_input})
    confidence, input_box, best_index, raw_location = decode_best(scores, boxes, make_priors())
    display_box = map_from_letterbox(input_box, x_scale, y_scale)
    expected_box = [float(value) for value in fixture["expectedFace"]]
    overlap = iou(display_box, expected_box)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    annotated = preview.copy()
    draw = ImageDraw.Draw(annotated)
    draw_box(draw, expected_box, annotated.size, "red")
    draw_box(draw, display_box, annotated.size, "lime")
    output_path = OUTPUT_DIR / "front_camera_face_actual.jpeg"
    annotated.save(output_path, quality=95)

    assert_mapping_round_trip()
    print(f"confidence={confidence:.6f}")
    print(f"best_index={best_index}")
    print(f"raw_location={[round(value, 6) for value in raw_location]}")
    print(f"input_box={[round(value, 6) for value in input_box]}")
    print(f"display_box={[round(value, 6) for value in display_box]}")
    print(f"expected_box={expected_box}")
    print(f"iou={overlap:.6f}")
    print(f"annotated={output_path}")

    failures: list[str] = []
    if confidence < float(fixture["minimumConfidence"]):
        failures.append("face confidence is below the fixture threshold")
    if overlap < float(fixture["minimumIou"]):
        failures.append("predicted box does not overlap the expected face")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print("PASS: offline face-detection regression")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
