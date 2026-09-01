# Face detection offline regression

This test replays a captured front-camera scene through the same UltraFace
preprocessing, SSD prior decoding, and letterbox coordinate mapping used by the
HarmonyOS demo.

Red is the expected face region. Green is the decoded model result.

```powershell
python tests/face_detection/test_face_detector.py
```

The annotated result is written to
`artifacts/face-detection-regression/front_camera_face_actual.jpeg`.

Required Python packages: `numpy`, `Pillow`, `onnxruntime`.

To regenerate the deterministic input used by the on-device NNRT self-test:

```powershell
python tests/face_detection/test_face_detector.py --export-npu-fixture
```
