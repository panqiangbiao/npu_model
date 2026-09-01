# MediaPipe Face Mesh assets

- `face_mesh_192.onnx` comes from `yakhyo/mediapipe-face-mesh-onnx`.
- `canonical_face_model.obj` and `facepaint.pngblob` come from Google MediaPipe.
- The model, topology, canonical model, and reference effect are Apache-2.0 licensed.
- `tools/prepare_face_mesh_onnx.py` removes the default MaxPool dilation attribute that MindSpore Lite 2.0 cannot parse.
- `tools/generate_face_mesh_assets.py` generates the native UV/index table and application mask texture.

The generated mask is an original demo texture. It does not include third-party character artwork.
