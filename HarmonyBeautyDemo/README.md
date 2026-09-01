# HarmonyBeautyDemo

HarmonyOS 6.0 / API 21 real-time beauty camera demo for DevEco Studio 6.0.

## Features

- Front-camera preview with front/back camera switching.
- HarmonyOS Camera Control Center integration for the system face-aware beauty effect.
- Lightweight preview controls for smoothing, brightness, and complexion.
- Natural, clear, and warm presets.
- Preview-frame capture with an in-app thumbnail.
- Camera resource cleanup during camera switching and page destruction.
- Selectable NNRT/NPU, MNN-HiAI/NPU, MindSpore Lite CPU, and MNN OpenCL/GPU inference backends.

## Effect Boundary

The system beauty switch enables the Camera Control Center capability provided by HarmonyOS. Its beauty effect is implemented by the system camera pipeline and may use the system face models and NPU.

The three in-app sliders are compositor-level preview effects. They are intentionally kept separate from the system beauty state and should not be treated as face-aware skin processing.

## Run

1. Open this directory in DevEco Studio 6.0.
2. Select the connected HarmonyOS phone.
3. Enable automatic signing for bundle `com.example.harmonybeautydemo` when prompted.
4. Run the `entry` module and grant camera permission.

Command-line build used for verification:

```powershell
$env:DEVECO_SDK_HOME='C:\Program Files\Huawei\DevEco Studio\sdk'
$env:JAVA_HOME='C:\Program Files\Huawei\DevEco Studio\jbr'
& 'C:\Program Files\Huawei\DevEco Studio\tools\hvigor\bin\hvigorw.bat' assembleHap `
  --mode module -p product=default -p module=entry@default -p buildMode=debug --no-daemon
```

## NPU Face Detection

The demo packages the MIT-licensed `version-slim-320` model from
`Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB`. The ONNX model is converted to
MindSpore Lite format and loaded with `target: ['nnrt']`, selecting the NNRT accelerator device at runtime.

```text
Camera external texture
  -> 320x240 letterboxed OpenGL framebuffer every 8 frames
  -> RGB float32 normalization
  -> MindSpore Lite -> NNRT -> Kirin NPU
  -> SSD prior-box decode
  -> normalized face region consumed by the GPU beauty shader
```

Only the latest analysis frame is retained; stale frames are dropped so inference cannot block the preview.
On the HUAWEI Pura 90 Pro used for verification, live model execution was 3-14 ms per sampled frame.
The current demo uses a small synchronous `glReadPixels` readback. A production pipeline should replace it
with a camera analysis stream or asynchronous PBO/shared-buffer path.

## Backend Comparison

The in-app `NPU / MNN-NPU / CPU / GPU` segmented control keeps the camera input, sampling interval, models,
post-processing, and GPU effect renderer unchanged while selecting the inference backend:

- `NPU`: MindSpore Lite with NNRT accelerator context.
- `MNN-NPU`: MNN 3.6 with the mobiInfer HiAI backend (`MNN_FORWARD_USER_1`). The app verifies the
  actual session backend and rejects CPU fallback.
- `CPU`: MindSpore Lite CPU context with four threads, using the same face detector and 68-point model.
- `GPU`: MNN 3.6 OpenCL sessions for the same face detector and 68-point landmark model. Session creation
  verifies `MNN_FORWARD_OPENCL` before inference and reports an explicit error if OpenCL is unavailable.

`BeautyDemo` hilog entries expose `live`, `landmarks`, and `total` timings for trace correlation. The CPU path
was verified on-device with face detection and 68 landmarks updating correctly.

The MindSpore Lite conversion command used here is:

```powershell
converter_lite.exe --fmk=ONNX `
  --modelFile=models\npu_face\version-slim-320-clean.onnx `
  --outputFile=models\npu_face\version-slim-320
```

The GPU models are converted from the same ONNX sources with `mnnconvert`:

```powershell
mnnconvert.exe -f ONNX --modelFile models\npu_face\version-slim-320-clean.onnx `
  --MNNModel entry\src\main\resources\rawfile\gpu_face_detector.mnn --bizCode MNN
mnnconvert.exe -f ONNX --modelFile models\npu_landmark\landmarks_68_pfld_fixed.onnx `
  --MNNModel entry\src\main\resources\rawfile\gpu_landmarks_68.mnn --bizCode MNN
```

The packaged `libMNN.so` contains the mobiInfer HiAI backend. OpenCL is packaged as the separately built
`libMNN_CL.so` plugin, so the same app process can register both MNN NPU and GPU backends. MNN public headers and license are
stored under `entry/src/main/cpp/third_party/mnn/`.
