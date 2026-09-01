# MNN HiAI and OpenCL build notes

The packaged `entry/libs/arm64-v8a/libMNN.so` is built from the local mobiInfer MNN 3.6 fork for OHOS arm64
with the HiAI backend enabled. It uses `MNN_FORWARD_USER_1` and links the packaged HiAI runtime libraries.

`entry/libs/arm64-v8a/libMNN_CL.so` is built from the same source revision with `MNN_OPENCL=ON`,
`MNN_SEP_BUILD=ON`, and `MNN_NPU=OFF`. `beauty_pipeline` explicitly links this plugin so its static backend
registration runs alongside the HiAI backend in `libMNN.so`.


The upstream OHOS loader list in `source/backend/opencl/core/runtime/OpenCLWrapper.cpp` does not include the
OpenCL library paths exposed by the target Huawei phone. Add these entries to its `__OHOS__` path list before
building:

```cpp
"/vendor/lib64/libOpenCL.so",
"/vendor/lib64/passthrough/libOpenCL.so",
"/vendor/lib64/chipsetsdk/libOpenCL_impl.so",
```

On the test phone, the application process maps confirmed that MNN loaded
`/vendor/lib64/passthrough/libOpenCL.so`. Without this adaptation, MNN reports `OpenCL init error` and session
creation is rejected by the app rather than silently using the CPU backend.
