#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <sstream>
#include <vector>

#include <napi/native_api.h>
#include <neural_network_runtime/neural_network_core.h>

#include "beauty_renderer.h"
#include "gpu_face_inference.h"

namespace {
std::mutex g_mutex;
std::mutex g_gpuMutex;
std::mutex g_mnnNpuMutex;
std::unique_ptr<BeautyRenderer> g_renderer;
std::unique_ptr<MnnFaceInference> g_gpuInference;
std::unique_ptr<MnnFaceInference> g_mnnNpuInference;

bool ReadArrayBuffer(napi_env env, napi_value value, void *&data, size_t &byteLength)
{
    data = nullptr;
    byteLength = 0;
    return napi_get_arraybuffer_info(env, value, &data, &byteLength) == napi_ok && data != nullptr;
}

napi_value CreateFloatBuffer(napi_env env, const std::vector<float> &values)
{
    void *data = nullptr;
    napi_value result = nullptr;
    const size_t byteLength = values.size() * sizeof(float);
    napi_create_arraybuffer(env, byteLength, &data, &result);
    if (byteLength > 0) std::memcpy(data, values.data(), byteLength);
    return result;
}

std::string ReadString(napi_env env, napi_value value)
{
    size_t length = 0;
    napi_get_value_string_utf8(env, value, nullptr, 0, &length);
    std::string result(length, '\0');
    napi_get_value_string_utf8(env, value, result.data(), length + 1, &length);
    return result;
}

napi_value Create(napi_env env, napi_callback_info info)
{
    size_t argc = 1;
    napi_value args[1] = { nullptr };
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    if (argc != 1) {
        napi_throw_error(env, nullptr, "Output surface ID is required");
        return nullptr;
    }

    const uint64_t outputSurfaceId = std::strtoull(ReadString(env, args[0]).c_str(), nullptr, 10);
    std::lock_guard<std::mutex> lock(g_mutex);
    g_renderer = std::make_unique<BeautyRenderer>();
    uint64_t inputSurfaceId = 0;
    std::string error;
    if (!g_renderer->Start(outputSurfaceId, inputSurfaceId, error)) {
        g_renderer.reset();
        napi_throw_error(env, nullptr, error.c_str());
        return nullptr;
    }
    const std::string inputSurfaceText = std::to_string(inputSurfaceId);
    napi_value result = nullptr;
    napi_create_string_utf8(env, inputSurfaceText.c_str(), inputSurfaceText.size(), &result);
    return result;
}

napi_value SetParameters(napi_env env, napi_callback_info info)
{
    size_t argc = 4;
    napi_value args[4] = { nullptr };
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    if (argc != 4) return nullptr;
    bool enabled = false;
    double smooth = 0.0;
    double whiten = 0.0;
    double rosy = 0.0;
    napi_get_value_bool(env, args[0], &enabled);
    napi_get_value_double(env, args[1], &smooth);
    napi_get_value_double(env, args[2], &whiten);
    napi_get_value_double(env, args[3], &rosy);
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_renderer) {
        g_renderer->SetParameters(enabled, static_cast<float>(smooth), static_cast<float>(whiten),
            static_cast<float>(rosy));
    }
    return nullptr;
}

napi_value SetReshapeParameters(napi_env env, napi_callback_info info)
{
    size_t argc = 2;
    napi_value args[2] = { nullptr };
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    if (argc != 2) return nullptr;
    double slimFace = 0.0;
    double bigEye = 0.0;
    napi_get_value_double(env, args[0], &slimFace);
    napi_get_value_double(env, args[1], &bigEye);
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_renderer) {
        g_renderer->SetReshapeParameters(static_cast<float>(slimFace), static_cast<float>(bigEye));
    }
    return nullptr;
}

napi_value SetSpiderMaskEnabled(napi_env env, napi_callback_info info)
{
    size_t argc = 1;
    napi_value args[1] = { nullptr };
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    if (argc != 1) return nullptr;
    bool enabled = false;
    napi_get_value_bool(env, args[0], &enabled);
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_renderer) g_renderer->SetSpiderMaskEnabled(enabled);
    return nullptr;
}

napi_value SetFaceRegion(napi_env env, napi_callback_info info)
{
    size_t argc = 5;
    napi_value args[5] = { nullptr };
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    if (argc != 5) return nullptr;
    bool hasFace = false;
    double left = 0.0;
    double top = 0.0;
    double width = 1.0;
    double height = 1.0;
    napi_get_value_bool(env, args[0], &hasFace);
    napi_get_value_double(env, args[1], &left);
    napi_get_value_double(env, args[2], &top);
    napi_get_value_double(env, args[3], &width);
    napi_get_value_double(env, args[4], &height);
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_renderer) {
        g_renderer->SetFaceRegion(hasFace, static_cast<float>(left), static_cast<float>(top),
            static_cast<float>(width), static_cast<float>(height));
    }
    return nullptr;
}

napi_value SetLandmarks(napi_env env, napi_callback_info info)
{
    size_t argc = 1;
    napi_value args[1] = { nullptr };
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    if (argc != 1) return nullptr;
    void *data = nullptr;
    size_t byteLength = 0;
    napi_get_arraybuffer_info(env, args[0], &data, &byteLength);
    std::vector<float> landmarks;
    if (byteLength == 136 * sizeof(float) && data != nullptr) {
        const float *values = static_cast<const float *>(data);
        landmarks.assign(values, values + 136);
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_renderer) g_renderer->SetLandmarks(landmarks);
    return nullptr;
}

napi_value SetFaceMesh(napi_env env, napi_callback_info info)
{
    size_t argc = 1;
    napi_value args[1] = { nullptr };
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    if (argc != 1) return nullptr;
    void *data = nullptr;
    size_t byteLength = 0;
    napi_get_arraybuffer_info(env, args[0], &data, &byteLength);
    std::vector<float> mesh;
    if (byteLength == 468 * 3 * sizeof(float) && data != nullptr) {
        const float *values = static_cast<const float *>(data);
        mesh.assign(values, values + 468 * 3);
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_renderer) g_renderer->SetFaceMesh(mesh);
    return nullptr;
}

napi_value SetFaceMeshTexture(napi_env env, napi_callback_info info)
{
    size_t argc = 1;
    napi_value args[1] = { nullptr };
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    if (argc != 1) return nullptr;
    void *data = nullptr;
    size_t byteLength = 0;
    napi_get_arraybuffer_info(env, args[0], &data, &byteLength);
    if (data == nullptr) return nullptr;
    const auto *bytes = static_cast<const uint8_t *>(data);
    std::vector<uint8_t> rgba(bytes, bytes + byteLength);
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_renderer) g_renderer->SetFaceMeshTexture(rgba);
    return nullptr;
}

napi_value SetDebugOverlay(napi_env env, napi_callback_info info)
{
    size_t argc = 1;
    napi_value args[1] = { nullptr };
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    bool enabled = false;
    if (argc == 1) napi_get_value_bool(env, args[0], &enabled);
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_renderer) g_renderer->SetDebugOverlay(enabled);
    return nullptr;
}

napi_value SetBeautyLuts(napi_env env, napi_callback_info info)
{
    size_t argc = 4;
    napi_value args[4] = { nullptr };
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    if (argc != 4) return nullptr;
    std::vector<std::vector<uint8_t>> data(4);
    for (size_t index = 0; index < 4; ++index) {
        void *bytes = nullptr;
        size_t byteLength = 0;
        if (napi_get_arraybuffer_info(env, args[index], &bytes, &byteLength) != napi_ok || bytes == nullptr) {
            return nullptr;
        }
        const auto *first = static_cast<const uint8_t *>(bytes);
        data[index].assign(first, first + byteLength);
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_renderer) g_renderer->SetBeautyLuts(data[0], data[1], data[2], data[3]);
    return nullptr;
}

napi_value Release(napi_env env, napi_callback_info info)
{
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_renderer.reset();
    }
    {
        std::lock_guard<std::mutex> lock(g_gpuMutex);
        g_gpuInference.reset();
    }
    {
        std::lock_guard<std::mutex> lock(g_mnnNpuMutex);
        g_mnnNpuInference.reset();
    }
    return nullptr;
}

napi_value InitializeGpuInference(napi_env env, napi_callback_info info)
{
    size_t argc = 2;
    napi_value args[2] = { nullptr };
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    void *faceModel = nullptr;
    void *landmarkModel = nullptr;
    size_t faceModelSize = 0;
    size_t landmarkModelSize = 0;
    if (argc != 2 || !ReadArrayBuffer(env, args[0], faceModel, faceModelSize) ||
        !ReadArrayBuffer(env, args[1], landmarkModel, landmarkModelSize)) {
        napi_throw_type_error(env, nullptr, "Two MNN model ArrayBuffers are required");
        return nullptr;
    }

    std::string error;
    std::lock_guard<std::mutex> lock(g_gpuMutex);
    auto inference = std::make_unique<MnnFaceInference>(MnnInferenceBackend::OpenCl);
    if (!inference->Initialize(faceModel, faceModelSize, landmarkModel, landmarkModelSize, error)) {
        napi_throw_error(env, nullptr, error.c_str());
        return nullptr;
    }
    g_gpuInference = std::move(inference);
    napi_value result = nullptr;
    napi_create_string_utf8(env, "MNN OpenCL backend=3", NAPI_AUTO_LENGTH, &result);
    return result;
}

napi_value InitializeMnnNpuInference(napi_env env, napi_callback_info info)
{
    size_t argc = 3;
    napi_value args[3] = { nullptr };
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    void *faceModel = nullptr;
    void *landmarkModel = nullptr;
    size_t faceModelSize = 0;
    size_t landmarkModelSize = 0;
    int32_t mode = -1;
    if (argc != 3 || !ReadArrayBuffer(env, args[0], faceModel, faceModelSize) ||
        !ReadArrayBuffer(env, args[1], landmarkModel, landmarkModelSize)) {
        napi_throw_type_error(env, nullptr, "Two MNN model ArrayBuffers and a USER mode are required");
        return nullptr;
    }
    if (napi_get_value_int32(env, args[2], &mode) != napi_ok || (mode != 0 && mode != 1)) {
        napi_throw_range_error(env, nullptr, "MNN NPU mode must be 0 (USER_0) or 1 (USER_1)");
        return nullptr;
    }

    std::string error;
    std::lock_guard<std::mutex> lock(g_mnnNpuMutex);
    const MnnInferenceBackend selected = mode == 0 ? MnnInferenceBackend::NpuUser0 :
        MnnInferenceBackend::NpuUser1;
    auto inference = std::make_unique<MnnFaceInference>(selected);
    if (!inference->Initialize(faceModel, faceModelSize, landmarkModel, landmarkModelSize, error)) {
        napi_throw_error(env, nullptr, error.c_str());
        return nullptr;
    }
    g_mnnNpuInference = std::move(inference);
    napi_value result = nullptr;
    const char *description = mode == 0 ? "MNN HiAI USER_0 backend=8 full-graph" :
        "MNN HiAI USER_1 backend=9 per-convolution delegate";
    napi_create_string_utf8(env, description, NAPI_AUTO_LENGTH, &result);
    return result;
}

napi_value RunGpuFace(napi_env env, napi_callback_info info)
{
    size_t argc = 1;
    napi_value args[1] = { nullptr };
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    void *input = nullptr;
    size_t byteLength = 0;
    if (argc != 1 || !ReadArrayBuffer(env, args[0], input, byteLength) || byteLength % sizeof(float) != 0) {
        napi_throw_type_error(env, nullptr, "Face input must be a Float32 ArrayBuffer");
        return nullptr;
    }

    std::vector<float> scores;
    std::vector<float> boxes;
    double elapsedMs = 0.0;
    std::string error;
    std::lock_guard<std::mutex> lock(g_gpuMutex);
    if (!g_gpuInference || !g_gpuInference->RunFace(static_cast<const float *>(input),
        byteLength / sizeof(float), scores, boxes, elapsedMs, error)) {
        napi_throw_error(env, nullptr, error.empty() ? "GPU inference is not initialized" : error.c_str());
        return nullptr;
    }

    napi_value result = nullptr;
    napi_create_object(env, &result);
    napi_set_named_property(env, result, "scores", CreateFloatBuffer(env, scores));
    napi_set_named_property(env, result, "boxes", CreateFloatBuffer(env, boxes));
    napi_value elapsed = nullptr;
    napi_create_double(env, elapsedMs, &elapsed);
    napi_set_named_property(env, result, "elapsedMs", elapsed);
    return result;
}

napi_value RunGpuLandmarks(napi_env env, napi_callback_info info)
{
    size_t argc = 1;
    napi_value args[1] = { nullptr };
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    void *input = nullptr;
    size_t byteLength = 0;
    if (argc != 1 || !ReadArrayBuffer(env, args[0], input, byteLength) || byteLength % sizeof(float) != 0) {
        napi_throw_type_error(env, nullptr, "Landmark input must be a Float32 ArrayBuffer");
        return nullptr;
    }

    std::vector<float> landmarks;
    double elapsedMs = 0.0;
    std::string error;
    std::lock_guard<std::mutex> lock(g_gpuMutex);
    if (!g_gpuInference || !g_gpuInference->RunLandmarks(static_cast<const float *>(input),
        byteLength / sizeof(float), landmarks, elapsedMs, error)) {
        napi_throw_error(env, nullptr, error.empty() ? "GPU inference is not initialized" : error.c_str());
        return nullptr;
    }

    napi_value result = nullptr;
    napi_create_object(env, &result);
    napi_set_named_property(env, result, "landmarks", CreateFloatBuffer(env, landmarks));
    napi_value elapsed = nullptr;
    napi_create_double(env, elapsedMs, &elapsed);
    napi_set_named_property(env, result, "elapsedMs", elapsed);
    return result;
}

napi_value RunMnnNpuFace(napi_env env, napi_callback_info info)
{
    size_t argc = 1;
    napi_value args[1] = { nullptr };
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    void *input = nullptr;
    size_t byteLength = 0;
    if (argc != 1 || !ReadArrayBuffer(env, args[0], input, byteLength) || byteLength % sizeof(float) != 0) {
        napi_throw_type_error(env, nullptr, "Face input must be a Float32 ArrayBuffer");
        return nullptr;
    }

    std::vector<float> scores;
    std::vector<float> boxes;
    double elapsedMs = 0.0;
    std::string error;
    std::lock_guard<std::mutex> lock(g_mnnNpuMutex);
    if (!g_mnnNpuInference || !g_mnnNpuInference->RunFace(static_cast<const float *>(input),
        byteLength / sizeof(float), scores, boxes, elapsedMs, error)) {
        napi_throw_error(env, nullptr, error.empty() ? "MNN NPU inference is not initialized" : error.c_str());
        return nullptr;
    }

    napi_value result = nullptr;
    napi_create_object(env, &result);
    napi_set_named_property(env, result, "scores", CreateFloatBuffer(env, scores));
    napi_set_named_property(env, result, "boxes", CreateFloatBuffer(env, boxes));
    napi_value elapsed = nullptr;
    napi_create_double(env, elapsedMs, &elapsed);
    napi_set_named_property(env, result, "elapsedMs", elapsed);
    return result;
}

napi_value RunMnnNpuLandmarks(napi_env env, napi_callback_info info)
{
    size_t argc = 1;
    napi_value args[1] = { nullptr };
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    void *input = nullptr;
    size_t byteLength = 0;
    if (argc != 1 || !ReadArrayBuffer(env, args[0], input, byteLength) || byteLength % sizeof(float) != 0) {
        napi_throw_type_error(env, nullptr, "Landmark input must be a Float32 ArrayBuffer");
        return nullptr;
    }

    std::vector<float> landmarks;
    double elapsedMs = 0.0;
    std::string error;
    std::lock_guard<std::mutex> lock(g_mnnNpuMutex);
    if (!g_mnnNpuInference || !g_mnnNpuInference->RunLandmarks(static_cast<const float *>(input),
        byteLength / sizeof(float), landmarks, elapsedMs, error)) {
        napi_throw_error(env, nullptr, error.empty() ? "MNN NPU inference is not initialized" : error.c_str());
        return nullptr;
    }

    napi_value result = nullptr;
    napi_create_object(env, &result);
    napi_set_named_property(env, result, "landmarks", CreateFloatBuffer(env, landmarks));
    napi_value elapsed = nullptr;
    napi_create_double(env, elapsedMs, &elapsed);
    napi_set_named_property(env, result, "elapsedMs", elapsed);
    return result;
}

napi_value GetNpuDevices(napi_env env, napi_callback_info info)
{
    const size_t *deviceIds = nullptr;
    uint32_t deviceCount = 0;
    std::ostringstream result;
    if (OH_NNDevice_GetAllDevicesID(&deviceIds, &deviceCount) != OH_NN_SUCCESS) {
        result << "NNRT device query failed";
    } else if (deviceCount == 0) {
        result << "No NNRT accelerator";
    } else {
        for (uint32_t index = 0; index < deviceCount; ++index) {
            const char *name = nullptr;
            OH_NN_DeviceType type = OH_NN_OTHERS;
            OH_NNDevice_GetName(deviceIds[index], &name);
            OH_NNDevice_GetType(deviceIds[index], &type);
            if (index > 0) result << ", ";
            result << (name == nullptr ? "unknown" : name) << "(id=" << deviceIds[index]
                   << ",type=" << static_cast<int>(type) << ")";
        }
    }
    const std::string text = result.str();
    napi_value value = nullptr;
    napi_create_string_utf8(env, text.c_str(), text.size(), &value);
    return value;
}

napi_value ConsumeFaceFrame(napi_env env, napi_callback_info info)
{
    std::vector<float> input;
    float xScale = 1.0f;
    float yScale = 1.0f;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (!g_renderer || !g_renderer->ConsumeFaceInput(input, xScale, yScale)) {
            napi_value empty = nullptr;
            napi_get_null(env, &empty);
            return empty;
        }
    }

    void *bufferData = nullptr;
    napi_value buffer = nullptr;
    const size_t byteSize = input.size() * sizeof(float);
    napi_create_arraybuffer(env, byteSize, &bufferData, &buffer);
    std::memcpy(bufferData, input.data(), byteSize);

    napi_value result = nullptr;
    napi_create_object(env, &result);
    napi_set_named_property(env, result, "data", buffer);
    napi_value xScaleValue = nullptr;
    napi_value yScaleValue = nullptr;
    napi_create_double(env, xScale, &xScaleValue);
    napi_create_double(env, yScale, &yScaleValue);
    napi_set_named_property(env, result, "xScale", xScaleValue);
    napi_set_named_property(env, result, "yScale", yScaleValue);
    return result;
}

napi_value Init(napi_env env, napi_value exports)
{
    napi_property_descriptor properties[] = {
        { "create", nullptr, Create, nullptr, nullptr, nullptr, napi_default, nullptr },
        { "setParameters", nullptr, SetParameters, nullptr, nullptr, nullptr, napi_default, nullptr },
        { "setReshapeParameters", nullptr, SetReshapeParameters, nullptr, nullptr, nullptr, napi_default, nullptr },
        { "setSpiderMaskEnabled", nullptr, SetSpiderMaskEnabled, nullptr, nullptr, nullptr, napi_default, nullptr },
        { "setFaceRegion", nullptr, SetFaceRegion, nullptr, nullptr, nullptr, napi_default, nullptr },
        { "setLandmarks", nullptr, SetLandmarks, nullptr, nullptr, nullptr, napi_default, nullptr },
        { "setFaceMesh", nullptr, SetFaceMesh, nullptr, nullptr, nullptr, napi_default, nullptr },
        { "setFaceMeshTexture", nullptr, SetFaceMeshTexture, nullptr, nullptr, nullptr, napi_default, nullptr },
        { "setDebugOverlay", nullptr, SetDebugOverlay, nullptr, nullptr, nullptr, napi_default, nullptr },
        { "setBeautyLuts", nullptr, SetBeautyLuts, nullptr, nullptr, nullptr, napi_default, nullptr },
        { "getNpuDevices", nullptr, GetNpuDevices, nullptr, nullptr, nullptr, napi_default, nullptr },
        { "consumeFaceFrame", nullptr, ConsumeFaceFrame, nullptr, nullptr, nullptr, napi_default, nullptr },
        { "initializeGpuInference", nullptr, InitializeGpuInference, nullptr, nullptr, nullptr, napi_default, nullptr },
        { "runGpuFace", nullptr, RunGpuFace, nullptr, nullptr, nullptr, napi_default, nullptr },
        { "runGpuLandmarks", nullptr, RunGpuLandmarks, nullptr, nullptr, nullptr, napi_default, nullptr },
        { "initializeMnnNpuInference", nullptr, InitializeMnnNpuInference, nullptr, nullptr, nullptr,
            napi_default, nullptr },
        { "runMnnNpuFace", nullptr, RunMnnNpuFace, nullptr, nullptr, nullptr, napi_default, nullptr },
        { "runMnnNpuLandmarks", nullptr, RunMnnNpuLandmarks, nullptr, nullptr, nullptr, napi_default, nullptr },
        { "release", nullptr, Release, nullptr, nullptr, nullptr, napi_default, nullptr },
    };
    napi_define_properties(env, exports, sizeof(properties) / sizeof(properties[0]), properties);
    return exports;
}
}

static napi_module beautyPipelineModule = {
    .nm_version = 1,
    .nm_flags = 0,
    .nm_filename = nullptr,
    .nm_register_func = Init,
    .nm_modname = "beauty_pipeline",
    .nm_priv = nullptr,
    .reserved = { nullptr },
};

extern "C" __attribute__((constructor)) void RegisterBeautyPipelineModule()
{
    napi_module_register(&beautyPipelineModule);
}
