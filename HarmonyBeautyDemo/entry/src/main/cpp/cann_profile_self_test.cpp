#include "cann_profile_self_test.h"

#include <cerrno>
#include <cmath>
#include <cstring>
#include <dlfcn.h>
#include <sstream>
#include <sys/stat.h>
#include <vector>

#include <CANNKit/hiai_options.h>
#include <CANNKit/hiai_helper.h>
#include <hilog/log.h>
#include <neural_network_runtime/neural_network_core.h>
#include <neural_network_runtime/neural_network_runtime.h>

#undef LOG_DOMAIN
#undef LOG_TAG
#define LOG_DOMAIN 0xB001
#define LOG_TAG "CannProfileTest"

namespace {
constexpr size_t kElementCount = 256;

using SetOmOptionsFn = OH_NN_ReturnCode (*)(OH_NNCompilation *, HiAI_OmType, const char *);
using SetBandModeFn = OH_NN_ReturnCode (*)(OH_NNCompilation *, HiAI_BandMode);
using SetModelDeviceOrderFn = OH_NN_ReturnCode (*)(OH_NNCompilation *, HiAI_ExecuteDevice *, size_t);
using CheckCompatibilityFn = HiAI_Compatibility (*)(const void *, size_t);

std::string Failure(const char *stage, OH_NN_ReturnCode code)
{
    std::ostringstream stream;
    stream << "CANN Profiling failed at " << stage << ", code=" << static_cast<int>(code);
    OH_LOG_ERROR(LOG_APP, "PQB:CANN_PROFILE %{public}s", stream.str().c_str());
    return stream.str();
}

bool EnsureOutputDirectory(const std::string &path, std::string &error)
{
    if (path.empty()) {
        error = "CANN Profiling output directory is empty";
        return false;
    }
    if (mkdir(path.c_str(), 0700) == 0 || errno == EEXIST) {
        return true;
    }
    error = "Cannot create CANN Profiling directory: " + path + ", errno=" + std::to_string(errno);
    return false;
}

void DestroyAll(OH_NNExecutor *&executor, OH_NNCompilation *&compilation, OH_NNModel *&model)
{
    if (executor != nullptr) OH_NNExecutor_Destroy(&executor);
    if (compilation != nullptr) OH_NNCompilation_Destroy(&compilation);
    if (model != nullptr) OH_NNModel_Destroy(&model);
}

void DestroyTensors(std::vector<NN_Tensor *> &tensors)
{
    for (NN_Tensor *tensor : tensors) OH_NNTensor_Destroy(&tensor);
    tensors.clear();
}
} // namespace

std::string RunCannProfileSelfTest(const void *modelData, size_t modelSize,
    const std::string &outputDir, uint32_t repeatCount, const void *inputData, size_t inputSize)
{
    std::string directoryError;
    if (!EnsureOutputDirectory(outputDir, directoryError)) return directoryError;
    repeatCount = repeatCount == 0 ? 10 : repeatCount;

    if (modelData == nullptr || modelSize == 0) {
        return "CANN Profiling official OM buffer is empty";
    }
    OH_LOG_INFO(LOG_APP, "PQB:CANN_PROFILE MNN-U0 exported OM bytes=%{public}zu", modelSize);

    OH_NNModel *model = nullptr;
    OH_NNCompilation *compilation = nullptr;
    OH_NNExecutor *executor = nullptr;

    const size_t *deviceIds = nullptr;
    uint32_t deviceCount = 0;
    OH_NN_ReturnCode code = OH_NNDevice_GetAllDevicesID(&deviceIds, &deviceCount);
    size_t targetDevice = 0;
    bool foundAccelerator = false;
    if (code == OH_NN_SUCCESS) {
        for (uint32_t index = 0; index < deviceCount; ++index) {
            const char *name = nullptr;
            OH_NN_DeviceType type = OH_NN_OTHERS;
            OH_NNDevice_GetName(deviceIds[index], &name);
            OH_NNDevice_GetType(deviceIds[index], &type);
            OH_LOG_INFO(LOG_APP, "PQB:CANN_PROFILE device id=%{public}zu name=%{public}s type=%{public}d",
                deviceIds[index], name == nullptr ? "unknown" : name, static_cast<int>(type));
            if (name != nullptr && std::strcmp(name, "HIAI_F") == 0) {
                targetDevice = deviceIds[index];
                foundAccelerator = true;
                break;
            }
        }
    }
    if (!foundAccelerator) {
        DestroyAll(executor, compilation, model);
        return "CANN Profiling failed: NNRT accelerator not found";
    }

    compilation = OH_NNCompilation_ConstructWithOfflineModelBuffer(modelData, modelSize);
    if (compilation == nullptr) {
        DestroyAll(executor, compilation, model);
        return "CANN Profiling failed to construct compilation";
    }
    void *foundation = dlopen("libhiai_foundation.so", RTLD_NOW | RTLD_LOCAL);
    auto setBandMode = foundation == nullptr ? nullptr :
        reinterpret_cast<SetBandModeFn>(dlsym(foundation, "HMS_HiAIOptions_SetBandMode"));
    auto setModelDeviceOrder = foundation == nullptr ? nullptr :
        reinterpret_cast<SetModelDeviceOrderFn>(dlsym(foundation, "HMS_HiAIOptions_SetModelDeviceOrder"));
    auto setOmOptions = foundation == nullptr ? nullptr :
        reinterpret_cast<SetOmOptionsFn>(dlsym(foundation, "HMS_HiAIOptions_SetOmOptions"));
    auto checkCompatibility = foundation == nullptr ? nullptr :
        reinterpret_cast<CheckCompatibilityFn>(dlsym(foundation, "HMS_HiAICompatibility_CheckFromBuffer"));
    if (setBandMode == nullptr || setModelDeviceOrder == nullptr || setOmOptions == nullptr) {
        const char *loadError = dlerror();
        std::string result = "CANN Profiling API unavailable";
        if (loadError != nullptr) result += ": " + std::string(loadError);
        if (foundation != nullptr) dlclose(foundation);
        DestroyAll(executor, compilation, model);
        OH_LOG_ERROR(LOG_APP, "PQB:CANN_PROFILE %{public}s", result.c_str());
        return result;
    }
    if (checkCompatibility != nullptr) {
        const HiAI_Compatibility compatibility = checkCompatibility(modelData, modelSize);
        OH_LOG_INFO(LOG_APP, "PQB:CANN_PROFILE compatibility=%{public}d", static_cast<int>(compatibility));
    }

    code = setOmOptions(compilation, HIAI_OM_TYPE_PROFILING, outputDir.c_str());
    OH_LOG_INFO(LOG_APP, "PQB:CANN_PROFILE SetOmOptions code=%{public}d dir=%{public}s",
        static_cast<int>(code), outputDir.c_str());
    if (code != OH_NN_SUCCESS) {
        dlclose(foundation);
        DestroyAll(executor, compilation, model);
        return Failure("SetOmOptions", code);
    }

    code = OH_NNCompilation_SetDevice(compilation, targetDevice);
    OH_LOG_INFO(LOG_APP, "PQB:CANN_PROFILE SetDevice code=%{public}d id=%{public}zu",
        static_cast<int>(code), targetDevice);
    if (code != OH_NN_SUCCESS) {
        dlclose(foundation);
        DestroyAll(executor, compilation, model);
        return Failure("SetDevice", code);
    }

    code = setBandMode(compilation, HiAI_BandMode::HIAI_BANDMODE_NORMAL);
    OH_LOG_INFO(LOG_APP, "PQB:CANN_PROFILE SetBandMode code=%{public}d", static_cast<int>(code));
    if (code == OH_NN_SUCCESS) {
        HiAI_ExecuteDevice devices[] = {HiAI_ExecuteDevice::HIAI_EXECUTE_DEVICE_NPU};
        code = setModelDeviceOrder(compilation, devices, 1);
        OH_LOG_INFO(LOG_APP, "PQB:CANN_PROFILE SetModelDeviceOrder code=%{public}d", static_cast<int>(code));
    }
    dlclose(foundation);
    if (code != OH_NN_SUCCESS) {
        DestroyAll(executor, compilation, model);
        return Failure("SetNpuOptions", code);
    }

    code = OH_NNCompilation_Build(compilation);
    OH_LOG_INFO(LOG_APP, "PQB:CANN_PROFILE CompilationBuild code=%{public}d", static_cast<int>(code));
    if (code != OH_NN_SUCCESS) {
        DestroyAll(executor, compilation, model);
        return Failure("CompilationBuild", code);
    }
    executor = OH_NNExecutor_Construct(compilation);
    if (executor == nullptr) {
        DestroyAll(executor, compilation, model);
        return "CANN Profiling failed to construct executor";
    }

    size_t inputCount = 0;
    size_t outputCount = 0;
    code = OH_NNExecutor_GetInputCount(executor, &inputCount);
    if (code == OH_NN_SUCCESS) code = OH_NNExecutor_GetOutputCount(executor, &outputCount);
    std::vector<NN_Tensor *> inputs;
    std::vector<NN_Tensor *> outputs;
    for (size_t index = 0; code == OH_NN_SUCCESS && index < inputCount; ++index) {
        NN_TensorDesc *desc = OH_NNExecutor_CreateInputTensorDesc(executor, index);
        NN_Tensor *tensor = desc == nullptr ? nullptr : OH_NNTensor_Create(targetDevice, desc);
        if (desc != nullptr) OH_NNTensorDesc_Destroy(&desc);
        if (tensor == nullptr) code = OH_NN_FAILED;
        else inputs.push_back(tensor);
    }
    for (size_t index = 0; code == OH_NN_SUCCESS && index < outputCount; ++index) {
        NN_TensorDesc *desc = OH_NNExecutor_CreateOutputTensorDesc(executor, index);
        NN_Tensor *tensor = desc == nullptr ? nullptr : OH_NNTensor_Create(targetDevice, desc);
        if (desc != nullptr) OH_NNTensorDesc_Destroy(&desc);
        if (tensor == nullptr) code = OH_NN_FAILED;
        else outputs.push_back(tensor);
    }
    if (code != OH_NN_SUCCESS || inputs.empty() || outputs.empty()) {
        DestroyTensors(inputs);
        DestroyTensors(outputs);
        DestroyAll(executor, compilation, model);
        return "CANN Profiling failed to create offline-model tensors";
    }

    for (size_t index = 0; index < inputs.size(); ++index) {
        NN_Tensor *input = inputs[index];
        void *buffer = OH_NNTensor_GetDataBuffer(input);
        size_t size = 0;
        OH_NNTensor_GetSize(input, &size);
        if (buffer != nullptr && index == 0 && inputData != nullptr && inputSize == size) {
            std::memcpy(buffer, inputData, size);
            OH_LOG_INFO(LOG_APP, "PQB:CANN_PROFILE input[0] copied bytes=%{public}zu", size);
        } else if (buffer != nullptr) {
            std::memset(buffer, 0, size);
        }
    }
    for (uint32_t index = 0; code == OH_NN_SUCCESS && index < repeatCount; ++index) {
        code = OH_NNExecutor_RunSync(executor, inputs.data(), inputs.size(), outputs.data(), outputs.size());
    }
    DestroyTensors(inputs);
    DestroyTensors(outputs);
    DestroyAll(executor, compilation, model);
    if (code != OH_NN_SUCCESS) return Failure("ExecutorRunSync", code);

    std::ostringstream result;
    result << "CANN Profiling completed with MNN-U0 exported OM: bytes=" << modelSize
           << ", inputs=" << inputCount << ", outputs=" << outputCount
           << ", runs=" << repeatCount << ", dir=" << outputDir;
    OH_LOG_INFO(LOG_APP, "PQB:CANN_PROFILE %{public}s", result.str().c_str());
    return result.str();
}
