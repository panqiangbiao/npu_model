#include "gpu_face_inference.h"
#include "beauty_trace.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>

#include <hilog/log.h>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

namespace {
constexpr int kOpenClBackend = static_cast<int>(MNN_FORWARD_OPENCL);
constexpr int kNpuUser0Backend = static_cast<int>(MNN_FORWARD_USER_0);
constexpr int kNpuUser1Backend = static_cast<int>(MNN_FORWARD_USER_1);

bool IsNpu(MnnInferenceBackend backend)
{
    return backend != MnnInferenceBackend::OpenCl;
}

MNNForwardType ForwardType(MnnInferenceBackend backend)
{
    if (backend == MnnInferenceBackend::NpuUser0) return MNN_FORWARD_USER_0;
    if (backend == MnnInferenceBackend::NpuUser1) return MNN_FORWARD_USER_1;
    return MNN_FORWARD_OPENCL;
}

const char *BackendName(MnnInferenceBackend backend)
{
    if (backend == MnnInferenceBackend::NpuUser0) return "MNN HiAI USER_0";
    if (backend == MnnInferenceBackend::NpuUser1) return "MNN HiAI USER_1";
    return "MNN OpenCL";
}

std::string TensorSummary(const MNN::Tensor &tensor, const std::string &name)
{
    const float *data = tensor.host<float>();
    const size_t size = tensor.elementSize();
    float minValue = std::numeric_limits<float>::infinity();
    float maxValue = -std::numeric_limits<float>::infinity();
    double sum = 0.0;
    size_t finiteCount = 0;
    size_t nanCount = 0;
    size_t infCount = 0;
    std::vector<std::pair<float, size_t>> top;
    top.reserve(8);
    for (size_t index = 0; index < size; ++index) {
        const float value = data[index];
        if (std::isnan(value)) {
            ++nanCount;
            continue;
        }
        if (std::isinf(value)) {
            ++infCount;
            continue;
        }
        ++finiteCount;
        minValue = std::min(minValue, value);
        maxValue = std::max(maxValue, value);
        sum += value;
        top.emplace_back(value, index);
    }
    std::partial_sort(top.begin(), top.begin() + std::min<size_t>(8, top.size()), top.end(),
        [](const auto &first, const auto &second) { return first.first > second.first; });

    std::ostringstream stream;
    stream << std::fixed << std::setprecision(5)
           << "name=" << name
           << " dims=" << tensor.dimensions()
           << " shape=";
    for (int index = 0; index < tensor.dimensions(); ++index) {
        if (index > 0) stream << "x";
        stream << tensor.length(index);
    }
    stream << " hostLayout=CAFFE"
           << " count=" << size
           << " finite=" << finiteCount
           << " nan=" << nanCount
           << " inf=" << infCount;
    if (finiteCount > 0) {
        stream << " min=" << minValue << " max=" << maxValue << " mean=" << (sum / finiteCount);
    }
    stream << " top=";
    const size_t topCount = std::min<size_t>(8, top.size());
    for (size_t index = 0; index < topCount; ++index) {
        if (index > 0) stream << ",";
        stream << top[index].second << ":" << top[index].first;
    }
    return stream.str();
}
}

MnnFaceInference::MnnFaceInference(MnnInferenceBackend backend) : backend_(backend) {}

MnnFaceInference::~MnnFaceInference()
{
    Release();
}

bool MnnFaceInference::CreateSession(const void *model, size_t modelSize, ModelSession &target,
    std::string &error)
{
    const bool npu = IsNpu(backend_);
    BeautyTrace::Scope trace(backend_ == MnnInferenceBackend::NpuUser0 ? "MNNNPU/User0/CreateSession" :
        backend_ == MnnInferenceBackend::NpuUser1 ? "MNNNPU/User1/CreateSession" : "GPU/CreateOpenClSession");
    target.interpreter.reset(MNN::Interpreter::createFromBuffer(model, modelSize));
    if (!target.interpreter) {
        error = "MNN model parsing failed";
        return false;
    }

    MNN::BackendConfig backendConfig;
    backendConfig.precision = MNN::BackendConfig::Precision_High;
    backendConfig.power = MNN::BackendConfig::Power_High;
    backendConfig.memory = MNN::BackendConfig::Memory_Normal;
    MNN::ScheduleConfig schedule;
    schedule.type = ForwardType(backend_);
    schedule.backupType = schedule.type;
    schedule.backendConfig = &backendConfig;
    target.session = target.interpreter->createSession(schedule);
    if (!target.session) {
        error = npu ? "MNN NPU session creation failed" : "MNN OpenCL session creation failed";
        target.interpreter.reset();
        return false;
    }

    const int expectedBackend = backend_ == MnnInferenceBackend::NpuUser0 ? kNpuUser0Backend :
        backend_ == MnnInferenceBackend::NpuUser1 ? kNpuUser1Backend : kOpenClBackend;
    int backends[4] = { -1, -1, -1, -1 };
    if (!target.interpreter->getSessionInfo(target.session, MNN::Interpreter::BACKENDS, backends) ||
        backends[0] != expectedBackend) {
        std::ostringstream stream;
        stream << "MNN backend mismatch: expected " << BackendName(backend_) << "("
               << expectedBackend << "), actual " << backends[0];
        error = stream.str();
        target.interpreter->releaseSession(target.session);
        target.session = nullptr;
        target.interpreter.reset();
        return false;
    }

    target.input = target.interpreter->getSessionInput(target.session, nullptr);
    if (!target.input) {
        error = "MNN model has no input tensor";
        target.interpreter->releaseSession(target.session);
        target.session = nullptr;
        target.interpreter.reset();
        return false;
    }
    return true;
}

bool MnnFaceInference::Initialize(const void *faceModel, size_t faceModelSize, const void *landmarkModel,
    size_t landmarkModelSize, std::string &error)
{
    BeautyTrace::Scope trace(backend_ == MnnInferenceBackend::NpuUser0 ? "MNNNPU/User0/Initialize" :
        backend_ == MnnInferenceBackend::NpuUser1 ? "MNNNPU/User1/Initialize" : "GPU/Initialize");
    Release();
    if (!faceModel || faceModelSize == 0 || !landmarkModel || landmarkModelSize == 0) {
        error = "MNN model buffers are empty";
        return false;
    }
    if (!CreateSession(faceModel, faceModelSize, face_, error)) {
        return false;
    }
    if (!CreateSession(landmarkModel, landmarkModelSize, landmarks_, error)) {
        Release();
        return false;
    }
    return true;
}

bool MnnFaceInference::Run(ModelSession &model, const float *input, size_t inputCount,
    const std::vector<std::string> &outputNames, std::vector<std::vector<float>> &outputs,
    double &elapsedMs, std::string &error)
{
    if (!model.interpreter || !model.session || !model.input) {
        error = IsNpu(backend_) ? std::string(BackendName(backend_)) + " session is not initialized" :
            "MNN OpenCL session is not initialized";
        return false;
    }
    if (!input || inputCount != static_cast<size_t>(model.input->elementSize())) {
        std::ostringstream stream;
        stream << "MNN input size mismatch: " << inputCount << "/" << model.input->elementSize();
        error = stream.str();
        return false;
    }

    {
        BeautyTrace::Scope upload(backend_ == MnnInferenceBackend::NpuUser0 ? "MNNNPU/User0/TensorUpload" :
            backend_ == MnnInferenceBackend::NpuUser1 ? "MNNNPU/User1/TensorUpload" : "GPU/TensorUpload");
        MNN::Tensor hostInput(model.input, MNN::Tensor::CAFFE);
        std::memcpy(hostInput.host<float>(), input, inputCount * sizeof(float));
        if (!model.input->copyFromHostTensor(&hostInput)) {
            error = "MNN failed to upload input";
            return false;
        }
    }

    const auto startedAt = std::chrono::steady_clock::now();
    MNN::ErrorCode code = MNN::NO_ERROR;
    {
        BeautyTrace::Scope execute(backend_ == MnnInferenceBackend::NpuUser0 ? "MNNNPU/User0/RunSession" :
            backend_ == MnnInferenceBackend::NpuUser1 ? "MNNNPU/User1/RunSession" : "GPU/RunSession");
        code = model.interpreter->runSession(model.session);
    }
    const auto finishedAt = std::chrono::steady_clock::now();
    elapsedMs = std::chrono::duration<double, std::milli>(finishedAt - startedAt).count();
    if (code != MNN::NO_ERROR) {
        std::ostringstream stream;
        stream << "MNN execution failed: " << static_cast<int>(code);
        error = stream.str();
        return false;
    }

    outputs.clear();
    outputs.reserve(outputNames.size());
    {
        BeautyTrace::Scope download(backend_ == MnnInferenceBackend::NpuUser0 ? "MNNNPU/User0/TensorDownload" :
            backend_ == MnnInferenceBackend::NpuUser1 ? "MNNNPU/User1/TensorDownload" : "GPU/TensorDownload");
        for (const std::string &name : outputNames) {
            MNN::Tensor *deviceOutput = model.interpreter->getSessionOutput(model.session, name.c_str());
            if (!deviceOutput) {
                error = "MNN output not found: " + name;
                return false;
            }
            MNN::Tensor hostOutput(deviceOutput, MNN::Tensor::CAFFE);
            if (!deviceOutput->copyToHostTensor(&hostOutput)) {
                error = "MNN failed to download output: " + name;
                return false;
            }
            if (!model.outputSummaryLogged) {
                const std::string summary = TensorSummary(hostOutput, name);
                OH_LOG_INFO(LOG_APP, "MNN_DIAG backend=%{public}s %{public}s", BackendName(backend_), summary.c_str());
            }
            const float *first = hostOutput.host<float>();
            outputs.emplace_back(first, first + hostOutput.elementSize());
        }
        model.outputSummaryLogged = true;
    }
    return true;
}

bool MnnFaceInference::RunFace(const float *input, size_t inputCount, std::vector<float> &scores,
    std::vector<float> &boxes, double &elapsedMs, std::string &error)
{
    BeautyTrace::Scope trace(backend_ == MnnInferenceBackend::NpuUser0 ? "MNNNPU/User0/Face/Total" :
        backend_ == MnnInferenceBackend::NpuUser1 ? "MNNNPU/User1/Face/Total" : "GPU/Face/Total");
    std::vector<std::vector<float>> outputs;
    if (!Run(face_, input, inputCount, { "scores", "boxes" }, outputs, elapsedMs, error)) {
        return false;
    }
    if (outputs.size() != 2 || outputs[0].size() != 4420 * 2 || outputs[1].size() != 4420 * 4) {
        std::ostringstream stream;
        stream << "MNN face output mismatch: " << (outputs.empty() ? 0 : outputs[0].size()) << "/"
               << (outputs.size() < 2 ? 0 : outputs[1].size());
        error = stream.str();
        return false;
    }
    scores = std::move(outputs[0]);
    boxes = std::move(outputs[1]);
    return true;
}

bool MnnFaceInference::RunLandmarks(const float *input, size_t inputCount, std::vector<float> &landmarks,
    double &elapsedMs, std::string &error)
{
    BeautyTrace::Scope trace(backend_ == MnnInferenceBackend::NpuUser0 ? "MNNNPU/User0/Landmarks/Total" :
        backend_ == MnnInferenceBackend::NpuUser1 ? "MNNNPU/User1/Landmarks/Total" : "GPU/Landmarks/Total");
    std::vector<std::vector<float>> outputs;
    if (!Run(landmarks_, input, inputCount, { "output" }, outputs, elapsedMs, error)) {
        return false;
    }
    if (outputs.size() != 1 || outputs[0].size() != 136) {
        error = "MNN landmark output mismatch";
        return false;
    }
    landmarks = std::move(outputs[0]);
    return true;
}

void MnnFaceInference::Release()
{
    auto release = [](ModelSession &model) {
        if (model.interpreter && model.session) {
            model.interpreter->releaseSession(model.session);
        }
        model.input = nullptr;
        model.session = nullptr;
        model.interpreter.reset();
    };
    release(landmarks_);
    release(face_);
}
