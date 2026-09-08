#include "tiny_gpt2_profiler.h"

#include "cann_profile_self_test.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <vector>

#include <hilog/log.h>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

#undef LOG_DOMAIN
#undef LOG_TAG
#define LOG_DOMAIN 0xB001
#define LOG_TAG "TinyGpt2Profile"

namespace {
constexpr int kSequenceLength = 16;
constexpr int kHiddenSize = 2;
constexpr int kInputElementCount = kSequenceLength * kHiddenSize;
constexpr int kVocabularySize = 50257;
constexpr int kPredictionIndex = 3;
constexpr int kReferenceToken = 5087;
constexpr uint32_t kWarmupCount = 5;
constexpr const char *kExportedOms[] = {
    "/data/storage/el2/base/haps/entry/files/mnn_user0_outputs_5.om",
    "/data/storage/el2/base/haps/entry/files/mnn_user0_outputs_4.om",
    "/data/storage/el2/base/haps/entry/files/mnn_user0_outputs_1.om",
};

struct InterpreterDeleter {
    void operator()(MNN::Interpreter *interpreter) const
    {
        if (interpreter != nullptr) delete interpreter;
    }
};

struct BenchmarkResult {
    int token = -1;
    double setupUs = 0.0;
    std::vector<double> inferenceUs;
    std::vector<std::vector<float>> diagnostics;
    std::string error;
};

struct Statistics {
    double mean = 0.0;
    double p50 = 0.0;
    double p95 = 0.0;
    double minimum = 0.0;
    double maximum = 0.0;
};

bool ReadFile(const std::string &path, std::vector<uint8_t> &data)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) return false;
    const std::streamsize size = input.tellg();
    if (size <= 0) return false;
    data.resize(static_cast<size_t>(size));
    input.seekg(0, std::ios::beg);
    return static_cast<bool>(input.read(reinterpret_cast<char *>(data.data()), size));
}

Statistics CalculateStatistics(const std::vector<double> &samples)
{
    Statistics result;
    if (samples.empty()) return result;
    std::vector<double> sorted(samples);
    std::sort(sorted.begin(), sorted.end());
    double sum = 0.0;
    for (double sample : samples) sum += sample;
    auto percentile = [&sorted](double fraction) {
        const size_t index = static_cast<size_t>(std::ceil(fraction * sorted.size())) - 1;
        return sorted[std::min(index, sorted.size() - 1)];
    };
    result.mean = sum / samples.size();
    result.p50 = percentile(0.50);
    result.p95 = percentile(0.95);
    result.minimum = sorted.front();
    result.maximum = sorted.back();
    return result;
}

BenchmarkResult RunMnnBenchmark(const void *modelData, size_t modelSize, const void *inputData,
    MNNForwardType forwardType, const char *backend, uint32_t repeatCount)
{
    BenchmarkResult result;
    const auto setupStart = std::chrono::steady_clock::now();
    std::unique_ptr<MNN::Interpreter, InterpreterDeleter> interpreter(
        MNN::Interpreter::createFromBuffer(modelData, modelSize));
    if (!interpreter) {
        result.error = std::string(backend) + " model parsing failed";
        return result;
    }
    MNN::BackendConfig backendConfig;
    backendConfig.precision = MNN::BackendConfig::Precision_High;
    backendConfig.power = MNN::BackendConfig::Power_High;
    MNN::ScheduleConfig schedule;
    schedule.type = forwardType;
    schedule.backupType = forwardType;
    schedule.numThread = 1;
    schedule.backendConfig = &backendConfig;
    MNN::Session *session = interpreter->createSession(schedule);
    if (session == nullptr) {
        result.error = std::string(backend) + " session/OM build failed";
        return result;
    }
    result.setupUs = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - setupStart).count();
    MNN::Tensor *input = interpreter->getSessionInput(session, "input_embeddings");
    MNN::Tensor *output = interpreter->getSessionOutput(session, "logits");
    if (input == nullptr || output == nullptr || input->elementSize() != kInputElementCount) {
        interpreter->releaseSession(session);
        result.error = std::string(backend) + " input_embeddings/logits tensor shape mismatch";
        return result;
    }
    MNN::Tensor hostInput(input, MNN::Tensor::CAFFE);
    std::copy_n(static_cast<const float *>(inputData), kInputElementCount, hostInput.host<float>());
    input->copyFromHostTensor(&hostInput);
    MNN::ErrorCode runCode = MNN::NO_ERROR;
    for (uint32_t index = 0; index < kWarmupCount && runCode == MNN::NO_ERROR; ++index) {
        runCode = interpreter->runSession(session);
    }
    repeatCount = repeatCount == 0 ? 50 : repeatCount;
    result.inferenceUs.reserve(repeatCount);
    for (uint32_t index = 0; index < repeatCount && runCode == MNN::NO_ERROR; ++index) {
        const auto start = std::chrono::steady_clock::now();
        runCode = interpreter->runSession(session);
        const auto end = std::chrono::steady_clock::now();
        result.inferenceUs.push_back(std::chrono::duration<double, std::micro>(end - start).count());
    }
    if (runCode == MNN::NO_ERROR) {
        MNN::Tensor hostOutput(output, MNN::Tensor::CAFFE);
        output->copyToHostTensor(&hostOutput);
        if (hostOutput.elementSize() >= kSequenceLength * kVocabularySize) {
            const float *logits = hostOutput.host<float>() + kPredictionIndex * kVocabularySize;
            result.token = static_cast<int>(std::max_element(logits, logits + kVocabularySize) - logits);
        }
        const char *diagnosticNames[] = { "hidden_embedding", "hidden_block0", "hidden_block1" };
        for (const char *name : diagnosticNames) {
            MNN::Tensor *deviceTensor = interpreter->getSessionOutput(session, name);
            if (deviceTensor == nullptr) continue;
            MNN::Tensor hostTensor(deviceTensor, MNN::Tensor::CAFFE);
            deviceTensor->copyToHostTensor(&hostTensor);
            const float *values = hostTensor.host<float>();
            result.diagnostics.emplace_back(values, values + hostTensor.elementSize());
        }
    }
    interpreter->releaseSession(session);
    if (runCode != MNN::NO_ERROR || result.token < 0) {
        result.error = std::string(backend) + " inference/output failed";
        result.token = -1;
        return result;
    }
    const Statistics stats = CalculateStatistics(result.inferenceUs);
    OH_LOG_INFO(LOG_APP,
        "PQB:TINY_GPT2_BENCH backend=%{public}s setup_us=%{public}.1f warmup=%{public}u runs=%{public}zu "
        "mean_us=%{public}.1f p50_us=%{public}.1f p95_us=%{public}.1f min_us=%{public}.1f max_us=%{public}.1f token=%{public}d",
        backend, result.setupUs, kWarmupCount, result.inferenceUs.size(), stats.mean, stats.p50,
        stats.p95, stats.minimum, stats.maximum, result.token);
    return result;
}

float MaxAbsDifference(const std::vector<float> &left, const std::vector<float> &right)
{
    if (left.size() != right.size()) return -1.0f;
    float result = 0.0f;
    for (size_t index = 0; index < left.size(); ++index) {
        result = std::max(result, std::abs(left[index] - right[index]));
    }
    return result;
}

bool WriteMnnBenchmarkCsv(const std::string &outputDir,
    const BenchmarkResult &cpu, const BenchmarkResult &npu)
{
    std::ofstream output(outputDir + "/mnn_benchmark.csv", std::ios::trunc);
    if (!output) return false;
    output << "backend,phase,run,duration_us\n" << std::fixed << std::setprecision(3);
    output << "CPU,setup,-1," << cpu.setupUs << "\n";
    for (size_t index = 0; index < cpu.inferenceUs.size(); ++index) {
        output << "CPU,inference," << index << "," << cpu.inferenceUs[index] << "\n";
    }
    output << "MNN_USER0,setup,-1," << npu.setupUs << "\n";
    for (size_t index = 0; index < npu.inferenceUs.size(); ++index) {
        output << "MNN_USER0,inference," << index << "," << npu.inferenceUs[index] << "\n";
    }
    return static_cast<bool>(output);
}
} // namespace

std::string RunTinyGpt2Profiler(const void *modelData, size_t modelSize,
    const void *inputData, size_t inputSize, const std::string &outputDir, uint32_t repeatCount)
{
    if (modelData == nullptr || modelSize == 0) return "TinyGPT2: MNN model is empty";
    if (inputData == nullptr || inputSize != kInputElementCount * sizeof(float)) {
        return "TinyGPT2: input_embeddings must be float32[1,16,2]";
    }

    OH_LOG_INFO(LOG_APP, "PQB:TINY_GPT2 start model=%{public}zu input=%{public}zu",
        modelSize, inputSize);
    repeatCount = repeatCount == 0 ? 50 : repeatCount;
    BenchmarkResult cpu = RunMnnBenchmark(
        modelData, modelSize, inputData, MNN_FORWARD_CPU, "CPU", repeatCount);
    if (cpu.token < 0) return std::string("TinyGPT2: ") + cpu.error;
    OH_LOG_INFO(LOG_APP, "PQB:TINY_GPT2 CPU token=%{public}d reference=%{public}d",
        cpu.token, kReferenceToken);

    for (const char *path : kExportedOms) std::remove(path);
    BenchmarkResult npu = RunMnnBenchmark(
        modelData, modelSize, inputData, MNN_FORWARD_USER_0, "MNN_USER0", repeatCount);
    if (npu.token < 0) return std::string("TinyGPT2: ") + npu.error + "; inspect HIAI_V";
    std::vector<float> stageErrors;
    const size_t stageCount = std::min(cpu.diagnostics.size(), npu.diagnostics.size());
    for (size_t index = 0; index < stageCount; ++index) {
        stageErrors.push_back(MaxAbsDifference(cpu.diagnostics[index], npu.diagnostics[index]));
    }
    if (stageErrors.size() == 3) {
        OH_LOG_INFO(LOG_APP,
            "PQB:TINY_GPT2 stage max_abs embedding=%{public}f block0=%{public}f block1=%{public}f",
            stageErrors[0], stageErrors[1], stageErrors[2]);
        const size_t sampleCount = std::min<size_t>(8, cpu.diagnostics[0].size());
        for (size_t index = 0; index < sampleCount; ++index) {
            OH_LOG_INFO(LOG_APP,
                "PQB:TINY_GPT2 embedding[%{public}zu] cpu=%{public}f npu=%{public}f",
                index, cpu.diagnostics[0][index], npu.diagnostics[0][index]);
        }
    }

    std::vector<uint8_t> om;
    for (const char *path : kExportedOms) {
        if (ReadFile(path, om)) break;
    }
    if (om.empty()) {
        return "TinyGPT2 inference passed, but no USER_0 OM was exported";
    }
    OH_LOG_INFO(LOG_APP,
        "PQB:TINY_GPT2 OM ready bytes=%{public}zu cpu=%{public}d npu=%{public}d reference=%{public}d",
        om.size(), cpu.token, npu.token, kReferenceToken);
    const std::string profile = RunCannProfileSelfTest(
        om.data(), om.size(), outputDir, repeatCount, inputData, inputSize);
    const bool benchmarkSaved = WriteMnnBenchmarkCsv(outputDir, cpu, npu);
    OH_LOG_INFO(LOG_APP, "PQB:TINY_GPT2 benchmark_csv=%{public}s",
        benchmarkSaved ? "saved" : "failed");

    std::ostringstream result;
    const Statistics cpuStats = CalculateStatistics(cpu.inferenceUs);
    const Statistics npuStats = CalculateStatistics(npu.inferenceUs);
    result << std::fixed << std::setprecision(1);
    result << "tiny-gpt2: source=sshleifer/tiny-gpt2, seq=16, MNN=" << modelSize
           << " bytes, OM=" << om.size() << " bytes"
           << ", CPU mean/P50/P95=" << cpuStats.mean << "/" << cpuStats.p50 << "/" << cpuStats.p95 << " us"
           << ", NPU mean/P50/P95=" << npuStats.mean << "/" << npuStats.p50 << "/" << npuStats.p95 << " us"
           << ", CPU token=" << cpu.token << ", NPU token=" << npu.token << ", reference=" << kReferenceToken
           << ", correctness="
           << (cpu.token == kReferenceToken && npu.token == cpu.token ? "PASS" : "MISMATCH");
    if (stageErrors.size() == 3) {
        result << ", stage max_abs=" << stageErrors[0] << "/" << stageErrors[1]
               << "/" << stageErrors[2];
    }
    result << "; " << profile;
    result << ", benchmark_csv=" << (benchmarkSaved ? "saved" : "failed");
    return result.str();
}
