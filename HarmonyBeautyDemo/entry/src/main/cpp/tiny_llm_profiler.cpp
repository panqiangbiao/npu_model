#include "tiny_llm_profiler.h"

#include "cann_profile_self_test.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <vector>

#include <hilog/log.h>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/MathOp.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>

#undef LOG_DOMAIN
#undef LOG_TAG
#define LOG_DOMAIN 0xB001
#define LOG_TAG "TinyLlmProfile"

namespace {
using namespace MNN;
using namespace MNN::Express;

constexpr int kSequence = 4;
constexpr int kHidden = 32;
constexpr int kFfn = 64;
constexpr int kVocab = 64;
constexpr const char *kExportedOm =
    "/data/storage/el2/base/haps/entry/files/mnn_user0_outputs_1.om";

struct InterpreterDeleter {
    void operator()(MNN::Interpreter *interpreter) const
    {
        if (interpreter != nullptr) delete interpreter;
    }
};

float NextWeight(uint32_t &state)
{
    state = state * 1664525u + 1013904223u;
    const float unit = static_cast<float>((state >> 8) & 0xffffu) / 65535.0f;
    return (unit * 2.0f - 1.0f) * 0.08f;
}

std::vector<float> MakeWeights(size_t count, uint32_t seed)
{
    std::vector<float> result(count);
    for (float &value : result) value = NextWeight(seed);
    return result;
}

VARP Weight(const std::vector<float> &values, INTS shape, const char *name)
{
    VARP value = _Const(values.data(), std::move(shape), NCHW, halide_type_of<float>());
    value->setName(name);
    return value;
}

std::vector<int8_t> BuildTinyTransformer()
{
    uint32_t seed = 0x20260907u;
    VARP inputIds = _Input({1, kSequence}, NCHW, halide_type_of<int32_t>());
    inputIds->setName("input_ids");

    VARP embedding = Weight(MakeWeights(kVocab * kHidden, seed), {kVocab, kHidden}, "token_embedding.weight");
    VARP hidden = _GatherV2(embedding, inputIds, _Scalar<int32_t>(0));
    hidden->setName("token_embedding");

    std::vector<float> gamma(kHidden, 1.0f);
    std::vector<float> beta(kHidden, 0.0f);
    VARP norm1 = _LayerNorm(hidden, {-1}, 1.0e-5f, gamma, beta);
    norm1->setName("attention_norm");

    VARP q = _MatMul(norm1, Weight(MakeWeights(kHidden * kHidden, seed), {kHidden, kHidden}, "q_proj.weight"));
    VARP k = _MatMul(norm1, Weight(MakeWeights(kHidden * kHidden, seed), {kHidden, kHidden}, "k_proj.weight"));
    VARP v = _MatMul(norm1, Weight(MakeWeights(kHidden * kHidden, seed), {kHidden, kHidden}, "v_proj.weight"));
    q->setName("q_proj");
    k->setName("k_proj");
    v->setName("v_proj");

    q = _Reshape(q, {1, 1, kSequence, kHidden});
    k = _Reshape(k, {1, 1, kSequence, kHidden});
    v = _Reshape(v, {1, 1, kSequence, kHidden});
    VARP scores = _BatchMatMul(q, k, false, true);
    scores = _Multiply(scores, _Const(1.0f / std::sqrt(static_cast<float>(kHidden))));
    std::vector<float> mask(kSequence * kSequence, 0.0f);
    for (int row = 0; row < kSequence; ++row) {
        for (int column = row + 1; column < kSequence; ++column) {
            mask[row * kSequence + column] = -10000.0f;
        }
    }
    scores = _Add(scores, Weight(mask, {1, 1, kSequence, kSequence}, "causal_mask"));
    VARP probabilities = _Softmax(scores, -1);
    probabilities->setName("attention_softmax");
    VARP context = _BatchMatMul(probabilities, v);
    context = _Reshape(context, {1, kSequence, kHidden});
    VARP attentionOutput = _MatMul(context,
        Weight(MakeWeights(kHidden * kHidden, seed), {kHidden, kHidden}, "o_proj.weight"));
    VARP residual = _Add(hidden, attentionOutput);
    residual->setName("attention_residual");

    VARP norm2 = _LayerNorm(residual, {-1}, 1.0e-5f, gamma, beta);
    norm2->setName("ffn_norm");
    VARP up = _MatMul(norm2, Weight(MakeWeights(kHidden * kFfn, seed), {kHidden, kFfn}, "ffn_up.weight"));
    VARP activated = _Relu(up);
    activated->setName("ffn_relu");
    VARP down = _MatMul(activated,
        Weight(MakeWeights(kFfn * kHidden, seed), {kFfn, kHidden}, "ffn_down.weight"));
    VARP blockOutput = _Add(residual, down);
    blockOutput->setName("transformer_block_output");

    VARP finalNorm = _LayerNorm(blockOutput, {-1}, 1.0e-5f, gamma, beta);
    VARP logits = _MatMul(finalNorm,
        Weight(MakeWeights(kHidden * kVocab, seed), {kHidden, kVocab}, "lm_head.weight"));
    logits->setName("logits");
    return Variable::save({logits});
}

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
} // namespace

std::string RunTinyLlmProfiler(const std::string &outputDir, uint32_t repeatCount)
{
    OH_LOG_INFO(LOG_APP, "PQB:TINY_LLM build start seq=%{public}d hidden=%{public}d ffn=%{public}d vocab=%{public}d",
        kSequence, kHidden, kFfn, kVocab);
    const std::vector<int8_t> model = BuildTinyTransformer();
    if (model.empty()) return "Tiny LLM: MNN model generation failed";

    std::remove(kExportedOm);
    std::unique_ptr<MNN::Interpreter, InterpreterDeleter> interpreter(
        MNN::Interpreter::createFromBuffer(model.data(), model.size()));
    if (!interpreter) return "Tiny LLM: MNN model parsing failed";

    MNN::BackendConfig backendConfig;
    backendConfig.precision = MNN::BackendConfig::Precision_High;
    backendConfig.power = MNN::BackendConfig::Power_High;
    MNN::ScheduleConfig schedule;
    schedule.type = MNN_FORWARD_USER_0;
    schedule.backupType = MNN_FORWARD_USER_0;
    schedule.backendConfig = &backendConfig;
    MNN::Session *session = interpreter->createSession(schedule);
    if (session == nullptr) return "Tiny LLM: MNN USER_0 session/OM build failed; inspect PQB:TINY_LLM and HIAI_V logs";

    MNN::Tensor *input = interpreter->getSessionInput(session, "input_ids");
    MNN::Tensor *output = interpreter->getSessionOutput(session, "logits");
    if (input == nullptr || output == nullptr) {
        interpreter->releaseSession(session);
        return "Tiny LLM: input_ids or logits tensor missing";
    }
    MNN::Tensor hostInput(input, MNN::Tensor::CAFFE);
    const int32_t prompt[kSequence] = {1, 7, 11, 23};
    std::copy(prompt, prompt + kSequence, hostInput.host<int32_t>());
    input->copyFromHostTensor(&hostInput);
    const MNN::ErrorCode runCode = interpreter->runSession(session);
    MNN::Tensor hostOutput(output, MNN::Tensor::CAFFE);
    output->copyToHostTensor(&hostOutput);
    int token = -1;
    if (runCode == MNN::NO_ERROR && hostOutput.elementSize() >= kVocab) {
        const float *last = hostOutput.host<float>() + (kSequence - 1) * kVocab;
        token = static_cast<int>(std::max_element(last, last + kVocab) - last);
    }
    interpreter->releaseSession(session);
    if (runCode != MNN::NO_ERROR) return "Tiny LLM: first MNN USER_0 inference failed";

    std::vector<uint8_t> om;
    if (!ReadFile(kExportedOm, om)) {
        return std::string("Tiny LLM inference passed, but profiling OM was not exported: ") + kExportedOm;
    }
    OH_LOG_INFO(LOG_APP, "PQB:TINY_LLM OM ready bytes=%{public}zu next_token=%{public}d", om.size(), token);
    const std::string profile = RunCannProfileSelfTest(om.data(), om.size(), outputDir, repeatCount);

    std::ostringstream result;
    result << "Tiny Transformer: seq=4, hidden=32, FFN=64, vocab=64; MNN=" << model.size()
           << " bytes; OM=" << om.size() << " bytes; next token=" << token << "; " << profile;
    return result.str();
}
