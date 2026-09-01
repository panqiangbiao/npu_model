#ifndef HARMONY_BEAUTY_GPU_FACE_INFERENCE_H
#define HARMONY_BEAUTY_GPU_FACE_INFERENCE_H

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace MNN {
class Interpreter;
class Session;
class Tensor;
}

enum class MnnInferenceBackend {
    OpenCl,
    NpuUser0,
    NpuUser1,
};

class MnnFaceInference {
public:
    explicit MnnFaceInference(MnnInferenceBackend backend);
    ~MnnFaceInference();

    bool Initialize(const void *faceModel, size_t faceModelSize, const void *landmarkModel,
        size_t landmarkModelSize, std::string &error);
    bool RunFace(const float *input, size_t inputCount, std::vector<float> &scores,
        std::vector<float> &boxes, double &elapsedMs, std::string &error);
    bool RunLandmarks(const float *input, size_t inputCount, std::vector<float> &landmarks,
        double &elapsedMs, std::string &error);
    void Release();

private:
    struct ModelSession {
        std::unique_ptr<MNN::Interpreter> interpreter;
        MNN::Session *session = nullptr;
        MNN::Tensor *input = nullptr;
        bool outputSummaryLogged = false;
    };

    bool CreateSession(const void *model, size_t modelSize, ModelSession &target,
        std::string &error);
    bool Run(ModelSession &model, const float *input, size_t inputCount,
        const std::vector<std::string> &outputNames, std::vector<std::vector<float>> &outputs,
        double &elapsedMs, std::string &error);

    ModelSession face_;
    ModelSession landmarks_;
    MnnInferenceBackend backend_;
};

#endif
