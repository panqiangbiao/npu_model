#ifndef HARMONY_BEAUTY_TINY_GPT2_PROFILER_H
#define HARMONY_BEAUTY_TINY_GPT2_PROFILER_H

#include <cstddef>
#include <cstdint>
#include <string>

std::string RunTinyGpt2Profiler(const void *modelData, size_t modelSize,
    const void *inputData, size_t inputSize, const std::string &outputDir, uint32_t repeatCount);

#endif
