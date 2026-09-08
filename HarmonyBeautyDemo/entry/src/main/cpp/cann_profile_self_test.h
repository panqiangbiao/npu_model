#ifndef HARMONY_BEAUTY_DEMO_CANN_PROFILE_SELF_TEST_H
#define HARMONY_BEAUTY_DEMO_CANN_PROFILE_SELF_TEST_H

#include <cstddef>
#include <cstdint>
#include <string>

std::string RunCannProfileSelfTest(const void *modelData, size_t modelSize,
    const std::string &outputDir, uint32_t repeatCount,
    const void *inputData = nullptr, size_t inputSize = 0);

#endif
