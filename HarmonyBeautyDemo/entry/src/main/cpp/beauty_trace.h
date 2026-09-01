#ifndef HARMONY_BEAUTY_TRACE_H
#define HARMONY_BEAUTY_TRACE_H

#include <cstdint>
#include <string>

#include <hitrace/trace.h>

namespace BeautyTrace {
constexpr const char *PREFIX = "HBAI_TRACE/";

class Scope {
public:
    explicit Scope(const char *stage) : name_(PREFIX + std::string(stage))
    {
        OH_HiTrace_StartTrace(name_.c_str());
    }

    ~Scope()
    {
        OH_HiTrace_FinishTrace();
    }

    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;

private:
    std::string name_;
};

inline void Count(const char *name, int64_t value)
{
    const std::string fullName = PREFIX + std::string(name);
    OH_HiTrace_CountTrace(fullName.c_str(), value);
}
}

#endif
