#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace vectorpulse::detail {
inline void cuda_check(cudaError_t status, const char* operation, const char* file, int line) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string{operation} + ": " + cudaGetErrorName(status) +
            " (" + cudaGetErrorString(status) + ") at " + file + ":" + std::to_string(line));
    }
}
// Destructors cannot throw during unwinding. Failures are still checked/reported.
inline void cuda_check_cleanup(cudaError_t status, const char* operation) noexcept {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA cleanup error in %s: %s (%s)\n", operation,
                     cudaGetErrorName(status), cudaGetErrorString(status));
    }
}
}
#define VP_CUDA_CHECK(call) ::vectorpulse::detail::cuda_check((call), #call, __FILE__, __LINE__)
