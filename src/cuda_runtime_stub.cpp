#include "cuda_runtime_bridge.h"
#include <stdexcept>
namespace vectorpulse {
bool CudaSearchBackend::is_compiled() noexcept { return false; }
CudaDeviceInfo CudaSearchBackend::device_info() {
    CudaDeviceInfo info;
    info.reason = "CUDA support was not compiled (enable VECTORPULSE_ENABLE_CUDA with a CUDA toolkit)";
    return info;
}
std::vector<float> detail::cuda_scores(const float*, const float*, std::size_t, std::size_t,
                                      double, CudaSearchTimings&) {
    throw std::runtime_error("CUDA support was not compiled");
}
std::shared_ptr<const detail::CudaIndex> detail::cuda_build_index(const float*,
    std::size_t, std::size_t, CudaIndexBuildTimings&) {
    throw std::runtime_error("CUDA support was not compiled");
}
std::vector<float> detail::cuda_persistent_scores(const CudaIndex&, const float*,
    double, CudaKernel, CudaSearchTimings&) {
    throw std::runtime_error("CUDA support was not compiled");
}
}
