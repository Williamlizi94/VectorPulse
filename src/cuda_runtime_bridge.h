#pragma once
#include "vectorpulse/cuda_search_backend.h"

namespace vectorpulse::detail {
std::shared_ptr<const CudaIndex> cuda_build_index(const float* vectors,
    std::size_t count, std::size_t dimension, CudaIndexBuildTimings& timings);
std::vector<float> cuda_persistent_scores(const CudaIndex& index, const float* query,
    double query_squared_norm, CudaKernel kernel, CudaSearchTimings& timings);
// Validated finite inputs, nonzero count/dimension, checked byte size.
std::vector<float> cuda_scores(const float* vectors, const float* query,
    std::size_t count, std::size_t dimension, double query_squared_norm,
    CudaSearchTimings& timings);
}
