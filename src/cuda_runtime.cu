#include "cuda_runtime_bridge.h"
#include "cuda_error.h"
#include <chrono>
#include <cmath>
#include <limits>
#include <utility>

namespace vectorpulse {
namespace {
using Clock = std::chrono::steady_clock;
double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
class DeviceScope {
    int previous_ = 0;
    bool changed_ = false;
public:
    DeviceScope() {
        VP_CUDA_CHECK(cudaGetDevice(&previous_));
        if (previous_ != 0) { VP_CUDA_CHECK(cudaSetDevice(0)); changed_ = true; }
    }
    DeviceScope(const DeviceScope&) = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;
    ~DeviceScope() { if (changed_) detail::cuda_check_cleanup(cudaSetDevice(previous_), "cudaSetDevice restore"); }
    void close() {
        if (changed_) { changed_ = false; VP_CUDA_CHECK(cudaSetDevice(previous_)); }
    }
};
class Buffer {
public:
    float* data = nullptr;
    explicit Buffer(std::size_t bytes) { VP_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data), bytes)); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() { if (data) detail::cuda_check_cleanup(cudaFree(data), "cudaFree"); }
    void close() {
        if (data) { auto* pointer = std::exchange(data, nullptr); VP_CUDA_CHECK(cudaFree(pointer)); }
    }
};
class Event {
public:
    cudaEvent_t event = nullptr;
    Event() { VP_CUDA_CHECK(cudaEventCreate(&event)); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    ~Event() { if (event) detail::cuda_check_cleanup(cudaEventDestroy(event), "cudaEventDestroy"); }
    void close() {
        if (event) { const auto handle = std::exchange(event, nullptr); VP_CUDA_CHECK(cudaEventDestroy(handle)); }
    }
};

// Deliberately naive: one thread owns one row and loops through all dimensions.
__global__ void cosine_similarity_kernel(const float* vectors, const float* query,
    float* scores, std::size_t count, std::size_t dimension, double query_norm) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    if (query_norm == 0.0) { scores[index] = 0.0F; return; }
    double dot = 0.0, norm = 0.0;
    for (std::size_t d = 0; d < dimension; ++d) {
        const double a = query[d];
        const double b = vectors[index * dimension + d];
        dot += a * b;
        norm += b * b;
    }
    scores[index] = norm == 0.0 ? 0.0F : static_cast<float>(dot / sqrt(query_norm * norm));
}
// One block per row. Every dimension is visited once; inactive lanes contribute zero.
// All reduction stages use shared memory and full-block barriers (no warp primitives).
__global__ void block_cosine_similarity_kernel(const float* vectors, const float* query,
    float* scores, std::size_t count, std::size_t dimension, double query_norm) {
    const auto row = static_cast<std::size_t>(blockIdx.x);
    if (row >= count) return; // Uniform across the block, before any barrier.
    const auto lane = threadIdx.x;
    if (query_norm == 0.0) {
        if (lane == 0) scores[row] = 0.0F;
        return; // Also uniform across the block.
    }
    __shared__ double dots[CudaSearchBackend::block_size];
    __shared__ double norms[CudaSearchBackend::block_size];
    double dot = 0.0, norm = 0.0;
    for (std::size_t d = lane; d < dimension; d += blockDim.x) {
        const double a = query[d];
        const double b = vectors[row * dimension + d];
        dot += a * b;
        norm += b * b;
    }
    dots[lane] = dot;
    norms[lane] = norm;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride != 0; stride /= 2) {
        if (lane < stride) {
            dots[lane] += dots[lane + stride];
            norms[lane] += norms[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        scores[row] = norms[0] == 0.0 ? 0.0F :
            static_cast<float>(dots[0] / sqrt(query_norm * norms[0]));
    }
}
// FP32 accumulation variants. Final normalization stays double to avoid norm-product overflow.
__global__ void cosine_similarity_fp32_kernel(const float* vectors, const float* query,
    float* scores, std::size_t count, std::size_t dimension, double query_norm) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    if (query_norm == 0.0) { scores[index] = 0.0F; return; }
    float dot = 0.0F, norm = 0.0F;
    for (std::size_t d = 0; d < dimension; ++d) {
        const float a = query[d];
        const float b = vectors[index * dimension + d];
        dot += a * b;
        norm += b * b;
    }
    scores[index] = norm == 0.0 ? 0.0F : static_cast<float>(dot / sqrt(query_norm * norm));
}
// One block per row. Every dimension is visited once; inactive lanes contribute zero.
// All reduction stages use shared memory and full-block barriers (no warp primitives).
__global__ void block_cosine_similarity_fp32_kernel(const float* vectors, const float* query,
    float* scores, std::size_t count, std::size_t dimension, double query_norm) {
    const auto row = static_cast<std::size_t>(blockIdx.x);
    if (row >= count) return; // Uniform across the block, before any barrier.
    const auto lane = threadIdx.x;
    if (query_norm == 0.0) {
        if (lane == 0) scores[row] = 0.0F;
        return; // Also uniform across the block.
    }
    __shared__ float dots[CudaSearchBackend::block_size];
    __shared__ float norms[CudaSearchBackend::block_size];
    float dot = 0.0F, norm = 0.0F;
    for (std::size_t d = lane; d < dimension; d += blockDim.x) {
        const float a = query[d];
        const float b = vectors[row * dimension + d];
        dot += a * b;
        norm += b * b;
    }
    dots[lane] = dot;
    norms[lane] = norm;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride != 0; stride /= 2) {
        if (lane < stride) {
            dots[lane] += dots[lane + stride];
            norms[lane] += norms[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        scores[row] = norms[0] == 0.0 ? 0.0F :
            static_cast<float>(dots[0] / sqrt(query_norm * norms[0]));
    }
}
static_assert((CudaSearchBackend::block_size & (CudaSearchBackend::block_size - 1)) == 0);
}  // namespace

bool CudaSearchBackend::is_compiled() noexcept { return true; }
CudaDeviceInfo CudaSearchBackend::device_info() {
    CudaDeviceInfo info;
    info.compiled = true;
    try {
        VP_CUDA_CHECK(cudaRuntimeGetVersion(&info.runtime_version));
        VP_CUDA_CHECK(cudaDriverGetVersion(&info.driver_version));
        int count = 0;
        VP_CUDA_CHECK(cudaGetDeviceCount(&count));
        if (count == 0) { info.reason = "no CUDA devices detected"; return info; }
        DeviceScope device;
        cudaDeviceProp properties{};
        VP_CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
        info.name = properties.name;
        info.compute_major = properties.major;
        info.compute_minor = properties.minor;
        info.global_memory_bytes = properties.totalGlobalMem;
        if (properties.computeMode == cudaComputeModeProhibited) {
            info.reason = "CUDA device 0 prohibits compute";
            device.close();
            return info;
        }
        // Resolve the actual compiled kernel to detect unsupported device images.
        cudaFuncAttributes attributes{};
        VP_CUDA_CHECK(cudaFuncGetAttributes(&attributes, cosine_similarity_kernel));
        VP_CUDA_CHECK(cudaFuncGetAttributes(&attributes, block_cosine_similarity_kernel));
        VP_CUDA_CHECK(cudaFuncGetAttributes(&attributes, cosine_similarity_fp32_kernel));
        VP_CUDA_CHECK(cudaFuncGetAttributes(&attributes, block_cosine_similarity_fp32_kernel));
        device.close();
        info.available = true;
    } catch (const std::exception& error) { info.reason = error.what(); }
    return info;
}

std::vector<float> detail::cuda_scores(const float* vectors, const float* query,
    std::size_t count, std::size_t dimension, double query_norm, CudaSearchTimings& timings) {
    DeviceScope device;
    cudaDeviceProp properties{};
    VP_CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    const auto blocks = count / CudaSearchBackend::block_size + (count % CudaSearchBackend::block_size != 0);
    if (blocks > static_cast<std::size_t>(properties.maxGridSize[0])) {
        throw std::length_error("CUDA grid exceeds device limit");
    }
    Buffer device_vectors(count * dimension * sizeof(float));
    Buffer device_query(dimension * sizeof(float));
    Buffer device_scores(count * sizeof(float));
    Event begin, end;
    std::vector<float> scores(count);
    const auto h2d_start = Clock::now();
    VP_CUDA_CHECK(cudaMemcpy(device_vectors.data, vectors, count * dimension * sizeof(float), cudaMemcpyHostToDevice));
    VP_CUDA_CHECK(cudaMemcpy(device_query.data, query, dimension * sizeof(float), cudaMemcpyHostToDevice));
    // Pageable H2D may return before DMA completes; include completion in host timing.
    VP_CUDA_CHECK(cudaDeviceSynchronize());
    timings.h2d_ms = milliseconds(h2d_start);
    timings.database_h2d_bytes = count * dimension * sizeof(float);
    timings.query_h2d_bytes = dimension * sizeof(float);

    VP_CUDA_CHECK(cudaEventRecord(begin.event));
    cosine_similarity_kernel<<<static_cast<unsigned int>(blocks), CudaSearchBackend::block_size>>>(
        device_vectors.data, device_query.data, device_scores.data, count, dimension, query_norm);
    VP_CUDA_CHECK(cudaGetLastError());
    VP_CUDA_CHECK(cudaEventRecord(end.event));
    VP_CUDA_CHECK(cudaEventSynchronize(end.event));
    float elapsed = 0.0F;
    VP_CUDA_CHECK(cudaEventElapsedTime(&elapsed, begin.event, end.event));
    timings.kernel_ms = elapsed;

    const auto d2h_start = Clock::now();
    VP_CUDA_CHECK(cudaMemcpy(scores.data(), device_scores.data, count * sizeof(float), cudaMemcpyDeviceToHost));
    timings.d2h_ms = milliseconds(d2h_start);
    // Normal-path cleanup failures propagate. Unwinding cleanup reports to stderr.
    end.close(); begin.close();
    device_scores.close(); device_query.close(); device_vectors.close();
    device.close();
    return scores;
}
namespace detail {
// No CUDA types escape the private bridge. The immutable matrix owns its allocation
// independently of any query and is always released on its owning device (device 0).
class CudaIndex {
public:
    Buffer vectors;
    const std::size_t count, dimension;
    const unsigned int blocks;
    const std::size_t max_grid_blocks;
    CudaIndex(std::size_t n, std::size_t d, unsigned int grid, std::size_t max_grid)
        : vectors(n * d * sizeof(float)), count(n), dimension(d), blocks(grid), max_grid_blocks(max_grid) {}
    ~CudaIndex() {
        try {
            DeviceScope device;
            vectors.close();
        } catch (const std::exception& error) {
            std::fprintf(stderr, "CUDA index cleanup: %s\n", error.what());
        }
    }
};
std::shared_ptr<const CudaIndex> cuda_build_index(const float* vectors,
    std::size_t count, std::size_t dimension, CudaIndexBuildTimings& timings) {
    DeviceScope device;
    cudaDeviceProp properties{};
    VP_CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    const auto blocks = count / CudaSearchBackend::block_size + (count % CudaSearchBackend::block_size != 0);
    if (blocks > static_cast<std::size_t>(properties.maxGridSize[0])) {
        throw std::length_error("CUDA grid exceeds device limit");
    }
    auto index = std::make_shared<CudaIndex>(count, dimension, static_cast<unsigned int>(blocks),
                                             static_cast<std::size_t>(properties.maxGridSize[0]));
    const auto start = Clock::now();
    VP_CUDA_CHECK(cudaMemcpy(index->vectors.data, vectors, count * dimension * sizeof(float), cudaMemcpyHostToDevice));
    VP_CUDA_CHECK(cudaDeviceSynchronize());
    timings.h2d_ms = milliseconds(start);
    timings.database_bytes = count * dimension * sizeof(float);
    device.close();
    return index;
}
std::vector<float> cuda_persistent_scores(const CudaIndex& index, const float* query,
    double query_norm, CudaKernel kernel, CudaSearchTimings& timings) {
    DeviceScope device;
    const auto count = index.count, dimension = index.dimension;
    const auto blocks = (kernel == CudaKernel::BlockParallel || kernel == CudaKernel::BlockParallelFP32) ? count : index.blocks;
    if (blocks > index.max_grid_blocks) throw std::length_error("CUDA grid exceeds device limit");
    Buffer device_query(dimension * sizeof(float));
    Buffer device_scores(count * sizeof(float));
    Event begin, end;
    std::vector<float> scores(count);
    const auto h2d_start = Clock::now();
    VP_CUDA_CHECK(cudaMemcpy(device_query.data, query, dimension * sizeof(float), cudaMemcpyHostToDevice));
    // Pageable H2D may return before DMA completes; include completion in host timing.
    VP_CUDA_CHECK(cudaDeviceSynchronize());
    timings.h2d_ms = milliseconds(h2d_start);
    timings.query_h2d_bytes = dimension * sizeof(float);

    VP_CUDA_CHECK(cudaEventRecord(begin.event));
    if (kernel == CudaKernel::NaiveFP32) {
        cosine_similarity_fp32_kernel<<<static_cast<unsigned int>(blocks), CudaSearchBackend::block_size>>>(
            index.vectors.data, device_query.data, device_scores.data, count, dimension, query_norm);
    } else if (kernel == CudaKernel::BlockParallelFP32) {
        block_cosine_similarity_fp32_kernel<<<static_cast<unsigned int>(blocks), CudaSearchBackend::block_size>>>(
            index.vectors.data, device_query.data, device_scores.data, count, dimension, query_norm);
    } else if (kernel == CudaKernel::BlockParallel) {
        block_cosine_similarity_kernel<<<static_cast<unsigned int>(blocks), CudaSearchBackend::block_size>>>(
            index.vectors.data, device_query.data, device_scores.data, count, dimension, query_norm);
    } else {
        cosine_similarity_kernel<<<static_cast<unsigned int>(blocks), CudaSearchBackend::block_size>>>(
            index.vectors.data, device_query.data, device_scores.data, count, dimension, query_norm);
    }
    VP_CUDA_CHECK(cudaGetLastError());
    VP_CUDA_CHECK(cudaEventRecord(end.event));
    VP_CUDA_CHECK(cudaEventSynchronize(end.event));
    float elapsed = 0.0F;
    VP_CUDA_CHECK(cudaEventElapsedTime(&elapsed, begin.event, end.event));
    timings.kernel_ms = elapsed;

    const auto d2h_start = Clock::now();
    VP_CUDA_CHECK(cudaMemcpy(scores.data(), device_scores.data, count * sizeof(float), cudaMemcpyDeviceToHost));
    timings.d2h_ms = milliseconds(d2h_start);
    // Normal-path cleanup failures propagate. Unwinding cleanup reports to stderr.
    end.close(); begin.close();
    device_scores.close(); device_query.close();
    device.close();
    return scores;
}
}  // namespace detail
}  // namespace vectorpulse
