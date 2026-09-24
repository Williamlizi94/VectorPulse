#include <pybind11/pybind11.h>
#include <pybind11/stl/filesystem.h>

#include "vectorpulse/vector_index.h"
#include "vectorpulse/avx2_search_backend.h"
#include "vectorpulse/cuda_search_backend.h"
#include "vectorpulse/multithreaded_avx2_search_backend.h"
#include "vectorpulse/multithreaded_scalar_search_backend.h"
#include "vectorpulse/scalar_search_backend.h"

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace py = pybind11;
namespace vp = vectorpulse;

namespace {

// Copy input before entering the C++ API. The index never retains Python memory.
// The buffer protocol handles NumPy without a build-time or runtime dependency.
std::vector<float> copy_vector(py::handle value, std::size_t dimension) {
    if (PyUnicode_Check(value.ptr()) || PyBytes_Check(value.ptr()) ||
        PyByteArray_Check(value.ptr())) {
        throw py::type_error("vector must be a one-dimensional numeric sequence or buffer");
    }
    if (PyObject_CheckBuffer(value.ptr())) {
        const auto buffer = py::reinterpret_borrow<py::buffer>(value).request();
        if (buffer.ndim != 1) throw py::value_error("vector must be one-dimensional");
        if (buffer.shape[0] < 0 || static_cast<std::size_t>(buffer.shape[0]) != dimension) {
            throw py::value_error("vector dimension does not match the index");
        }
        if (buffer.itemsize == sizeof(float) &&
            buffer.format == py::format_descriptor<float>::format()) {
            std::vector<float> values(dimension);
            const auto* source = static_cast<const char*>(buffer.ptr);
            if (buffer.strides[0] == sizeof(float)) {
                std::memcpy(values.data(), source, values.size() * sizeof(float));
            } else {
                // memcpy also supports unaligned, readonly, reversed and strided views.
                for (std::size_t i = 0; i < dimension; ++i) {
                    std::memcpy(&values[i], source + static_cast<py::ssize_t>(i) * buffer.strides[0],
                                sizeof(float));
                }
            }
            return values;
        }
    }
    if (!PySequence_Check(value.ptr())) {
        throw py::type_error("vector must be a one-dimensional numeric sequence or buffer");
    }
    const auto sequence = py::reinterpret_borrow<py::sequence>(value);
    if (sequence.size() != dimension) {
        throw py::value_error("vector dimension does not match the index");
    }
    std::vector<float> values;
    values.reserve(dimension);
    for (const auto item : sequence) {
        try {
            values.push_back(py::cast<float>(item));
        } catch (const py::cast_error&) {
            throw py::type_error("vector components must be real numbers");
        }
    }
    return values;
}

std::unique_ptr<vp::SearchBackend> make_backend(std::size_t dimension,
                                               std::string_view name,
                                               std::size_t threads) {
    if (threads == 0) throw std::invalid_argument("threads must be greater than zero");
    if (name == "scalar") return std::make_unique<vp::ScalarSearchBackend>(dimension);
    if (name == "avx2") return std::make_unique<vp::AVX2SearchBackend>(dimension);
    if (name == "multithreaded_scalar") {
        return std::make_unique<vp::MultithreadedScalarSearchBackend>(dimension, threads);
    }
    if (name == "multithreaded_avx2") {
        return std::make_unique<vp::MultithreadedAVX2SearchBackend>(dimension, threads);
    }
    auto kernel = vp::CudaKernel::Naive;
    auto storage = vp::CudaStorageMode::Persistent;
    if (name == "cuda") { /* Existing default CUDA kernel. */ }
    else if (name == "cuda_block_parallel") kernel = vp::CudaKernel::BlockParallel;
    else if (name == "cuda_fp32") kernel = vp::CudaKernel::NaiveFP32;
    else if (name == "cuda_block_parallel_fp32") kernel = vp::CudaKernel::BlockParallelFP32;
    else if (name == "cuda_per_query") storage = vp::CudaStorageMode::PerQuery;
    else throw std::invalid_argument("unknown VectorPulse backend: " + std::string{name});
    return std::make_unique<vp::CudaSearchBackend>(dimension, storage, kernel);
}

py::list available_backends() {
    py::list names;
    names.append("scalar");
    names.append("multithreaded_scalar");
    if (vp::AVX2SearchBackend::is_supported()) {
        names.append("avx2");
        names.append("multithreaded_avx2");
    }
    if (vp::CudaSearchBackend::is_supported()) {
        for (const auto* name : {"cuda", "cuda_block_parallel", "cuda_fp32",
                                 "cuda_block_parallel_fp32", "cuda_per_query"}) {
            names.append(name);
        }
    }
    return names;
}

}  // namespace

PYBIND11_MODULE(vectorpulse, module) {
    module.doc() = "Python bindings for the existing VectorPulse C++20 exact search library.";
    module.attr("__version__") = VECTORPULSE_VERSION;
    module.def("available_backends", &available_backends,
               "Return backend names supported by this build and machine.");

    // Keep the GIL: callers cannot race add() against search()/save() from Python.
    py::class_<vp::VectorIndex>(module, "VectorIndex")
        .def(py::init([](std::size_t dimension, const std::string& backend, std::size_t threads) {
            return vp::VectorIndex{dimension, make_backend(dimension, backend, threads)};
        }), py::arg("dimension"), py::kw_only(), py::arg("backend") = "scalar",
            py::arg("threads") = 1,
            "Create an index; threads configures the multithreaded CPU backends.")
        .def("add", [](vp::VectorIndex& index, const py::str& id, py::handle vector) {
            index.add(id.cast<std::string>(), copy_vector(vector, index.dimension()));
        }, py::arg("id"), py::arg("vector"), "Copy a vector into the C++ index under a unique ID.")
        .def("search", [](const vp::VectorIndex& index, py::handle query, std::size_t k) {
            const auto values = copy_vector(query, index.dimension());
            const auto results = index.search(values, k);
            py::list output;
            for (const auto& result : results) {
                py::dict entry;
                entry["id"] = result.id;
                entry["score"] = result.score;
                output.append(std::move(entry));
            }
            return output;
        }, py::arg("query"), py::arg("k"), "Return ordered dictionaries containing id and score.")
        .def("size", &vp::VectorIndex::size)
        .def("dimension", &vp::VectorIndex::dimension)
        .def("save", &vp::VectorIndex::save, py::arg("path"),
             "Save the existing portable C++ index format (no GPU state).")
        .def_static("load", [](const std::filesystem::path& path) {
            return vp::VectorIndex::load(path);
        }, py::arg("path"), "Load a saved index using the C++ default scalar backend.");
}
