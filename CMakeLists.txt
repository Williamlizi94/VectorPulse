cmake_minimum_required(VERSION 3.20)

project(VectorPulse VERSION 0.3.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(VECTORPULSE_BUILD_PYTHON "Build optional pybind11 Python module" OFF)
option(VECTORPULSE_BUILD_EXAMPLES "Build VectorPulse examples" ON)

option(VECTORPULSE_BUILD_TESTS "Build VectorPulse unit tests" ON)
option(VECTORPULSE_BUILD_BENCHMARKS "Build VectorPulse benchmarks" ON)
option(VECTORPULSE_ENABLE_AVX2 "Build optional AVX2/FMA kernel on x86" ON)
option(VECTORPULSE_ENABLE_CUDA "Build optional naive CUDA backend when a compiler is available" OFF)

add_library(vectorpulse
    src/avx2_search_backend.cpp
    src/avx2_support.cpp
    src/cuda_search_backend.cpp
    src/multithreaded_scalar_search_backend.cpp
    src/multithreaded_avx2_search_backend.cpp
    src/scalar_search_backend.cpp
    src/similarity.cpp
    src/vector_store.cpp
    src/vector_index.cpp
    src/hnsw_index.cpp
)
add_library(VectorPulse::vectorpulse ALIAS vectorpulse)
find_package(Threads REQUIRED)
target_link_libraries(vectorpulse PUBLIC Threads::Threads)

# CUDA is entirely optional: CPU-only builds neither enable CUDA nor find its toolkit.
set(VECTORPULSE_CUDA_COMPILED OFF)
if(VECTORPULSE_ENABLE_CUDA)
    include(CheckLanguage)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA target architectures")
    endif()
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
        target_sources(vectorpulse PRIVATE src/cuda_runtime.cu)
        target_link_libraries(vectorpulse PRIVATE CUDA::cudart_static)
        set_target_properties(vectorpulse PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON
                              CUDA_RUNTIME_LIBRARY Static)
        # Match the scalar accumulation order; no fast math in this correctness baseline.
        target_compile_options(vectorpulse PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
        set(VECTORPULSE_CUDA_COMPILED ON)
    else()
        message(WARNING "CUDA requested but no compiler found; building CPU-only CUDA stub")
    endif()
endif()
if(NOT VECTORPULSE_CUDA_COMPILED)
    target_sources(vectorpulse PRIVATE src/cuda_runtime_stub.cpp)
endif()
message(STATUS "VectorPulse CUDA compiled: ${VECTORPULSE_CUDA_COMPILED}")

# Only the kernel may contain AVX instructions; dispatch and scalar remain portable.
if(VECTORPULSE_ENABLE_AVX2 AND
   (CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|amd64|x86_64|X86|x86|i[3-6]86)$" OR
    CMAKE_CXX_COMPILER_ARCHITECTURE_ID MATCHES "^(x64|X86)$"))
    include(CheckCXXCompilerFlag)
    if(MSVC)
        set(avx2_flags /arch:AVX2)
    else()
        set(avx2_flags -mavx2 -mfma)
    endif()
    string(JOIN " " avx2_flag_check ${avx2_flags})
    check_cxx_compiler_flag("${avx2_flag_check}" VECTORPULSE_COMPILER_HAS_AVX2)
    if(VECTORPULSE_COMPILER_HAS_AVX2)
        target_sources(vectorpulse PRIVATE src/avx2_kernel.cpp)
        set_source_files_properties(src/avx2_kernel.cpp PROPERTIES COMPILE_OPTIONS "${avx2_flags}")
        target_compile_definitions(vectorpulse PRIVATE VECTORPULSE_HAS_AVX2_KERNEL=1)
    endif()
endif()

target_compile_features(vectorpulse PUBLIC cxx_std_20)
target_include_directories(vectorpulse
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if(MSVC)
    target_compile_options(vectorpulse PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/W4;/permissive->")
else()
    target_compile_options(vectorpulse PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Wconversion;-Wshadow>")
endif()

if(VECTORPULSE_BUILD_TESTS)
    include(CTest)
    include(FetchContent)

    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(googletest)

    add_executable(vectorpulse_tests
        tests/avx2_search_backend_test.cpp
        tests/multithreaded_scalar_search_backend_test.cpp
        tests/multithreaded_avx2_search_backend_test.cpp
        tests/cuda_search_backend_test.cpp
        tests/similarity_test.cpp
        tests/vector_store_test.cpp
        tests/vector_index_test.cpp
        tests/vector_index_persistence_test.cpp
        tests/hnsw_index_test.cpp
    )
    target_link_libraries(vectorpulse_tests PRIVATE VectorPulse::vectorpulse GTest::gtest_main)
    target_compile_features(vectorpulse_tests PRIVATE cxx_std_20)

    if(MSVC)
        target_compile_options(vectorpulse_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(vectorpulse_tests PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    endif()

    include(GoogleTest)
    gtest_discover_tests(vectorpulse_tests)
endif()

if(VECTORPULSE_BUILD_BENCHMARKS)
    add_executable(vectorpulse_hnsw_diagnostics benchmarks/hnsw_diagnostics.cpp)
    target_link_libraries(vectorpulse_hnsw_diagnostics PRIVATE VectorPulse::vectorpulse)
    add_executable(vectorpulse_hnsw_benchmark benchmarks/hnsw_benchmark.cpp)
    target_link_libraries(vectorpulse_hnsw_benchmark PRIVATE VectorPulse::vectorpulse)
    if(WIN32)
        target_link_libraries(vectorpulse_hnsw_benchmark PRIVATE psapi)
    endif()

    add_executable(vectorpulse_benchmark benchmarks/scalar_benchmark.cpp)
    target_link_libraries(vectorpulse_benchmark PRIVATE VectorPulse::vectorpulse)
    target_compile_features(vectorpulse_benchmark PRIVATE cxx_std_20)

    if(MSVC)
        target_compile_options(vectorpulse_benchmark PRIVATE /W4 /permissive-)
        target_compile_options(vectorpulse_hnsw_benchmark PRIVATE /W4 /permissive-)
        target_compile_options(vectorpulse_hnsw_diagnostics PRIVATE /W4 /permissive-)
    else()
        target_compile_options(vectorpulse_benchmark PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
        target_compile_options(vectorpulse_hnsw_benchmark PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
        target_compile_options(vectorpulse_hnsw_diagnostics PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    endif()
endif()

# Isolate existing CUDA kernels for Nsight; the CPU-only stub remains linkable.
if(VECTORPULSE_BUILD_BENCHMARKS)
    add_executable(vectorpulse_cuda_profile benchmarks/cuda_profile.cpp)
    target_link_libraries(vectorpulse_cuda_profile PRIVATE VectorPulse::vectorpulse)
    target_compile_features(vectorpulse_cuda_profile PRIVATE cxx_std_20)
endif()

if(VECTORPULSE_BUILD_EXAMPLES)
    add_executable(vectorpulse_basic_search examples/basic_search.cpp)
    target_link_libraries(vectorpulse_basic_search PRIVATE VectorPulse::vectorpulse)
endif()

# Export only the library and its public usage requirements, not development tools.
set_target_properties(vectorpulse PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
install(TARGETS vectorpulse EXPORT VectorPulseTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT VectorPulse
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT VectorPulse
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT VectorPulse)
install(DIRECTORY include/vectorpulse DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT VectorPulse FILES_MATCHING PATTERN "*.h")

set(VectorPulse_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/VectorPulse")
configure_package_config_file(cmake/VectorPulseConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/VectorPulseConfig.cmake"
    INSTALL_DESTINATION "${VectorPulse_INSTALL_CMAKEDIR}")
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/VectorPulseConfigVersion.cmake"
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(EXPORT VectorPulseTargets NAMESPACE VectorPulse::
    DESTINATION "${VectorPulse_INSTALL_CMAKEDIR}" COMPONENT VectorPulse)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/VectorPulseConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/VectorPulseConfigVersion.cmake"
    DESTINATION "${VectorPulse_INSTALL_CMAKEDIR}" COMPONENT VectorPulse)

if(VECTORPULSE_BUILD_TESTS AND BUILD_TESTING)
    configure_file(cmake/RunInstalledConsumer.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/RunInstalledConsumer.cmake" @ONLY)
    add_test(NAME VectorPulse.InstalledConsumer
        COMMAND ${CMAKE_COMMAND} "-DTEST_CONFIG=$<CONFIG>"
            -P "${CMAKE_CURRENT_BINARY_DIR}/RunInstalledConsumer.cmake")
endif()

if(VECTORPULSE_BUILD_PYTHON)
    option(VECTORPULSE_BUILD_PYTHON_TESTS "Build Python binding tests" ${VECTORPULSE_BUILD_TESTS})
    option(VECTORPULSE_BUILD_PYTHON_INSTALL_TESTS "Test pip install in a clean environment (downloads build dependencies)" OFF)
    if(VECTORPULSE_BUILD_PYTHON_TESTS OR VECTORPULSE_BUILD_PYTHON_INSTALL_TESTS)
        include(CTest)
    endif()
    add_subdirectory(python)
endif()

if(VECTORPULSE_BUILD_TESTS AND BUILD_TESTING AND VECTORPULSE_BUILD_BENCHMARKS)
    foreach(dataset random clustered)
        add_test(NAME VectorPulse.HnswDiagnostics.${dataset}
            COMMAND vectorpulse_hnsw_diagnostics --vectors 200 --dimension 16 --queries 5
                --m 16 --ef-construction 100 --dataset ${dataset} --repetitions 1)
    endforeach()
    add_test(NAME VectorPulse.HnswBenchmarkSmoke COMMAND vectorpulse_hnsw_benchmark
        --preset large --vectors 128 --dimension 7 --queries 5 --top-k 5 --m 4 --threads 2
        --ef-construction 24 --ef-search 5,16,128 --warmups 0 --repetitions 1)
    add_test(NAME VectorPulse.HnswBenchmarkInvalidThreads COMMAND vectorpulse_hnsw_benchmark --threads 0)
    add_test(NAME VectorPulse.HnswBenchmarkInvalidPreset COMMAND vectorpulse_hnsw_benchmark --preset unknown)
    set_tests_properties(VectorPulse.HnswBenchmarkInvalidThreads VectorPulse.HnswBenchmarkInvalidPreset
        PROPERTIES WILL_FAIL TRUE)
endif()
