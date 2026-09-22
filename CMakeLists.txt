cmake_minimum_required(VERSION 3.20)

project(VectorPulse VERSION 0.1.0 LANGUAGES CXX)

option(VECTORPULSE_BUILD_TESTS "Build VectorPulse unit tests" ON)
option(VECTORPULSE_BUILD_BENCHMARKS "Build VectorPulse benchmarks" ON)
option(VECTORPULSE_ENABLE_AVX2 "Build optional AVX2/FMA kernel on x86" ON)

add_library(vectorpulse
    src/avx2_search_backend.cpp
    src/avx2_support.cpp
    src/scalar_search_backend.cpp
    src/similarity.cpp
    src/vector_store.cpp
)
add_library(VectorPulse::vectorpulse ALIAS vectorpulse)

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
        $<INSTALL_INTERFACE:include>
)

if(MSVC)
    target_compile_options(vectorpulse PRIVATE /W4 /permissive-)
else()
    target_compile_options(vectorpulse PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
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
        tests/similarity_test.cpp
        tests/vector_store_test.cpp
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
    add_executable(vectorpulse_benchmark benchmarks/scalar_benchmark.cpp)
    target_link_libraries(vectorpulse_benchmark PRIVATE VectorPulse::vectorpulse)
    target_compile_features(vectorpulse_benchmark PRIVATE cxx_std_20)

    if(MSVC)
        target_compile_options(vectorpulse_benchmark PRIVATE /W4 /permissive-)
    else()
        target_compile_options(vectorpulse_benchmark PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    endif()
endif()
