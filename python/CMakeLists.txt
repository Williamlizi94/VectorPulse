find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    # Also find pybind11 installed by pip into the selected interpreter's environment.
    execute_process(COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE vectorpulse_pybind11_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET RESULT_VARIABLE vectorpulse_pybind11_result)
    if("${vectorpulse_pybind11_result}" STREQUAL "0")
        find_package(pybind11 CONFIG REQUIRED HINTS "${vectorpulse_pybind11_dir}")
    else()
        message(FATAL_ERROR
            "Python bindings need pybind11 >= 2.13. Install it with '${Python_EXECUTABLE} -m pip install pybind11' or set pybind11_DIR.")
    endif()
endif()
if(pybind11_VERSION VERSION_LESS 2.13)
    message(FATAL_ERROR "VectorPulse Python bindings require pybind11 >= 2.13")
endif()

# Static library objects must be usable inside the extension on ELF platforms.
set_target_properties(vectorpulse PROPERTIES POSITION_INDEPENDENT_CODE ON)
pybind11_add_module(vectorpulse_python MODULE NO_EXTRAS bindings.cpp)
target_link_libraries(vectorpulse_python PRIVATE VectorPulse::vectorpulse)
target_compile_definitions(vectorpulse_python PRIVATE VECTORPULSE_VERSION="${PROJECT_VERSION}")
set_target_properties(vectorpulse_python PROPERTIES
    OUTPUT_NAME vectorpulse
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>")

if(WIN32 AND BUILD_SHARED_LIBS)
    add_custom_command(TARGET vectorpulse_python POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:vectorpulse>" "$<TARGET_FILE_DIR:vectorpulse_python>"
        VERBATIM)
endif()

if(VECTORPULSE_BUILD_PYTHON_TESTS)
    if(BUILD_TESTING)
        add_test(NAME VectorPulse.Python
            COMMAND "${Python_EXECUTABLE}" -I
                "${PROJECT_SOURCE_DIR}/tests/python/test_vectorpulse.py"
                --module-dir "$<TARGET_FILE_DIR:vectorpulse_python>")
        # Ensure ordinary sequences do not acquire a runtime NumPy dependency.
        add_test(NAME VectorPulse.PythonWithoutNumPy
            COMMAND "${Python_EXECUTABLE}" -I -S
                "${PROJECT_SOURCE_DIR}/tests/python/test_vectorpulse.py"
                --module-dir "$<TARGET_FILE_DIR:vectorpulse_python>" --without-numpy)
    endif()
endif()

# A wheel contains only the existing extension, with the C++ library linked in.
# Keep standalone CMake install/export behavior separate from Python packaging.
if(SKBUILD)
    if(BUILD_SHARED_LIBS)
        message(FATAL_ERROR "Python wheels require BUILD_SHARED_LIBS=OFF to bundle the C++ implementation")
    endif()
    install(TARGETS vectorpulse_python
        LIBRARY DESTINATION . COMPONENT VectorPulsePython
        RUNTIME DESTINATION . COMPONENT VectorPulsePython)
endif()

if(VECTORPULSE_BUILD_PYTHON_INSTALL_TESTS AND BUILD_TESTING)
    add_test(NAME VectorPulse.PythonInstall
        COMMAND "${Python_EXECUTABLE}" -I
            "${PROJECT_SOURCE_DIR}/tests/python/test_install.py"
            --source "${PROJECT_SOURCE_DIR}")
    set_tests_properties(VectorPulse.PythonInstall PROPERTIES
        LABELS "packaging" RUN_SERIAL TRUE TIMEOUT 600)
endif()