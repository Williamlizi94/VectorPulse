# Python bindings

The optional `vectorpulse` extension uses pybind11 and links the existing C++20
library. All storage, similarity scoring, Top-K ordering and persistence execute
in C++. CPU-only C++ builds and installed CMake consumers do not require Python
or pybind11.

## Install with pip

From the repository root:

```powershell
python -m pip install .
```

Then import directly, with no source/build path setup:

```python
from vectorpulse import VectorIndex

index = VectorIndex(2)
index.add("east", [1.0, 0.0])
print(index.search([1.0, 0.0], 1))
index.save("index.vp")
restored = VectorIndex.load("index.vp")
```

The source installation requires Python 3.9+ with development headers/libraries
and a C++20 compiler (MSVC Build Tools on Windows). Pip uses the
[scikit-build-core backend](https://scikit-build-core.readthedocs.io/en/stable/guide/cmakelists.html)
from `pyproject.toml` to obtain isolated build dependencies and build the same
`vectorpulse_python` CMake target. The default wheel statically links the CPU
library, needs no CUDA toolkit, and has no required Python runtime dependencies.
Install optional NumPy support with `python -m pip install ".[numpy]"`.

The wheel installs only the extension and Python distribution metadata; C++
headers, libraries, examples and CMake exports remain part of the separate C++
installation. Python package builds disable C++ tests/benchmarks/examples and
Python tests, so installing does not download GoogleTest or run tests recursively.
Standalone CMake defaults are unchanged. Wheel builds require
`BUILD_SHARED_LIBS=OFF`; standalone CMake builds still support shared libraries.

To request CUDA, use `python -m pip install . -Ccmake.define.VECTORPULSE_ENABLE_CUDA=ON`
and pass any necessary toolkit/compiler/architecture settings with additional
`-Ccmake.define.NAME=value` options, following the main README's CUDA setup.
The existing CUDA compiler detection and runtime support rules apply.

To produce distributable artifacts:

```powershell
python -m pip install build
python -m build --outdir dist
```

This builds an sdist and a platform-specific wheel from that sdist. Generated
build directories and large benchmark reports are excluded from the sdist.

## Test an installation

```powershell
python tests/python/test_install.py
```

The test creates a temporary virtual environment without system site packages,
runs isolated `pip install .`, then imports from an unrelated working directory.
It verifies module ownership under that environment, package metadata, and an
add/search/save/load round trip without NumPy. Build dependencies may be
downloaded. The environment is removed afterward. To test an already-built
wheel or sdist, pass `--artifact path/to/distribution`.

The same test is available as `VectorPulse.PythonInstall` in CTest when configured
with `VECTORPULSE_BUILD_PYTHON=ON` and `VECTORPULSE_BUILD_PYTHON_INSTALL_TESTS=ON`.
It is opt-in so ordinary C++/Python test runs do not create environments or access
the network.

## Build directly with CMake

Use Python 3.9+ with development headers/libraries and pybind11 2.13+. NumPy is
optional and needed only for NumPy inputs/tests. Install the Python dependencies
in your preferred environment:

```powershell
python -m pip install "pybind11>=2.13" numpy
cmake -S . -B out/python -DVECTORPULSE_BUILD_PYTHON=ON -DCMAKE_BUILD_TYPE=Release
cmake --build out/python --config Release
ctest --test-dir out/python -C Release -R "VectorPulse.Python" --output-on-failure
```

Set `-DPython_EXECUTABLE=/path/to/python` when multiple Python installations are
present. CMake discovers pip-installed pybind11 through that interpreter; a
separate CMake installation can be selected with `-Dpybind11_DIR=...`.
Discovery uses the [pybind11 CMake integration](https://pybind11.readthedocs.io/en/stable/cmake/).

The CMake target is `vectorpulse_python`; its import name is `vectorpulse`.
The commands above produce the extension in `out/python/python/Release` for both
Visual Studio and single-config Release generators. Add that directory to
`PYTHONPATH` or `sys.path`, using the same Python interpreter that built it.
For shared-library builds on Windows, CMake copies `vectorpulse.dll` beside the
extension. The existing C++ install/export remains independent of this module.

`VECTORPULSE_BUILD_PYTHON` defaults to OFF. `VECTORPULSE_BUILD_PYTHON_TESTS`
defaults to `VECTORPULSE_BUILD_TESTS`; it can be enabled independently with
`-DVECTORPULSE_BUILD_TESTS=OFF -DVECTORPULSE_BUILD_PYTHON_TESTS=ON` to run Python
tests without downloading GoogleTest. Tests use the standard-library `unittest`
module. NumPy-specific tests skip when it is absent; a second isolated test run
explicitly checks operation without NumPy.

For CUDA, add `-DVECTORPULSE_ENABLE_CUDA=ON` and the normal toolkit/architecture
options described in the main README. The Python module uses that same compiled
library; it does not contain separate kernels.

## Use

```python
from vectorpulse import VectorIndex

index = VectorIndex(3)
index.add("east", [1.0, 0.0, 0.0])
index.add("diagonal", (1.0, 1.0, 0.0))
index.add("north", [0.0, 1.0, 0.0])
print(index.size(), index.dimension())  # 3 3
print(index.search([1.0, 0.0, 0.0], 2))
# [{'id': 'east', 'score': 1.0}, {'id': 'diagonal', 'score': 0.7071067690849304}]

index.save("index.vp")
restored = VectorIndex.load("index.vp")
assert restored.search([1.0, 0.0, 0.0], 2) == index.search([1.0, 0.0, 0.0], 2)
```

`add(id, vector)` accepts string IDs and one-dimensional numeric sequences such
as lists, tuples and `array.array`. `search(query, k)` accepts the same vector
inputs and returns a list of dictionaries with `id` (str) and `score` (float).
The list preserves the C++ result order, including ascending IDs for exact ties.

Native FP32 buffers use a direct copy, including contiguous, strided, reversed,
read-only and unaligned NumPy float32 arrays. Other numeric sequence elements
are converted to C++ float. Multidimensional arrays and dimension mismatches are
rejected. Input memory is copied, so changing an array after `add()` does not
change stored vectors. No NumPy headers or NumPy import are required for the
native buffer/ordinary sequence paths; this uses Python's
[buffer protocol](https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html#buffer-protocol).

```python
import numpy as np
index.add("numpy", np.array([0.5, 0.5, 0.0], dtype=np.float32))
results = index.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), 2)
```

Paths accept strings and `pathlib.Path`. `load(path)` uses the C++ default scalar
backend and the existing `.vp` format, including its corruption checks. Saved
files contain no backend configuration or GPU state. Scalar round trips preserve
exact results; moving a CUDA/AVX2-created file to scalar retains the existing
cross-backend rounding caveats documented in [persistence.md](persistence.md).

C++ `invalid_argument` errors (duplicate IDs, invalid dimensions, backend options)
become `ValueError`; malformed vector types raise `TypeError`. File/format errors
and unavailable backends raise `RuntimeError`. Negative values for unsigned API
arguments are rejected. Calls retain the GIL, so Python callers cannot run
insertion concurrently with search/save through this module.

## Existing backend selection

The constructor defaults to scalar and accepts optional keyword-only backend
selection. There is no automatic fallback when an explicitly selected backend
is unavailable.

```python
import vectorpulse
print(vectorpulse.available_backends())
parallel = vectorpulse.VectorIndex(768, backend="multithreaded_scalar", threads=4)
if "cuda" in vectorpulse.available_backends():
    gpu = vectorpulse.VectorIndex(768, backend="cuda")
```

Supported names are `scalar`, `avx2`, `multithreaded_scalar`,
`multithreaded_avx2`, `cuda`, `cuda_block_parallel`, `cuda_fp32`,
`cuda_block_parallel_fp32`, and `cuda_per_query`. CUDA names map directly to the
existing storage/kernel configurations. `threads` is a positive thread budget
for the two multithreaded CPU backends; other backends ignore it. Existing CUDA
input range checks and hardware requirements still apply.
