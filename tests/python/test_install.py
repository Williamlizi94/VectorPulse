"""Install the project (or a distribution artifact) into a fresh Python environment."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import venv


SMOKE_TEST = r'''
from importlib import metadata, util
from pathlib import Path
import sys
import vectorpulse
from vectorpulse import VectorIndex

prefix = Path(sys.prefix).resolve()
module = Path(vectorpulse.__file__).resolve()
assert prefix in module.parents, (prefix, module)
assert sys.prefix != sys.base_prefix, "Expected an isolated virtual environment"
assert util.find_spec("numpy") is None, "NumPy must not be a required runtime dependency"
assert metadata.version("vectorpulse") == vectorpulse.__version__
files = metadata.distribution("vectorpulse").files
assert files and not any(str(path).endswith((".lib", ".a", ".h", ".cmake")) for path in files)

index = VectorIndex(3)
index.add("north", [0.0, 1.0, 0.0])
index.add("east", (1.0, 0.0, 0.0))
index.add("diagonal", [1.0, 1.0, 0.0])
assert index.size() == 3 and index.dimension() == 3
query = [1.0, 0.0, 0.0]
results = index.search(query, 2)
assert [result["id"] for result in results] == ["east", "diagonal"]
assert results[0]["score"] == 1.0
assert abs(results[1]["score"] - 2 ** -0.5) < 1e-6

path = Path("index.vp")
index.save(path)
assert path.read_bytes()[:8] == b"VPINDEX\0"
loaded = VectorIndex.load(path)
assert loaded.size() == index.size() and loaded.dimension() == index.dimension()
for query in ([1, 0, 0], [0.25, -0.5, 0.75], [0, 0, 0]):
    for k in (0, 1, 2, 10):
        assert loaded.search(query, k) == index.search(query, k)
loaded.add("after-load", [0, 0, 1])
assert loaded.search([0, 0, 1], 1)[0]["id"] == "after-load"
print(f"Installed VectorPulse {vectorpulse.__version__}: {module}")
print("Clean-environment add/search/save/load passed")
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--artifact", type=Path, help="Optional wheel or sdist to install instead")
    args = parser.parse_args()
    source = args.source.resolve()
    artifact = args.artifact.resolve() if args.artifact else None
    if artifact is not None and not artifact.is_file():
        parser.error(f"Distribution artifact does not exist: {artifact}")
    if artifact is None and not (source / "pyproject.toml").is_file():
        parser.error(f"No pyproject.toml in {source}")

    env = os.environ.copy()
    # Preserve compiler/toolchain setup, but exclude Python/build overrides that
    # could make this test accidentally consume an existing development build.
    for key in list(env):
        if key.upper() in {"PYTHONPATH", "PYTHONHOME", "CMAKE_ARGS"} or key.upper().startswith("SKBUILD_"):
            env.pop(key)
    with tempfile.TemporaryDirectory(prefix="vectorpulse-install-") as temporary:
        root = Path(temporary)
        environment = root / "venv"
        venv.EnvBuilder(with_pip=True, system_site_packages=False).create(environment)
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        work = root / "consumer"
        work.mkdir()
        # Source installs deliberately exercise the exact `pip install .` route,
        # with PEP 517 build isolation and no preinstalled pybind11 or NumPy.
        subprocess.run(
            [str(python), "-I", "-m", "pip", "--isolated", "install", "--no-deps",
             "--no-cache-dir", str(artifact) if artifact else "."],
            cwd=work if artifact else source, env=env, check=True,
        )
        subprocess.run([str(python), "-I", "-c", SMOKE_TEST], cwd=work, env=env, check=True)


if __name__ == "__main__":
    main()
