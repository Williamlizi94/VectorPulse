"""Run with --module-dir pointing at the freshly built extension; no pytest needed."""

import argparse
from array import array
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

parser = argparse.ArgumentParser()
parser.add_argument("--module-dir", type=Path, required=True)
parser.add_argument("--without-numpy", action="store_true")
options, test_args = parser.parse_known_args()
module_dir = options.module_dir.resolve()
sys.path.insert(0, str(module_dir))
import vectorpulse as vp

if Path(vp.__file__).resolve().parent != module_dir:
    raise RuntimeError("Tests must exercise the requested built module")
if options.without_numpy and importlib.util.find_spec("numpy") is not None:
    raise RuntimeError("NumPy must be absent for the no-NumPy test")
try:
    import numpy as np
except ImportError:
    np = None


class VectorIndexTests(unittest.TestCase):
    def test_add_search_sequences_and_result_order(self):
        index = vp.VectorIndex(2)
        self.assertEqual(index.size(), 0)
        self.assertEqual(index.dimension(), 2)
        index.add("west", [-1.0, 0.0])
        index.add("b", (1.0, 1.0))
        index.add("a", array("f", [1.0, 1.0]))
        index.add("east", [1, 0])
        index.add("north", range(2))
        results = index.search(query=(1, 0), k=4)
        self.assertEqual([r["id"] for r in results], ["east", "a", "b", "north"])
        self.assertEqual(results[0], {"id": "east", "score": 1.0})
        self.assertEqual(set(results[1]), {"id", "score"})
        self.assertIsInstance(results[1]["score"], float)
        self.assertAlmostEqual(results[1]["score"], 2 ** -0.5, places=6)
        self.assertEqual(index.size(), 5)
        scores = [r["score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        results[0]["id"] = "changed"
        self.assertEqual(index.search([1, 0], 1)[0]["id"], "east")

    def test_zero_vectors_empty_index_and_k_edges(self):
        index = vp.VectorIndex(2)
        self.assertEqual(index.search([1, 0], 10), [])
        index.add("zero", [0, 0])
        self.assertEqual(index.search([1, 0], 0), [])
        self.assertEqual(index.search([0, 0], 99), [{"id": "zero", "score": 0.0}])
        with self.assertRaises((TypeError, ValueError)):
            index.search([1, 0], -1)

    def test_validation_and_exception_translation(self):
        with self.assertRaises(ValueError):
            vp.VectorIndex(0)
        with self.assertRaises((TypeError, ValueError)):
            vp.VectorIndex(-1)
        index = vp.VectorIndex(2)
        index.add("same", [1, 0])
        with self.assertRaises(ValueError):
            index.add("same", [0, 1])
        for bad in ([1], [1, 2, 3], []):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    index.add("bad", bad)
                with self.assertRaises(ValueError):
                    index.search(bad, 1)
        for bad in ("12", b"12", None, 42, {1, 2}, [[1], [2]], [object(), 2]):
            with self.subTest(bad=bad):
                with self.assertRaises((TypeError, ValueError)):
                    index.add("bad", bad)
        self.assertEqual(index.size(), 1)
        self.assertEqual(index.search([1, 0], 1)[0]["id"], "same")

    def test_buffer_and_sequence_inputs_are_copied(self):
        index = vp.VectorIndex(2)
        values = [1.0, 0.0]
        index.add("list", values)
        values[:] = [-1, 0]
        values = array("f", [1, 0])
        index.add("buffer", memoryview(values))
        values[0] = -1
        self.assertEqual([r["score"] for r in index.search([1, 0], 2)], [1.0, 1.0])

    def test_save_load_exact_round_trip_and_pathlike(self):
        index = vp.VectorIndex(3)
        for id_, vector in (("向量", [1.25, -2, 0.5]), ("", [0, 0, 0]),
                            ("nul\0id", [1, 2, 3]), ("tie", [1.25, -2, 0.5])):
            index.add(id_, vector)
        queries = ([0.25, 0.5, -0.75], [1.25, -2, 0.5], [0, 0, 0])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "索引.vp"
            index.save(path)
            self.assertEqual(path.read_bytes()[:8], b"VPINDEX\0")
            loaded = vp.VectorIndex.load(str(path))
            self.assertEqual(loaded.size(), index.size())
            self.assertEqual(loaded.dimension(), index.dimension())
            for query in queries:
                for k in (0, 1, 3, 99):
                    self.assertEqual(loaded.search(query, k), index.search(query, k))
            second = Path(tmp) / "copy.vp"
            loaded.save(str(second))
            self.assertEqual(path.read_bytes(), second.read_bytes())
            # Loading in a fresh process verifies this is a real disk round trip.
            code = (
                "import sys; sys.path.insert(0, sys.argv[1]); import vectorpulse; "
                "i = vectorpulse.VectorIndex.load(sys.argv[2]); "
                "assert i.dimension() == 3 and i.size() == 4"
            )
            subprocess.run([sys.executable, "-I", "-S", "-c", code, str(module_dir), str(path)],
                           check=True, capture_output=True, text=True)
            loaded.add("after-load", [0, 1, 0])
            self.assertEqual(loaded.size(), 5)
            self.assertEqual(index.size(), 4)

    def test_empty_round_trip_and_invalid_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.vp"
            vp.VectorIndex(7).save(path)
            loaded = vp.VectorIndex.load(path)
            self.assertEqual((loaded.size(), loaded.dimension()), (0, 7))
            with self.assertRaises(RuntimeError):
                vp.VectorIndex.load(Path(tmp) / "missing.vp")
            path.write_bytes(b"invalid file")
            with self.assertRaises(RuntimeError):
                vp.VectorIndex.load(path)
            vp.VectorIndex(7).save(path)
            corrupted = bytearray(path.read_bytes())
            corrupted[0] ^= 1
            path.write_bytes(corrupted)
            with self.assertRaises(RuntimeError):
                vp.VectorIndex.load(path)
            with self.assertRaises(RuntimeError):
                loaded.save(Path(tmp) / "missing" / "index.vp")

    def test_existing_backend_selection(self):
        available = vp.available_backends()
        self.assertIn("scalar", available)
        self.assertIn("multithreaded_scalar", available)
        expected = [{"id": "exact", "score": 1.0}, {"id": "zero", "score": 0.0}]
        for backend in available:
            with self.subTest(backend=backend):
                index = vp.VectorIndex(2, backend=backend, threads=2)
                index.add("zero", [0, 0])
                index.add("exact", [1, 0])
                self.assertEqual(index.search([1, 0], 2), expected)
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "backend.vp"
                    index.save(path)
                    self.assertEqual(vp.VectorIndex.load(path).search([1, 0], 2), expected)
        with self.assertRaises(ValueError):
            vp.VectorIndex(2, backend="unknown")
        with self.assertRaises(ValueError):
            vp.VectorIndex(2, backend="multithreaded_scalar", threads=0)
        if "cuda" not in available:
            with self.assertRaises(RuntimeError):
                vp.VectorIndex(2, backend="cuda")
        if "avx2" not in available:
            with self.assertRaises(RuntimeError):
                vp.VectorIndex(2, backend="avx2")


@unittest.skipIf(np is None, "NumPy is optional")
class NumPyInputTests(unittest.TestCase):
    def test_float32_contiguous_readonly_strided_reversed_and_unaligned(self):
        expected = np.array([1, 2, 3], dtype=np.float32)
        readonly = expected.copy()
        readonly.flags.writeable = False
        views = [expected, readonly,
                 np.array([1, 9, 2, 9, 3, 9], dtype=np.float32)[::2],
                 np.array([3, 2, 1], dtype=np.float32)[::-1],
                 np.ndarray((3,), dtype=np.float32, buffer=bytearray(13), offset=1)]
        views[-1][:] = expected
        for values in views:
            with self.subTest(strides=values.strides, aligned=values.flags.aligned):
                index = vp.VectorIndex(3)
                index.add("array", values)
                self.assertEqual(index.search(values, 1), [{"id": "array", "score": 1.0}])
                self.assertEqual(index.search(values, 1), index.search([1, 2, 3], 1))

    def test_numpy_storage_is_copied(self):
        values = np.array([1, 0], dtype=np.float32)
        index = vp.VectorIndex(2)
        index.add("owned", values)
        values[:] = [-1, 0]
        self.assertEqual(index.search([1, 0], 1), [{"id": "owned", "score": 1.0}])

    def test_other_numeric_dtypes_and_numpy_round_trip(self):
        index = vp.VectorIndex(3)
        for dtype in (np.float32, np.float64, np.int32, ">f4"):
            with self.subTest(dtype=dtype):
                values = np.array([1, 2, 3], dtype=dtype)
                index.add(str(dtype), values)
                self.assertAlmostEqual(index.search(values, 1)[0]["score"], 1.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "numpy.vp"
            index.save(path)
            loaded = vp.VectorIndex.load(path)
            query = np.array([0.25, 1, -0.5], dtype=np.float32)
            self.assertEqual(index.search(query, 10), loaded.search(query, 10))

    def test_rejects_multidimensional_and_wrong_size_arrays(self):
        index = vp.VectorIndex(3)
        for values in (np.ones((1, 3), dtype=np.float32), np.ones((3, 1), dtype=np.float32),
                       np.ones((2, 3), dtype=np.float32), np.array(1, dtype=np.float32),
                       np.ones(2, dtype=np.float32)):
            with self.subTest(shape=values.shape):
                with self.assertRaises(ValueError):
                    index.add("bad", values)
                with self.assertRaises(ValueError):
                    index.search(values, 1)
        self.assertEqual(index.size(), 0)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], *test_args], verbosity=2)
