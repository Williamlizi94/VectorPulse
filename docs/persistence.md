# VectorPulse persistence format, version 1

`VectorIndex::save(path)` writes only host vectors and IDs in insertion order.
`VectorIndex::load(path, backend = nullptr)` restores them through normal `add()`
validation. The default backend is scalar; an explicitly supplied backend must
be empty and have the dimension recorded in the file. Backend selection, thread
counts, CUDA kernel selection, device allocations, caches and timing state are
not stored. Persistent CUDA matrices are built lazily on the first search.

All integer fields are unsigned and little-endian. Float components are IEEE-754
binary32 bit patterns in little-endian byte order. No C++ struct layout, pointers,
padding, native `size_t`, or null terminators are written.

| Offset | Width | Field |
|---|---:|---|
| 0 | 8 | Magic bytes: `56 50 49 4e 44 45 58 00` (`VPINDEX` plus NUL) |
| 8 | 4 | Format version: 1 |
| 12 | 4 | Reserved flags: 0 |
| 16 | 8 | Positive vector dimension |
| 24 | 8 | Vector count (zero is valid) |
| 32 | variable | Records, in insertion order |
| End minus 4 | 4 | CRC-32/ISO-HDLC of all preceding bytes |

Each record consists of a uint64 ID byte length, that many raw string bytes,
and exactly `dimension` FP32 components (four bytes each). Empty IDs, embedded
NULs and UTF-8 bytes are preserved. IDs must be unique. Zero and signed-zero
components are preserved bit for bit. The CRC uses reflected polynomial
`0xedb88320`, initial value `0xffffffff`, and final XOR `0xffffffff`.
An empty file-format index occupies 36 bytes, regardless of its dimension.

Loading rejects bad magic, unsupported versions, nonzero reserved flags, zero or
unrepresentable dimensions, impossible counts/lengths, duplicate IDs, truncation,
trailing data and checksum mismatches. Declared record and ID sizes are checked
against remaining file bytes before allocation. Input is processed in bounded
I/O chunks, with only one decoded record held outside the destination backend.

File I/O, format and invalid-record errors throw `std::runtime_error`. A supplied
backend with the wrong dimension or existing entries throws
`std::invalid_argument`. Backend construction/support and allocation failures
retain their usual exceptions. CUDA input validation remains in force, so CPU
files containing nonfinite values or magnitudes outside an FP32 kernel's accepted
range cannot load into those CUDA configurations.

Saving truncates an existing target file; failed saves can leave partial output.
Save to a separate path if an existing snapshot must be retained. Loading returns
an index only after complete validation. Callers must exclude insertion during
save, just as during search. Custom SearchBackend implementations may override
`for_each_vector()` for saving; its default implementation throws without changing
their existing search/add interface.

With the same backend/settings and execution environment, round trips preserve
Top-K IDs, ranking and score bits. Switching between scalar, AVX2, FP64 CUDA and
FP32 CUDA retains their existing arithmetic differences, including possible
reordering of arbitrarily close scores. Persistence does not rescore results or
change search algorithms to conceal those differences.
