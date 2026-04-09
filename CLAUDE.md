# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a scalable matrix-matrix multiplication (GEMM) kernel for Xilinx FPGAs, implemented in Vitis HLS. It uses a systolic array architecture with configurable processing elements (PEs). Published at FPGA'20.

Theoretical peak: `2 × MM_PARALLELISM_N × MM_PARALLELISM_M × Frequency (GHz)` FLOP/s.

## Build System

All configuration is done through CMake parameters. There is no default platform — you must specify one.

```bash
git submodule update --init   # Required: fetch hlslib and powermeter

mkdir build && cd build
cmake ../ \
  -DMM_PLATFORM=xilinx_u250_gen3x16_xdma_3_1_202020_1 \
  -DMM_DATA_TYPE=float \
  -DMM_PARALLELISM_N=32 \
  -DMM_PARALLELISM_M=8 \
  -DMM_MEMORY_TILE_SIZE_N=512 \
  -DMM_MEMORY_TILE_SIZE_M=512

make                # Builds TestSimulation and RunHardware.exe
make synthesis      # HLS synthesis only (requires Vitis, ~30 min)
make hw_emu         # Hardware emulation (requires Vitis, hours)
make hw             # Full place & route (requires Vitis, many hours)
```

### Key CMake Parameters

| Parameter | Default | Description |
|---|---|---|
| `MM_DATA_TYPE` | `float` | Element type: `float`, `double`, `half`, `uint8_t` |
| `MM_PARALLELISM_N` | 32 | Number of PEs (systolic array length) |
| `MM_PARALLELISM_M` | 8 | Vector width per PE; max = 64 / sizeof(Data_t) |
| `MM_MEMORY_TILE_SIZE_N/M` | 256 | Outer tile sizes in elements |
| `MM_MEMORY_BUS_WIDTH_N/K/M` | 64 | Memory bus widths in bytes |
| `MM_DYNAMIC_SIZES` | ON | Allow runtime matrix dimensions |
| `MM_SIZE_N/K/M` | 512 | Static dimensions (when DYNAMIC_SIZES=OFF) |
| `MM_TRANSPOSED_A` | OFF | If ON, reads A pre-transposed (skips on-chip transpose) |
| `MM_TWO_DIMMS` | OFF | Use two DDR DIMMs |
| `MM_MAP_OP` | `Multiply` | hlslib operator for map step |
| `MM_REDUCE_OP` | `Add` | hlslib operator for reduce step |
### Per-platform HLS synthesis

Each `platforms/<name>.json` file defines a `synthesize_<name>` CMake target. Required JSON fields: `part`, `parallelism_n`, `memory_tile_size_n`. Optional: `board`, `clock` (default 300), `parallelism_m`, `memory_tile_size_m`, `memory_bus_width_n/k/m`. Adding a new platform only requires a JSON file and re-running cmake.

Constraints CMake enforces (issues a WARNING and skips the platform if violated):
- `memory_tile_size_n % parallelism_n == 0`
- tile sizes divisible by bus width in elements
- `tile_n/parallelism_n ≤ tile_m/parallelism_m` (double-buffer constraint in `Compute.cpp`)

Each platform generates `build/<name>/Config.h` at configure time; it is placed first on the HLS include path to shadow the default `Config.h`.

## Running Tests

```bash
cd build
ctest                        # Runs TestSimulation with configured dimensions
./TestSimulation 512 512 512 # Direct invocation (requires MM_DYNAMIC_SIZES=ON)
```

`test/TestSimulation.cpp` generates random matrices, runs the HLS kernel simulation, and verifies against BLAS (if available) or a naive reference. Tolerance: 0.1% for floating-point, exact for integers.

## Running on Hardware

```bash
./RunHardware.exe 1024 1024 1024 hw on       # Real FPGA, with verification
./RunHardware.exe 1024 1024 1024 hw_emu on   # Hardware emulation
./RunHardware.exe hw                          # Static size mode (MM_DYNAMIC_SIZES=OFF)
```

## Architecture

### Dataflow Pipeline

```
DDR → ReadA → [TransposeA / ConvertWidthA] → aPipes[0..N-1] → ProcessingElement[0..N-1]
DDR → ReadB → [ConvertWidthB] → bPipes                       ↑ systolic pass-through
                              WriteC ← [ConvertWidthC] ← cPipes
```

### Key Files

- **`kernel/Top.cpp`** — Kernel entry point `MatrixMultiplicationKernel()`. Declares AXI-M ports, instantiates hlslib Streams, and invokes dataflow.
- **`kernel/Memory.cpp`** — All DDR I/O: `ReadA`, `ReadATransposed`, `TransposeA`, `ConvertWidthA/B/C`, `ReadB`, `WriteC`. Width conversions between memory bus and compute bus widths live here.
- **`kernel/Compute.cpp`** — `ProcessingElement()`: double-buffered accumulation of partial outer products. Each PE holds a local A buffer and a C accumulator buffer in BRAM.
- **`include/MatrixMultiplication.h`** — Central header: `Data_t`, all pack types (`MemoryPackK_t`, `ComputePackN_t`, etc.), tile size constants, HLS resource pragmas.
- **`include/Utility.h`** — Reference implementations (`Naive()`, `CallBLAS()`), `Pack<W>()` / `Unpack<W>()` helpers.
- **`host/RunHardware.cpp`** — OpenCL host code using hlslib wrappers: loads xclbin, allocates aligned buffers, transfers data, launches kernel, measures time.

### Tiling Strategy

- **Outer tiles** (memory): `MM_MEMORY_TILE_SIZE_N × MM_MEMORY_TILE_SIZE_M` — swept over the output matrix.
- **Inner tiles** (compute): `MM_PARALLELISM_N × MM_PARALLELISM_M` — mapped to the systolic array.
- The K-dimension is streamed through each PE; A values are double-buffered for perfect pipelining.

### On-Chip Transpose

When `MM_TRANSPOSED_A=OFF` (default), matrix A is stored row-major in DDR but must be read column-major for the systolic array. `TransposeA()` performs this on-chip using a stream-based ping-pong buffer with width `MM_TRANSPOSE_WIDTH` bytes.

### HLS Pragmas Used

`#pragma HLS DATAFLOW`, `PIPELINE II=1`, `LOOP_FLATTEN`, `UNROLL`, `ARRAY_PARTITION` — primarily in `kernel/Compute.cpp` and `kernel/Memory.cpp`.

## Dependencies

- **`hlslib/`** (submodule) — Provides CMake macros `add_vitis_kernel` / `add_vitis_program`, `hlslib::Stream`, `hlslib::DataPack`, OpenCL wrappers, and operator types (`hlslib::op::Multiply`, etc.).
- **Xilinx Vitis** — Required for synthesis/emulation/hardware targets. Tested on 2021.1.
- **BLAS** (optional) — Used for reference verification in tests and host code.
- **`powermeter/`** (submodule) — Optional Corsair power measurement; enabled with `-DMM_POWER_METER=ON`.

## Multi-Configuration Builds

`scripts/build_manager.py` automates parameter sweeps across multiple configurations, writing builds under `scan/` and results under `benchmark/`. `scripts/optimal_memory_tile_size.py` helps compute tile sizes that maximize utilization.
