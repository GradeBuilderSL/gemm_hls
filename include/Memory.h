/// @author    Johannes de Fine Licht (definelicht@inf.ethz.ch)
/// @copyright This software is copyrighted under the BSD 3-Clause License.

#pragma once

#include "MatrixMultiplication.h"
#include "hlslib/xilinx/Stream.h"

// Non-transposed A path: read row-major A and transpose on-chip.
void ReadA(MemoryPackK_t const a[], Stream<Data_t> aSplit[kTransposeWidth],
           unsigned n, unsigned k, unsigned m);

void TransposeA(Stream<Data_t> aSplit[kTransposeWidth],
                Stream<ComputePackN_t> &toKernel, unsigned n, unsigned k,
                unsigned m);

// Pre-transposed A path: A is stored as K×N (column-major original).
// The buffer uses MemoryPackK_t (same width as MemoryPackN_t, per static_assert).
void ReadATransposed(MemoryPackK_t const memory[], Stream<MemoryPackK_t> &pipe,
                     const unsigned size_n, const unsigned size_k,
                     const unsigned size_m);

void ConvertWidthATransposed(Stream<MemoryPackK_t> &pipe_in,
                             Stream<ComputePackN_t> &pipe_out,
                             const unsigned size_n, const unsigned size_k,
                             const unsigned size_m);

void ReadB(MemoryPackM_t const memory[], Stream<MemoryPackM_t> &pipe,
           unsigned n, unsigned k, unsigned m);

#ifdef MM_CONVERT_B

void ConvertWidthB(Stream<MemoryPackM_t> &wide, Stream<ComputePackM_t> &narrow,
                   unsigned n, unsigned k, unsigned m);

void FeedB(Stream<ComputePackM_t> &converted, Stream<ComputePackM_t> &toKernel,
           unsigned n, unsigned k, unsigned m);

#else

void FeedB(Stream<ComputePackM_t> &fromMemory, Stream<ComputePackM_t> &toKernel,
           unsigned n, unsigned k, unsigned m);

#endif

void ConvertWidthC(Stream<OutputPack_t> &narrow, Stream<MemoryPackM_t> &wide,
                   unsigned n, unsigned k, unsigned m);

void WriteC(Stream<MemoryPackM_t> &pipe, MemoryPackM_t memory[], unsigned n,
            unsigned k, unsigned m);
