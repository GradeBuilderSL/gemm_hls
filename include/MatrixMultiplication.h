/// @author    Johannes de Fine Licht (definelicht@inf.ethz.ch)
/// @copyright This software is copyrighted under the BSD 3-Clause License. 

#pragma once

#include <type_traits>
#include "Config.h"
#include "hlslib/xilinx/DataPack.h"
#include "hlslib/xilinx/Resource.h"
#include "hlslib/xilinx/Stream.h"

using hlslib::Stream;

constexpr int kSeed = 5; // For initializing matrices for testing
constexpr unsigned kPipeDepth = 4;

// Memory bus in K-dimension
constexpr int kMemoryWidthK = kMemoryWidthBytesK / sizeof(Data_t);
static_assert(kMemoryWidthBytesK % sizeof(Data_t) == 0,
              "Memory width in K not divisable by size of data type.");
using MemoryPackK_t = hlslib::DataPack<Data_t, kMemoryWidthK>;

// Memory bus in M-dimension
constexpr int kMemoryWidthM = kMemoryWidthBytesM / sizeof(Data_t);
static_assert(kMemoryWidthBytesM % sizeof(Data_t) == 0,
              "Memory width in M not divisable by size of data type.");
using MemoryPackM_t = hlslib::DataPack<Data_t, kMemoryWidthM>;

// Internal compute buses
using ComputePackN_t = hlslib::DataPack<Data_t, kComputeTileSizeN>;
using ComputePackM_t = hlslib::DataPack<Data_t, kComputeTileSizeM>;
using OutputPack_t = hlslib::DataPack<Data_t, kComputeTileSizeM>;
// Accumulator bus: wider integer part to prevent overflow during K-reduction.
// When AccData_t == Data_t (float/double/int), this is identical to ComputePackM_t.
using AccComputePackM_t = hlslib::DataPack<AccData_t, kComputeTileSizeM>;

// On-chip transpose of A (used when A is row-major in DDR)
constexpr int kTransposeWidth = kTransposeWidthBytes / sizeof(Data_t);
static_assert(kTransposeWidthBytes % sizeof(Data_t) == 0,
              "Transpose width must be divisible by data size.");
static_assert(kTransposeWidthBytes % kMemoryWidthBytesK == 0,
              "Transpose width must be divisible by memory port width.");

// N-dimension memory bus (used when A is pre-transposed in DDR).
// Runtime transposed-A mode requires equal K and N bus widths so a single
// AXI port can serve both modes without changing the port type.
constexpr int kMemoryWidthN = kMemoryWidthBytesN / sizeof(Data_t);
static_assert(kMemoryWidthBytesN % sizeof(Data_t) == 0,
              "Memory width in N not divisible by size of data type.");
static_assert(kMemoryWidthBytesN == kMemoryWidthBytesK,
              "Runtime transposed-A requires equal K and N memory bus widths "
              "(MM_MEMORY_BUS_WIDTH_N == MM_MEMORY_BUS_WIDTH_K).");
using MemoryPackN_t = hlslib::DataPack<Data_t, kMemoryWidthN>;

// A always uses the K-width bus (identical to the N-width bus per the assert).
using MemoryPackA_t = MemoryPackK_t;
constexpr decltype(kMemoryWidthK) kMemoryWidthA = kMemoryWidthK;

constexpr unsigned long kOuterTileSizeNMemory = kOuterTileSizeN / kMemoryWidthN;
static_assert(kOuterTileSizeN % kMemoryWidthN == 0,
              "Outer memory tile size in N must be divisible by memory port width.");

inline unsigned SizeNMemory(unsigned n) {
  #pragma HLS INLINE
  return n / kMemoryWidthN;
}

constexpr unsigned long kOuterTileSizeMMemory = kOuterTileSizeM / kMemoryWidthM;
static_assert(
    kOuterTileSizeM % kMemoryWidthM == 0,
    "Outer memory tile size in M must be divisable by memory port width.");

constexpr unsigned long kInnerTilesN = kOuterTileSizeN / kInnerTileSizeN;
static_assert(kOuterTileSizeN % kInnerTileSizeN == 0,
              "Outer tile size must be divisable by the inner tile size.");

constexpr unsigned long kInnerTilesM = kOuterTileSizeM / kComputeTileSizeM;
static_assert(kOuterTileSizeM % kComputeTileSizeM == 0,
              "Outer tile size must be divisable by compute tile size in M.");

constexpr unsigned long kComputeTilesN = kInnerTileSizeN / kComputeTileSizeN;
static_assert(kInnerTileSizeN % kComputeTileSizeN == 0,
              "Inner tile size must be divisable by compute tile size.");

#ifndef MM_DYNAMIC_SIZES

static_assert(kSizeK % kMemoryWidthK == 0,
              "K must be divisable by memory width.");
static_assert(kSizeK % kTransposeWidth == 0,
              "K must be divisable by the transpose width.");

#endif

inline unsigned SizeKMemory(unsigned k) {
  #pragma HLS INLINE
  return k / kMemoryWidthK;
}

inline unsigned SizeMMemory(unsigned m) {
  #pragma HLS INLINE
  return m / kMemoryWidthM;
}

inline unsigned OuterTilesN(unsigned n) {
  #pragma HLS INLINE
  return (n + kOuterTileSizeN - 1) / kOuterTileSizeN;
}

inline unsigned OuterTilesM(unsigned m) {
  #pragma HLS INLINE
  return (m + kOuterTileSizeM - 1) / kOuterTileSizeM;
}

inline unsigned long TotalReadsFromA(const unsigned size_n,
                                     const unsigned size_k,
                                     const unsigned size_m) {
  #pragma HLS INLINE
  return static_cast<unsigned long>(OuterTilesN(size_n)) * OuterTilesM(size_m) *
         kOuterTileSizeN * size_k;
}

inline unsigned long TotalReadsFromB(const unsigned size_n,
                                     const unsigned size_k,
                                     const unsigned size_m) {
  #pragma HLS INLINE
  return static_cast<unsigned long>(OuterTilesN(size_n)) * OuterTilesM(size_m) *
         kOuterTileSizeM * size_k;
}

template <typename T,
          class = typename std::enable_if<std::is_integral<T>::value, T>::type>
constexpr T PowerOfTwo(T number, unsigned char power) {
  return (number > 0) ? PowerOfTwo(number >> 1, power + 1) : (1 << (power - 1));
}

#ifdef MM_ADD_RESOURCE
#define MM_ADD_RESOURCE_PRAGMA(var)                                 \
  HLSLIB_RESOURCE_PRAGMA(var, MM_ADD_RESOURCE)
#else
#define MM_ADD_RESOURCE_PRAGMA(var)
#endif

#ifdef MM_MULT_RESOURCE
#define MM_MULT_RESOURCE_PRAGMA(var)                                 \
  HLSLIB_RESOURCE_PRAGMA(var, MM_MULT_RESOURCE)
#else
#define MM_MULT_RESOURCE_PRAGMA(var)
#endif

extern "C" {

void MatrixMultiplicationKernel(MemoryPackK_t const a[],
                                MemoryPackM_t const b[], MemoryPackM_t c[]
#ifdef MM_DYNAMIC_SIZES
                                ,
                                const unsigned size_n, const unsigned size_k,
                                const unsigned size_m
#endif
                                ,
                                const bool transposed_a = false
);

}
