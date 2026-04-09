#include "MatrixMultiplication.h"
#include "Compute.h"
#include "Memory.h"
#include "hlslib/xilinx/Simulation.h"

// #ifndef HLSLIB_SYNTHESIS
// #define HLSLIB_SYNTHESIS
// #endif

#ifdef HLSLIB_SYNTHESIS
// Template-recursive PE chain — synthesis only.
//
// Each instantiation emits one ProcessingElement call with compile-time-
// constant stream indices and recurses to pe+1.  #pragma HLS INLINE on
// every instantiation collapses the whole chain into the DATAFLOW region so
// HLS sees only direct function calls — satisfying canonical DATAFLOW form
// (HLS 214-114).
//
// Not used in simulation because HLSLIB_DATAFLOW_FUNCTION expands to a
// thread-pool call that references __hlslib_dataflow_context, a local
// variable of the kernel function that would be out of scope inside a
// separately compiled helper.  A plain for loop handles simulation instead.
template <int pe, int max_pe>
struct _PEChain {
  static void run(
      Stream<ComputePackN_t> (&aPipes)[kComputeTilesN + 1],
      Stream<ComputePackM_t> (&bPipes)[kComputeTilesN + 1],
      Stream<ComputePackM_t> (&cPipes)[kComputeTilesN + 1],
      const unsigned size_n, const unsigned size_k, const unsigned size_m) {
    #pragma HLS INLINE
    ProcessingElement(aPipes[pe], aPipes[pe + 1],
                      bPipes[pe], bPipes[pe + 1],
                      cPipes[pe], cPipes[pe + 1],
                      pe, size_n, size_k, size_m);
    _PEChain<pe + 1, max_pe>::run(aPipes, bPipes, cPipes,
                                  size_n, size_k, size_m);
  }
};

template <int max_pe>
struct _PEChain<max_pe, max_pe> {
  static void run(
      Stream<ComputePackN_t> (&)[kComputeTilesN + 1],
      Stream<ComputePackM_t> (&)[kComputeTilesN + 1],
      Stream<ComputePackM_t> (&)[kComputeTilesN + 1],
      const unsigned, const unsigned, const unsigned) {
    #pragma HLS INLINE
  }
};
#endif  // HLSLIB_SYNTHESIS

void MatrixMultiplicationKernel(MemoryPackK_t const a[],
                                MemoryPackM_t const b[], MemoryPackM_t c[]
#ifdef MM_DYNAMIC_SIZES
                                ,
                                const unsigned size_n, const unsigned size_k,
                                const unsigned size_m
#endif
                                ,
                                const bool transposed_a
) {

  #pragma HLS INTERFACE m_axi port=a offset=slave bundle=gmem0
  #pragma HLS INTERFACE m_axi port=b offset=slave bundle=gmem1
  #pragma HLS INTERFACE m_axi port=c offset=slave bundle=gmem2
  #pragma HLS INTERFACE s_axilite port=transposed_a

  #pragma HLS DATAFLOW

#ifndef MM_DYNAMIC_SIZES
  const unsigned size_n = kSizeN;
  const unsigned size_k = kSizeK;
  const unsigned size_m = kSizeM;
#endif

  // All stream declarations must precede all function calls for canonical
  // DATAFLOW form (HLS 214-114).

  // Streams for A — both sets always declared; runtime transposed_a selects
  // which DATAFLOW path is taken.
  // Synthesis uses Stream<T> (no template depth) to avoid the auto-DISAGGREGATE
  // issue in Vitis HLS 2022+ (HLS 214-177); depth is set via external pragma.
  // Simulation keeps the full template depth for correct blocking semantics.
#ifdef HLSLIB_SYNTHESIS
  Stream<Data_t> aSplit[kTransposeWidth];
  #pragma HLS STREAM variable=aSplit depth=2*kOuterTileSizeN
  Stream<MemoryPackK_t> aMemory;
  #pragma HLS STREAM variable=aMemory depth=2*kOuterTileSizeNMemory
#else
  Stream<Data_t, 2 * kOuterTileSizeN> aSplit[kTransposeWidth];
  Stream<MemoryPackK_t, 2 * kOuterTileSizeNMemory> aMemory("aMemory");
#endif

#ifdef HLSLIB_SYNTHESIS
  Stream<ComputePackN_t> aPipes[kComputeTilesN + 1];
#else
  Stream<ComputePackN_t, kPipeDepth> aPipes[kComputeTilesN + 1];
#endif
  #pragma HLS STREAM variable=aPipes depth=kPipeDepth

  // Streams for B
  Stream<MemoryPackM_t, 2 * kOuterTileSizeMMemory> bMemory("bMemory");

#ifdef HLSLIB_SYNTHESIS
  Stream<ComputePackM_t> bPipes[kComputeTilesN + 1];
#else
  Stream<ComputePackM_t, kPipeDepth> bPipes[kComputeTilesN + 1];
#endif
  #pragma HLS STREAM variable=bPipes depth=kPipeDepth

#ifdef MM_CONVERT_B
  Stream<ComputePackM_t> bFeed("bFeed");
#endif

  // Streams for C
  Stream<ComputePackM_t> cPipes[kComputeTilesN + 1];
  Stream<MemoryPackM_t, 2 * kOuterTileSizeMMemory> cMemory("cMemory");

#ifndef HLSLIB_SYNTHESIS
  // Name the arrays of channels for debugging purposes
  for (unsigned i = 0; i < kTransposeWidth; ++i) {
    aSplit[i].set_name(("aSplit[" + std::to_string(i) + "]").c_str());
  }
  for (unsigned n = 0; n < kComputeTilesN; ++n) {
    aPipes[n].set_name(("aPipes[" + std::to_string(n) + "]").c_str());
  }
  for (unsigned n = 0; n < kComputeTilesN + 1; ++n) {
    bPipes[n].set_name(("bPipes[" + std::to_string(n) + "]").c_str());
  }
  for (unsigned n = 0; n < kComputeTilesN + 1; ++n) {
    cPipes[n].set_name(("cPipes[" + std::to_string(n) + "]").c_str());
  }
#endif

  HLSLIB_DATAFLOW_INIT();

  // Feed A: runtime transposed_a selects which path to execute.
  // In synthesis transposed_a is an AXI-lite scalar (constant during execution)
  // so HLS synthesises only the selected branch.
  if (!transposed_a) {
    HLSLIB_DATAFLOW_FUNCTION(ReadA, a, aSplit, size_n, size_k, size_m);
    HLSLIB_DATAFLOW_FUNCTION(TransposeA, aSplit, aPipes[0], size_n, size_k,
                             size_m);
  } else {
    HLSLIB_DATAFLOW_FUNCTION(ReadATransposed, a, aMemory, size_n, size_k,
                             size_m);
    HLSLIB_DATAFLOW_FUNCTION(ConvertWidthATransposed, aMemory, aPipes[0],
                             size_n, size_k, size_m);
  }

  HLSLIB_DATAFLOW_FUNCTION(ReadB, b, bMemory, size_n, size_k, size_m);

  // Only convert memory width if necessary
#ifdef MM_CONVERT_B
  HLSLIB_DATAFLOW_FUNCTION(ConvertWidthB, bMemory, bFeed, size_n, size_k,
                           size_m);
  HLSLIB_DATAFLOW_FUNCTION(FeedB, bFeed, bPipes[0], size_n, size_k, size_m);
#else
  HLSLIB_DATAFLOW_FUNCTION(FeedB, bMemory, bPipes[0], size_n, size_k, size_m);
#endif

  // Synthesis: _PEChain unrolls to kComputeTilesN direct ProcessingElement
  // calls with constant indices, inlined into the DATAFLOW region for
  // canonical form.
  // Simulation: plain loop; HLSLIB_DATAFLOW_FUNCTION spawns each PE as a
  // thread using the local dataflow context.
#ifdef HLSLIB_SYNTHESIS
  _PEChain<0, (int)kComputeTilesN>::run(aPipes, bPipes, cPipes,
                                        size_n, size_k, size_m);
#else
  for (unsigned pe = 0; pe < kComputeTilesN; ++pe) {
    HLSLIB_DATAFLOW_FUNCTION(ProcessingElement,
                             aPipes[pe], aPipes[pe + 1],
                             bPipes[pe], bPipes[pe + 1],
                             cPipes[pe], cPipes[pe + 1],
                             pe, size_n, size_k, size_m);
  }
#endif

  HLSLIB_DATAFLOW_FUNCTION(ConvertWidthC, cPipes[0], cMemory, size_n, size_k,
                           size_m);
  HLSLIB_DATAFLOW_FUNCTION(WriteC, cMemory, c, size_n, size_k, size_m);

  HLSLIB_DATAFLOW_FINALIZE();
}
