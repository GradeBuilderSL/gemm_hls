#include "MatrixMultiplication.h"
#include "Compute.h"
#include "Memory.h"
#include "hlslib/xilinx/Simulation.h"

// ── PE chain ─────────────────────────────────────────────────────────────────
// Synthesis: template recursion expands the PE loop into direct function calls
// so the DATAFLOW region contains only calls (canonical form, no for-loop body).
// Each instantiation is inlined into the DATAFLOW caller.
//
// Simulation: plain for-loop with HLSLIB_DATAFLOW_FUNCTION (the dataflow
// context is a local variable of the DATAFLOW function — out of scope inside
// a separately compiled struct, so _PEChain is synthesis-only).
// The synthesis/simulation split also sidesteps the Stream<T> vs Stream<T,D>
// array type mismatch between the two build modes.
#ifdef HLSLIB_SYNTHESIS
template <int pe, int kPEs>
struct _PEChain {
  static void run(Stream<ComputePackN_t> aPipes[],
                  Stream<ComputePackM_t> bPipes[],
                  Stream<ComputePackM_t> cPipes[],
                  const unsigned size_n,
                  const unsigned size_k,
                  const unsigned size_m) {
    #pragma HLS INLINE
    ProcessingElement(aPipes[pe], aPipes[pe + 1],
                      bPipes[pe], bPipes[pe + 1],
                      cPipes[pe], cPipes[pe + 1],
                      pe, size_n, size_k, size_m);
    _PEChain<pe + 1, kPEs>::run(aPipes, bPipes, cPipes,
                                size_n, size_k, size_m);
  }
};
template <int kPEs>
struct _PEChain<kPEs, kPEs> {
  static void run(Stream<ComputePackN_t>[],
                  Stream<ComputePackM_t>[],
                  Stream<ComputePackM_t>[],
                  const unsigned, const unsigned, const unsigned) {
    #pragma HLS INLINE
  }
};
#endif  // HLSLIB_SYNTHESIS

// ── Non-transposed A dataflow ─────────────────────────────────────────────────
// All streams are local variables (canonical DATAFLOW: only declarations +
// function calls, no if/else, no loops).
// Stream ARRAYS use Stream<T> (depth=0, no internal STREAM pragma) in synthesis
// so Vitis HLS 2022+ does not auto-DISAGGREGATE them before loop unrolling
// (avoids HLS 214-177).  The FIFO depth is set by the external STREAM pragma.
// Simulation uses Stream<T,depth> so hlslib blocking Push/Pop has correct depth.
static void DataflowNormal(MemoryPackK_t const a[],
                           MemoryPackM_t const b[],
                           MemoryPackM_t       c[],
                           const unsigned size_n,
                           const unsigned size_k,
                           const unsigned size_m) {
  #pragma HLS DATAFLOW
  HLSLIB_DATAFLOW_INIT();

#ifdef HLSLIB_SYNTHESIS
  Stream<Data_t>         aSplit[kTransposeWidth];
  Stream<ComputePackN_t> aPipes[kComputeTilesN + 1];
  Stream<ComputePackM_t> bPipes[kComputeTilesN + 1];
  Stream<ComputePackM_t> cPipes[kComputeTilesN + 1];
#else
  Stream<Data_t,         2 * kOuterTileSizeN> aSplit[kTransposeWidth];
  Stream<ComputePackN_t, kPipeDepth>          aPipes[kComputeTilesN + 1];
  Stream<ComputePackM_t, kPipeDepth>          bPipes[kComputeTilesN + 1];
  Stream<ComputePackM_t, kPipeDepth>          cPipes[kComputeTilesN + 1];
#endif
  #pragma HLS STREAM variable=aSplit depth=2*kOuterTileSizeN
  #pragma HLS STREAM variable=aPipes depth=kPipeDepth
  #pragma HLS STREAM variable=bPipes depth=kPipeDepth
  #pragma HLS STREAM variable=cPipes depth=kPipeDepth

  // Scalar streams: single instance — DISAGGREGATE trivially succeeds,
  // depth template argument is safe.
  Stream<MemoryPackM_t, 2 * kOuterTileSizeMMemory> bMemory;
  Stream<MemoryPackM_t, 2 * kOuterTileSizeMMemory> cMemory;
#ifdef MM_CONVERT_B
  Stream<ComputePackM_t, kPipeDepth> bFeed;
#endif

  HLSLIB_DATAFLOW_FUNCTION(ReadA, a, aSplit, size_n, size_k, size_m);
  HLSLIB_DATAFLOW_FUNCTION(TransposeA, aSplit, aPipes[0], size_n, size_k, size_m);

  HLSLIB_DATAFLOW_FUNCTION(ReadB, b, bMemory, size_n, size_k, size_m);
#ifdef MM_CONVERT_B
  HLSLIB_DATAFLOW_FUNCTION(ConvertWidthB, bMemory, bFeed, size_n, size_k, size_m);
  HLSLIB_DATAFLOW_FUNCTION(FeedB, bFeed, bPipes[0], size_n, size_k, size_m);
#else
  HLSLIB_DATAFLOW_FUNCTION(FeedB, bMemory, bPipes[0], size_n, size_k, size_m);
#endif

#ifdef HLSLIB_SYNTHESIS
  _PEChain<0, kComputeTilesN>::run(aPipes, bPipes, cPipes,
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

  // PE 0 collects all C results (own + forwarded from PEs 1..N-1).
  HLSLIB_DATAFLOW_FUNCTION(ConvertWidthC, cPipes[0], cMemory,
                            size_n, size_k, size_m);
  HLSLIB_DATAFLOW_FUNCTION(WriteC, cMemory, c, size_n, size_k, size_m);

  HLSLIB_DATAFLOW_FINALIZE();
}

// ── Pre-transposed A dataflow ─────────────────────────────────────────────────
// A is stored K×N in DDR; ReadATransposed + ConvertWidthATransposed replace
// ReadA + TransposeA.  B/PE/C path is identical to DataflowNormal.
static void DataflowTransposed(MemoryPackK_t const a[],
                               MemoryPackM_t const b[],
                               MemoryPackM_t       c[],
                               const unsigned size_n,
                               const unsigned size_k,
                               const unsigned size_m) {
  #pragma HLS DATAFLOW
  HLSLIB_DATAFLOW_INIT();

#ifdef HLSLIB_SYNTHESIS
  Stream<ComputePackN_t> aPipes[kComputeTilesN + 1];
  Stream<ComputePackM_t> bPipes[kComputeTilesN + 1];
  Stream<ComputePackM_t> cPipes[kComputeTilesN + 1];
#else
  Stream<ComputePackN_t, kPipeDepth> aPipes[kComputeTilesN + 1];
  Stream<ComputePackM_t, kPipeDepth> bPipes[kComputeTilesN + 1];
  Stream<ComputePackM_t, kPipeDepth> cPipes[kComputeTilesN + 1];
#endif
  #pragma HLS STREAM variable=aPipes depth=kPipeDepth
  #pragma HLS STREAM variable=bPipes depth=kPipeDepth
  #pragma HLS STREAM variable=cPipes depth=kPipeDepth

  Stream<MemoryPackK_t, 2 * kOuterTileSizeNMemory> aMemory;
  Stream<MemoryPackM_t, 2 * kOuterTileSizeMMemory> bMemory;
  Stream<MemoryPackM_t, 2 * kOuterTileSizeMMemory> cMemory;
#ifdef MM_CONVERT_B
  Stream<ComputePackM_t, kPipeDepth> bFeed;
#endif

  HLSLIB_DATAFLOW_FUNCTION(ReadATransposed, a, aMemory, size_n, size_k, size_m);
  HLSLIB_DATAFLOW_FUNCTION(ConvertWidthATransposed, aMemory, aPipes[0],
                            size_n, size_k, size_m);

  HLSLIB_DATAFLOW_FUNCTION(ReadB, b, bMemory, size_n, size_k, size_m);
#ifdef MM_CONVERT_B
  HLSLIB_DATAFLOW_FUNCTION(ConvertWidthB, bMemory, bFeed, size_n, size_k, size_m);
  HLSLIB_DATAFLOW_FUNCTION(FeedB, bFeed, bPipes[0], size_n, size_k, size_m);
#else
  HLSLIB_DATAFLOW_FUNCTION(FeedB, bMemory, bPipes[0], size_n, size_k, size_m);
#endif

#ifdef HLSLIB_SYNTHESIS
  _PEChain<0, kComputeTilesN>::run(aPipes, bPipes, cPipes,
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

  HLSLIB_DATAFLOW_FUNCTION(ConvertWidthC, cPipes[0], cMemory,
                            size_n, size_k, size_m);
  HLSLIB_DATAFLOW_FUNCTION(WriteC, cMemory, c, size_n, size_k, size_m);

  HLSLIB_DATAFLOW_FINALIZE();
}

// ── Kernel entry point ────────────────────────────────────────────────────────
// Not a DATAFLOW function: dispatches to the appropriate pipeline based on
// transposed_a, which is an AXI-Lite scalar constant during kernel execution.
void MatrixMultiplicationKernel(MemoryPackK_t const a[],
                                MemoryPackM_t const b[],
                                MemoryPackM_t       c[]
#ifdef MM_DYNAMIC_SIZES
                                ,
                                const unsigned size_n,
                                const unsigned size_k,
                                const unsigned size_m
#endif
                                ,
                                const bool transposed_a) {

  #pragma HLS INTERFACE m_axi port=a offset=slave bundle=gmem0
  #pragma HLS INTERFACE m_axi port=b offset=slave bundle=gmem1
  #pragma HLS INTERFACE m_axi port=c offset=slave bundle=gmem2
  #pragma HLS INTERFACE s_axilite port=transposed_a

#ifndef MM_DYNAMIC_SIZES
  const unsigned size_n = kSizeN;
  const unsigned size_k = kSizeK;
  const unsigned size_m = kSizeM;
#endif

  if (!transposed_a)
    DataflowNormal(a, b, c, size_n, size_k, size_m);
  else
    DataflowTransposed(a, b, c, size_n, size_k, size_m);
}
