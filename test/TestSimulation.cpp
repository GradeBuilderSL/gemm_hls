/// @author    Johannes de Fine Licht (definelicht@inf.ethz.ch)
/// @copyright This software is copyrighted under the BSD 3-Clause License.

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>
#include "MatrixMultiplication.h"
#include "Utility.h"

struct TestResult {
  bool passed;
  double diff_min;
  double diff_max;
  double diff_avg;
  unsigned mismatches;
  unsigned total;
};

/// Runs one GEMM test: random A (N×K) and B (K×M), calls the kernel, verifies
/// against the reference.  Returns diff statistics and pass/fail.
/// transposed_a=true: A is passed to the kernel in K×N layout (pre-transposed).
static TestResult RunTest(const unsigned size_n, const unsigned size_k,
                          const unsigned size_m, const unsigned seed,
                          const bool transposed_a = false) {
  // Validate hardware constraints
  if (size_k % kMemoryWidthK != 0) {
    std::cerr << "  K=" << size_k << " not divisible by kMemoryWidthK="
              << kMemoryWidthK << "\n";
    return {false, 0, 0, 0, 0, 0};
  }
  if (!transposed_a && size_k % kTransposeWidth != 0) {
    std::cerr << "  K=" << size_k << " not divisible by kTransposeWidth="
              << kTransposeWidth << " (required for non-transposed path)\n";
    return {false, 0, 0, 0, 0, 0};
  }
  if (size_m % kMemoryWidthM != 0) {
    std::cerr << "  M=" << size_m << " not divisible by kMemoryWidthM="
              << kMemoryWidthM << "\n";
    return {false, 0, 0, 0, 0, 0};
  }

  // ReadA / ReadB iterate full outer tiles regardless of the actual matrix
  // size; PEs discard out-of-bounds results via inBoundsN/inBoundsM checks.
  // Pad the input buffers to the nearest outer-tile boundary so the simulation
  // never reads past the end of an allocated std::vector.
  const unsigned size_n_padded = OuterTilesN(size_n) * kOuterTileSizeN;
  const unsigned size_m_padded = OuterTilesM(size_m) * kOuterTileSizeM;

  // Row-major A (N×K) used for reference; may be transposed for kernel below.
  std::vector<Data_t> a(size_n_padded * size_k, 0);
  std::vector<Data_t> b(size_k * size_m_padded, 0);
  std::vector<Data_t> cReference(size_n * size_m, 0);

  std::default_random_engine rng(seed);
  // For fixed-point types (ap_fixed etc.) use a small range [0, 1] so that
  // accumulation over the K dimension does not overflow the limited integer
  // part.  Standard int / float / double keep the original [1, 10] range.
  constexpr bool kUseSmallRange =
      !std::is_integral<Data_t>::value &&
      !std::is_floating_point<Data_t>::value &&
      !std::is_same<Data_t, half>::value;
  typename std::conditional<std::is_integral<Data_t>::value,
                            std::uniform_int_distribution<unsigned long>,
                            std::uniform_real_distribution<double>>::type
      dist(kUseSmallRange ? 0 : 1, kUseSmallRange ? 1 : 10);

  // Fill only the valid sub-matrix; padding stays zero
  for (unsigned i = 0; i < size_n; ++i)
    for (unsigned j = 0; j < size_k; ++j)
      a[i * size_k + j] = Data_t(dist(rng));
  std::for_each(b.begin(), b.begin() + size_k * size_m,
                [&dist, &rng](Data_t &in) { in = Data_t(dist(rng)); });

  // When transposed_a is true, rearrange A to K×N_padded layout so the
  // ReadATransposed path reads A^T row-by-row (each row = one column of A).
  std::vector<Data_t> a_kernel_buf;
  if (transposed_a) {
    a_kernel_buf.assign(size_k * size_n_padded, Data_t(0));
    for (unsigned n = 0; n < size_n; ++n)
      for (unsigned k = 0; k < size_k; ++k)
        a_kernel_buf[k * size_n_padded + n] = a[n * size_k + k];
  }
  const auto &a_kernel_src = transposed_a ? a_kernel_buf : a;
  const auto aKernel = Pack<kMemoryWidthA>(a_kernel_src);
  const auto bKernel = Pack<kMemoryWidthM>(b);
  auto cKernel = Pack<kMemoryWidthM>(cReference);  // zero-initialised

  ReferenceImplementation(a.data(), b.data(), cReference.data(),
                          size_n, size_k, size_m);

#ifdef MM_DYNAMIC_SIZES
  MatrixMultiplicationKernel(aKernel.data(), bKernel.data(), cKernel.data(),
                             size_n, size_k, size_m);
#else
  MatrixMultiplicationKernel(aKernel.data(), bKernel.data(), cKernel.data());
#endif

  const auto cTest = Unpack<kMemoryWidthM>(cKernel);

  double diff_min = std::numeric_limits<double>::max();
  double diff_max = 0.0;
  double diff_sum = 0.0;
  unsigned mismatches = 0;
  const unsigned total = size_n * size_m;

  for (unsigned i = 0; i < size_n; ++i) {
    for (unsigned j = 0; j < size_m; ++j) {
      // Route comparison through double so that types without std::abs or
      // operator- (e.g. ap_fixed) are handled uniformly.
      const double testDouble =
          static_cast<double>(make_signed(cTest[i * size_m + j]));
      const double refDouble =
          static_cast<double>(make_signed(cReference[i * size_m + j]));
      const double diff = std::abs(testDouble - refDouble);

      if (diff < diff_min) diff_min = diff;
      if (diff > diff_max) diff_max = diff;
      diff_sum += diff;

      bool mismatch;
      if (std::is_floating_point<Data_t>::value) {
        // float/double: relative tolerance 0.1%
        const double absRef = std::abs(refDouble);
        mismatch = (absRef > 1e-9)
                       ? diff / absRef > 1e-3
                       : diff > 1e-3;
      } else if (!std::is_integral<Data_t>::value &&
                 !std::is_same<Data_t, half>::value) {
        // Fixed-point via float reference: ReferenceViaFloat accumulates in
        // float and quantises once at the end, while the hardware truncates
        // each partial product to Data_t precision.  The accumulated absolute
        // error is bounded by size_k * (1 Data_t LSB).
        // For ap_fixed<16,8>: 1 LSB = 2^-8 = 1/256; tolerance = (K+1)/256.
        const double abs_tol = (size_k + 1) / 256.0;
        mismatch = diff > abs_tol;
      } else {
        // Integer types: exact match
        mismatch = diff != 0.0;
      }
      if (mismatch) {
        ++mismatches;
        if (mismatches <= 3) {
          std::cerr << "  Mismatch at (" << i << ", " << j << "): "
                    << testDouble << " vs. " << refDouble << "\n";
        }
      }
    }
  }

  if (total == 0) diff_min = 0.0;
  const double diff_avg = total > 0 ? diff_sum / total : 0.0;
  return {mismatches == 0, diff_min, diff_max, diff_avg, mismatches, total};
}

void printTypesInfo() {
  std::cout << "Input data type: " << typeid(Data_t).name() << std::endl;
  std::cout << "Output data type: " << typeid(SatData_t).name() << std::endl;
  std::cout << "Accumulator data type: " << typeid(AccData_t).name() << std::endl;
  std::cout << std::endl;
}

// ---------------------------------------------------------------------------
// Comprehensive test suite (MM_DYNAMIC_SIZES required for varying dimensions)
// ---------------------------------------------------------------------------
#ifdef MM_DYNAMIC_SIZES

static bool RunAllTests(bool transposed_a = false) {
  struct TC { unsigned n, k, m; const char *desc; };

  // K must be a multiple of kMemoryWidthK (and kTransposeWidth when A is not
  // pre-transposed; they are equal for all standard float/half/int8 configs).
  // M must be a multiple of kMemoryWidthM.
  // N is unconstrained; the kernel pads internally and discards extras.
  //
  // Naming convention for tile counts:
  //   kOuterTileSizeN/M = outer (memory) tile size in N / M
  //   kComputeTilesN    = number of PEs  = kInnerTileSizeN / kComputeTileSizeN
  //   kMemoryWidthK/M   = elements per memory bus word in K / M

  const TC tests[] = {
    // ── N-dimension boundaries ──────────────────────────────────────────────
    // Hold K and M at minimum valid values; vary N across tile boundaries.
    { 1,
      kMemoryWidthK, kMemoryWidthM,
      "N=1 (single row, well inside first outer tile)" },

    { kComputeTilesN,
      kMemoryWidthK, kMemoryWidthM,
      "N=kComputeTilesN (exact inner N tile, all PEs active)" },

    { kOuterTileSizeN,
      kMemoryWidthK, kMemoryWidthM,
      "N=kOuterTileSizeN (exact single outer N tile, no partial)" },

    { kOuterTileSizeN + 1,
      kMemoryWidthK, kMemoryWidthM,
      "N=kOuterTileSizeN+1 (first row of second outer N tile)" },

    { 2 * kOuterTileSizeN,
      kMemoryWidthK, kMemoryWidthM,
      "N=2*kOuterTileSizeN (two exact outer N tiles, no partial)" },

    { 2 * kOuterTileSizeN + 1,
      kMemoryWidthK, kMemoryWidthM,
      "N=2*kOuterTileSizeN+1 (partial third outer N tile)" },

    // ── M-dimension boundaries ──────────────────────────────────────────────
    // Hold N=kComputeTilesN and K=kMemoryWidthK; vary M across tile boundaries.
    { kComputeTilesN, kMemoryWidthK,
      kOuterTileSizeM,
      "M=kOuterTileSizeM (exact single outer M tile, no partial)" },

    { kComputeTilesN, kMemoryWidthK,
      kOuterTileSizeM + kMemoryWidthM,
      "M=kOuterTileSizeM+kMemoryWidthM (partial second outer M tile)" },

    { kComputeTilesN, kMemoryWidthK,
      2 * kOuterTileSizeM,
      "M=2*kOuterTileSizeM (two exact outer M tiles, no partial)" },

    { kComputeTilesN, kMemoryWidthK,
      2 * kOuterTileSizeM + kMemoryWidthM,
      "M=2*kOuterTileSizeM+kMemoryWidthM (partial third outer M tile)" },

    // ── K-dimension variation ───────────────────────────────────────────────
    // Minimum N and M; vary K to test the reduction loop and double-buffering.
    { kComputeTilesN,
      2 * kMemoryWidthK, kMemoryWidthM,
      "K=2*kMemoryWidthK (two K chunks, exercises K loop)" },

    // ── Combined: all three dimensions simultaneously partial ───────────────
    { 2 * kOuterTileSizeN + 1,
      2 * kMemoryWidthK,
      2 * kOuterTileSizeM + kMemoryWidthM,
      "all partial, small K (fast combined boundary check)" },

    // Standard stress test: sizes chosen so every dimension requires handling
    // a partial last tile.  Matches the CMake-computed test sizes for default
    // config (513 x 528 x 528 with MM_PARALLELISM_N=32, MM_PARALLELISM_M=8,
    // MM_MEMORY_TILE_SIZE_N/M=256, MM_MEMORY_BUS_WIDTH_K=64, Data_t=float).
    { 2 * kOuterTileSizeN + 1,
      2 * kComputeTilesN * kComputeTileSizeM + kMemoryWidthK,
      2 * kOuterTileSizeM + kMemoryWidthM,
      "stress test: all dimensions partial, large K" },
  };

  const unsigned kNumTests = sizeof(tests) / sizeof(tests[0]);
  bool all_passed = true;

  for (unsigned i = 0; i < kNumTests; ++i) {
    const TC &tc = tests[i];
    std::cout << "[" << (i + 1) << "/" << kNumTests << "] "
              << tc.desc << "\n    ("
              << tc.n << " x " << tc.k << " x " << tc.m
              << (transposed_a ? ", transposed_a" : "") << ")... " << std::flush;
    const auto r = RunTest(tc.n, tc.k, tc.m, kSeed + i, transposed_a);
    std::cout << (r.passed ? "PASS" : "FAIL")
              << std::scientific << std::setprecision(2)
              << "  diff min=" << r.diff_min
              << " max=" << r.diff_max
              << " avg=" << r.diff_avg
              << std::defaultfloat
              << "  mismatches=" << r.mismatches << "/" << r.total << "\n";
    all_passed &= r.passed;
  }

  std::cout << "\n"
            << (all_passed ? "All " : "FAILED: ")
            << kNumTests << " tests"
            << (all_passed ? " passed.\n" : " (see mismatches above).\n");
  return all_passed;
}

int main(int argc, char **argv) {
  printTypesInfo();

  // Parse optional trailing "transposed" flag present in any invocation mode.
  bool transposed_a = false;
  if (argc > 1 && std::string(argv[argc - 1]) == "transposed") {
    transposed_a = true;
    --argc;  // hide from further argument parsing
  }

  if (argc == 4) {
    // Single-run mode: ./TestSimulation N K M [transposed]
    const unsigned size_n = std::stoul(argv[1]);
    const unsigned size_k = std::stoul(argv[2]);
    const unsigned size_m = std::stoul(argv[3]);
    std::cout << "Running single test: "
              << size_n << " x " << size_k << " x " << size_m
              << (transposed_a ? " (transposed A)" : "") << "\n" << std::flush;
    const auto r = RunTest(size_n, size_k, size_m, kSeed, transposed_a);
    std::cout << (r.passed ? "PASS" : "FAIL")
              << std::scientific << std::setprecision(2)
              << "  diff min=" << r.diff_min
              << " max=" << r.diff_max
              << " avg=" << r.diff_avg
              << std::defaultfloat
              << "  mismatches=" << r.mismatches << "/" << r.total << "\n";
    return r.passed ? 0 : 1;
  }
  if (argc != 1) {
    std::cerr << "Usage: " << argv[0] << " [N K M] [transposed]\n"
              << "  No dimensions: run the full test suite.\n"
              << "  N K M: run a single test with the given dimensions.\n"
              << "  transposed: use the pre-transposed A read path.\n";
    return 1;
  }
  return RunAllTests(transposed_a) ? 0 : 1;
}

#else  // !MM_DYNAMIC_SIZES — static dimensions baked in at compile time

int main(int argc, char **argv) {
  printTypesInfo();
  bool transposed_a = (argc > 1 && std::string(argv[1]) == "transposed");
  std::cout << "Running test: "
            << kSizeN << " x " << kSizeK << " x " << kSizeM
            << (transposed_a ? " (transposed A)" : "") << "\n" << std::flush;
  const auto r = RunTest(kSizeN, kSizeK, kSizeM, kSeed, transposed_a);
  std::cout << (r.passed ? "PASS" : "FAIL")
            << std::scientific << std::setprecision(2)
            << "  diff min=" << r.diff_min
            << " max=" << r.diff_max
            << " avg=" << r.diff_avg
            << std::defaultfloat
            << "  mismatches=" << r.mismatches << "/" << r.total << "\n";
  return r.passed ? 0 : 1;
}

#endif  // MM_DYNAMIC_SIZES
