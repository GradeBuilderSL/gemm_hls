# Author:  Johannes de Fine Licht (definelicht@inf.ethz.ch)
# Copyright: This software is copyrighted under the BSD 3-Clause License.

cmake_minimum_required(VERSION 3.19)  # 3.19+ required for string(JSON ...)
project(sdaccel_mm)

# Target options 
set(MM_PLATFORM "xilinx_u250_gen3x16_xdma_3_1_202020_1" CACHE STRING "Platform string for Vitis.")
set(MM_TARGET_CLOCK "" CACHE STRING "Target clock for kernel (uses default is left empty).")
set(MM_ENABLE_PROFILING OFF CACHE BOOL "Collect profiling information.")
set(MM_ENABLE_DEBUGGING OFF CACHE BOOL "Inject debugging cores to design.")
set(MM_VITIS_FLAGS "" CACHE STRING "Extra flags for Vitis.")
set(MM_POWER_METER OFF CACHE BOOL "Enable Corsair power meter for measuring power consumption with Corsair RMi power supplies.")
set(MM_ENABLE_BLAS ON CACHE BOOL "Use BLAS library for validation if available.")

# Domain options
set(MM_DATA_TYPE "float" CACHE STRING "Matrix data type.")
set(MM_MEMORY_BUS_WIDTH_N 64 CACHE STRING "Width of memory bus in bytes in N.")
set(MM_MEMORY_BUS_WIDTH_K 64 CACHE STRING "Width of memory bus in bytes in K.")
set(MM_MEMORY_BUS_WIDTH_M 64 CACHE STRING "Width of memory bus in bytes in M.")
set(MM_DYNAMIC_SIZES ON CACHE BOOL "Use dynamic matrix dimension sizes.")
set(MM_SIZE_N 512 CACHE STRING "Size of matrix dimension.")
set(MM_SIZE_K 512 CACHE STRING "Size of matrix dimension.")
set(MM_SIZE_M 512 CACHE STRING "Size of matrix dimension.")
set(MM_MEMORY_TILE_SIZE_N 256 CACHE STRING "Tile size of outer memory tile in N.")
set(MM_MEMORY_TILE_SIZE_M 256 CACHE STRING "Tile size of outer memory tile in M.")
set(MM_PARALLELISM_N 32 CACHE STRING "Number of parallel compute in N.")
set(MM_PARALLELISM_M 8 CACHE STRING "Number of parallel compute in M.")
# set(MM_GRANULARITY_N 1 CACHE STRING "Granularity of processing elements in N.")
set(MM_TRANSPOSE_WIDTH 64 CACHE STRING "Burst width when transposing reads from A.")
set(MM_TWO_DIMMS OFF CACHE BOOL "Use two DDR DIMMs instead of one")
set(MM_MAP_OP "Multiply" CACHE STRING "Map operation to perform between A and B matrices.")
set(MM_REDUCE_OP "Add" CACHE STRING "Reduction operation to write back to C.")
set(MM_ADD_RESOURCE OFF CACHE STRING "")
set(MM_MULT_RESOURCE OFF CACHE STRING "")
set(MM_ACCUMULATOR_EXTRA_INT_BITS 16 CACHE STRING
    "Extra integer bits added to the internal accumulator for ap_fixed types. \
AccData_t = ap_fixed<W+extra, I+extra> where Data_t = ap_fixed<W,I>. \
Prevents accumulation overflow for large K; no effect for float/double/integer types.")

# Internal
set(MM_KERNEL_NAME MatrixMultiplicationKernel)
include(CheckTypeSize)
check_type_size(${MM_DATA_TYPE} MM_DATA_WIDTH_${MM_DATA_TYPE})
if(NOT MM_DATA_WIDTH_${MM_DATA_TYPE}) # Non-primitive data type
  if(MM_DATA_TYPE STREQUAL "half")
    set(MM_DATA_WIDTH_${MM_DATA_TYPE} 2)
  elseif(MM_DATA_TYPE STREQUAL "uint8_t")
    set(MM_DATA_WIDTH_${MM_DATA_TYPE} 1)
  else()
    message(FATAL_ERROR "Could not get size of data type ${MM_DATA_TYPE}.") 
  endif()
endif()
math(EXPR MM_KERNEL_WIDTH_N "${MM_DATA_WIDTH_${MM_DATA_TYPE}} * ${MM_PARALLELISM_N}")
math(EXPR MM_KERNEL_WIDTH_M "${MM_DATA_WIDTH_${MM_DATA_TYPE}} * ${MM_PARALLELISM_M}")

# Flat byte-width variable (avoids nested ${MM_DATA_WIDTH_${MM_DATA_TYPE}} expansion
# which breaks for template type names like ap_fixed<16,8>).
set(MM_DATA_WIDTH_BYTES "${MM_DATA_WIDTH_${MM_DATA_TYPE}}")
# Include directive injected into Config.h for types that need a special header.
if(MM_DATA_TYPE STREQUAL "half")
  set(MM_DATA_TYPE_INCLUDE "#include \"hls_half.h\"")
elseif(MM_DATA_TYPE MATCHES "^ap_(u?)fixed")
  set(MM_DATA_TYPE_INCLUDE "#include \"ap_fixed.h\"")
else()
  set(MM_DATA_TYPE_INCLUDE "")
endif()

# Accumulator type: for ap_fixed<W,I>, widen integer part by MM_ACCUMULATOR_EXTRA_INT_BITS
# to avoid overflow when accumulating K products.  Fraction bits stay the same.
# SatData_t = ap_fixed<W,I,AP_TRN,AP_SAT> — used when writing from AccData_t to output.
# For all other types AccData_t = SatData_t = Data_t (no-op widening).
if(MM_DATA_TYPE MATCHES "^ap_(u?)fixed<([0-9]+),([0-9]+)")
  set(_fx_u "${CMAKE_MATCH_1}")
  set(_fx_w "${CMAKE_MATCH_2}")
  set(_fx_i "${CMAKE_MATCH_3}")
  math(EXPR _acc_w "${_fx_w} + ${MM_ACCUMULATOR_EXTRA_INT_BITS}")
  math(EXPR _acc_i "${_fx_i} + ${MM_ACCUMULATOR_EXTRA_INT_BITS}")
  set(MM_ACC_DATA_TYPE "ap_${_fx_u}fixed<${_acc_w},${_acc_i}>")
  set(MM_SAT_DATA_TYPE "ap_${_fx_u}fixed<${_fx_w},${_fx_i},AP_TRN,AP_SAT>")
else()
  set(MM_ACC_DATA_TYPE "${MM_DATA_TYPE}")
  set(MM_SAT_DATA_TYPE "${MM_DATA_TYPE}")
endif()

# Validation checks
math(EXPR MM_INNER_TILES "(${MM_MEMORY_TILE_SIZE_N} / ${MM_PARALLELISM_N}) * (${MM_MEMORY_TILE_SIZE_M} / ${MM_PARALLELISM_M})")
if(MM_MEMORY_TILE_SIZE_N GREATER MM_INNER_TILES)
  message(WARNING "In-memory transposition for A cannot keep up with the instantiated number of compute units. The number of inner tiles (currently ${MM_INNER_TILES}) must be greater than or equal to the outer tile size in N (currently ${MM_MEMORY_TILE_SIZE_N}).")
endif()
math(EXPR MM_MEMORY_WIDTH_M "${MM_MEMORY_BUS_WIDTH_M} / ${MM_DATA_WIDTH_${MM_DATA_TYPE}}")
math(EXPR MM_REM "${MM_MEMORY_WIDTH_M} % ${MM_PARALLELISM_M}")
if(MM_REM GREATER 0)
  message(FATAL_ERROR "Bus width in M (${MM_MEMORY_WIDTH_M}) must be a multiple of the parallelism in M (${MM_PARALLELISM_M}).")
endif()
math(EXPR MM_REM "${MM_MEMORY_TILE_SIZE_M} % ${MM_MEMORY_WIDTH_M}")
if(MM_REM GREATER 0)
  message(FATAL_ERROR "Outer memory tile size must be divisible by element width of memory bus.")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake
    ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/hlslib/cmake)

find_package(Vitis REQUIRED)
find_package(Threads REQUIRED)

# Check if BLAS is available
if(MM_ENABLE_BLAS)
  find_package(BLAS)
  if(BLAS_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMM_HAS_BLAS")
  else()
    set(BLAS_LIBRARIES)
  endif()
else()
  set(BLAS_LIBRARIES)
endif()

# Include power meter
if(MM_POWER_METER)
  add_subdirectory(powermeter)
  include_directories(SYSTEM powermeter/include powermeter/OpenCorsairLink/include ${LIBUSB_INCLUDE_DIR})
  add_definitions("-DMM_POWER_METER")
endif()

include_directories(include ${CMAKE_BINARY_DIR} SYSTEM hlslib/include ${Vitis_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
if(MM_DYNAMIC_SIZES)
  add_definitions("-DMM_DYNAMIC_SIZES")
  set(MM_SYNTHESIS_FLAGS "${MM_SYNTHESIS_FLAGS} -DMM_DYNAMIC_SIZES")
endif()
add_definitions("-DHLSLIB_STREAM_TIMEOUT=16")
if(((${Vitis_MAJOR_VERSION} LESS 2018) AND (${Vitis_MINOR_VERSION} LESS 3)) OR ${Vitis_MAJOR_VERSION} LESS 2017)
  add_definitions(-DHLSLIB_LEGACY_SDX=1)
else()
  add_definitions(-DHLSLIB_LEGACY_SDX=0)
endif()
if(MM_DATA_TYPE STREQUAL "half")
  add_definitions("-DMM_HALF_PRECISION")
endif()

# Query default clock of platform
if(NOT "${MM_TARGET_CLOCK}")
  if(NOT "${MM_PLATFORM_INTERNAL}" STREQUAL "${MM_PLATFORM}")
    message(STATUS "Querying default clock for ${MM_PLATFORM}.")
    execute_process(COMMAND ${Vitis_PLATFORMINFO} --platform ${MM_PLATFORM} -jhardwarePlatform.systemClocks
                    OUTPUT_VARIABLE SYSTEM_CLOCKS
                    RESULT_VARIABLE RET)
    if(RET EQUAL 0)
      string(FIND "${SYSTEM_CLOCKS}" "\"default\": \"true\"" LOC)
      string(SUBSTRING "${SYSTEM_CLOCKS}" ${LOC} -1 SYSTEM_CLOCKS)
      string(REGEX MATCH "\"frequency\": \"[0-9\\.]+\"" SYSTEM_CLOCKS "${SYSTEM_CLOCKS}")
      string(REGEX REPLACE "\"frequency\": \"([0-9\\.]+)\"" "\\1" CLOCK "${SYSTEM_CLOCKS}")
      set(MM_CLOCK_INTERNAL "${CLOCK}" CACHE INTERNAL "")
      set(MM_PLATFORM_INTERNAL "${MM_PLATFORM}" CACHE INTERNAL "")
    else()
      message(WARNING "Failed to query default frequency for platform ${MM_PLATFORM}. Assuming 300 MHz.")
      set(MM_CLOCK_INTERNAL "300" CACHE INTERNAL "")
    endif()
  endif()
endif()

# Hardware configuration header for HLS
configure_file(include/Config.h.in Config.h)

# C++ source code 
set(MM_KERNEL_SRC
    ${CMAKE_SOURCE_DIR}/kernel/Compute.cpp
    ${CMAKE_SOURCE_DIR}/kernel/Memory.cpp
    ${CMAKE_SOURCE_DIR}/kernel/Top.cpp)
add_library(mmkernel ${MM_KERNEL_SRC})
target_link_libraries(mmkernel ${CMAKE_THREAD_LIBS_INIT})

# Executables 
add_executable(PrintSpecifications src/PrintSpecifications.cpp)

# Software test
add_executable(TestSimulation test/TestSimulation.cpp)
target_link_libraries(TestSimulation ${Vitis_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${BLAS_LIBRARIES} ${Vitis_LIBRARIES} mmkernel)
if(MM_DATA_TYPE STREQUAL "half")
  target_link_libraries(TestSimulation ${Vitis_FLOATING_POINT_LIBRARY})
endif()
enable_testing()

# Synthesis flags
set(MM_SYNTHESIS_FLAGS "${MM_SYNTHESIS_FLAGS} -O3 -DMM_SYNTHESIS")
if(MM_DATA_TYPE STREQUAL "half")
  set(MM_SYNTHESIS_FLAGS "${MM_SYNTHESIS_FLAGS} -DMM_HALF_PRECISION")
endif()

if(MM_ADD_RESOURCE)
  set(MM_SYNTHESIS_FLAGS "${MM_SYNTHESIS_FLAGS} -DMM_ADD_RESOURCE=${MM_ADD_RESOURCE}") 
endif() 
if(MM_MULT_RESOURCE) 
  set(MM_SYNTHESIS_FLAGS "${MM_SYNTHESIS_FLAGS} -DMM_MULT_RESOURCE=${MM_MULT_RESOURCE}") 
endif() 

if(MM_TWO_DIMMS)
  set(MM_PORT_MAPPING m_axi_gmem0:DDR[0] m_axi_gmem1:DDR[1] m_axi_gmem2:DDR[1])
else()
  set(MM_PORT_MAPPING m_axi_gmem0:DDR[1] m_axi_gmem1:DDR[1] m_axi_gmem2:DDR[1])
endif()

# Hardware kernel
add_vitis_kernel(MatrixMultiplication
                 KERNEL "MatrixMultiplicationKernel"
                 FILES ${MM_KERNEL_SRC}
                 HLS_FLAGS ${MM_SYNTHESIS_FLAGS}
                 INCLUDE_DIRS include
                              hlslib/include
                              ${CMAKE_BINARY_DIR}
                 PORT_MAPPING ${MM_PORT_MAPPING} 
                 DEPENDS include/Compute.h
                         include/MatrixMultiplication.h
                         include/Memory.h
                         ${CMAKE_BINARY_DIR}/Config.h)
add_vitis_program(MatrixMultiplication ${MM_PLATFORM}
                  CLOCK ${MM_TARGET_CLOCK}
                  BUILD_FLAGS ${MM_VITIS_FLAGS}
                  PROFILING ${MM_ENABLE_PROFILING}
                  DEBUGGING ${MM_ENABLE_DEBUGGING})

# ---------------------------------------------------------------------------
# Per-platform HLS synthesis targets driven by platforms/*.json
#
# Each JSON file in platforms/ defines one synthesis target.
# Fields (all required unless noted):
#   part               — Xilinx FPGA part string
#   parallelism_n      — number of PEs (systolic array length)
#   memory_tile_size_n — outer tile size in N (must satisfy constraints below)
#   board              — (optional) board identifier for set_part -board
#   clock              — (optional) target MHz, default 300
#   parallelism_m      — (optional) vector width per PE, default MM_PARALLELISM_M
#   memory_tile_size_m — (optional) outer tile size in M, default MM_MEMORY_TILE_SIZE_M
#   memory_bus_width_n/k/m — (optional) bus widths in bytes, default MM_MEMORY_BUS_WIDTH_*
#
# Constraints checked at configure time (WARNING skips that platform):
#   (1) memory_tile_size_n % parallelism_n == 0
#   (2) memory_tile_size_n % (memory_bus_width_n / sizeof(Data_t)) == 0
#   (3) memory_tile_size_m % (memory_bus_width_m / sizeof(Data_t)) == 0
#   (4) memory_bus_width_m % (parallelism_m * sizeof(Data_t)) == 0
#   (5) kInnerTilesM >= kInnerTilesN  (double-buffer in Compute.cpp)
#       i.e. tile_m/parallelism_m >= tile_n/parallelism_n
#
# Usage: make synthesize_<name>   (e.g. make synthesize_kv260)
# ---------------------------------------------------------------------------
get_target_property(_MM_BASE_HLS_FLAGS MatrixMultiplication HLS_FLAGS)
string(REPLACE ";" " " _MM_KERNEL_SRC_STR "${MM_KERNEL_SRC}")

# Determine the part string for csim set_part before the platform loop so
# that per-platform csim_<name> targets can reference MM_CSIM_PART.
# Prefer the part queried from the installed MM_PLATFORM; fall back to the
# first platform JSON so the targets are available without a full install.
if(PROGRAM_PLATFORM_PART)
  set(MM_CSIM_PART "${PROGRAM_PLATFORM_PART}")
else()
  set(MM_CSIM_PART "")
  file(GLOB _csim_platform_files "${CMAKE_SOURCE_DIR}/platforms/*.json")
  foreach(_csim_pf IN LISTS _csim_platform_files)
    if(NOT MM_CSIM_PART)
      file(READ "${_csim_pf}" _csim_pjson)
      string(JSON _csim_part_try ERROR_VARIABLE _e GET "${_csim_pjson}" part)
      if(NOT _e AND _csim_part_try)
        set(MM_CSIM_PART "${_csim_part_try}")
      endif()
    endif()
  endforeach()
  if(MM_CSIM_PART)
    message(STATUS "csim: PROGRAM_PLATFORM_PART not set; using fallback part '${MM_CSIM_PART}' from platforms/")
  else()
    message(WARNING "csim: no part found in platforms/*.json — csim target unavailable.")
  endif()
endif()

file(GLOB _MM_PLATFORM_FILES "${CMAKE_SOURCE_DIR}/platforms/*.json")
foreach(_MM_PLATFORM_FILE IN LISTS _MM_PLATFORM_FILES)
  get_filename_component(_pname "${_MM_PLATFORM_FILE}" NAME_WE)
  file(READ "${_MM_PLATFORM_FILE}" _pjson)

  # --- Required fields ---
  string(JSON _p_part   ERROR_VARIABLE _e1 GET "${_pjson}" part)
  string(JSON _p_para_n ERROR_VARIABLE _e2 GET "${_pjson}" parallelism_n)
  string(JSON _p_tile_n ERROR_VARIABLE _e3 GET "${_pjson}" memory_tile_size_n)
  if(_e1 OR _e2 OR _e3)
    message(WARNING
      "Platform '${_pname}': missing required field "
      "(part / parallelism_n / memory_tile_size_n) — skipping.")
    continue()
  endif()

  # --- Optional fields with defaults ---
  string(JSON _p_board  ERROR_VARIABLE _e GET "${_pjson}" board)
  if(_e)
    set(_p_board "")
  endif()

  string(JSON _p_clock  ERROR_VARIABLE _e GET "${_pjson}" clock)
  if(_e)
    set(_p_clock "300")
  endif()

  string(JSON _p_para_m ERROR_VARIABLE _e GET "${_pjson}" parallelism_m)
  if(_e)
    set(_p_para_m "${MM_PARALLELISM_M}")
  endif()

  string(JSON _p_tile_m ERROR_VARIABLE _e GET "${_pjson}" memory_tile_size_m)
  if(_e)
    set(_p_tile_m "${MM_MEMORY_TILE_SIZE_M}")
  endif()

  string(JSON _p_bus_n  ERROR_VARIABLE _e GET "${_pjson}" memory_bus_width_n)
  if(_e)
    set(_p_bus_n "${MM_MEMORY_BUS_WIDTH_N}")
  endif()

  string(JSON _p_bus_k  ERROR_VARIABLE _e GET "${_pjson}" memory_bus_width_k)
  if(_e)
    set(_p_bus_k "${MM_MEMORY_BUS_WIDTH_K}")
  endif()

  string(JSON _p_bus_m  ERROR_VARIABLE _e GET "${_pjson}" memory_bus_width_m)
  if(_e)
    set(_p_bus_m "${MM_MEMORY_BUS_WIDTH_M}")
  endif()

  # --- Data type (optional; defaults to global MM_DATA_TYPE) ---
  string(JSON _p_data_type ERROR_VARIABLE _e GET "${_pjson}" data_type)
  if(_e)
    # Not in JSON — inherit global type
    set(_p_data_type         "${MM_DATA_TYPE}")
    set(_p_data_width_bytes  "${MM_DATA_WIDTH_BYTES}")
    set(_p_data_type_include "${MM_DATA_TYPE_INCLUDE}")
    set(_p_type_extra_flags  "")
    set(_p_acc_data_type     "${MM_ACC_DATA_TYPE}")
    set(_p_sat_data_type     "${MM_SAT_DATA_TYPE}")
  else()
    # Default: AccData_t = SatData_t = Data_t (overridden for ap_fixed below).
    set(_p_acc_data_type "${_p_data_type}")
    set(_p_sat_data_type "${_p_data_type}")
    # Determine byte width and required header from the type string.
    if(_p_data_type STREQUAL "float")
      set(_p_data_width_bytes 4)
      set(_p_data_type_include "")
      set(_p_type_extra_flags "")
    elseif(_p_data_type STREQUAL "double")
      set(_p_data_width_bytes 8)
      set(_p_data_type_include "")
      set(_p_type_extra_flags "")
    elseif(_p_data_type STREQUAL "half")
      set(_p_data_width_bytes 2)
      set(_p_data_type_include "#include \"hls_half.h\"")
      set(_p_type_extra_flags "-DMM_HALF_PRECISION")
    elseif(_p_data_type STREQUAL "uint8_t")
      set(_p_data_width_bytes 1)
      set(_p_data_type_include "")
      set(_p_type_extra_flags "")
    elseif(_p_data_type MATCHES "^ap_(u?)fixed<([0-9]+),([0-9]+)")
      # ap_fixed<W,I> / ap_ufixed<W,I>: byte width = ceil(W/8)
      set(_fx_u "${CMAKE_MATCH_1}")
      set(_p_total_bits "${CMAKE_MATCH_2}")
      set(_p_int_bits   "${CMAKE_MATCH_3}")
      math(EXPR _p_data_width_bytes "(${_p_total_bits} + 7) / 8")
      # Wide accumulator: add MM_ACCUMULATOR_EXTRA_INT_BITS to both W and I
      math(EXPR _acc_w "${_p_total_bits} + ${MM_ACCUMULATOR_EXTRA_INT_BITS}")
      math(EXPR _acc_i "${_p_int_bits}   + ${MM_ACCUMULATOR_EXTRA_INT_BITS}")
      set(_p_acc_data_type "ap_${_fx_u}fixed<${_acc_w},${_acc_i}>")
      set(_p_sat_data_type
          "ap_${_fx_u}fixed<${_p_total_bits},${_p_int_bits},AP_TRN,AP_SAT>")
      set(_p_data_type_include "#include \"ap_fixed.h\"")
      set(_p_type_extra_flags "")
    else()
      message(WARNING
        "Platform '${_pname}': unrecognised data_type '${_p_data_type}' "
        "(supported: float, double, half, uint8_t, ap_fixed<W,I>, ap_ufixed<W,I>) — skipping.")
      continue()
    endif()
  endif()

  # --- Optional acc_data_type override ---
  # If "acc_data_type" is present in the JSON it overrides the auto-computed
  # value (ap_fixed<W+extra,I+extra>).  This lets platforms pin the accumulator
  # width independently of MM_ACCUMULATOR_EXTRA_INT_BITS.
  string(JSON _p_acc_override ERROR_VARIABLE _e GET "${_pjson}" acc_data_type)
  if(NOT _e AND _p_acc_override)
    set(_p_acc_data_type "${_p_acc_override}")
  endif()

  # --- Derived quantities ---
  set(_p_elem_size "${_p_data_width_bytes}")
  math(EXPR _p_mem_elems_n "${_p_bus_n} / ${_p_elem_size}")
  math(EXPR _p_mem_elems_m "${_p_bus_m} / ${_p_elem_size}")
  math(EXPR _p_inner_n    "${_p_tile_n} / ${_p_para_n}")
  math(EXPR _p_inner_m    "${_p_tile_m} / ${_p_para_m}")
  math(EXPR _p_kw_n       "${_p_elem_size} * ${_p_para_n}")
  math(EXPR _p_kw_m       "${_p_elem_size} * ${_p_para_m}")

  # --- Constraint checks ---
  set(_p_skip FALSE)

  math(EXPR _r "${_p_tile_n} % ${_p_para_n}")
  if(_r GREATER 0)
    message(WARNING "Platform '${_pname}': memory_tile_size_n (${_p_tile_n}) not divisible by parallelism_n (${_p_para_n}) — skipping.")
    set(_p_skip TRUE)
  endif()

  math(EXPR _r "${_p_tile_n} % ${_p_mem_elems_n}")
  if(_r GREATER 0)
    message(WARNING "Platform '${_pname}': memory_tile_size_n (${_p_tile_n}) not divisible by bus width in N (${_p_mem_elems_n} elems) — skipping.")
    set(_p_skip TRUE)
  endif()

  math(EXPR _r "${_p_tile_m} % ${_p_mem_elems_m}")
  if(_r GREATER 0)
    message(WARNING "Platform '${_pname}': memory_tile_size_m (${_p_tile_m}) not divisible by bus width in M (${_p_mem_elems_m} elems) — skipping.")
    set(_p_skip TRUE)
  endif()

  math(EXPR _r "${_p_mem_elems_m} % ${_p_para_m}")
  if(_r GREATER 0)
    message(WARNING "Platform '${_pname}': bus width in M (${_p_mem_elems_m} elems) not divisible by parallelism_m (${_p_para_m}) — skipping.")
    set(_p_skip TRUE)
  endif()

  if(_p_inner_n GREATER _p_inner_m)
    message(WARNING
      "Platform '${_pname}': double-buffer constraint violated — "
      "kInnerTilesN (${_p_inner_n}) > kInnerTilesM (${_p_inner_m}). "
      "Reduce memory_tile_size_n or increase memory_tile_size_m / parallelism_n — skipping.")
    set(_p_skip TRUE)
  endif()

  if(_p_skip)
    continue()
  endif()

  # --- Generate platform-specific Config.h ---
  # Temporarily override the global synthesis parameters so that
  # configure_file produces a header with the platform's values.
  set(_sv_PARA_N        "${MM_PARALLELISM_N}")
  set(_sv_PARA_M        "${MM_PARALLELISM_M}")
  set(_sv_KW_N          "${MM_KERNEL_WIDTH_N}")
  set(_sv_KW_M          "${MM_KERNEL_WIDTH_M}")
  set(_sv_CLOCK         "${MM_CLOCK_INTERNAL}")
  set(_sv_TILE_N        "${MM_MEMORY_TILE_SIZE_N}")
  set(_sv_TILE_M        "${MM_MEMORY_TILE_SIZE_M}")
  set(_sv_BUS_N         "${MM_MEMORY_BUS_WIDTH_N}")
  set(_sv_BUS_K         "${MM_MEMORY_BUS_WIDTH_K}")
  set(_sv_BUS_M         "${MM_MEMORY_BUS_WIDTH_M}")
  set(_sv_DATA_TYPE     "${MM_DATA_TYPE}")
  set(_sv_DATA_WIDTH_B  "${MM_DATA_WIDTH_BYTES}")
  set(_sv_DATA_INCLUDE  "${MM_DATA_TYPE_INCLUDE}")
  set(_sv_ACC_DATA_TYPE "${MM_ACC_DATA_TYPE}")
  set(_sv_SAT_DATA_TYPE "${MM_SAT_DATA_TYPE}")

  set(MM_PARALLELISM_N      "${_p_para_n}")
  set(MM_PARALLELISM_M      "${_p_para_m}")
  set(MM_KERNEL_WIDTH_N     "${_p_kw_n}")
  set(MM_KERNEL_WIDTH_M     "${_p_kw_m}")
  set(MM_CLOCK_INTERNAL     "${_p_clock}")
  set(MM_MEMORY_TILE_SIZE_N "${_p_tile_n}")
  set(MM_MEMORY_TILE_SIZE_M "${_p_tile_m}")
  set(MM_MEMORY_BUS_WIDTH_N "${_p_bus_n}")
  set(MM_MEMORY_BUS_WIDTH_K "${_p_bus_k}")
  set(MM_MEMORY_BUS_WIDTH_M "${_p_bus_m}")
  set(MM_DATA_TYPE          "${_p_data_type}")
  set(MM_DATA_WIDTH_BYTES   "${_p_data_width_bytes}")
  set(MM_DATA_TYPE_INCLUDE  "${_p_data_type_include}")
  set(MM_ACC_DATA_TYPE      "${_p_acc_data_type}")
  set(MM_SAT_DATA_TYPE      "${_p_sat_data_type}")

  file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/${_pname}")
  configure_file(include/Config.h.in "${_pname}/Config.h")

  set(MM_PARALLELISM_N      "${_sv_PARA_N}")
  set(MM_PARALLELISM_M      "${_sv_PARA_M}")
  set(MM_KERNEL_WIDTH_N     "${_sv_KW_N}")
  set(MM_KERNEL_WIDTH_M     "${_sv_KW_M}")
  set(MM_CLOCK_INTERNAL     "${_sv_CLOCK}")
  set(MM_MEMORY_TILE_SIZE_N "${_sv_TILE_N}")
  set(MM_MEMORY_TILE_SIZE_M "${_sv_TILE_M}")
  set(MM_MEMORY_BUS_WIDTH_N "${_sv_BUS_N}")
  set(MM_MEMORY_BUS_WIDTH_K "${_sv_BUS_K}")
  set(MM_MEMORY_BUS_WIDTH_M "${_sv_BUS_M}")
  set(MM_DATA_TYPE          "${_sv_DATA_TYPE}")
  set(MM_DATA_WIDTH_BYTES   "${_sv_DATA_WIDTH_B}")
  set(MM_DATA_TYPE_INCLUDE  "${_sv_DATA_INCLUDE}")
  set(MM_ACC_DATA_TYPE      "${_sv_ACC_DATA_TYPE}")
  set(MM_SAT_DATA_TYPE      "${_sv_SAT_DATA_TYPE}")

  # --- Generate Vitis HLS TCL script ---
  # Prepend platform include dir so its Config.h shadows the default one.
  # _p_type_extra_flags carries any defines required by the data type
  # (e.g. -DMM_HALF_PRECISION) that are not already in the base flags.
  set(_p_hls_flags "-I${CMAKE_BINARY_DIR}/${_pname} ${_p_type_extra_flags} ${_MM_BASE_HLS_FLAGS}")
  if(_p_board)
    set(_p_set_part "set_part ${_p_part} -board ${_p_board}")
  else()
    set(_p_set_part "set_part ${_p_part}")
  endif()

  file(WRITE "${CMAKE_BINARY_DIR}/MatrixMultiplication_${_pname}.tcl"
"open_project MatrixMultiplication_${_pname}
open_solution -flow_target vitis ${_pname}
${_p_set_part}
add_files -cflags \"${_p_hls_flags}\" \"${_MM_KERNEL_SRC_STR}\"
set_top ${MM_KERNEL_NAME}
create_clock -period ${_p_clock}MHz -name default
csynth_design
export_design -rtl verilog -format ip_catalog
exit
")

  # --- Add synthesis target ---
  add_custom_target(synthesize_${_pname}
    COMMAND ${Vitis_HLS} ${Vitis_HLS_TCL_FLAG}
            "${CMAKE_BINARY_DIR}/MatrixMultiplication_${_pname}.tcl"
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    COMMENT "HLS synthesis for ${_pname} (${_p_part} @ ${_p_clock} MHz, ${_p_data_type}, N=${_p_para_n} PEs, tile ${_p_tile_n}x${_p_tile_m})"
    DEPENDS ${MM_KERNEL_SRC}
            "${CMAKE_BINARY_DIR}/Config.h"
            "${CMAKE_BINARY_DIR}/${_pname}/Config.h"
            "${_MM_PLATFORM_FILE}")
  set_property(TARGET synthesize_${_pname} APPEND PROPERTY ADDITIONAL_CLEAN_FILES
               "${CMAKE_BINARY_DIR}/MatrixMultiplication_${_pname}"
               "${CMAKE_BINARY_DIR}/${_pname}")

  message(STATUS
    "Added target: synthesize_${_pname} "
    "(${_p_para_n} PEs, tile ${_p_tile_n}x${_p_tile_m}, ${_p_clock} MHz)")

  # --- Per-platform TestSimulation ---
  # Compile kernel sources + testbench together with the platform's Config.h
  # (placed BEFORE the global build-dir so it shadows the default Config.h).
  # This gives a self-contained binary that exercises the exact data type,
  # tile sizes, and bus widths defined in the platform JSON.
  add_executable(TestSimulation_${_pname}
      test/TestSimulation.cpp
      ${MM_KERNEL_SRC})
  target_include_directories(TestSimulation_${_pname} BEFORE PRIVATE
      "${CMAKE_BINARY_DIR}/${_pname}")
  target_link_libraries(TestSimulation_${_pname}
      ${CMAKE_THREAD_LIBS_INIT}
      ${BLAS_LIBRARIES}
      ${Vitis_LIBRARIES})
  if(_p_data_type STREQUAL "half")
    target_compile_definitions(TestSimulation_${_pname} PRIVATE MM_HALF_PRECISION)
    target_link_libraries(TestSimulation_${_pname} ${Vitis_FLOATING_POINT_LIBRARY})
  endif()
  add_test(NAME TestSimulation_${_pname}
      COMMAND TestSimulation_${_pname}
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")

  # --- Per-platform csim target ---
  # Same logic as the global csim but uses the platform's Config.h (placed
  # first in the include path so it shadows the default one).
  if(MM_CSIM_PART AND Vitis_HLS)
    # Kernel flags: no -DMM_SYNTHESIS / -DHLSLIB_SYNTHESIS (hlslib threading
    # handles DATAFLOW concurrency for simulation).
    set(_p_csim_kflags "-DHLSLIB_XILINX -D__VITIS_HLS__ -DHLSLIB_STREAM_TIMEOUT=16")
    set(_p_csim_kflags "${_p_csim_kflags} -DVITIS_MAJOR_VERSION=${Vitis_MAJOR_VERSION}")
    set(_p_csim_kflags "${_p_csim_kflags} -DVITIS_MINOR_VERSION=${Vitis_MINOR_VERSION}")
    set(_p_csim_kflags "${_p_csim_kflags} -DVITIS_VERSION=${Vitis_MAJOR_VERSION}.${Vitis_MINOR_VERSION}")
    # Platform Config.h MUST come before the global build dir.
    set(_p_csim_kflags "${_p_csim_kflags} -I${CMAKE_BINARY_DIR}/${_pname} -I${CMAKE_SOURCE_DIR}/include -I${CMAKE_SOURCE_DIR}/hlslib/include -I${CMAKE_BINARY_DIR}")
    if(MM_DYNAMIC_SIZES)
      set(_p_csim_kflags "${_p_csim_kflags} -DMM_DYNAMIC_SIZES")
    endif()
    if(_p_type_extra_flags)
      set(_p_csim_kflags "${_p_csim_kflags} ${_p_type_extra_flags}")
    endif()

    set(_p_csim_tbflags "-DMM_DYNAMIC_SIZES -DMM_CSIM -std=c++14")
    if(MM_HAS_BLAS OR BLAS_FOUND)
      set(_p_csim_tbflags "${_p_csim_tbflags} -DMM_HAS_BLAS")
    endif()
    set(_p_csim_tbflags "${_p_csim_tbflags} -I${CMAKE_BINARY_DIR}/${_pname} -I${CMAKE_SOURCE_DIR}/include -I${CMAKE_BINARY_DIR} -I${CMAKE_SOURCE_DIR}/hlslib/include")

    string(REPLACE ";" " " MM_CSIM_LDFLAGS_TMP "${BLAS_LIBRARIES}")
    set(_p_csim_design_cmd "csim_design -ldflags \"${MM_CSIM_LDFLAGS_TMP} -lOpenCL\"")

    file(WRITE "${CMAKE_BINARY_DIR}/MatrixMultiplication_csim_${_pname}.tcl"
"open_project MatrixMultiplication_csim_${_pname}
open_solution -flow_target vitis csim_${_pname}
${_p_set_part}
add_files -cflags \"${_p_csim_kflags}\" \"${_MM_KERNEL_SRC_STR}\"
add_files -tb -cflags \"${_p_csim_tbflags}\" \"${CMAKE_SOURCE_DIR}/test/TestSimulation.cpp\"
set_top ${MM_KERNEL_NAME}
${_p_csim_design_cmd}
exit
")

    add_custom_target(csim_${_pname}
      COMMAND ${Vitis_HLS} ${Vitis_HLS_TCL_FLAG}
              "${CMAKE_BINARY_DIR}/MatrixMultiplication_csim_${_pname}.tcl"
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
      COMMENT "Running HLS C-simulation for ${_pname} (${_p_data_type})"
      DEPENDS ${MM_KERNEL_SRC}
              "${CMAKE_SOURCE_DIR}/test/TestSimulation.cpp"
              "${CMAKE_BINARY_DIR}/${_pname}/Config.h")
    set_property(TARGET csim_${_pname} APPEND PROPERTY ADDITIONAL_CLEAN_FILES
                 "${CMAKE_BINARY_DIR}/MatrixMultiplication_csim_${_pname}")
  endif()

endforeach()

# HLS C-simulation (csim) — global csim target using the default Config.h.
# Per-platform csim_<name> targets are created inside the platform foreach loop above.
if(MM_CSIM_PART AND Vitis_HLS)
  string(REPLACE ";" " " MM_KERNEL_SRC_STR "${MM_KERNEL_SRC}")

  # Csim kernel flags: deliberately omit -DMM_SYNTHESIS and -DHLSLIB_SYNTHESIS.
  # The kernel uses a loop (with #pragma HLS UNROLL) to instantiate PEs inside a
  # DATAFLOW region. Vitis HLS csim runs DATAFLOW loop bodies sequentially even
  # with UNROLL, causing stream deadlocks. Without HLSLIB_SYNTHESIS, hlslib's
  # threaded simulation handles HLSLIB_DATAFLOW_FUNCTION concurrently, which
  # correctly models the loop-based PE instantiation.
  set(MM_CSIM_KERNEL_FLAGS "-DHLSLIB_XILINX -D__VITIS_HLS__ -DHLSLIB_STREAM_TIMEOUT=16")
  set(MM_CSIM_KERNEL_FLAGS "${MM_CSIM_KERNEL_FLAGS} -DVITIS_MAJOR_VERSION=${Vitis_MAJOR_VERSION}")
  set(MM_CSIM_KERNEL_FLAGS "${MM_CSIM_KERNEL_FLAGS} -DVITIS_MINOR_VERSION=${Vitis_MINOR_VERSION}")
  set(MM_CSIM_KERNEL_FLAGS "${MM_CSIM_KERNEL_FLAGS} -DVITIS_VERSION=${Vitis_MAJOR_VERSION}.${Vitis_MINOR_VERSION}")
  set(MM_CSIM_KERNEL_FLAGS "${MM_CSIM_KERNEL_FLAGS} -I${CMAKE_SOURCE_DIR}/include -I${CMAKE_SOURCE_DIR}/hlslib/include -I${CMAKE_BINARY_DIR}")
  if(MM_DYNAMIC_SIZES)
    set(MM_CSIM_KERNEL_FLAGS "${MM_CSIM_KERNEL_FLAGS} -DMM_DYNAMIC_SIZES")
  endif()
  if(MM_DATA_TYPE STREQUAL "half")
    set(MM_CSIM_KERNEL_FLAGS "${MM_CSIM_KERNEL_FLAGS} -DMM_HALF_PRECISION")
  endif()

  # Testbench flags: same includes as kernel but no -DMM_SYNTHESIS.
  # -DMM_CSIM suppresses the OpenCL host include in Utility.h (not needed for csim).
  set(MM_CSIM_TB_FLAGS "-DMM_DYNAMIC_SIZES -DMM_CSIM -std=c++14")
  if(MM_HAS_BLAS OR BLAS_FOUND)
    set(MM_CSIM_TB_FLAGS "${MM_CSIM_TB_FLAGS} -DMM_HAS_BLAS")
  endif()
  set(MM_CSIM_TB_FLAGS "${MM_CSIM_TB_FLAGS} -I${CMAKE_SOURCE_DIR}/include -I${CMAKE_BINARY_DIR} -I${CMAKE_SOURCE_DIR}/hlslib/include")

  # Collect external link flags needed by the testbench (BLAS + OpenCL)
  string(REPLACE ";" " " MM_CSIM_LDFLAGS "${BLAS_LIBRARIES}")
  set(MM_CSIM_LDFLAGS "${MM_CSIM_LDFLAGS} -lOpenCL")

  # The testbench runs all cases internally; no -argv needed.
  set(MM_CSIM_DESIGN_CMD "csim_design -ldflags \"${MM_CSIM_LDFLAGS}\"")

  file(WRITE ${CMAKE_BINARY_DIR}/MatrixMultiplication_csim.tcl
"open_project MatrixMultiplication
open_solution -flow_target vitis csim
set_part ${MM_CSIM_PART}
add_files -cflags \"${MM_CSIM_KERNEL_FLAGS}\" \"${MM_KERNEL_SRC_STR}\"
add_files -tb -cflags \"${MM_CSIM_TB_FLAGS}\" \"${CMAKE_SOURCE_DIR}/test/TestSimulation.cpp\"
set_top ${MM_KERNEL_NAME}
${MM_CSIM_DESIGN_CMD}
exit
")

  add_custom_target(csim
    COMMAND ${Vitis_HLS} ${Vitis_HLS_TCL_FLAG} ${CMAKE_BINARY_DIR}/MatrixMultiplication_csim.tcl
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running HLS C-simulation for MatrixMultiplication"
    DEPENDS ${MM_KERNEL_SRC}
            ${CMAKE_SOURCE_DIR}/test/TestSimulation.cpp
            ${CMAKE_BINARY_DIR}/Config.h)
  set_property(TARGET csim APPEND PROPERTY ADDITIONAL_CLEAN_FILES
               ${CMAKE_BINARY_DIR}/MatrixMultiplication)

  add_test(NAME csim
    COMMAND ${Vitis_HLS} ${Vitis_HLS_TCL_FLAG}
            ${CMAKE_BINARY_DIR}/MatrixMultiplication_csim.tcl
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endif()  # MM_CSIM_PART AND Vitis_HLS

# Host code to launch kernel
add_executable(RunHardware.exe host/RunHardware.cpp)
target_link_libraries(RunHardware.exe ${Vitis_LIBRARIES} ${BLAS_LIBRARIES} mmkernel)
if(MM_TWO_DIMMS)
  target_compile_definitions(RunHardware.exe PRIVATE MM_TWO_DIMMS)
endif()
if(MM_DATA_TYPE STREQUAL "half")
  target_link_libraries(RunHardware.exe ${Vitis_FLOATING_POINT_LIBRARY})
endif()
if(MM_POWER_METER)
  target_link_libraries(RunHardware.exe powermeter)
endif()
