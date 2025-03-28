# Synopsis:
#   CUDA_SELECT_NVCC_ARCH_FLAGS(out_variable [target_CUDA_architectures])
#   -- Selects GPU arch flags for nvcc based on target_CUDA_architectures
#      target_CUDA_architectures : Auto | Common | All | LIST(ARCH_AND_PTX ...)
#       - "Auto" detects local machine GPU compute arch at runtime.
#       - "Common" and "All" cover common and entire subsets of architectures
#      ARCH_AND_PTX : NAME | NUM.NUM | NUM.NUM(NUM.NUM) | NUM.NUM+PTX
#      NAME: Fermi Kepler Maxwell Kepler+Tegra Kepler+Tesla Maxwell+Tegra Pascal
#      NUM: Any number. Only those pairs are currently accepted by NVCC though:
#            2.0 2.1 3.0 3.2 3.5 3.7 5.0 5.2 5.3 6.0 6.2
#      Returns LIST of flags to be added to CUDA_NVCC_FLAGS in ${out_variable}
#      Additionally, sets ${out_variable}_readable to the resulting numeric list
#      Example:
#       CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS 3.0 3.5+PTX 5.2(5.0) Maxwell)
#        LIST(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
#
#      More info on CUDA architectures: https://en.wikipedia.org/wiki/CUDA
#

# This list will be used for CUDA_ARCH_NAME = All option
set(CUDA_KNOWN_GPU_ARCHITECTURES "Fermi" "Kepler" "Maxwell")

# This list will be used for CUDA_ARCH_NAME = Common option (enabled by default)
set(CUDA_COMMON_GPU_ARCHITECTURES "3.0" "3.5" "5.0")

if(CUDA_VERSION VERSION_GREATER "6.5")
  list(APPEND CUDA_KNOWN_GPU_ARCHITECTURES "Kepler+Tegra" "Kepler+Tesla" "Maxwell+Tegra")
  list(APPEND CUDA_COMMON_GPU_ARCHITECTURES "5.2")
endif()

if(CUDA_VERSION VERSION_GREATER "7.5")
  list(APPEND CUDA_KNOWN_GPU_ARCHITECTURES "Pascal")
  list(APPEND CUDA_COMMON_GPU_ARCHITECTURES "6.0" "6.1" "6.1+PTX")
else()
  list(APPEND CUDA_COMMON_GPU_ARCHITECTURES "5.2+PTX")
endif()

################################################################################################
# A function for automatic detection of GPUs installed  (if autodetection is enabled)
# Usage:
#   CUDA_DETECT_INSTALLED_GPUS(OUT_VARIABLE)
#
function(CUDA_DETECT_INSTALLED_GPUS OUT_VARIABLE)
  if(NOT CUDA_GPU_DETECT_OUTPUT)
    set(cufile ${PROJECT_BINARY_DIR}/detect_cuda_archs.cu)

    file(
      WRITE ${cufile}
      ""
      "#include <cstdio>\n"
      "int main()\n"
      "{\n"
      "  int count = 0;\n"
      "  if (cudaSuccess != cudaGetDeviceCount(&count)) return -1;\n"
      "  if (count == 0) return -1;\n"
      "  for (int device = 0; device < count; ++device)\n"
      "  {\n"
      "    cudaDeviceProp prop;\n"
      "    if (cudaSuccess == cudaGetDeviceProperties(&prop, device))\n"
      "      std::printf(\"%d.%d \", prop.major, prop.minor);\n"
      "  }\n"
      "  return 0;\n"
      "}\n")

    execute_process(
      COMMAND "${CUDA_NVCC_EXECUTABLE}" "--run" "${cufile}"
      WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/CMakeFiles/"
      RESULT_VARIABLE nvcc_res
      OUTPUT_VARIABLE nvcc_out
      ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(nvcc_res EQUAL 0)
      string(REPLACE "2.1" "2.1(2.0)" nvcc_out "${nvcc_out}")
      set(CUDA_GPU_DETECT_OUTPUT
          ${nvcc_out}
          CACHE INTERNAL "Returned GPU architetures from detect_gpus tool" FORCE)
    endif()
  endif()

  if(NOT CUDA_GPU_DETECT_OUTPUT)
    message(STATUS "Automatic GPU detection failed. Building for common architectures.")
    set(${OUT_VARIABLE}
        ${CUDA_COMMON_GPU_ARCHITECTURES}
        PARENT_SCOPE)
  else()
    set(${OUT_VARIABLE}
        ${CUDA_GPU_DETECT_OUTPUT}
        PARENT_SCOPE)
  endif()
endfunction()

################################################################################################
# Function for selecting GPU arch flags for nvcc based on CUDA architectures from parameter list
# Usage:
#   SELECT_NVCC_ARCH_FLAGS(out_variable [list of CUDA compute archs])
function(CUDA_SELECT_NVCC_ARCH_FLAGS out_variable)
  set(CUDA_ARCH_LIST "${ARGN}")

  if("X${CUDA_ARCH_LIST}" STREQUAL "X")
    set(CUDA_ARCH_LIST "Auto")
  endif()

  set(cuda_arch_bin)
  set(cuda_arch_ptx)

  if("${CUDA_ARCH_LIST}" STREQUAL "All")
    set(CUDA_ARCH_LIST ${CUDA_KNOWN_GPU_ARCHITECTURES})
  elseif("${CUDA_ARCH_LIST}" STREQUAL "Common")
    set(CUDA_ARCH_LIST ${CUDA_COMMON_GPU_ARCHITECTURES})
  elseif("${CUDA_ARCH_LIST}" STREQUAL "Auto")
    cuda_detect_installed_gpus(CUDA_ARCH_LIST)
    message(STATUS "Autodetected CUDA architecture(s): ${CUDA_ARCH_LIST}")
  endif()

  # Now process the list and look for names
  string(REGEX REPLACE "[ \t]+" ";" CUDA_ARCH_LIST "${CUDA_ARCH_LIST}")
  list(REMOVE_DUPLICATES CUDA_ARCH_LIST)
  foreach(arch_name ${CUDA_ARCH_LIST})
    set(arch_bin)
    set(add_ptx FALSE)
    # Check to see if we are compiling PTX
    if(arch_name MATCHES "(.*)\\+PTX$")
      set(add_ptx TRUE)
      set(arch_name ${CMAKE_MATCH_1})
    endif()
    if(arch_name MATCHES "^([0-9]\\.[0-9](\\([0-9]\\.[0-9]\\))?)$")
      set(arch_bin ${CMAKE_MATCH_1})
      set(arch_ptx ${arch_bin})
    else()
      # Look for it in our list of known architectures
      if(${arch_name} STREQUAL "Fermi")
        set(arch_bin 2.0 "2.1(2.0)")
      elseif(${arch_name} STREQUAL "Kepler+Tegra")
        set(arch_bin 3.2)
      elseif(${arch_name} STREQUAL "Kepler+Tesla")
        set(arch_bin 3.7)
      elseif(${arch_name} STREQUAL "Kepler")
        set(arch_bin 3.0 3.5)
        set(arch_ptx 3.5)
      elseif(${arch_name} STREQUAL "Maxwell+Tegra")
        set(arch_bin 5.3)
      elseif(${arch_name} STREQUAL "Maxwell")
        set(arch_bin 5.0 5.2)
        set(arch_ptx 5.2)
      elseif(${arch_name} STREQUAL "Pascal")
        set(arch_bin 6.0 6.1)
        set(arch_ptx 6.1)
      else()
        message(
          SEND_ERROR "Unknown CUDA Architecture Name ${arch_name} in CUDA_SELECT_NVCC_ARCH_FLAGS")
      endif()
    endif()
    if(NOT arch_bin)
      message(SEND_ERROR "arch_bin wasn't set for some reason")
    endif()
    list(APPEND cuda_arch_bin ${arch_bin})
    if(add_ptx)
      if(NOT arch_ptx)
        set(arch_ptx ${arch_bin})
      endif()
      list(APPEND cuda_arch_ptx ${arch_ptx})
    endif()
  endforeach()

  # remove dots and convert to lists
  string(REGEX REPLACE "\\." "" cuda_arch_bin "${cuda_arch_bin}")
  string(REGEX REPLACE "\\." "" cuda_arch_ptx "${cuda_arch_ptx}")
  string(REGEX MATCHALL "[0-9()]+" cuda_arch_bin "${cuda_arch_bin}")
  string(REGEX MATCHALL "[0-9]+" cuda_arch_ptx "${cuda_arch_ptx}")

  if(cuda_arch_bin)
    list(REMOVE_DUPLICATES cuda_arch_bin)
  endif()
  if(cuda_arch_ptx)
    list(REMOVE_DUPLICATES cuda_arch_ptx)
  endif()

  set(nvcc_flags "")
  set(nvcc_archs_readable "")

  # Tell NVCC to add binaries for the specified GPUs
  foreach(arch ${cuda_arch_bin})
    if(arch MATCHES "([0-9]+)\\(([0-9]+)\\)")
      # User explicitly specified ARCH for the concrete CODE
      list(APPEND nvcc_flags -gencode arch=compute_${CMAKE_MATCH_2},code=sm_${CMAKE_MATCH_1})
      list(APPEND nvcc_archs_readable sm_${CMAKE_MATCH_1})
    else()
      # User didn't explicitly specify ARCH for the concrete CODE, we assume ARCH=CODE
      list(APPEND nvcc_flags -gencode arch=compute_${arch},code=sm_${arch})
      list(APPEND nvcc_archs_readable sm_${arch})
    endif()
  endforeach()

  # Tell NVCC to add PTX intermediate code for the specified architectures
  foreach(arch ${cuda_arch_ptx})
    list(APPEND nvcc_flags -gencode arch=compute_${arch},code=compute_${arch})
    list(APPEND nvcc_archs_readable compute_${arch})
  endforeach()

  string(REPLACE ";" " " nvcc_archs_readable "${nvcc_archs_readable}")
  set(${out_variable}
      ${nvcc_flags}
      PARENT_SCOPE)
  set(${out_variable}_readable
      ${nvcc_archs_readable}
      PARENT_SCOPE)
endfunction()
