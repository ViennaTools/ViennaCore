# Create a PTX file from a .cu source and copy it to VIENNACORE_NVCC_PTX_DIR.
#
# Usage:
#   set(VIENNACORE_NVCC_PTX_DIR "${CMAKE_BINARY_DIR}/ptx")  # or any path
#   viennacore_add_ptx(my_kernel_ptx "${CMAKE_CURRENT_SOURCE_DIR}/kernel.cu")
#
# Result:
#   - Builds: <binary_dir>/<target_name>.ptx
#   - Copies to: ${VIENNACORE_NVCC_PTX_DIR}/<target_name>.ptx
#   - Creates a custom target <target_name> you can depend on

function(viennacore_add_ptx target_name cu_file)
  if(NOT DEFINED VIENNACORE_NVCC_PTX_DIR OR VIENNACORE_NVCC_PTX_DIR STREQUAL "")
    message(
      FATAL_ERROR "VIENNACORE_NVCC_PTX_DIR is not set. Set it before calling viennacore_add_ptx().")
  endif()

  if(NOT EXISTS "${cu_file}")
    message(FATAL_ERROR "viennacore_add_ptx: CUDA file not found: ${cu_file}")
  endif()

  # Ensure CUDA compiler is available (works even if CUDA isn't enabled globally)
  enable_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "viennacore_add_ptx: CUDA language enabled but no CUDA compiler found.")
  endif()

  # Output PTX in the current binary dir with a stable name
  set(ptx_out "${CMAKE_CURRENT_BINARY_DIR}/${target_name}.ptx")
  set(ptx_dst "${VIENNACORE_NVCC_PTX_DIR}/${target_name}.ptx")

  # Gather include dirs from the directory scope (optional but useful)
  # You can also pass include dirs via target_link_libraries to an INTERFACE target and use that instead.
  get_directory_property(dir_includes INCLUDE_DIRECTORIES)

  # Convert include dirs to -I flags
  set(nvcc_includes "")
  foreach(inc IN LISTS VIENNACORE_NVCC_INCLUDE_DIRS)
    list(APPEND nvcc_includes "-I${inc}")
  endforeach()

  # Configure preprocessor definitions
  foreach(def IN LISTS VIENNACORE_NVCC_DEFINES)
    list(APPEND nvcc_includes "-D${def}")
  endforeach()

  # Allow user to provide extra NVCC flags via a cache/list variable
  # Example:
  #   set(VIENNACORE_NVCC_FLAGS --use_fast_math --expt-relaxed-constexpr)
  if(NOT DEFINED VIENNACORE_NVCC_FLAGS)
    set(VIENNACORE_NVCC_FLAGS "")
  endif()

  # Allow user to choose architectures (optional). If empty, NVCC default applies.
  # Example:
  #   set(VIENNACORE_CUDA_ARCH "75")  # or "native" if you handle that yourself
  set(arch_flag "")
  if(DEFINED VIENNACORE_CUDA_ARCH AND NOT VIENNACORE_CUDA_ARCH STREQUAL "")
    set(arch_flag "-gencode=code=sm_${VIENNACORE_CUDA_ARCH},arch=compute_${VIENNACORE_CUDA_ARCH}")
  endif()

  add_custom_command(
    OUTPUT "${ptx_out}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${VIENNACORE_NVCC_PTX_DIR}"
    COMMAND "${CMAKE_CUDA_COMPILER}" --ptx -std=c++17 ${arch_flag} ${VIENNACORE_NVCC_FLAGS}
            ${nvcc_includes} "${cu_file}" -o "${ptx_out}"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${ptx_out}" "${ptx_dst}"
    DEPENDS "${cu_file}"
    VERBATIM
    COMMENT "Building NVCC PTX: ${ptx_dst}")

  add_custom_target(${target_name} ALL DEPENDS "${ptx_out}")

  # Expose paths to parent scope (handy for consumers)
  set(${target_name}_PTX_BUILD_PATH
      "${ptx_out}"
      PARENT_SCOPE)
  set(${target_name}_PTX_OUTPUT_PATH
      "${ptx_dst}"
      PARENT_SCOPE)
endfunction()
