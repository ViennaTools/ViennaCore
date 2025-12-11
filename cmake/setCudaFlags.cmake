set(CUDA_GENERATED_OUTPUT_DIR ${VIENNACORE_PTX_DIR})
set(CUDA_MIN_SM_TARGET
    sm_60
    CACHE STRING "Minimum CUDA SM architecture to use for compilation.")

# Present the CUDA_64_BIT_DEVICE_CODE on the default set of options.
mark_as_advanced(CLEAR CUDA_64_BIT_DEVICE_CODE)

macro(add_cuda_flag_config config flag)
  string(TOUPPER "${config}" config)
  list(FIND CUDA_NVCC_FLAGS${config} ${flag} index)
  if(index EQUAL -1)
    list(APPEND CUDA_NVCC_FLAGS${config} ${flag})
    set(CUDA_NVCC_FLAGS${config}
        ${CUDA_NVCC_FLAGS${config}}
        CACHE STRING ${CUDA_NVCC_FLAGS_DESCRIPTION} FORCE)
  endif()
endmacro()

macro(add_cuda_flag flag)
  add_cuda_flag_config("" ${flag})
endmacro()

# Add some useful default arguments to the NVCC and NVRTC flags.  This is an example of
# how we use PASSED_FIRST_CONFIGURE.  Once you have configured, this variable is TRUE
# and following block of code will not be executed leaving you free to edit the values
# as much as you wish from the GUI or from ccmake.
if(NOT PASSED_FIRST_CONFIGURE)
  message(STATUS "[ViennaCore] Setting default NVCC flags for first time configuration.")

  set(CUDA_NVCC_FLAGS_DESCRIPTION "Semi-colon delimit multiple arguments.")
  string(REPLACE "sm_" "compute_" CUDA_MIN_COMPUTE_TARGET ${CUDA_MIN_SM_TARGET})
  list(FIND CUDA_NVCC_FLAGS "arch" index)
  if(index EQUAL -1)
    list(APPEND CUDA_NVCC_FLAGS -gencode=arch=${CUDA_MIN_COMPUTE_TARGET},code=${CUDA_MIN_SM_TARGET})
    set(CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS}
        CACHE STRING ${CUDA_NVCC_FLAGS_DESCRIPTION} FORCE)
  endif()
  
  add_cuda_flag("-use_fast_math")
  add_cuda_flag("-lineinfo")
  add_cuda_flag("-expt-relaxed-constexpr")
  add_cuda_flag("-diag-suppress 20044")
  add_cuda_flag("-rdc=true") # Enable relocatable device code for separate compilation.

  # CMAKE_CONFIGURATION_TYPES is only defined for multi-config build systems like
  # MSVC and Ninja, but we need to generate flags for each configuration
  # regardless.
  if(DEFINED CMAKE_CONFIGURATION_TYPES)
    set(OPTIX_CONFIGURATION_TYPES ${CMAKE_CONFIGURATION_TYPES})
  else()
    set(OPTIX_CONFIGURATION_TYPES "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
  endif()

  foreach(config ${OPTIX_CONFIGURATION_TYPES})
    if(${config} STREQUAL "Debug")
      add_cuda_flag_config(_${config} "-G")
      add_cuda_flag_config(_${config} "-O0")
    endif()
  endforeach()

  if(CUDA_USE_LOCAL_ENV)
    add_cuda_flag("--use-local-env")
  endif()
endif()

# Now that everything is done, indicate that we have finished configuring at least once.
# We use this variable to set certain defaults only on the first pass, so that we don't
# continually set them over and over again.
set(PASSED_FIRST_CONFIGURE
    ON
    CACHE INTERNAL "Already Configured once?")
