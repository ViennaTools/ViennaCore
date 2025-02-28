# Helper macros to configure CUDA and OptiX

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
