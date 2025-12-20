function(viennacore_detect_native_sm)
  find_program(NVIDIA_SMI_EXECUTABLE nvidia-smi)

  if (NOT NVIDIA_SMI_EXECUTABLE)
    message(WARNING "[ViennaCore] nvidia-smi not found. Cannot detect native SM.")
    return()
  endif()

  execute_process(
    COMMAND ${NVIDIA_SMI_EXECUTABLE} --query-gpu=compute_cap --format=csv,noheader
    OUTPUT_VARIABLE _cc
    RESULT_VARIABLE _res
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  if (NOT _res EQUAL 0 OR _cc STREQUAL "")
    message(WARNING "[ViennaCore] Failed to query compute capability via nvidia-smi.")
    return()
  endif()

  # Example: "8.6" -> "86"
  string(REPLACE "." "" _cc_nodot "${_cc}")

  set(VIENNACORE_NATIVE_SM "${_cc_nodot}" PARENT_SCOPE)
  message(STATUS "[ViennaCore] Detected native SM: ${_cc_nodot}")
endfunction()
