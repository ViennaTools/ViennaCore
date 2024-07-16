function(viennacore_setup SOURCE_TARGET DEST_TARGET OUTPUT)
  if(NOT TARGET ${SOURCE_TARGET})
    message(WARNING "Could not find target ${SOURCE_TARGET}")
    return()
  endif()

  add_custom_command(
    TARGET ${DEST_TARGET}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory $<TARGET_FILE_DIR:${SOURCE_TARGET}> ${OUTPUT})
endfunction()

function(viennacore_setup_vtk_env TARGET OUTPUT)
  message(STATUS "Setting up VTK-Environment for ${TARGET}")

  # We expect all of the VTK binaries to be present in the same directory to which "vtksys" is
  # built. This is currently the case, and has been the case for prior vtk versions - However we
  # should keep an eye on this.

  viennacore_setup(vtksys ${TARGET} ${OUTPUT})
endfunction()

function(viennacore_setup_tbb_env TARGET OUTPUT)
  message(STATUS "Setting up TBB-Environment for ${TARGET}")
  viennacore_setup(tbb ${TARGET} ${OUTPUT})
endfunction()

function(viennacore_setup_embree_env TARGET OUTPUT)
  message(STATUS "Setting up Embree-Environment for ${TARGET}")
  viennacore_setup(embree ${TARGET} ${OUTPUT})
endfunction()
