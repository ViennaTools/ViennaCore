function(viennacore_add_subdirs FOLDER)
  file(GLOB entries "${FOLDER}/*")

  foreach(entry IN LISTS entries)
    if(NOT IS_DIRECTORY "${entry}")
      message(STATUS "Skipping non directory '${entry}'")
      continue()
    endif()

    message(STATUS "Adding ${entry}")
    add_subdirectory(${entry})
  endforeach()
endfunction()
