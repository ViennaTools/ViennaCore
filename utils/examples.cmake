function(viennacore_quick_examples FOLDER)
  file(GLOB entries "*")

  foreach(entry IN LISTS ${entries})
    if(NOT IS_DIRECTORY "${entry}")
      message(STATUS "Skipping non directory '${entry}'")
      continue()
    endif()

    message(STATUS "Adding example ${entry}")
    add_subdirectory(${entry})
  endforeach()

  message(STATUS "Collected all examples")
endfunction()
