function(viennacore_setup_binding NAME OUTPUT_DIR)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_DIR}/${NAME})

  pybind11_add_module(${NAME} "pyWrap.cpp")
  configure_file(__init__.py.in ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/__init__.py)

  install(TARGETS ${NAME} LIBRARY DESTINATION ${NAME})
  install(DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/" DESTINATION ${NAME})

  message(STATUS "Added Python module '${NAME}'")
endfunction()
