find_program(CLANG_FORMAT clang-format)
find_program(CMAKE_FORMAT cmake-format)
find_program(GIT_COMMAND git)

set(FORMAT_NAME "format")
set(CHECK_NAME "check-format")

if(NOT CMAKE_FORMAT
   OR NOT CLANG_FORMAT
   OR NOT GIT_COMMAND)
  # cmake-format: off
  set(ERROR_DUMMY
    COMMAND ${CMAKE_COMMAND} -E echo
    COMMAND ${CMAKE_COMMAND} -E echo "Could not find git, cmake-format or clang-format!"
    COMMAND ${CMAKE_COMMAND} -E echo
    COMMAND ${CMAKE_COMMAND} -E echo "Git:          ${GIT_COMMAND}"
    COMMAND ${CMAKE_COMMAND} -E echo "Clang-Format: ${CLANG_FORMAT}"
    COMMAND ${CMAKE_COMMAND} -E echo "CMake-Format: ${CMAKE_FORMAT}"
    COMMAND ${CMAKE_COMMAND} -E echo
    COMMAND ${CMAKE_COMMAND} -E echo "Please refer to their documentation for installation instructions:"
    COMMAND ${CMAKE_COMMAND} -E echo
    COMMAND ${CMAKE_COMMAND} -E echo "  - https://cmake-format.readthedocs.io/en/latest/installation.html"
    COMMAND ${CMAKE_COMMAND} -E echo "  - https://clang.llvm.org/docs/ClangFormat.html"
    COMMAND ${CMAKE_COMMAND} -E echo "  - https://git-scm.com/downloads"
    COMMAND ${CMAKE_COMMAND} -E echo
    COMMAND ${CMAKE_COMMAND} -E false
  )
  # cmake-format: on

  add_custom_target(${FORMAT_NAME} ${ERROR_DUMMY})
  add_custom_target(${CHECK_NAME} ${ERROR_DUMMY})

  return()
endif()

add_custom_target(
  ${FORMAT_NAME}
  COMMAND
    ${CMAKE_COMMAND} -DMODE=FORMAT -DCLANG_FORMAT=${CLANG_FORMAT}
    -DCMAKE_FORMAT=${CMAKE_FORMAT} -DGIT_COMMAND=${GIT_COMMAND} -P
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/format.cmake
  VERBATIM
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_custom_target(
  ${CHECK_NAME}
  COMMAND
    ${CMAKE_COMMAND} -DMODE=CHECK -DCLANG_FORMAT=${CLANG_FORMAT}
    -DCMAKE_FORMAT=${CMAKE_FORMAT} -DGIT_COMMAND=${GIT_COMMAND} -P
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/format.cmake
  VERBATIM
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
