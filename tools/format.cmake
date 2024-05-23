cmake_policy(SET CMP0007 NEW)
cmake_policy(SET CMP0009 NEW)

# --------------------------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------------------------

function(check_changes FILE)
  execute_process(COMMAND ${GIT_COMMAND} --no-pager diff --exit-code --color
                          ${FILE} RESULT_VARIABLE RESULT)

  set(FILE_CHANGED
      ${RESULT}
      PARENT_SCOPE)
endfunction()

function(get_root)
  get_filename_component(PARENT_DIR ${CMAKE_CURRENT_FUNCTION_LIST_DIR}
                         DIRECTORY)
  set(ROOT_DIR
      ${PARENT_DIR}
      PARENT_SCOPE)
endfunction()

# --------------------------------------------------------------------------------------------------------
# Formatter
# --------------------------------------------------------------------------------------------------------

file(GLOB_RECURSE LIST_FILES ${CMAKE_SOURCE_DIR}/**)

list(FILTER LIST_FILES EXCLUDE REGEX "build/")

if(EXCLUDE)
  message(STATUS "[Format] Excluding: ${EXCLUDE}")
  list(FILTER LIST_FILES EXCLUDE REGEX "${EXCLUDE}")
endif()

foreach(file IN LISTS LIST_FILES)
  if(NOT EXISTS "${file}" OR IS_DIRECTORY "${file}")
    list(REMOVE_ITEM LIST_FILES ${file})
  endif()
endforeach()

set(CMAKE_FILES "${LIST_FILES}")
list(FILTER CMAKE_FILES INCLUDE REGEX "(CMakeLists.txt|.*\\.cmake(\\.in)?)$")

set(SOURCE_FILES "${LIST_FILES}")
list(FILTER SOURCE_FILES INCLUDE REGEX ".*\\.(cpp|hpp)$")

get_root()

set(CMAKE_FORMAT_CONFIG "${ROOT_DIR}/config/.cmake-format")
set(CLANG_FORMAT_CONFIG "file:${ROOT_DIR}/config/.clang-format")

message(STATUS "[Format] Core Root: '${ROOT_DIR}'")
message(STATUS "[Format] -- CMake-Format Config: '${CMAKE_FORMAT_CONFIG}'")
message(STATUS "[Format] -- Clang-Format Config: '${CLANG_FORMAT_CONFIG}'")

if(MODE STREQUAL "LIST")
  string(REPLACE ";" "\n-- " CMAKE_FILES "${CMAKE_FILES}")
  string(REPLACE ";" "\n-- " SOURCE_FILES "${SOURCE_FILES}")

  message(STATUS "[Format] CMake-Files: \n-- ${CMAKE_FILES}")
  message(STATUS "[Format] Source-Files: \n-- ${SOURCE_FILES}")

  return()
endif()

set(ALL_FINE TRUE)

foreach(file IN LISTS CMAKE_FILES)
  execute_process(COMMAND ${CMAKE_FORMAT} -c=${CMAKE_FORMAT_CONFIG} -i ${file})

  if(NOT MODE STREQUAL "CHECK")
    continue()
  endif()

  check_changes(${file})

  if(NOT FILE_CHANGED)
    continue()
  endif()

  set(ALL_FINE FALSE)
endforeach()

foreach(file IN LISTS SOURCE_FILES)
  execute_process(COMMAND ${CLANG_FORMAT} --style=${CLANG_FORMAT_CONFIG} -i
                          ${file})

  if(NOT MODE STREQUAL "CHECK")
    continue()
  endif()

  check_changes(${file})

  if(NOT FILE_CHANGED)
    continue()
  endif()

  set(ALL_FINE FALSE)
endforeach()

if(ALL_FINE)
  message(STATUS "[Format] All files are properly formatted!")
  return()
endif()

message(FATAL_ERROR "[Format] One or more files need formatting!")
