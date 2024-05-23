cmake_policy(SET CMP0007 NEW)

function(check_changes FILE)
  execute_process(COMMAND ${GIT_COMMAND} --no-pager diff --exit-code --color
                          ${FILE} RESULT_VARIABLE RESULT)

  set(FILE_CHANGED
      ${RESULT}
      PARENT_SCOPE)
endfunction()

execute_process(
  COMMAND ${GIT_COMMAND} ls-files --exclude-standard
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE TRACKED_FILES)

string(REPLACE "\n" ";" LIST_FILES "${TRACKED_FILES}")

foreach(file IN LISTS LIST_FILES)
  if(NOT EXISTS "${file}" OR IS_DIRECTORY "${file}")
    list(REMOVE_ITEM LIST_FILES ${file})
  endif()
endforeach()

set(CMAKE_FILES "${LIST_FILES}")
list(FILTER CMAKE_FILES INCLUDE REGEX "(CMakeLists.txt|.*\\.cmake(\\.in)?)$")

set(SOURCE_FILES "${LIST_FILES}")
list(FILTER SOURCE_FILES INCLUDE REGEX ".*\\.(cpp|hpp)$")

set(ALL_FINE TRUE)

foreach(file IN LISTS CMAKE_FILES)
  execute_process(
    COMMAND ${CMAKE_FORMAT} -c=${CMAKE_CURRENT_SOURCE_DIR}/config/.cpm-format
            -i ${file})

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
  execute_process(
    COMMAND ${CLANG_FORMAT} --style=file:${CMAKE_CURRENT_SOURCE_DIR}/config/.clang-format -i ${file})

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
