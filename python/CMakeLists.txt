cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(ViennaCore_Python LANGUAGES CXX)

add_custom_target(${PROJECT_NAME} ALL)

# --------------------------------------------------------------------------------------------------------
# Setup Dependencies
# --------------------------------------------------------------------------------------------------------

include("../cmake/cpm.cmake")

set(PYBIND11_FINDPYTHON ON)

CPMFindPackage(
  NAME pybind11
  VERSION 2.12.0
  GIT_REPOSITORY "https://github.com/pybind/pybind11")

# --------------------------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------------------------

set(VIENNACORE_PYTHON_MODULE_NAME "viennacore")
set(VIENNACORE_LIBRARY_OUTPUT_DIR ${CMAKE_BINARY_DIR})

if(CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  set(VIENNACORE_LIBRARY_OUTPUT_DIR "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
  message(STATUS "[ViennaCore] Using Library Output Directory for bindings")
endif()

# --------------------------------------------------------------------------------------------------------
# Setup Bindings
# --------------------------------------------------------------------------------------------------------

viennacore_setup_binding(viennacore ${VIENNACORE_LIBRARY_OUTPUT_DIR})

target_link_libraries(viennacore PRIVATE ViennaCore)

target_compile_definitions(viennacore PRIVATE -DVIENNACORE_MODULE_NAME=viennacore)
