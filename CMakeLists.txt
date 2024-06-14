cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(
  ViennaCore
  LANGUAGES CXX
  VERSION 1.0.0)

# --------------------------------------------------------------------------------------------------------
# Library Options
# --------------------------------------------------------------------------------------------------------

option(VIENNACORE_BUILD_TESTS "Build tests" OFF)

set(VIENNACORE_FORMAT_EXCLUDE
    ""
    CACHE STRING "Regex to further filter formatted files")

# --------------------------------------------------------------------------------------------------------
# Setup Library
# --------------------------------------------------------------------------------------------------------

add_library(${PROJECT_NAME} INTERFACE)
add_library(ViennaTools::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)

set_target_properties(
  ${PROJECT_NAME}
  PROPERTIES CXX_STANDARD 17
             CXX_EXTENSIONS OFF
             CXX_STANDARD_REQUIRED ON
             WINDOWS_EXPORT_ALL_SYMBOLS ON)

# --------------------------------------------------------------------------------------------------------
# Include directories
# --------------------------------------------------------------------------------------------------------

target_include_directories(
  ${PROJECT_NAME} INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/viennacore>
                            $<INSTALL_INTERFACE:include/viennacore-${PROJECT_VERSION}>)

# --------------------------------------------------------------------------------------------------------
# Setup Dependencies
# --------------------------------------------------------------------------------------------------------

include(cmake/cpm.cmake)

CPMAddPackage(
  NAME PackageProject
  VERSION 1.11.2
  GIT_REPOSITORY "https://github.com/TheLartians/PackageProject.cmake")

find_package(OpenMP REQUIRED)
target_link_libraries(${PROJECT_NAME} INTERFACE OpenMP::OpenMP_CXX)

# --------------------------------------------------------------------------------------------------------
# CMake Utilities
# --------------------------------------------------------------------------------------------------------

include(utils/format.cmake)

include(utils/batch.cmake)
include(utils/python.cmake)
include(utils/subdirs.cmake)
include(utils/sanitizer.cmake)

# --------------------------------------------------------------------------------------------------------
# Setup Tests
# --------------------------------------------------------------------------------------------------------

if(VIENNACORE_BUILD_TESTS)
  message(STATUS "[ViennaCore] Building Tests")

  enable_testing()
  add_subdirectory(tests)
endif()

# --------------------------------------------------------------------------------------------------------
# Install Target
# --------------------------------------------------------------------------------------------------------

packageProject(
  NAME ${PROJECT_NAME} NAMESPACE ViennaTools
  VERSION ${PROJECT_VERSION}
  BINARY_DIR ${PROJECT_BINARY_DIR}
  INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/viennacore
  INCLUDE_DESTINATION include/viennacore-${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)
