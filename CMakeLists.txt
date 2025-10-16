cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(
  ViennaCore
  LANGUAGES CXX
  VERSION 1.6.2)

# --------------------------------------------------------------------------------------------------------
# Library Options
# --------------------------------------------------------------------------------------------------------

option(VIENNACORE_USE_OPENMP "Enable OpenMP parallelization" ON)

option(VIENNACORE_USE_GPU "Enable GPU support" OFF)
option(VIENNACORE_FETCH_OPTIX "Fetch OptiX from GitHub" OFF)

option(VIENNACORE_BUILD_TESTS "Build tests" OFF)

set(VIENNACORE_FORMAT_EXCLUDE
    "build"
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
  VERSION 1.13.0
  GIT_REPOSITORY "https://github.com/TheLartians/PackageProject.cmake")

if(VIENNACORE_USE_OPENMP)
  find_package(OpenMP REQUIRED)
  target_link_libraries(${PROJECT_NAME} INTERFACE OpenMP::OpenMP_CXX)
endif()

# --------------------------------------------------------------------------------------------------------
# Setup GPU
# --------------------------------------------------------------------------------------------------------

if(VIENNACORE_USE_GPU)
  # Add ViennaCore cmake paths to our CMake code to the module path, so they can be found automatically by
  # CMake (i.e. FindCUDA and FindOptiX).
  set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

  # Find the CUDA Toolkit
  find_package(CUDA 12.0 QUIET)

  if(VIENNACORE_FETCH_OPTIX)
    CPMAddPackage(
      NAME OptiX
      VERSION 8.0.0
      GIT_REPOSITORY "https://github.com/NVIDIA/optix-dev")
    set(OptiX_INSTALL_DIR
        ${CPM_PACKAGE_OptiX_SOURCE_DIR}
        CACHE PATH "Path to OptiX installation" FORCE)
    set(OptiX_INCLUDE_DIR
        ${OptiX_INSTALL_DIR}/include
        CACHE PATH "Path to OptiX include directory." FORCE)
  else()
    # Search for the OptiX include files.
    find_package(OptiX 8.0 QUIET)
  endif()

  # Requires both CUDA and OptiX
  if(CUDA_FOUND AND OptiX_INCLUDE_DIR)
    message(STATUS "[ViennaCore] Enabling GPU support")
    include("cmake/setCudaFlags.cmake")

    set(VIENNACORE_PTX_DIR
        "${CMAKE_BINARY_DIR}/lib/ptx"
        CACHE STRING "Directory for compiled PTX files.")
    set(CUDA_GENERATED_OUTPUT_DIR
        ${VIENNACORE_PTX_DIR}
        CACHE STRING "Generated PTX output directory" FORCE)
    set(VIENNACORE_GPU_LIBS
        ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY}
        CACHE STRING "ViennaCore CUDA libraries" FORCE)

    target_compile_definitions(${PROJECT_NAME} INTERFACE VIENNACORE_COMPILE_GPU)
    target_include_directories(${PROJECT_NAME} INTERFACE $<BUILD_INTERFACE:${OptiX_INCLUDE_DIR}>
                                                         $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>)
    target_link_libraries(${PROJECT_NAME} INTERFACE ${VIENNACORE_GPU_LIBS})
  else()
    message(FATAL_ERROR "CUDA Toolkit or OptiX not found. Cannot build GPU module.")
  endif()
endif()

# --------------------------------------------------------------------------------------------------------
# CMake Utilities
# --------------------------------------------------------------------------------------------------------

include(utils/setup.cmake)
include(utils/python.cmake)

include(utils/batch.cmake)
include(utils/subdirs.cmake)

include(utils/format.cmake)
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
