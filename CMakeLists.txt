cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(
  ViennaCore
  LANGUAGES CXX
  VERSION 1.9.0)

# --------------------------------------------------------------------------------------------------------
# Library Options
# --------------------------------------------------------------------------------------------------------

option(VIENNACORE_USE_OPENMP "Enable OpenMP parallelization" ON)
option(VIENNACORE_USE_GPU "Enable NVIDIA GPU support" OFF)
option(VIENNACORE_CUDA_LOG_DEBUG "Enable CUDA debug logging" OFF)

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

if(MSVC)
  # https://learn.microsoft.com/cpp/c-runtime-library/math-constants TODO: In case C++20 is adopted
  # any time soon: https://cppreference.com/w/cpp/numeric/constants
  target_compile_definitions(${PROJECT_NAME} INTERFACE _USE_MATH_DEFINES=1 NOMINMAX
                                                       WIN32_LEAN_AND_MEAN)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -openmp:llvm /bigobj")
endif()

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

set(VIENNACORE_DEPENDENCIES "")
if(VIENNACORE_USE_OPENMP)
  find_package(OpenMP REQUIRED)
  target_link_libraries(${PROJECT_NAME} INTERFACE OpenMP::OpenMP_CXX)
  list(APPEND VIENNACORE_DEPENDENCIES OpenMP)
endif()

# --------------------------------------------------------------------------------------------------------
# Setup GPU
# --------------------------------------------------------------------------------------------------------

if(VIENNACORE_USE_GPU)

  # Find the CUDA Toolkit
  find_package(CUDAToolkit QUIET)

  if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    list(APPEND VIENNACORE_DEPENDENCIES CUDAToolkit)
  else()
    message(FATAL_ERROR "CUDA Toolkit not found. Cannot build GPU module.")
  endif()

  # Find OptiX
  find_package(OptiX 8.0.0 QUIET)

  if(NOT OptiX_INCLUDE_DIR)
    # If OptiX is not found, download it using CPM
    CPMAddPackage(
      NAME OptiX
      VERSION 8.0.0
      GIT_REPOSITORY "https://github.com/NVIDIA/optix-dev")
    set(OptiX_INCLUDE_DIR
        ${CPM_PACKAGE_OptiX_SOURCE_DIR}/include
        CACHE PATH "Path to OptiX include directory." FORCE)
  endif()

  # CUDA PTX compilation setup
  set(VIENNACORE_NVCC_FLAGS
      "-use_fast_math" "-expt-relaxed-constexpr" "-rdc=true" "-lineinfo" "-diag-suppress=20044"
      CACHE STRING "NVCC flags for NVCC compilation")

  include(utils/CUDAUtil.cmake)
  viennacore_detect_native_sm()
  if(DEFINED VIENNACORE_NATIVE_SM)
    set(VIENNACORE_CUDA_ARCH
        ${VIENNACORE_NATIVE_SM}
        CACHE STRING "CUDA architecture to compile for")
  else()
    set(VIENNACORE_CUDA_ARCH
        "60"
        CACHE STRING "CUDA architecture to compile for")
  endif()

  set(VIENNACORE_NVCC_INCLUDE_DIRS
      ${CMAKE_CURRENT_SOURCE_DIR}/include/viennacore ${OptiX_INCLUDE_DIR}
      CACHE STRING "Include directories for NVCC compilation")

  set(VIENNACORE_NVCC_DEFINES
      "_USE_MATH_DEFINES=1" "VIENNACORE_COMPILE_GPU=1"
      CACHE STRING "Preprocessor definitions for NVCC compilation")

  set(VIENNACORE_NVCC_PTX_DIR
      "${CMAKE_BINARY_DIR}/lib/ptx"
      CACHE STRING "Directory for compiled PTX/optixir files.")

  # Handle OptiX include directories (build-only if from CPM, otherwise installed)
  if(OptiX_INCLUDE_DIR MATCHES "^${CMAKE_BINARY_DIR}")
    # OptiX downloaded via CPM: install headers and use install interface
    install(DIRECTORY ${OptiX_INCLUDE_DIR}/ DESTINATION include/viennacore-${PROJECT_VERSION}/optix)
    target_include_directories(${PROJECT_NAME} INTERFACE
      $<BUILD_INTERFACE:${OptiX_INCLUDE_DIR}>
      $<INSTALL_INTERFACE:include/viennacore-${PROJECT_VERSION}/optix>)
  else()
    # OptiX found via system or OptiX_INSTALL_DIR: use as-is for both interfaces
    target_include_directories(${PROJECT_NAME} INTERFACE ${OptiX_INCLUDE_DIR})
  endif()
  target_link_libraries(${PROJECT_NAME} INTERFACE CUDA::cuda_driver CUDA::cudart)
  target_compile_definitions(${PROJECT_NAME} INTERFACE VIENNACORE_COMPILE_GPU=1)
  target_compile_definitions(${PROJECT_NAME} INTERFACE
    $<BUILD_INTERFACE:VIENNACORE_KERNELS_PATH_DEFINE=${VIENNACORE_NVCC_PTX_DIR}>
    $<INSTALL_INTERFACE:VIENNACORE_KERNELS_PATH_DEFINE=${CMAKE_INSTALL_PREFIX}/lib/ptx>)
  if(VIENNACORE_CUDA_LOG_DEBUG)
    target_compile_definitions(${PROJECT_NAME} INTERFACE VIENNACORE_CUDA_LOG_DEBUG=1)
  endif()

  # CUDA and OptiX should now be found and configured
  message(STATUS "[ViennaCore] Enabling GPU support")
  include(utils/compilePTX.cmake)
  include(utils/compileOptixir.cmake)
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
  COMPATIBILITY SameMajorVersion
  DEPENDENCIES ${VIENNACORE_DEPENDENCIES})
