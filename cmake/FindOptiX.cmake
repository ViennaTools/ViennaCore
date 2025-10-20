#
# SPDX-FileCopyrightText: Copyright (c) 2010 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

if(OptiX_FOUND)
  # Already found, so just return.
  return()
endif()

# The distribution contains only 64 bit libraries.  Error when we have been mis-configured.
if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  if(WIN32)
    message(SEND_ERROR "Make sure when selecting the generator, you select one with Win64 or x64.")
  endif()
  message(STATUS "OptiX only supports builds configured for 64 bits.")
  return()
endif()

# search path based on the bit-ness of the build.  (i.e. 64: bin64, lib64; 32:
# bin, lib).  Note that on Mac, the OptiX library is a universal binary, so we
# only need to look in lib and not lib64 for 64 bit builds.
if(NOT APPLE)
  set(bit_dest "64")
else()
  set(bit_dest "")
endif()

if(NOT OptiX_FIND_VERSION)
  set(OptiX_FIND_VERSION "*")
endif()

if("${is_cached}" STREQUAL "" AND DEFINED OptiX_INSTALL_DIR)
  # Required for windows to convert backslashes to forward slashes
  file(TO_CMAKE_PATH "${OptiX_INSTALL_DIR}" OptiX_INSTALL_DIR)
  set(OptiX_INSTALL_DIR
      "${OptiX_INSTALL_DIR}"
      CACHE PATH "Path to OptiX installation" FORCE)
else()
  set(OptiX_INSTALL_DIR
      $ENV{OptiX_INSTALL_DIR}
      CACHE PATH "Path to OptiX installation.")
endif()

if("${OptiX_INSTALL_DIR}" STREQUAL "")
  if(CMAKE_HOST_WIN32)
    # This is the default OptiX SDK install location on Windows.
    file(GLOB OptiX_INSTALL_DIR
         "$ENV{ProgramData}/NVIDIA Corporation/OptiX SDK ${OptiX_FIND_VERSION}*")
  else()
    # On linux, there is no default install location for the SDK, but it does have a default subdir name.
    foreach(dir "/opt" "/usr/local" "$ENV{HOME}" "$ENV{HOME}/Downloads")
      file(GLOB OptiX_INSTALL_DIR "${dir}/NVIDIA-OptiX-SDK-${OptiX_FIND_VERSION}*")
      if(OptiX_INSTALL_DIR)
        break()
      endif()
    endforeach()
  endif()
endif()

# Include
if(OptiX_FIND_QUIETLY)
  find_path(
    OptiX_ROOT_DIR
    NAMES include/optix.h
    PATHS ${OptiX_INSTALL_DIR} QUIET)
else()
  find_path(
    OptiX_ROOT_DIR
    NAMES include/optix.h
    PATHS ${OptiX_INSTALL_DIR})
endif()

if(NOT OptiX_ROOT_DIR AND OptiX_FIND_REQUIRED)
  message(
    FATAL_ERROR
      "OptiX installation not found. Please set CMAKE_PREFIX_PATH or OptiX_INSTALL_DIR to locate 'include/optix.h'."
  )
endif()

if(OptiX_ROOT_DIR)
  file(READ "${OptiX_ROOT_DIR}/include/optix.h" header)

  # Extract version parts
  string(REGEX REPLACE "^.*OPTIX_VERSION ([0-9]+)([0-9][0-9])([0-9][0-9])[^0-9].*$" "\\1;\\2;\\3"
                       OPTIX_VERSION_LIST "${header}")

  # Split into list
  list(GET OPTIX_VERSION_LIST 0 OPTIX_VERSION_MAJOR)
  list(GET OPTIX_VERSION_LIST 1 OPTIX_VERSION_MINOR)
  list(GET OPTIX_VERSION_LIST 2 OPTIX_VERSION_PATCH)

  # Strip leading zeros
  math(EXPR OPTIX_VERSION_MINOR "${OPTIX_VERSION_MINOR}")
  math(EXPR OPTIX_VERSION_PATCH "${OPTIX_VERSION_PATCH}")

  # Build normalized version string
  set(OPTIX_VERSION "${OPTIX_VERSION_MAJOR}.${OPTIX_VERSION_MINOR}.${OPTIX_VERSION_PATCH}")

  set(OptiX_INCLUDE_DIR
      ${OptiX_ROOT_DIR}/include
      CACHE PATH "Path to OptiX include directory." FORCE)
endif()
