#  James Bigler, NVIDIA Corp (nvidia.com - jbigler)
#
#  Copyright (c) 2008 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
#  This code is licensed under the MIT License.  See the FindCUDA.cmake script
#  for the text of the license.

# The MIT License
#
# SPDX-FileCopyrightText: Copyright (c) 2008 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

##########################################################################
# This file runs the nvcc commands to produce the desired output file along with
# the dependency file needed by CMake to compute dependencies.  In addition the
# file checks the output of each command and if the command fails it deletes the
# output files.

# Input variables
#
# verbose:BOOL=<>          OFF: Be as quiet as possible (default)
#                          ON : Describe each step
#
# build_configuration:STRING=<> Typically one of Debug, MinSizeRel, Release, or
#                               RelWithDebInfo, but it should match one of the
#                               entries in CUDA_HOST_FLAGS. This is the build
#                               configuration used when compiling the code.  If
#                               blank or unspecified Debug is assumed as this is
#                               what CMake does.
#
# generated_file:STRING=<> File to generate.  This argument must be passed in.
#
# generated_cubin_file:STRING=<> File to generate.  This argument must be passed
#                                                   in if build_cubin is true.
# generate_dependency_only:BOOL=<> Only generate the dependency file.
#
# check_dependencies:BOOL=<> Check the dependencies.  If everything is up to
#                            date, simply touch the output file instead of
#                            generating it.

# Support IN_LIST
cmake_policy(SET CMP0057 NEW)

if(NOT generated_file)
  message(FATAL_ERROR "You must specify generated_file on the command line")
endif()

# Set these up as variables to make reading the generated file easier
set(CMAKE_COMMAND "@CMAKE_COMMAND@") # path
set(source_file "@source_file@") # path
set(NVCC_generated_dependency_file "@NVCC_generated_dependency_file@") # path
set(cmake_dependency_file "@cmake_dependency_file@") # path
set(CUDA_make2cmake "@CUDA_make2cmake@") # path
set(CUDA_parse_cubin "@CUDA_parse_cubin@") # path
set(build_cubin @build_cubin@) # bool
set(CUDA_HOST_COMPILER "@CUDA_HOST_COMPILER@") # path
# We won't actually use these variables for now, but we need to set this, in
# order to force this file to be run again if it changes.
set(generated_file_path "@generated_file_path@") # path
set(generated_file_internal "@generated_file@") # path
set(generated_cubin_file_internal "@generated_cubin_file@") # path

set(CUDA_REMOVE_GLOBAL_MEMORY_SPACE_WARNING @CUDA_REMOVE_GLOBAL_MEMORY_SPACE_WARNING@)

set(CUDA_NVCC_EXECUTABLE "@CUDA_NVCC_EXECUTABLE@") # path
set(CUDA_NVCC_FLAGS @CUDA_NVCC_FLAGS@ ;; @CUDA_WRAP_OPTION_NVCC_FLAGS@) # list
@CUDA_NVCC_FLAGS_CONFIG@
set(nvcc_flags @nvcc_flags@) # list
set(CUDA_NVCC_INCLUDE_ARGS "@CUDA_NVCC_INCLUDE_ARGS@"
)# list (needs to be in quotes to handle spaces properly).
set(format_flag "@format_flag@") # string
set(cuda_language_flag @cuda_language_flag@) # list

if(build_cubin AND NOT generated_cubin_file)
  message(FATAL_ERROR "You must specify generated_cubin_file on the command line")
endif()

# This is the list of host compilation flags.  It C or CXX should already have
# been chosen by FindCUDA.cmake.
@CUDA_HOST_FLAGS@

# Take the compiler flags and package them up to be sent to the compiler via -Xcompiler
set(nvcc_host_compiler_flags "")
# If we weren't given a build_configuration, use Debug.
if(NOT build_configuration)
  set(build_configuration Debug)
endif()
string(TOUPPER "${build_configuration}" build_configuration)
#message("CUDA_NVCC_HOST_COMPILER_FLAGS = ${CUDA_NVCC_HOST_COMPILER_FLAGS}")
foreach(flag ${CMAKE_HOST_FLAGS} ${CMAKE_HOST_FLAGS_${build_configuration}})
  # Extra quotes are added around each flag to help nvcc parse out flags with spaces.
  if("${nvcc_host_compiler_flags}" STREQUAL "")
    set(nvcc_host_compiler_flags "\"${flag}\"")
  else()
    set(nvcc_host_compiler_flags "${nvcc_host_compiler_flags},\"${flag}\"")
  endif()
endforeach()
if(nvcc_host_compiler_flags)
  set(nvcc_host_compiler_flags "-Xcompiler" ${nvcc_host_compiler_flags})
endif()
#message("nvcc_host_compiler_flags = \"${nvcc_host_compiler_flags}\"")

set(depends_nvcc_host_compiler_flags "")
foreach(flag ${CMAKE_HOST_FLAGS})
  # Extra quotes are added around each flag to help nvcc parse out flags with spaces.
  if("${depends_nvcc_host_compiler_flags}" STREQUAL "")
    set(depends_nvcc_host_compiler_flags "\"${flag}\"")
  else()
    set(depends_nvcc_host_compiler_flags "${depends_nvcc_host_compiler_flags},\"${flag}\"")
  endif()
endforeach()
if(depends_nvcc_host_compiler_flags)
  set(depends_nvcc_host_compiler_flags "-Xcompiler" ${depends_nvcc_host_compiler_flags})
endif()

list(APPEND depends_CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})
# Add the build specific configuration flags
list(APPEND CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS_${build_configuration}})

# Any -ccbin existing in CUDA_NVCC_FLAGS gets highest priority
list(FIND CUDA_NVCC_FLAGS "-ccbin" ccbin_found0)
list(FIND CUDA_NVCC_FLAGS "--compiler-bindir" ccbin_found1)
if(ccbin_found0 LESS 0
   AND ccbin_found1 LESS 0
   AND CUDA_HOST_COMPILER)
  if(CUDA_HOST_COMPILER STREQUAL "@_CUDA_MSVC_HOST_COMPILER@" AND DEFINED CCBIN)
    set(CCBIN -ccbin "${CCBIN}")
  else()
    set(CCBIN -ccbin "${CUDA_HOST_COMPILER}")
  endif()
endif()

# cuda_execute_process - Executes a command with optional command echo and status message.
#
#   status  - Status message to print if verbose is true
#   command - COMMAND argument from the usual execute_process argument structure
#   ARGN    - Remaining arguments are the command with arguments
#
#   CUDA_result - return value from running the command
#
# Make this a macro instead of a function, so that things like RESULT_VARIABLE
# and other return variables are present after executing the process.
macro(cuda_execute_process status command)
  set(_command ${command})
  if(NOT "x${_command}" STREQUAL "xCOMMAND")
    message(
      FATAL_ERROR
        "Malformed call to cuda_execute_process.  Missing COMMAND as second argument. (command = ${command})"
    )
  endif()
  # ARGN isn't like a normal variable in macros, so use a proxy variable that we can use instead.
  set(_arguments ${ARGN})
  # nvcc warns when specifying -G and --lineinfo, which is annoying.
  if("-G" IN_LIST _arguments)
    list(REMOVE_ITEM _arguments "-lineinfo")
    list(REMOVE_ITEM _arguments "--lineinfo")
  endif()
  if(verbose)
    execute_process(COMMAND "${CMAKE_COMMAND}" -E echo -- ${status})
    # Now we need to build up our command string.  We are accounting for quotes
    # and spaces, anything else is left up to the user to fix if they want to
    # copy and paste a runnable command line.
    set(cuda_execute_process_string)
    foreach(arg ${_arguments})
      # If there are quotes, excape them, so they come through.
      string(REPLACE "\"" "\\\"" arg ${arg})
      # Args with spaces need quotes around them to get them to be parsed as a single argument.
      if(arg MATCHES " ")
        list(APPEND cuda_execute_process_string "\"${arg}\"")
      else()
        list(APPEND cuda_execute_process_string ${arg})
      endif()
    endforeach()
    # Echo the command
    execute_process(COMMAND ${CMAKE_COMMAND} -E echo ${cuda_execute_process_string})
  endif()
  # Run the command
  execute_process(COMMAND ${_arguments} RESULT_VARIABLE CUDA_result)
endmacro()

# For CUDA 2.3 and below, -G -M doesn't work, so remove the -G flag
# for dependency generation and hope for the best.
set(CUDA_VERSION @CUDA_VERSION@)
if(CUDA_VERSION VERSION_LESS "3.0")
  cmake_policy(PUSH)
  # CMake policy 0007 NEW states that empty list elements are not
  # ignored.  I'm just setting it to avoid the warning that's printed.
  cmake_policy(SET CMP0007 NEW)
  # Note that this will remove all occurances of -G.
  list(REMOVE_ITEM depends_CUDA_NVCC_FLAGS "-G")
  cmake_policy(POP)
endif()

# If we need to create relocatible code, the dependency phase doesn't like this argument.
# We need to filter it out here.
list(REMOVE_ITEM depends_CUDA_NVCC_FLAGS "-dc")
list(REMOVE_ITEM depends_CUDA_NVCC_FLAGS "--device-c")

# nvcc doesn't define __CUDACC__ for some reason when generating dependency files.  This
# can cause incorrect dependencies when #including files based on this macro which is
# defined in the generating passes of nvcc invokation.  We will go ahead and manually
# define this for now until a future version fixes this bug.
set(CUDACC_DEFINE -D__CUDACC__)

if(check_dependencies)
  set(rebuild FALSE)
  include(${cmake_dependency_file})
  if(NOT CUDA_NVCC_DEPEND)
    # CUDA_NVCC_DEPEND should have something useful in it by now.  If not we
    # should force the rebuild.
    if(verbose)
      message(WARNING "CUDA_NVCC_DEPEND not found for ${generated_file}")
    endif()
    set(rebuild TRUE)
  endif()
  # Rebuilding is also dependent on this file changing.
  list(APPEND CUDA_NVCC_DEPEND "${CMAKE_CURRENT_LIST_FILE}")
  foreach(f ${CUDA_NVCC_DEPEND})
    # True if file1 is newer than file2 or if one of the two files doesn't
    # exist. Behavior is well-defined only for full paths. If the file time
    # stamps are exactly the same, an IS_NEWER_THAN comparison returns true, so
    # that any dependent build operations will occur in the event of a tie. This
    # includes the case of passing the same file name for both file1 and file2.
    if("${f}" IS_NEWER_THAN "${generated_file}")
      #message("file ${f} is newer than ${generated_file}")
      set(rebuild TRUE)
    endif()
  endforeach()
  if(NOT rebuild)
    #message("Not rebuilding ${generated_file}")
    cuda_execute_process("Dependencies up to date.  Not rebuilding ${generated_file}" COMMAND
                         "${CMAKE_COMMAND}" -E touch "${generated_file}")
    return()
  endif()
endif()

# Generate the dependency file
cuda_execute_process(
  "Generating dependency file: ${NVCC_generated_dependency_file}"
  COMMAND
  "${CUDA_NVCC_EXECUTABLE}"
  -M
  ${CUDACC_DEFINE}
  "${source_file}"
  -o
  "${NVCC_generated_dependency_file}"
  ${CCBIN}
  ${nvcc_flags}
  ${depends_nvcc_host_compiler_flags}
  ${depends_CUDA_NVCC_FLAGS}
  -DNVCC
  ${CUDA_NVCC_INCLUDE_ARGS})

if(CUDA_result)
  message(FATAL_ERROR "Error generating ${generated_file}")
endif()

# Generate the cmake readable dependency file to a temp file.  Don't put the
# quotes just around the filenames for the input_file and output_file variables.
# CMake will pass the quotes through and not be able to find the file.
cuda_execute_process(
  "Generating temporary cmake readable file: ${cmake_dependency_file}.tmp"
  COMMAND
  "${CMAKE_COMMAND}"
  -D
  "input_file:FILEPATH=${NVCC_generated_dependency_file}"
  -D
  "output_file:FILEPATH=${cmake_dependency_file}.tmp"
  -P
  "${CUDA_make2cmake}")

if(CUDA_result)
  message(FATAL_ERROR "Error generating ${generated_file}")
endif()

# Copy the file if it is different
cuda_execute_process(
  "Copy if different ${cmake_dependency_file}.tmp to ${cmake_dependency_file}"
  COMMAND
  "${CMAKE_COMMAND}"
  -E
  copy_if_different
  "${cmake_dependency_file}.tmp"
  "${cmake_dependency_file}")

if(CUDA_result)
  message(FATAL_ERROR "Error generating ${generated_file}")
endif()

# Delete the temporary file
cuda_execute_process("Removing ${cmake_dependency_file}.tmp" COMMAND "${CMAKE_COMMAND}" -E remove
                     "${cmake_dependency_file}.tmp")

if(CUDA_result)
  message(FATAL_ERROR "Error generating ${generated_file}")
endif()

if(generate_dependency_only)
  return()
endif()

if(CUDA_REMOVE_GLOBAL_MEMORY_SPACE_WARNING)
  set(get_error ERROR_VARIABLE stderr)
endif()

# Delete the target file
cuda_execute_process("Removing ${generated_file}" COMMAND "${CMAKE_COMMAND}" -E remove
                     "${generated_file}")

# Generate the code
cuda_execute_process(
  "Generating ${generated_file}"
  COMMAND
  "${CUDA_NVCC_EXECUTABLE}"
  "${source_file}"
  ${cuda_language_flag}
  ${format_flag}
  -o
  "${generated_file}"
  ${CCBIN}
  ${nvcc_flags}
  ${nvcc_host_compiler_flags}
  ${CUDA_NVCC_FLAGS}
  -DNVCC
  ${CUDA_NVCC_INCLUDE_ARGS}
  ${get_error})

if(get_error)
  if(stderr)
    # Filter out the annoying Advisory about pointer stuff.
    # Advisory: Cannot tell what pointer points to, assuming global memory space
    string(
      REGEX
      REPLACE
        "(^|\n)[^\n]*\(Advisory|Warning\): Cannot tell what pointer points to, assuming global memory space\n\n"
        "\\1"
        stderr
        "${stderr}")

    # Filter out warning we do not care about
    string(
      REGEX
      REPLACE
        "(^|\n)[^\n]*: Warning: Function [^\n]* has a large return size, so overriding noinline attribute. The function may be inlined when called.\n\n"
        "\\1"
        stderr
        "${stderr}")

    # To be investigated (OP-1999)
    string(
      REGEX
      REPLACE
        "(^|\n)[^\n]*: warning: function [^\n]*\n[^\n]*: here was declared deprecated \(.[^\n]* is not valid on compute_70 and above, and should be replaced with [^\n]*.To continue using [^\n]*, specify virtual architecture compute_60 when targeting sm_70 and above, for example, using the pair of compiler options.[^\n]*..\)\n( *detected during instantiation of [^\n]*\n[^\n]*: here\n)?( *detected during:\n( *instantiation of [^\n]*\n[^\n]*: here\n([^\n]*instantiation contexts not shown[^\n]*\n)?)+)?\n"
        "\\1"
        stderr
        "${stderr}")
    string(
      REGEX
      REPLACE
        "(^|\n)[^\n]*: warning: function [^\n]*\n[^\n]*: here was declared deprecated \(.[^\n]* is deprecated in favor of [^\n]* and may be removed in a future release [^\n]*\)\n( *detected during instantiation of [^\n]*\n[^\n]*: here\n)?( *detected during:\n( *instantiation of [^\n]*\n[^\n]*: here\n([^\n]*instantiation contexts not shown[^\n]*\n)?)+)?\n"
        "\\1"
        stderr
        "${stderr}")

    # If there is error output, there is usually a stray newline at the end. Eliminate it if it is the only content of ${stderr}.
    string(REGEX REPLACE "^\n$" "" stderr "${stderr}")

    if(stderr)
      message("${stderr}")
    endif()
  endif()
endif()

if(CUDA_result)
  # Since nvcc can sometimes leave half done files make sure that we delete the output file.
  cuda_execute_process("Removing ${generated_file}" COMMAND "${CMAKE_COMMAND}" -E remove
                       "${generated_file}")
  message(FATAL_ERROR "Error generating file ${generated_file}")
else()
  if(verbose)
    message("Generated ${generated_file} successfully.")
  endif()
endif()

# Cubin resource report commands.
if(build_cubin)
  # Run with -cubin to produce resource usage report.
  cuda_execute_process(
    "Generating ${generated_cubin_file}"
    COMMAND
    "${CUDA_NVCC_EXECUTABLE}"
    "${source_file}"
    ${CUDA_NVCC_FLAGS}
    ${nvcc_flags}
    ${CCBIN}
    ${nvcc_host_compiler_flags}
    -DNVCC
    -cubin
    -o
    "${generated_cubin_file}"
    ${CUDA_NVCC_INCLUDE_ARGS})

  # Execute the parser script.
  cuda_execute_process(
    "Executing the parser script"
    COMMAND
    "${CMAKE_COMMAND}"
    -D
    "input_file:STRING=${generated_cubin_file}"
    -P
    "${CUDA_parse_cubin}")

endif()
