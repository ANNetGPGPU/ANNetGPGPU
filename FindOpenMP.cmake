# - Finds OpenMP support
# This module can be used to detect OpenMP support in a compiler.
# If the compiler supports OpenMP, the flags required to compile with
# openmp support are set.  
#
# The following variables are set:
#   OpenMP_C_FLAGS - flags to add to the C compiler for OpenMP support
#   OpenMP_CXX_FLAGS - flags to add to the CXX compiler for OpenMP support
#   OPENMP_FOUND - true if openmp is detected
#
# Supported compilers can be found at http://openmp.org/wp/openmp-compilers/

#=============================================================================
# CMake - Cross Platform Makefile Generator
# Copyright 2000-2009 Kitware, Inc., Insight Software Consortium
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
#
# * Neither the names of Kitware, Inc., the Insight Software Consortium,
#   nor the names of their contributors may be used to endorse or promote
#   products derived from this software without specific prior written
#   permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#=============================================================================

# Modified in RSTM from original source to provide CXX-tm flags

set(OpenMP_C_FLAG_CANDIDATES
  #Gnu
  "-fopenmp"
  #Microsoft Visual Studio
  "/openmp"
  #Intel windows
  "-Qopenmp" 
  #Intel
  "-openmp" 
  #Empty, if compiler automatically accepts openmp
  " "
  #Sun
  "-xopenmp"
  #HP
  "+Oopenmp"
  #IBM XL C/c++
  "-qsmp"
  #Portland Group
  "-mp"
)
set(OpenMP_CXX_FLAG_CANDIDATES ${OpenMP_C_FLAG_CANDIDATES})
set(OpenMP_CXX-tm_FLAG_CANDIDATES ${OpenMP_CXX_FLAG_CANDIDATES})

# sample openmp source code to test
set(OpenMP_C_TEST_SOURCE 
"
#include <omp.h>
int main() { 
#ifdef _OPENMP
  return 0; 
#else
  breaks_on_purpose
#endif
}
")
# use the same source for CXX as C for now
set(OpenMP_CXX_TEST_SOURCE ${OpenMP_C_TEST_SOURCE})
set(OpenMP_CXX-tm_TEST_SOURCE ${OpenMP_CXX_TEST_SOURCE})
# if these are set then do not try to find them again,
# by avoiding any try_compiles for the flags
if(DEFINED OpenMP_C_FLAGS AND DEFINED OpenMP_CXX_FLAGS AND DEFINED OpenMP_CXX-tm_FLAGS)
  set(OpenMP_C_FLAG_CANDIDATES)
  set(OpenMP_CXX_FLAG_CANDIDATES)
  set(OpenMP_CXX-tm_FLAG_CANDIDATES)
endif()

# check c compiler
IF (CMAKE_C_COMPILER)
  include (CheckCSourceCompiles)
  foreach(FLAG ${OpenMP_C_FLAG_CANDIDATES})
    set(SAFE_CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS}")
    set(CMAKE_REQUIRED_FLAGS "${FLAG}")
    unset(OpenMP_FLAG_DETECTED CACHE)
    message(STATUS "Try OpenMP C flag = [${FLAG}]")
    check_c_source_compiles("${OpenMP_C_TEST_SOURCE}" OpenMP_FLAG_DETECTED)
    set(CMAKE_REQUIRED_FLAGS "${SAFE_CMAKE_REQUIRED_FLAGS}")
    if(OpenMP_FLAG_DETECTED)
      set(OpenMP_C_FLAGS_INTERNAL "${FLAG}")
      break()
    endif(OpenMP_FLAG_DETECTED) 
  endforeach(FLAG ${OpenMP_C_FLAG_CANDIDATES})
ENDIF (CMAKE_C_COMPILER)

# check cxx compiler
IF (CMAKE_CXX_COMPILER)
  include (CheckCXXSourceCompiles)
  foreach(FLAG ${OpenMP_CXX_FLAG_CANDIDATES})
    set(SAFE_CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS}")
    set(CMAKE_REQUIRED_FLAGS "${FLAG}")
    unset(OpenMP_FLAG_DETECTED CACHE)
    message(STATUS "Try OpenMP CXX flag = [${FLAG}]")
    check_cxx_source_compiles("${OpenMP_CXX_TEST_SOURCE}" OpenMP_FLAG_DETECTED)
    set(CMAKE_REQUIRED_FLAGS "${SAFE_CMAKE_REQUIRED_FLAGS}")
    if(OpenMP_FLAG_DETECTED)
      set(OpenMP_CXX_FLAGS_INTERNAL "${FLAG}")
      break()
    endif(OpenMP_FLAG_DETECTED)
  endforeach(FLAG ${OpenMP_CXX_FLAG_CANDIDATES})
ENDIF (CMAKE_CXX_COMPILER)

# check cxxtm compiler
IF (CMAKE_CXX-tm_COMPILER)
  include (CheckCXX-tmSourceCompiles)
  foreach(FLAG ${OpenMP_CXX-tm_FLAG_CANDIDATES})
    set(SAFE_CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS}")
    set(CMAKE_REQUIRED_FLAGS "${FLAG}")
    unset(OpenMP_FLAG_DETECTED CACHE)
    message(STATUS "Try OpenMP CXX-tm flag = [${FLAG}]")
    check_cxxtm_source_compiles("${OpenMP_CXX-tm_TEST_SOURCE}" OpenMP_FLAG_DETECTED)
    set(CMAKE_REQUIRED_FLAGS "${SAFE_CMAKE_REQUIRED_FLAGS}")
    if(OpenMP_FLAG_DETECTED)
      set(OpenMP_CXX-tm_FLAGS_INTERNAL "${FLAG}")
      break()
    endif()
  endforeach()
ENDIF ()

include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)
macro (OpenMP_complete_lang_flags lang langstring)
  IF (CMAKE_${lang}_COMPILER)
    set(OpenMP_${lang}_FLAGS "${OpenMP_${lang}_FLAGS_INTERNAL}"
      CACHE STRING "${langstring} compiler flags for OpenMP parallization")
    find_package_handle_standard_args(OpenMP DEFAULT_MSG 
      OpenMP_${lang}_FLAGS )
    mark_as_advanced(OpenMP_${lang}_FLAGS)
  ENDIF ()
endmacro ()

OpenMP_complete_lang_flags (C "C")
OpenMP_complete_lang_flags (CXX "C++")
OpenMP_complete_lang_flags (CXX-tm "C++ TM")
