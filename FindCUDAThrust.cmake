# - Tools for building CUDA Thrust files
#   For more info on Thrust, see http://code.google.com/p/thrust/
#
# This script locates the Thrust template library files.
# It looks in CUDA_TOOLKIT_ROOT_DIR, CUDA_SDK_ROOT_DIR, CUDA_INCLUDE_DIRS.
# This script relies on the FindCUDA.cmake script being run first;
# for more info on FindCUDA.cmake, see:
#   https://gforge.sci.utah.edu/gf/project/findcuda/scmsvn/
#
# This script makes use of the standard find_package arguments of
# REQUIRED and QUIET.
#
# The script defines the following variables:
#  CUDATHRUST_FOUND        -- Set to TRUE if found; set to FALSE if not found.
#  CUDATHRUST_INCLUDE_DIR  -- Include directory for Thrust headers.
#
#
#  Joseph K. Bradley, Carnegie Mellon University
#      -- http://www.cs.cmu.edu/~jkbradle
#
#  Copyright (c) 2010. Joseph K. Bradley. All rights reserved.
#
# This code is licensed under the MIT License.
# See the FindCUDAThrust.cmake script for the text of the license.
#
# The MIT License
#
# License for the specific language governing rights and limitations under
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
###############################################################################

# FindCUDAThrust.cmake

# CMake must find CUDA before looking for Thrust.
if(CUDA_FOUND)
  # Look for Thrust in CUDA directories.
  find_path(CUDATHRUST_INCLUDE
    thrust/version.h
    PATHS ${CUDA_INCLUDE_DIRS} ${CUDA_TOOLKIT_ROOT_DIR} ${CUDA_SDK_ROOT_DIR}
    NO_DEFAULT_PATH
    )
  # Look for Thrust in default search paths.
  find_path(CUDATHRUST_INCLUDE thrust/version.h)
  mark_as_advanced(CUDATHRUST_INCLUDE)
  if(CUDATHRUST_INCLUDE)
    # Thrust was found.
    message(STATUS "CUDA Thrust found: " ${CUDATHRUST_INCLUDE})
    set(CUDATHRUST_FOUND TRUE)
    set (CUDATHRUST_INCLUDE_DIRS ${CUDATHRUST_INCLUDE})
  else(CUDATHRUST_INCLUDE)
    # Thrust was not found.
    set(CUDATHRUST_FOUND FALSE)
    if(CUDATHRUST_FIND_REQUIRED)
      message(FATAL_ERROR "CUDA Thrust not found!")
    else(CUDATHRUST_FIND_REQUIRED)
      if (NOT CUDATHRUST_FIND_QUIETLY)
        message(STATUS "CUDA Thrust not found")
      endif(NOT CUDATHRUST_FIND_QUIETLY)
    endif(CUDATHRUST_FIND_REQUIRED)
  endif(CUDATHRUST_INCLUDE)
else(CUDA_FOUND)
  if(NOT CUDATHRUST_FIND_QUIETLY)
    message(STATUS "CUDA must be found before CMake looks for Thrust!")
  endif(NOT CUDATHRUST_FIND_QUIETLY)
  set(CUDATHRUST_FOUND FALSE)
endif(CUDA_FOUND) 
