# The name of our project is "ANNetGPGPU". CMakeLists files in this project can
# refer to the root source directory of the project as ${ANNetGPU_SOURCE_DIR} and
# to the root binary directory of the project as ${ANNetGPU_SOURCE_DIR}.
PROJECT (ANNetGPGPU)

option(VERBOSE "VERBOSE" ON)

# Make sure the compiler can find include files from our ANNet library.
INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR})
INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR}/math)
INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR}/containers)
INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR}/gpgpu)
INCLUDE (FindOpenMP)

IF (CUDATHRUST_FOUND)
  INCLUDE_DIRECTORIES (${CUDA_SDK_ROOT_DIR}/C/common/inc/)
  INCLUDE_DIRECTORIES (${CUDATHRUST_INCLUDE_DIR})
ENDIF (CUDATHRUST_FOUND)

SET( ANCUDASourceFiles
  SOMNetGPU.cu
)

IF(OPENMP_FOUND)
  MESSAGE(STATUS "\t- OpenMP found. We will link the library against -lgomp")

  add_compile_options(${OpenMP_CXX_FLAGS})
  IF(VERBOSE)
    add_compile_options(-DVERBOSE)
  ENDIF(VERBOSE)
  SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
ENDIF()

IF (CUDA_FOUND)
  MESSAGE(STATUS "\t- CUDA found. We will build the library with GPGPU support.")

  if ( CMAKE_COMPILER_IS_GNUCC )
    MESSAGE(INFO "\tImportant: ENSURE using a supported GCC version with CUDA")
  endif()
  
  IF(VERBOSE)
    LIST(APPEND CUDA_NVCC_FLAGS "-DVERBOSE")
  ENDIF(VERBOSE)
  
  ADD_DEFINITIONS("-DCUDA") # needed for conditional compilation of some files
  CUDA_ADD_LIBRARY (ANNet ${ANCUDASourceFiles} ${BZIP_INCLUDE_DIRS} OPTIONS -D __CUDA_CAB__=${__CUDA_CAB__})
  TARGET_LINK_LIBRARIES (ANNet ${BZIP2_LIBRARIES})
  
  # Link OpenMP to ANNet library
  IF(OPENMP_FOUND)
    TARGET_LINK_LIBRARIES (ANNet -lgomp)
  ENDIF(OPENMP_FOUND)
ENDIF(CUDA_FOUND)

# Build python interface
SET( ANPythonCPUInterfaceFiles PyNetCPU.i )
SET( ANPythonGPUInterfaceFiles PyNetGPU.i )

IF (SWIG_FOUND AND PYTHONLIBS_FOUND)
  MESSAGE(STATUS "\t-SWIG found. We will a python interface.")
  
  INCLUDE(${SWIG_USE_FILE})
  INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})
  
  SET_SOURCE_FILES_PROPERTIES(${ANPythonCPUInterfaceFiles} PROPERTIES CPLUSPLUS ON)
  SET_SOURCE_FILES_PROPERTIES(${ANPythonCPUInterfaceFiles} PROPERTIES CMAKE_SWIG_FLAGS "-includeall")
  SWIG_ADD_MODULE(ANPyNetCPU python ${ANPythonCPUInterfaceFiles})
  SWIG_LINK_LIBRARIES(ANPyNetCPU ${PYTHON_LIBRARIES} ${BZIP2_LIBRARIES} -lgomp)

  IF (CUDA_FOUND)
    SET_SOURCE_FILES_PROPERTIES(${ANPythonGPUInterfaceFiles} PROPERTIES CPLUSPLUS ON)
    SET_SOURCE_FILES_PROPERTIES(${ANPythonGPUInterfaceFiles} PROPERTIES CMAKE_SWIG_FLAGS "-includeall")
    SWIG_ADD_MODULE(ANPyNetGPU python ${ANPythonGPUInterfaceFiles})
    SWIG_LINK_LIBRARIES(ANPyNetGPU ${PYTHON_LIBRARIES} ANNet)
  ENDIF(CUDA_FOUND)
ENDIF(SWIG_FOUND AND PYTHONLIBS_FOUND)

#demos
ADD_SUBDIRECTORY(${ANNetGPGPU_SOURCE_DIR}/examples)
