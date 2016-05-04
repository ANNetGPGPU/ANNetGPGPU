# The name of our project is "ANNetGPGPU". CMakeLists files in this project can
# refer to the root source directory of the project as ${ANNetGPU_SOURCE_DIR} and
# to the root binary directory of the project as ${ANNetGPU_SOURCE_DIR}.
CMAKE_MINIMUM_REQUIRED (VERSION 2.8)
PROJECT (ANNetGPGPU)

# Standard CUDA architecture
SET(__CUDA_CAB__ "20")

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR})
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

FIND_PACKAGE(BZip2 REQUIRED)
FIND_PACKAGE(OpenMP)

FIND_PACKAGE(CUDA)
FIND_PACKAGE(CUDAThrust)

FIND_PACKAGE(SWIG)
FIND_PACKAGE(PythonLibs)

IF (SWIG_FOUND AND PYTHONLIBS_FOUND)
  SET(CMAKE_SWIG_FLAGS "")

  INCLUDE(${SWIG_USE_FILE})
  INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})
ENDIF()

INCLUDE (FindOpenMP)
IF(OPENMP_FOUND)
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
ENDIF()

IF (CUDA_FOUND)
  LIST(APPEND CUDA_NVCC_FLAGS -Xcompiler -fopenmp)
  LIST(APPEND CUDA_NVCC_FLAGS -arch=sm_${__CUDA_CAB__})

  ADD_DEFINITIONS("-DCUDA") # needed for conditional compilation of some files

  IF (CUDATHRUST_FOUND)
    INCLUDE_DIRECTORIES (${CUDA_SDK_ROOT_DIR}/C/common/inc/)
    INCLUDE_DIRECTORIES (${CUDATHRUST_INCLUDE_DIR})
  ENDIF (CUDATHRUST_FOUND)
ENDIF (CUDA_FOUND)

# Make sure the compiler can find include files from our ANNet library.
INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR})
INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR}/math)
INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR}/containers)
INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR}/gpgpu)

SET( ANPythonCPUInterfaceFiles
  PyNetCPU.i
)

SET( ANPythonGPUInterfaceFiles
  PyNetGPU.i
)

SET( ANSourceFiles 
  containers/2DArray.cpp
  containers/3DArray.cpp
  containers/TrainingSet.cpp
  math/Functions.cpp
  AbsLayer.cpp
  AbsNet.cpp
  AbsNeuron.cpp
  BPLayer.cpp
  BPNet.cpp
  BPNeuron.cpp
  Edge.cpp
  HFLayer.cpp
  HFNet.cpp
  HFNeuron.cpp
  SOMLayer.cpp
  SOMNet.cpp
  SOMNeuron.cpp
)

SET( ANCUDASourceFiles
  containers/2DArrayGPU.cu
  SOMNetGPU.cpp
  SOMKernel.cu
  SOMExport.cu
)

# Create a library called "ANNet" which includes the source files listed in "ANSourceFiles".
# Build ANNet C library
IF (CUDA_FOUND)  
  CUDA_ADD_LIBRARY (ANNet STATIC ${ANCUDASourceFiles} ${ANSourceFiles} ${BZIP_INCLUDE_DIRS} OPTIONS -D __CUDA_CAB__=${__CUDA_CAB__})
ELSEIF (NOT CUDA_FOUND)
  ADD_LIBRARY (ANNet STATIC ${ANSourceFiles} ${BZIP_INCLUDE_DIRS})
ENDIF(CUDA_FOUND)

# Linke OpenMP to ANNet library
IF(OPENMP_FOUND)
  TARGET_LINK_LIBRARIES (ANNet ${BZIP2_LIBRARIES} -lgomp)
ELSEIF(NOT OPENMP_FOUND)
  TARGET_LINK_LIBRARIES (ANNet ${BZIP2_LIBRARIES})
ENDIF(OPENMP_FOUND)

# Build python interface
IF (SWIG_FOUND AND PYTHONLIBS_FOUND)
  # Create sub directory for the interface intermediate
  FILE(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  SET_SOURCE_FILES_PROPERTIES(${ANPythonCPUInterfaceFiles} PROPERTIES CPLUSPLUS ON)
  SET_SOURCE_FILES_PROPERTIES(${ANPythonCPUInterfaceFiles} PROPERTIES CMAKE_SWIG_FLAGS "-includeall")
  SWIG_ADD_MODULE(ANPyNetCPU python ${ANPythonCPUInterfaceFiles})
  SWIG_LINK_LIBRARIES(ANPyNetCPU ${PYTHON_LIBRARIES} ANNet)

  IF (CUDA_FOUND)
    SET_SOURCE_FILES_PROPERTIES(${ANPythonGPUInterfaceFiles} PROPERTIES CPLUSPLUS ON)
    SET_SOURCE_FILES_PROPERTIES(${ANPythonGPUInterfaceFiles} PROPERTIES CMAKE_SWIG_FLAGS "-includeall")
    SWIG_ADD_MODULE(ANPyNetGPU python ${ANPythonGPUInterfaceFiles})
    SWIG_LINK_LIBRARIES(ANPyNetGPU ${PYTHON_LIBRARIES} ANNet)
  ENDIF(CUDA_FOUND)
ENDIF(SWIG_FOUND AND PYTHONLIBS_FOUND)

