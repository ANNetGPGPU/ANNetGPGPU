# The name of our project is "ANNetGPGPU". CMakeLists files in this project can
# refer to the root source directory of the project as ${ANNetGPU_SOURCE_DIR} and
# to the root binary directory of the project as ${ANNetGPU_SOURCE_DIR}.
CMAKE_MINIMUM_REQUIRED (VERSION 2.8)
PROJECT (ANNetMisc)

FIND_PACKAGE(BZip2 REQUIRED)
FIND_PACKAGE(Qt4)
FIND_PACKAGE(CUDA)

# SM_20 necessary for function pointer support as used in one example below
# SET(__CUDA_CAB__ "20")

IF (CUDA_FOUND)
  SET(CUDA_SEPARABLE_COMPILATION ON)

  LIST(APPEND CUDA_NVCC_FLAGS -Xcompiler -fopenmp)
  LIST(APPEND CUDA_NVCC_FLAGS -arch=sm_${__CUDA_CAB__})
  LIST(APPEND CUDA_NVCC_FLAGS --pre-include gcc_cuda_compat.h)

  ADD_DEFINITIONS("-DCUDA") # needed for conditional compilation of some files

  IF (CUDATHRUST_FOUND)
    INCLUDE_DIRECTORIES (${CUDA_SDK_ROOT_DIR}/C/common/inc/)
    INCLUDE_DIRECTORIES (${CUDATHRUST_INCLUDE_DIR})
  ENDIF (CUDATHRUST_FOUND)
ENDIF (CUDA_FOUND)

# Make sure the compiler can find include files from our ANNet library.
INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR})
INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR}/include/gpgpu)
INCLUDE_DIRECTORIES (${ANNetMisc_SOURCE_DIR})

SET( ANGUIHeaderFiles 
  QSOMReader.h
)

SET( ANGUISourceFiles
  QSOMReader.cpp
)

IF (QT4_FOUND)
  QT4_WRAP_CPP(ANGUIHeaderFiles_MOC ${ANGUIHeaderFiles} ${3rdPartyHeaderFiles})
  QT4_WRAP_UI(ANGUIHeaderFiles_MOC ${ANGUIFormularFiles})
  INCLUDE(${QT_USE_FILE})
  INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
  ADD_DEFINITIONS(${QT_DEFINITIONS})

  # Build Qt library
  add_library (SOMReader STATIC ${ANGUIHeaderFiles_MOC} ${ANGUISourceFiles})
  TARGET_LINK_LIBRARIES (SOMReader ${QT_LIBRARIES})
ENDIF (QT4_FOUND)

SET( ANSourceFiles 
  ${ANNetGPGPU_SOURCE_DIR}/2DArray.cpp
  ${ANNetGPGPU_SOURCE_DIR}/AbsLayer.cpp
  ${ANNetGPGPU_SOURCE_DIR}/AbsNet.cpp
  ${ANNetGPGPU_SOURCE_DIR}/AbsNeuron.cpp
  ${ANNetGPGPU_SOURCE_DIR}/Edge.cpp
  ${ANNetGPGPU_SOURCE_DIR}/Functions.cpp
  ${ANNetGPGPU_SOURCE_DIR}/SOMLayer.cpp
  ${ANNetGPGPU_SOURCE_DIR}/SOMNet.cpp
  ${ANNetGPGPU_SOURCE_DIR}/SOMNeuron.cpp
  ${ANNetGPGPU_SOURCE_DIR}/TrainingSet.cpp
)

SET( ANCUDASourceFiles
  ${ANNetGPGPU_SOURCE_DIR}/2DArray.cu
  ${ANNetGPGPU_SOURCE_DIR}/SOMNetGPU.cpp
  ${ANNetGPGPU_SOURCE_DIR}/SOMKernel.cu
  ${ANNetGPGPU_SOURCE_DIR}/SOMExport.cu
)

ADD_EXECUTABLE (BPNetCPU BPNetCPU.cpp)
TARGET_LINK_LIBRARIES (BPNetCPU ANNet) 

# Build misc examples
IF (CUDA_FOUND)
  IF (QT4_FOUND)
    CUDA_ADD_EXECUTABLE (SOMNetGPU SOMNetGPU.cpp)
    TARGET_LINK_LIBRARIES (SOMNetGPU ANNet SOMReader) 
    
#     CUDA_ADD_LIBRARY (foo STATIC SetFcn.cu ${ANCUDASourceFiles} ${ANSourceFiles} ${BZIP_INCLUDE_DIRS} OPTIONS -D __CUDA_CAB__=${__CUDA_CAB__})
#     CUDA_ADD_EXECUTABLE (SimpFnExtGPU SimpFnExtGPU.cpp)
#     TARGET_LINK_LIBRARIES (SimpFnExtGPU foo SOMReader ${BZIP2_LIBRARIES} -lgomp) 
  ENDIF (QT4_FOUND)
  
#  CUDA_ADD_EXECUTABLE (BPNetGPU BPNetGPU.cpp)
#  TARGET_LINK_LIBRARIES (BPNetGPU ANNet) 
ENDIF(CUDA_FOUND)

ADD_EXECUTABLE (LoadBPNet LoadBPNet.cpp)
TARGET_LINK_LIBRARIES (LoadBPNet ANNet) 

ADD_EXECUTABLE (HFNet HFNet.cpp)
TARGET_LINK_LIBRARIES (HFNet ANNet) 

IF (QT4_FOUND)
  ADD_EXECUTABLE (SOMNetCPU SOMNetCPU.cpp)
  TARGET_LINK_LIBRARIES (SOMNetCPU ANNet SOMReader) 
    
  ADD_EXECUTABLE (SimpFnExt SimpFnExt.cpp)
  TARGET_LINK_LIBRARIES (SimpFnExt ANNet SOMReader) 

  ADD_EXECUTABLE (SOMNetClustering SOMNetClustering.cpp)
  TARGET_LINK_LIBRARIES (SOMNetClustering ANNet SOMReader)  
ENDIF(QT4_FOUND)

