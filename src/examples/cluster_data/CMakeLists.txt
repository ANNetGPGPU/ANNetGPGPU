# The name of our project is "ANNetGPGPU". CMakeLists files in this project can
# refer to the root source directory of the project as ${ANNetGPU_SOURCE_DIR} and
# to the root binary directory of the project as ${ANNetGPU_SOURCE_DIR}.
PROJECT (Cluster_Analysis)

#FIND_PACKAGE(Qt5Widgets REQUIRED)
FIND_PACKAGE(CUDA REQUIRED)

SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CUDA_SEPARABLE_COMPILATION ON)

ADD_DEFINITIONS("-DCUDA") # needed for conditional compilation of some files

# Make sure the compiler can find include files from our ANNet library.
INCLUDE_DIRECTORIES (${CUDA_SDK_ROOT_DIR}/C/common/inc/)
INCLUDE_DIRECTORIES (${CUDATHRUST_INCLUDE_DIR})

INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR})
INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR}/include/math)
INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR}/include/gpgpu)

LIST(APPEND CUDA_NVCC_FLAGS "-arch=sm_20;-std=c++11;-O2;-DVERBOSE")

# Build misc examples
CUDA_ADD_EXECUTABLE (clustering clustering.cpp)
TARGET_LINK_LIBRARIES (clustering SOMReader ${BZIP2_LIBRARIES})
IF(OPENMP_FOUND)
  TARGET_LINK_LIBRARIES (clustering -lgomp)
ENDIF(OPENMP_FOUND)