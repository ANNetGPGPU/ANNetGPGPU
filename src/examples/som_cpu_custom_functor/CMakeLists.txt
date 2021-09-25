# The name of our project is "ANNetGPGPU". CMakeLists files in this project can
# refer to the root source directory of the project as ${ANNetGPU_SOURCE_DIR} and
# to the root binary directory of the project as ${ANNetGPU_SOURCE_DIR}.
PROJECT (SOM_GPU)

# Make sure the compiler can find include files from our ANNet library.
INCLUDE_DIRECTORIES (${CUDA_SDK_ROOT_DIR}/C/common/inc/)
INCLUDE_DIRECTORIES (${CUDATHRUST_INCLUDE_DIR})

INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR})
INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR}/include/math)
INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR}/include/gpgpu)

# Build misc examples
ADD_EXECUTABLE (som_cpu_custom_functor som_cpu_custom_functor.cpp)
TARGET_LINK_LIBRARIES (som_cpu_custom_functor ${BZIP2_LIBRARIES} SOMReader) 
IF(OPENMP_FOUND)
  TARGET_LINK_LIBRARIES (som_cpu_custom_functor -lgomp)
ENDIF(OPENMP_FOUND)
