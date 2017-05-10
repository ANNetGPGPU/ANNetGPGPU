# The name of our project is "ANNetGPGPU". CMakeLists files in this project can
# refer to the root source directory of the project as ${ANNetGPU_SOURCE_DIR} and
# to the root binary directory of the project as ${ANNetGPU_SOURCE_DIR}.
PROJECT (SOM_GPU_ADDON)

SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CUDA_SEPARABLE_COMPILATION ON)

# Make sure the compiler can find include files from our ANNet library.
INCLUDE_DIRECTORIES (${CUDA_SDK_ROOT_DIR}/C/common/inc/)
INCLUDE_DIRECTORIES (${CUDATHRUST_INCLUDE_DIR})

SET( ANCUDASourceFilesEX
  ${ANNetGPGPU_SOURCE_DIR}/SOMNetGPU.cu
)

# here we pass the extension headers to the build system
add_definitions(-D__Functions_ADDONS=<${SOM_GPU_ADDON_SOURCE_DIR}/NewFunctions.h>)
add_definitions(-D__SOMNetGPU_INSTANCES=<${SOM_GPU_ADDON_SOURCE_DIR}/NewInstances.h>)
# and rebuild the library with the extensions
CUDA_ADD_LIBRARY (ANNetEX STATIC ${ANSourceFilesEX} ${ANCUDASourceFilesEX} ${BZIP_INCLUDE_DIRS})
TARGET_LINK_LIBRARIES (ANNetEX ${BZIP2_LIBRARIES}) 
IF(OPENMP_FOUND)
  TARGET_LINK_LIBRARIES (ANNetEX -lgomp)
ENDIF(OPENMP_FOUND)

CUDA_ADD_EXECUTABLE (som_gpu_custom_functor som_gpu_custom_functor.cpp)
TARGET_LINK_LIBRARIES (som_gpu_custom_functor ANNetEX SOMReader) 
