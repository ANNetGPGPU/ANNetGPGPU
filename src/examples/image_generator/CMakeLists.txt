# The name of our project is "ANNetGPGPU". CMakeLists files in this project can
# refer to the root source directory of the project as ${ANNetGPU_SOURCE_DIR} and
# to the root binary directory of the project as ${ANNetGPU_SOURCE_DIR}.
PROJECT (SOM_IMAGE_GENERATOR)

FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)

SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CUDA_SEPARABLE_COMPILATION ON)

# Make sure the compiler can find include files from our ANNet library.
INCLUDE_DIRECTORIES (${CUDA_SDK_ROOT_DIR}/C/common/inc/)
INCLUDE_DIRECTORIES (${CUDATHRUST_INCLUDE_DIR})

INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR})
INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR}/include/math)
INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR}/include/gpgpu)

# Build misc examples
CUDA_ADD_EXECUTABLE (image_generator image_generator.cpp)
TARGET_LINK_LIBRARIES (image_generator ANNet ${OpenCV_LIBS}) 

configure_file(${SOM_IMAGE_GENERATOR_SOURCE_DIR}/vienna.png
               ${CMAKE_CURRENT_BINARY_DIR}/vienna.png COPYONLY)
               
configure_file(${SOM_IMAGE_GENERATOR_SOURCE_DIR}/wall.png
               ${CMAKE_CURRENT_BINARY_DIR}/wall.png COPYONLY)
