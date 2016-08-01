# The name of our project is "ANNetGPGPU". CMakeLists files in this project can
# refer to the root source directory of the project as ${ANNetGPU_SOURCE_DIR} and
# to the root binary directory of the project as ${ANNetGPU_SOURCE_DIR}.
PROJECT (Back_Propagation)

FIND_PACKAGE(BZip2 REQUIRED)

LIST(APPEND ${CMAKE_CXX_FLAGS} -std=gnu++11)

# Make sure the compiler can find include files from our ANNet library.
INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR})
INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR}/include/math)

ADD_EXECUTABLE (back_propagation back_propagation.cpp)
TARGET_LINK_LIBRARIES (back_propagation ANNet) 


