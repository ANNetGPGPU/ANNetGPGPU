# The name of our project is "ANNetGPGPU". CMakeLists files in this project can
# refer to the root source directory of the project as ${ANNetGPU_SOURCE_DIR} and
# to the root binary directory of the project as ${ANNetGPU_SOURCE_DIR}.
PROJECT (SOM_CPU)

FIND_PACKAGE(Qt5Widgets REQUIRED)

# Make sure the compiler can find include files from our ANNet library.
INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR})
INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR}/include/math)
INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR}/examples)

ADD_EXECUTABLE (som_cpu som_cpu.cpp)
TARGET_LINK_LIBRARIES (som_cpu SOMReader ${BZIP2_LIBRARIES})  
IF(OPENMP_FOUND)
  TARGET_LINK_LIBRARIES (som_cpu -lgomp)
ENDIF(OPENMP_FOUND)

