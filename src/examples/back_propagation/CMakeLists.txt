# The name of our project is "ANNetGPGPU". CMakeLists files in this project can
# refer to the root source directory of the project as ${ANNetGPU_SOURCE_DIR} and
# to the root binary directory of the project as ${ANNetGPU_SOURCE_DIR}.
PROJECT (Back_Propagation)

FIND_PACKAGE(BZip2 REQUIRED)

LIST (APPEND ${CMAKE_CXX_FLAGS} -std=gnu++11)

#INCLUDE_DIRECTORIES (${Examples_SOURCE_DIR}/QSOMReader.cpp)

ADD_EXECUTABLE (back_propagation back_propagation.cpp)
TARGET_LINK_LIBRARIES (back_propagation ${BZIP2_LIBRARIES}) 
IF(OPENMP_FOUND)
  TARGET_LINK_LIBRARIES (back_propagation -lgomp)
ENDIF(OPENMP_FOUND)

