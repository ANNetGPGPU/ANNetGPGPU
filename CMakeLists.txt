# The name of our project is "ANNetGPGPU". CMakeLists files in this project can
# refer to the root source directory of the project as ${ANNetGPU_SOURCE_DIR} and
# to the root binary directory of the project as ${ANNetGPU_SOURCE_DIR}.
CMAKE_MINIMUM_REQUIRED (VERSION 3.1)
PROJECT (ANNetGPGPU)

FIND_PACKAGE(BZip2 REQUIRED)
FIND_PACKAGE(OpenMP)
FIND_PACKAGE(CUDA)
FIND_PACKAGE(Doxygen)
FIND_PACKAGE(SWIG)
FIND_PACKAGE(PythonLibs)

SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

MESSAGE(STATUS "Checking for third party libraries:")

ADD_SUBDIRECTORY(src)

INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR}/include)
LINK_DIRECTORIES (${ANNetGPU_SOURCE_DIR}/ANNet) 

# Build docu
IF(DOXYGEN_FOUND)
  MESSAGE(STATUS "\t-Doxygen found. We can build a documentation")

  CONFIGURE_FILE(Doxyfile Doxyfile @ONLY)
  ADD_CUSTOM_TARGET(doc #ALL
    ${DOXYGEN_EXECUTABLE} Doxyfile
    WORKING_DIRECTORY ${ANNetGPU_CURRENT_BINARY_DIR}
    COMMENT "Generating API documentation with Doxygen" VERBATIM
  )
ENDIF(DOXYGEN_FOUND)
