# The name of our project is "ANNetGPGPU". CMakeLists files in this project can
# refer to the root source directory of the project as ${ANNetGPU_SOURCE_DIR} and
# to the root binary directory of the project as ${ANNetGPU_SOURCE_DIR}.
CMAKE_MINIMUM_REQUIRED (VERSION 2.8)
PROJECT (ANNetGPGPU)

FIND_PACKAGE(Doxygen)

ADD_SUBDIRECTORY (src) 			# ANNet library
# ADD_SUBDIRECTORY (examples/designer) 	# BP network designer
ADD_SUBDIRECTORY (examples/misc) 	# several examples

INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR}/include)
LINK_DIRECTORIES (${ANNetGPU_SOURCE_DIR}/ANNet) 

# Build docu
IF(DOXYGEN_FOUND)
  CONFIGURE_FILE(Doxyfile Doxyfile @ONLY)
  ADD_CUSTOM_TARGET(doc #ALL
    ${DOXYGEN_EXECUTABLE} Doxyfile
    WORKING_DIRECTORY ${ANNetGPU_CURRENT_BINARY_DIR}
    COMMENT "Generating API documentation with Doxygen" VERBATIM
  )
ENDIF(DOXYGEN_FOUND)
