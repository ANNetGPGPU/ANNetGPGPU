PROJECT (Examples) 

FIND_PACKAGE(Qt5Widgets REQUIRED)
FIND_PACKAGE(Qt4)
FIND_PACKAGE(CUDA)

SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_AUTOMOC ON)

INCLUDE_DIRECTORIES (${Examples_SOURCE_DIR})

SET( ANGUISourceFiles
  ${Examples_SOURCE_DIR}/QSOMReader.cpp
)

ADD_LIBRARY (SOMReader ${ANGUISourceFiles})
TARGET_LINK_LIBRARIES (SOMReader Qt5::Widgets)

ADD_SUBDIRECTORY(back_propagation)
ADD_SUBDIRECTORY(som_cpu)
ADD_SUBDIRECTORY(som_cpu_custom_functor)

IF (CUDA_FOUND)
  ADD_SUBDIRECTORY(som_gpu)
  ADD_SUBDIRECTORY(som_gpu_custom_functor)
  ADD_SUBDIRECTORY(cluster_data)
ELSE()
  MESSAGE(WARNING "CUDA not found! We will not build the GPU dependent examples")
ENDIF(CUDA_FOUND)

IF (Qt4_FOUND)
  ADD_SUBDIRECTORY(designer)
ELSE()
  MESSAGE(WARNING "Qt4 not found! We will not build the Qt designer example")
ENDIF(Qt4_FOUND)
