# The name of our project is "ANNetGPGPU". CMakeLists files in this project can
# refer to the root source directory of the project as ${ANNetGPU_SOURCE_DIR} and
# to the root binary directory of the project as ${ANNetGPU_SOURCE_DIR}.
CMAKE_MINIMUM_REQUIRED (VERSION 2.8)
PROJECT (ANNetDesigner)

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR})
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

FIND_PACKAGE(Qt4)
FIND_PACKAGE(CUDA)

# Make sure the compiler can find include files from our ANNet library.
INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES (${ANNetDesigner_SOURCE_DIR})

SET(ANGUIFormularFiles
  QTrainingForm.ui
)

SET( 3rdPartySourceFiles
  gui/3rdparty/qcustomplot.cpp
  gui/3rdparty/utils/fancymainwindow.cpp
  gui/3rdparty/utils/manhattanstyle.cpp
  gui/3rdparty/utils/minisplitter.cpp
  gui/3rdparty/utils/styleanimator.cpp
  gui/3rdparty/utils/styledbar.cpp
  gui/3rdparty/utils/stylehelper.cpp
  gui/3rdparty/utils/stringutils.cpp
  gui/3rdparty/fancytabwidget.cpp
  gui/3rdparty/fancyactionbar.cpp
  gui/3rdparty/delegate/spinboxdelegate.cpp
)

SET( 3rdPartyHeaderFiles
  gui/3rdparty/qcustomplot.h
  gui/3rdparty/utils/fancymainwindow.h
  gui/3rdparty/utils/manhattanstyle.h
  gui/3rdparty/utils/minisplitter.h
  gui/3rdparty/utils/styleanimator.h
  gui/3rdparty/utils/styledbar.h
  gui/3rdparty/utils/stylehelper.h
  gui/3rdparty/utils/stringutils.h
  gui/3rdparty/coreplugin/coreconstants.h
  gui/3rdparty/fancytabwidget.h
  gui/3rdparty/fancyactionbar.h
  gui/3rdparty/delegate/spinboxdelegate.h
)

SET( ANGUISourceFiles
  QMainWindow.cpp
  QEdge.cpp
  QNode.cpp
  QLayer.cpp
  QLabel.cpp
  QZLabel.cpp
  QScene.cpp
  QTrainingForm.cpp
  QIOForm.cpp
  QViewer.cpp
  QOutput.cpp
  QGraphTab.cpp
  QTableWidget.cpp
  QTrainingThread.cpp
)

SET( ANGUIHeaderFiles 
  gui/QMainWindow.h
  gui/QEdge.h
  gui/QNode.h
  gui/QLayer.h
  gui/QLabel.h
  gui/QZLabel.h
  gui/QScene.h
  gui/QTrainingForm.h
  gui/QIOForm.h
  gui/QViewer.h
  gui/QOutput.h
  gui/QGraphTab.h
  gui/QTableWidget.h
  gui/QTrainingThread.h
)

IF (QT4_FOUND)
  QT4_WRAP_CPP(ANGUIHeaderFiles_MOC ${ANGUIHeaderFiles} ${3rdPartyHeaderFiles})
  QT4_WRAP_UI(ANGUIHeaderFiles_MOC ${ANGUIFormularFiles})
  INCLUDE(${QT_USE_FILE})
  INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
  ADD_DEFINITIONS(${QT_DEFINITIONS})

  ADD_LIBRARY (ANNetGUI STATIC ${ANGUIHeaderFiles_MOC} ${ANGUISourceFiles} ${3rdPartySourceFiles})
  TARGET_LINK_LIBRARIES (ANNetGUI ANNet ${QT_LIBRARIES})
ENDIF (QT4_FOUND)

# Build Qt designer
IF (WIN32)
  ADD_EXECUTABLE (ANNetDesigner WIN32 ANNetDesigner.cpp)
ELSEIF(NOT WIN32)
  ADD_EXECUTABLE (ANNetDesigner ANNetDesigner.cpp)
ENDIF(WIN32)
TARGET_LINK_LIBRARIES (ANNetDesigner ANNetGUI) 

FILE(COPY ${ANNetDesigner_SOURCE_DIR}/gfx DESTINATION ${ANNetDesigner_BINARY_DIR})
