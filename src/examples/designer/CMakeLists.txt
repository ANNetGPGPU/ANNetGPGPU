# The name of our project is "ANNetGPGPU". CMakeLists files in this project can
# refer to the root source directory of the project as ${ANNetGPU_SOURCE_DIR} and
# to the root binary directory of the project as ${ANNetGPU_SOURCE_DIR}.
CMAKE_MINIMUM_REQUIRED (VERSION 3.0.0)
PROJECT (ANNetDesigner)

# Make sure the compiler can find include files from our ANNet library.
INCLUDE_DIRECTORIES (${ANNetGPGPU_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES (${ANNetDesigner_SOURCE_DIR})
INCLUDE_DIRECTORIES (${ANNetDesigner_SOURCE_DIR}/3rdparty/)
INCLUDE_DIRECTORIES (${ANNetDesigner_SOURCE_DIR}/3rdparty/utils)
INCLUDE_DIRECTORIES (${ANNetDesigner_SOURCE_DIR}/delegate)
INCLUDE_DIRECTORIES (${ANNetDesigner_SOURCE_DIR}/3rdparty/coreplugin)
INCLUDE_DIRECTORIES (${ANNetDesigner_SOURCE_DIR}/3rdparty/delegate)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

SET( 3rdPartySourceFiles
  3rdparty/qcustomplot.cpp
  3rdparty/utils/fancymainwindow.cpp
  3rdparty/utils/manhattanstyle.cpp
  3rdparty/utils/minisplitter.cpp
  3rdparty/utils/styleanimator.cpp
  3rdparty/utils/styledbar.cpp
  3rdparty/utils/stylehelper.cpp
  3rdparty/utils/stringutils.cpp
  3rdparty/fancytabwidget.cpp
  3rdparty/fancyactionbar.cpp
  3rdparty/delegate/spinboxdelegate.cpp
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

ADD_EXECUTABLE (ANNetDesigner ANNetDesigner.cpp ${ANGUISourceFiles} ${3rdPartySourceFiles})
TARGET_LINK_LIBRARIES (ANNetDesigner Qt4::QtGui Qt4::QtCore ${BZIP2_LIBRARIES})
IF(OPENMP_FOUND)
  TARGET_LINK_LIBRARIES (ANNetDesigner -lgomp)
ENDIF(OPENMP_FOUND)
FILE(COPY ${ANNetDesigner_SOURCE_DIR}/gfx DESTINATION ${ANNetDesigner_BINARY_DIR})
