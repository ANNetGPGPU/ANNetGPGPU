/*
#-------------------------------------------------------------------------------
# Copyright (c) 2012 Daniel <dgrat> Frenzel.
# All rights reserved. This program and the accompanying materials
# are made available under the terms of the GNU Lesser Public License v2.1
# which accompanies this distribution, and is available at
# http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
# 
# Contributors:
#     Daniel <dgrat> Frenzel - initial API and implementation
#-------------------------------------------------------------------------------
*/

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QtGui>
// own classes
#include <ANNet>
#include <ANContainers>
#include <QViewer.h>
#include <QScene.h>
#include <QTrainingForm.h>
#include <QIOForm.h>
#include <QOutput.h>
#include <QGraphTab.h>
#include <QTrainingThread.h>

//3rd party classes
#include <3rdparty/qcustomplot.h>
#include <3rdparty/fancytabwidget.h>
#include <3rdparty/fancyactionbar.h>

using namespace Core;
using namespace Core::Internal;


class MainWindow : public QMainWindow
{
    Q_OBJECT
private:
    ANN::BPNet<float, ANN::fcn_log<float>> *m_pANNet;
    ANN::TrainingSet<float> *m_pTrainingSet;
    TrainingThread *m_pTrainingThread;
    bool m_bBreakTraining;
    bool m_bAlreadyTrained;	// for warning dialog

    QTimer m_tTimer;

    /////////////////////////////////////////
    FancyActionBar *m_pActionBar;

    QAction *m_pStartStopTraining;
    QAction *m_pRunInput;
    QAction *m_pBuildNet;

    /////////////////////////////////////////
    QToolBar *m_ActionsBar;

    QAction *m_pAddLayer;
    QAction *m_pAddNeuron;
    QAction *m_pAddEdges;

    QAction *m_pRemoveLayers;
    QAction *m_pRemoveNeurons;

    QAction *m_pRemoveEdges;
    QAction *m_pRemoveAllEdges;

    QAction *m_pSetTrainingPairs;

    /////////////////////////////////////////
    FancyTabWidget *m_pTabBar;

    Viewer 	*m_pViewer;
    GraphTab *m_pCustomPlot;
    IOForm 	*m_pInputDial;
    TrainingForm *m_pTrainingDial;
    Output 	*m_pOutputTable;

    /////////////////////////////////////////
    QMenu 	*m_pFileMenu;

    QAction *m_pSave;
    QAction *m_pLoad;
    QAction *m_pNew;
    QAction *m_pQuit;
    
    QMenu 	*m_pViewMenu;
    QAction *m_pZoomIn;
    QAction *m_pZoomOut;
    QAction *m_pShowEdges;
    QAction *m_pShowNodes;

    /////////////////////////////////////////
    std::vector<float> m_vErrors;

private slots:
	void sl_tabChanged(int);
	void sl_updateGraph();
	void sl_switchStartStopTraining();

	void sl_updateProgr();

    void sl_createLayer();
    void sl_setTrainingSet();
    void sl_startTraining();
    void sl_stopTraining();
    void sl_run();
    void sl_build();

    // File menu
    void sl_newProject();
    void sl_saveANNet();
    void sl_loadANNet();

    // View menu
    void sl_zoomIn();
    void sl_zoomOut();
    void sl_ShowEdges(bool);
    void sl_ShowNodes(bool);

public:
    MainWindow(QWidget *parent = 0);
    virtual ~MainWindow() {}

    void createMenus();
    void createTabs();
    void createActions();

    static QCustomPlot *createGraph(float fXmin, float fXmax,
									float fYmin, float fYmax,
									QVector<double> x, QVector<double> y);
};

#endif // MAINWINDOW_H
