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

#ifndef SCENE_H
#define SCENE_H

#include <QtGui>
#include <ANNet>

class Edge;
class Node;
class Layer;


class Scene : public QGraphicsScene
{
	Q_OBJECT

private:
    QList<Node*> m_lNodes;
    QList<Edge*> m_lEdges;
    QList<Layer*> m_lLayers;

    ANN::BPNet<float, ANN::fcn_log<float>> *m_pANNet;

    void refreshLayerIDs();

signals:
	void si_netChanged(ANN::BPNet<float, ANN::fcn_log<float>> *m_pANNet);

public:
    Scene(QObject *parent = 0);
    virtual ~Scene() {}

    void clearAll();

    ANN::BPNet<float, ANN::fcn_log<float>> *getANNet(bool bDial = true);
    void setANNet(ANN::BPNet<float, ANN::fcn_log<float>> &);

    void addEdge(Edge*);
    void removeEdge(Edge*);
    QList<Edge*> edges();

    void addNode(Node*);
    void removeNode(Node*);
    QList<Node*> nodes();

    Layer* addLayer(const unsigned int &iNodes, const QPointF &fPos = QPointF(0, 0), const QString &sName = "");
    void removeLayer(Layer*);
    QList<Layer*> layers();

    void adjust();
};

#endif // SCENE_H
