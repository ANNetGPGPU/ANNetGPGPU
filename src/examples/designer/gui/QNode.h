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

#ifndef NODE_H
#define NODE_H

#include <QtGui>

class Edge;
class Viewer;
class Layer;


class Node : public QGraphicsItem
{
private: 
    int m_iID;			// index of neuron in layer
    QString m_sTransFunction;

    int m_iWidth;		// diameter of neuron in QGraphicsView
    bool m_bSelectedAsGroup;	// state variable for QGraphicsView

    QList<Edge *> m_EdgeListI;
    QList<Edge *> m_EdgeListO;
    Viewer *m_pGraph;
    Layer *m_pLayer;

public:
    Node(Viewer *parent = NULL);
    virtual ~Node();

    void setID(const int &iID);
    int getID() const;
    
    void setTransFunction(const QString &sFunction);
    QString getTransFunction() const;

    void addEdgeI(Edge *edge);
    QList<Edge *> edgesI() const;
    void addEdgeO(Edge *edge);
    QList<Edge *> edgesO() const;

    void removeEdge(Edge* pDelEdge);

    void setLayer(Layer *layer);
    Layer* getLayer() const;

    float getWidth();

    void setSelectedAsGroup(bool b);
    bool selectedAsGroup();

protected:
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);
    QRectF boundingRect() const;
    QPainterPath shape() const;

    void mousePressEvent(QGraphicsSceneMouseEvent *event);
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);
    void mouseMoveEvent(QGraphicsSceneMouseEvent *event);
};

#endif // NODE_H
