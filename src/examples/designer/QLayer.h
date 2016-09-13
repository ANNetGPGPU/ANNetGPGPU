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

#ifndef LAYER_H
#define LAYER_H

#include <QtGui>
#include <QZLabel.h>

class Node;
class Scene;
class Viewer;
class Edge;
class Label;
class ZLabel;


class Layer : public QGraphicsItem
{
private:
    int m_iID;	// ID of the layer
    
    Viewer *m_pGraph;
    QList<Node *> m_NodeList;
    QRectF m_BoundingRect;

    QRectF m_LabelRect;
    QRectF m_ZLabelRect;

    Label *m_pLabel;
    ZLabel *m_pZLabel;

    Scene *m_pScene;

    void refreshNodeIDs();

public:
    Layer(Viewer *parent = NULL);

    void setID(const int &iID);
    int getID() const;
    
    void addNode(Node *node);
    void addNodes(int iNeur);
    void removeNode(Node* pDelNode);
    QList<Node *> &nodes();

    void adjust();

    void shift(int dX, int dY);
    //void addNodes(const unsigned int &iNodes, const QString &sName);

    void setScene(Scene*);

    QList<Edge*> Connect(Layer*);
    QRectF getLabelRect();
    QRectF getZLabelRect();

    void setLabel(Label *pLabel);
    Label* getLabel();
    Label *addLabel(QString sName);

    ZLabel *addZLabel(const int &iNumber);
    void setZLabel(ZLabel *pLabel);
    ZLabel* getZLabel();

protected:
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);
    QRectF boundingRect() const;

    void mousePressEvent(QGraphicsSceneMouseEvent *event);
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);
    void mouseMoveEvent(QGraphicsSceneMouseEvent *event);

    void mouseDoubleClickEvent ( QGraphicsSceneMouseEvent * event );
};

#endif // LAYER_H
