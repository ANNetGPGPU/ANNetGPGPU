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

#ifndef EDGE_H
#define EDGE_H

#include <QtGui>

class Node;


class Edge : public QGraphicsItem
{
private:
    Node *m_pSource, *m_pDest;

    QPointF m_SourcePoint;
    QPointF m_DestPoint;
    qreal m_ArrowSize;

    QColor m_Color;

public:
    Edge(Node *pSourceNode, Node *pDestNode);
    virtual ~Edge();

    Node *sourceNode() const;
    Node *destNode() const;

    void adjust();

    void setColor(QColor color);

protected:
    QRectF boundingRect() const;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);
};

#endif // EDGE_H
