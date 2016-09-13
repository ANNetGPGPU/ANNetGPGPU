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


#ifndef QZLABEL_H_
#define QZLABEL_H_

#include <QtGui>


class ZLabel  : public QGraphicsItem {
private:
    int m_iZLayer;
    QRectF m_BRect;

public:
    ZLabel();

    void setZLayer(const int &iVal);
    int getZLayer();

    void setBRect(QRectF rect);

protected:
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);
    QRectF boundingRect() const;

    virtual void mouseDoubleClickEvent ( QGraphicsSceneMouseEvent * event );
};

#endif /* QZLABEL_H_ */
