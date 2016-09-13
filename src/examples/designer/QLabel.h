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

#ifndef LABEL_H
#define LABEL_H

#include <stdint.h>
#include <QtGui>


class Label : public QGraphicsItem
{
private:
    QRectF m_BRect;
    QString m_sName;

public:
    Label();
    void setBRect(QRectF rect);
    void SetName(QString sName);
    QString GetName();

    uint32_t getType();
    void setType(uint32_t);

protected:
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);
    QRectF boundingRect() const;

//    virtual void mousePressEvent ( QGraphicsSceneMouseEvent *event );
    virtual void mouseDoubleClickEvent ( QGraphicsSceneMouseEvent * event );
};

#endif // LABEL_H
