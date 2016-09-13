/*
 * ZLabel.cpp
 *
 *  Created on: 19.06.2012
 *      Author: dgrat
 */

#include <QZLabel.h>


ZLabel::ZLabel() {
	m_iZLayer = -1;
    setFlag(QGraphicsItem::ItemIsSelectable);
    setZValue(2);
}

void ZLabel::setZLayer(const int &iVal) {
	m_iZLayer = iVal;
}

int ZLabel::getZLayer() {
	return m_iZLayer;
}

void ZLabel::setBRect(QRectF rect) {
	m_BRect = rect;
}

void ZLabel::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) {
	QPainterPath path;
	path.setFillRule(Qt::WindingFill);

	path.moveTo(m_BRect.x(), m_BRect.y());
	path.lineTo(m_BRect.x()+m_BRect.width()/2, m_BRect.y());

	QRectF rect = m_BRect;
	rect.setX(rect.x()+rect.width()/2);

	path.arcTo(rect, 90, -180);
	path.lineTo(m_BRect.x(), m_BRect.y()+m_BRect.height());

	painter->setPen(QPen(Qt::black, 0) );
	painter->drawPath(path);

    if(option->state & QStyle::State_Selected) {
        painter->setPen(Qt::NoPen);
        QColor lightGray(192, 192, 192, 128);
        painter->setBrush(lightGray);
        painter->drawPath(path);
    }
    else {
    	painter->setPen(Qt::NoPen);
        QColor lightGray(128, 128, 128, 128);
        painter->setBrush(lightGray);
        painter->drawPath(path);
    }
    QFont font; font.setPixelSize(18);
    painter->setPen(QPen(Qt::black, 0));
    painter->setFont(font);

    painter->drawText(m_BRect, Qt::AlignCenter, QString::number(m_iZLayer) );
}

QRectF ZLabel::boundingRect() const {
	return m_BRect;
}

void ZLabel::mouseDoubleClickEvent ( QGraphicsSceneMouseEvent * event ) {
    bool ok;
    int i = QInputDialog::getInt(0, QObject::tr("QInputDialog::getInteger()"),
    								QObject::tr("Choose Z-value of this layer:"), m_iZLayer, -1, 99, 1, &ok);
    if (ok) {
        m_iZLayer = i;
        update();
    }
}
