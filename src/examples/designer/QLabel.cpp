#include <QLabel.h>
#include <QtCore>
#include <ANNet>


Label::Label() {
    m_sName = "";
    setFlag(QGraphicsItem::ItemIsSelectable);
    setZValue(2);
}

void Label::setBRect(QRectF rect) {
    m_BRect = rect;
}

void Label::SetName(QString sName) {
    m_sName = sName;
}

uint32_t Label::getType() {
	if(GetName() == "Input layer") {
		return ANN::ANLayerInput;
	}
	else if(GetName() == "Hidden layer") {
		return ANN::ANLayerHidden;
	}
	else if(GetName() == "Output layer") {
		return ANN::ANLayerOutput;
	}
	else {
		return -1;
	}
}

void Label::setType(uint32_t type) {
	if(type == ANN::ANLayerInput) {
		m_sName = "Input layer";
	}
	else if(type == ANN::ANLayerHidden) {
		m_sName = "Hidden layer";
	}
	else if(type == ANN::ANLayerOutput) {
		m_sName = "Output layer";
	}
	else {
		m_sName = "error";
	}
}

void Label::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) {
    if(option->state & QStyle::State_Selected) {
        painter->setPen(Qt::NoPen);
        QColor lightGray(192, 192, 192, 128);
        painter->setBrush(lightGray);
        painter->drawRect(m_BRect);
    }
    else {
        painter->setPen(Qt::NoPen);
        QColor lightGray(128, 128, 128, 128);
        painter->setBrush(lightGray);
        painter->drawRect(m_BRect);
    }
    QFont font; font.setPixelSize(18);
    painter->setPen(QPen(Qt::black, 0));
    painter->setFont(font);

    painter->drawText(m_BRect, Qt::AlignCenter, m_sName);
}

QRectF Label::boundingRect() const {
    return m_BRect;
}

QString Label::GetName() {
    return m_sName;
}

/*
void Label::mousePressEvent ( QGraphicsSceneMouseEvent *event ) {

}
*/
void Label::mouseDoubleClickEvent ( QGraphicsSceneMouseEvent * event ) {
    bool ok;
    QStringList items;
    items << QObject::tr("Input layer") << QObject::tr("Hidden layer") << QObject::tr("Output layer");

    QString item = QInputDialog::getItem(0, QObject::tr("Choose the type of the layer"),
                                         QObject::tr("Type of layer:"), items, 0, false, &ok);

    if (ok && !item.isEmpty())
        SetName(item);

    QGraphicsItem::mouseDoubleClickEvent(event);
}

