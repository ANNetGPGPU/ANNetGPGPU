#include <gui/QNode.h>
#include <gui/QEdge.h>
#include <gui/QViewer.h>
#include <gui/QLayer.h>


Node::Node(Viewer *parent) {
    m_iID = -1;
    m_sTransFunction = "Zen";
    
    setFlag(QGraphicsItem::ItemIsMovable);
    setFlag(QGraphicsItem::ItemIsSelectable);

    m_bSelectedAsGroup = false;

    m_pGraph = parent;
    m_iWidth = 20;
    m_pLayer = NULL;

    setZValue(2);
}

Node::~Node()
{
}

void Node::setTransFunction(const QString &sFunction) {
	m_sTransFunction = sFunction;
}

QString Node::getTransFunction() const {
	return m_sTransFunction;
}

int Node::getID() const {
    return m_iID;
}

void Node::setSelectedAsGroup(bool b) {
    m_bSelectedAsGroup = b;
}

bool Node::selectedAsGroup() {
    return m_bSelectedAsGroup;
}

void Node::setID(const int &iID) {
	m_iID = iID;
}

float Node::getWidth() {
    return m_iWidth;
}

void Node::addEdgeI(Edge *edge) {
    m_EdgeListI << edge;
    edge->adjust();
}

QList<Edge *> Node::edgesI() const {
    return m_EdgeListI;
}

void Node::addEdgeO(Edge *edge) {
    m_EdgeListO << edge;
    edge->adjust();
}

void Node::removeEdge(Edge* pDelEdge) {
    QList<Edge*> pNewListI;
    foreach(Edge *pEdge, m_EdgeListI) {
        if(pEdge != pDelEdge)
        	pNewListI << pEdge;
    }
    m_EdgeListI = pNewListI;

    QList<Edge*> pNewListO;
    foreach(Edge *pEdge, m_EdgeListO) {
        if(pEdge != pDelEdge)
        	pNewListO << pEdge;
    }
    m_EdgeListO = pNewListO;
}

QList<Edge *> Node::edgesO() const {
    return m_EdgeListO;
}

void Node::setLayer(Layer *layer) {
    m_pLayer = layer;
}

Layer* Node::getLayer() const {
    return m_pLayer;
}

void Node::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) {
    QRadialGradient gradient(-3, -3, m_iWidth/2);

    if (option->state & QStyle::State_Sunken) {
        gradient.setCenter(3, 3);
        gradient.setFocalPoint(3, 3);
        gradient.setColorAt(1, QColor(Qt::yellow).light(m_iWidth));
        gradient.setColorAt(0, QColor(Qt::darkYellow).light(m_iWidth));
    }
    if(option->state & QStyle::State_Selected || option->state & QStyle::State_Sunken) {
        gradient.setColorAt(0, Qt::white);
        gradient.setColorAt(1, Qt::red);

        /*foreach (Edge *edge, m_EdgeListI) {
            edge->setColor(Qt::yellow);
            edge->setZValue(1);
            edge->adjust();
        }*/
        foreach (Edge *edge, m_EdgeListO) {
            edge->setColor(Qt::red);
            edge->setZValue(1);
            edge->adjust();
        }
    }
    else {
        gradient.setColorAt(0, Qt::yellow);
        gradient.setColorAt(1, Qt::darkYellow);

        /*foreach (Edge *edge, m_EdgeListI) {
            edge->setColor(Qt::black);
            edge->setZValue(0);
            edge->adjust();
        }*/
        foreach (Edge *edge, m_EdgeListO) {
            edge->setColor(Qt::black);
            edge->setZValue(0);
            edge->adjust();
        }
    }
    if(m_bSelectedAsGroup) {
        gradient.setColorAt(0, Qt::white);
        QColor orange(255, 128, 0);
        gradient.setColorAt(1, orange);
    }

    painter->setBrush(gradient);

    painter->setPen(QPen(Qt::black, 0));
    painter->drawEllipse(-m_iWidth/2, -m_iWidth/2, m_iWidth, m_iWidth);
}

QPainterPath Node::shape() const
{
    QPainterPath path;
    path.addEllipse(-m_iWidth/2, -m_iWidth/2, m_iWidth, m_iWidth);
    return path;
}

QRectF Node::boundingRect() const {
    qreal adjust = 2;
    return QRectF( -m_iWidth/2 - adjust, -m_iWidth/2 - adjust, m_iWidth+3 + adjust, m_iWidth+3 + adjust);
}

void Node::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    foreach (Edge *edge, m_EdgeListI)
        edge->adjust();
    foreach (Edge *edge, m_EdgeListO)
        edge->adjust();

    if(m_pLayer != NULL)
        m_pLayer->adjust();

    update();
    QGraphicsItem::mousePressEvent(event);
}

void Node::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
    foreach (Edge *edge, m_EdgeListI)
        edge->adjust();
    foreach (Edge *edge, m_EdgeListO)
        edge->adjust();

    if(m_pLayer != NULL)
        m_pLayer->adjust();

    update();
    QGraphicsItem::mouseReleaseEvent(event);
}

void Node::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
    foreach (Edge *edge, m_EdgeListI)
        edge->adjust();
    foreach (Edge *edge, m_EdgeListO)
        edge->adjust();

    if(m_pLayer != NULL)
        m_pLayer->adjust();

    update();
    QGraphicsItem::mouseMoveEvent(event);
}
