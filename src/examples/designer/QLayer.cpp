#include <QLayer.h>
#include <QNode.h>
#include <QEdge.h>
#include <QLabel.h>
#include <QZLabel.h>
#include <QScene.h>
#include <iostream>


Layer::Layer(Viewer *parent) {
//    setFlag(QGraphicsItem::ItemIsSelectable);
    m_iID = -1;

    m_pLabel = NULL;
    m_pScene = NULL;
    m_pGraph = parent;
    setZValue(-1);
}

void Layer::setID(const int &iID) {
    m_iID = iID;
}
  
int Layer::getID() const {
    return m_iID;
}

void Layer::refreshNodeIDs() {
    for(unsigned int i = 0; i < m_NodeList.size(); i++) {
    	Node *pNode = m_NodeList.at(i);
    	pNode->setID(i);
    }
}

void Layer::addNode(Node *node) {
    m_NodeList << node;
    node->setID(nodes().size()-1);
}

QList<Node *> &Layer::nodes() {
    return m_NodeList;
}

void Layer::removeNode(Node* pDelNode) {
    QList<Node*> pNewList;
    foreach(Node *pNode, m_NodeList) {
        if(pNode != pDelNode)
            pNewList << pNode;
    }
    m_NodeList = pNewList;

    /*
     * Refresh the IDs of the nodes in the GUI
     */
    refreshNodeIDs();
}

void Layer::adjust() {
    m_BoundingRect = boundingRect();

    float   fTextBoxHeight = 24.f;
    m_LabelRect = QRectF(	m_BoundingRect.x(),
							m_BoundingRect.y()+m_BoundingRect.height(),
							m_BoundingRect.width(),
							fTextBoxHeight);

    m_ZLabelRect = QRectF(	m_BoundingRect.x()+m_BoundingRect.width(),
							m_BoundingRect.y()+m_BoundingRect.height(),
							fTextBoxHeight*1.5,
							fTextBoxHeight);

    if(m_pLabel != NULL) {
        m_pLabel->setBRect(m_LabelRect);
        m_pZLabel->setBRect(m_ZLabelRect);
    }
}

QRectF Layer::getLabelRect() {
    return m_LabelRect;
}

QRectF Layer::getZLabelRect() {
    return m_ZLabelRect;
}

void Layer::setLabel(Label *pLabel) {
    m_pLabel = pLabel;
}

Label* Layer::getLabel() {
    return m_pLabel;
}

Label *Layer::addLabel(QString sName) {
    Label *pLabel = new Label;
    pLabel->SetName(sName);
    setLabel(pLabel);
    return pLabel;
}

ZLabel *Layer::addZLabel(const int &iNumber) {
    ZLabel *pLabel = new ZLabel;
    pLabel->setZLayer(iNumber);
    setZLabel(pLabel);
    return pLabel;
}

void Layer::setZLabel(ZLabel *pLabel) {
    m_pZLabel = pLabel;
}

ZLabel* Layer::getZLabel() {
	return m_pZLabel;
}

void Layer::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) {
    float   fTextBoxHeight = 24.f;
    QRectF  boundingEdge(m_BoundingRect.x(), m_BoundingRect.y(), m_BoundingRect.width(), m_BoundingRect.height()+fTextBoxHeight);

    QLinearGradient linearGrad(QPointF(0, m_BoundingRect.y()), QPointF(0, m_BoundingRect.y()+m_BoundingRect.height()/2));
    linearGrad.setSpread(QGradient::ReflectSpread);
    linearGrad.setColorAt(0, Qt::darkGray);
    linearGrad.setColorAt(1, Qt::lightGray);

    painter->setPen(QPen(Qt::black, 0));
    painter->setBrush(Qt::NoBrush);
    painter->drawRect(boundingEdge);

    painter->setPen(Qt::NoPen);
    painter->setBrush(linearGrad);
    painter->drawRect(m_BoundingRect);
}

QRectF Layer::boundingRect() const {
    float fYmin = 9999;
    float fXmin = 9999;

    float fYmax = -9999;
    float fXmax = -9999;

    foreach(Node *node, m_NodeList) {
        QPointF pos = node->pos();

        float fW = node->getWidth();

        if(fYmin > pos.y() ) {
            fYmin = pos.y()-fW;
        }
        if(fXmin > pos.x() ) {
            fXmin = pos.x()-fW;
        }
        if(fYmax < pos.y() ) {
            fYmax = pos.y()+fW;
        }
        if(fXmax < pos.x() ) {
            fXmax = pos.x()+fW;
        }
    }

    QRectF boundingRect = QRectF(fXmin, fYmin, fXmax-fXmin, fYmax-fYmin);
    if(boundingRect.width() < m_pLabel->GetName().size()*18) {
        boundingRect.setWidth(m_pLabel->GetName().size()*18);
    }

    return boundingRect;
}

void Layer::shift(int dX, int dY) {
    for(int i = 0; i < nodes().size(); i++) {
        QPointF pos = nodes().at(i)->scenePos();
        pos.setX(pos.x() + dX);
        pos.setY(pos.y() + dY);
        nodes().at(i)->setPos(pos);
    }
    adjust();
}

/*
void Layer::addNodes(const unsigned int &iNodes, const QString &sName) {
   SetName(sName);

    for(unsigned int i = 0; i < iNodes; i++) {
        Node *pNode = new Node;
        pNode->setPos(i*24, 0);
        pNode->setLayer(this);
        addNode(pNode);
    }
    adjust();
}
*/

void Layer::setScene(Scene* pScene) {
    m_pScene = pScene;
}

QList<Edge*> Layer::Connect(Layer* pDest) {
    QList<Edge*> lEdges;
    for(int i = 0; i < nodes().size(); i++) {
        Node *pSrc = nodes().at(i);
        for(int j = 0; j < pDest->nodes().size(); j++) {
            Node *pDst = pDest->nodes().at(j);
            Edge *pEdge = new Edge(pSrc, pDst);
            lEdges << pEdge;
        }
    }
    return lEdges;
}

void Layer::mousePressEvent(QGraphicsSceneMouseEvent *event) {
    /*foreach (Node *node, m_NodeList) {
        node->setSelectedAsGroup(true);
        node->update();
    }*/

    adjust();
    QGraphicsItem::mousePressEvent(event);
}

void Layer::mouseReleaseEvent(QGraphicsSceneMouseEvent *event) {
    /*foreach (Node *node, m_NodeList) {
        node->setSelectedAsGroup(false);
        node->update();
    }*/

    adjust();
    QGraphicsItem::mouseReleaseEvent(event);
}

void Layer::mouseMoveEvent(QGraphicsSceneMouseEvent *event) {
    adjust();
    QGraphicsItem::mouseMoveEvent(event);
}

void Layer::addNodes(int iNumber) {
    int iNeuronsPerLine = 16;
    int iLines = m_NodeList.size()/iNeuronsPerLine;
    int iSlots = iLines*iNeuronsPerLine;

    int iRow = m_NodeList.size() - iSlots - 1;
    if (m_pScene != NULL) {
        for(int i = 0; i < iNumber; i++) {
            QPointF pos = m_NodeList.first()->pos();
            Node *pNode = new Node;
            pNode->setLayer(this);

            int iLine = m_NodeList.size() / iNeuronsPerLine;
            if((iLine+1)*iNeuronsPerLine - m_NodeList.size() == iNeuronsPerLine)
                iRow = 0;
            else iRow++;

            pNode->setPos(pos.x()+(iRow*(8+pNode->getWidth())), pos.y()+(iLine*(8+pNode->getWidth())) );
            addNode(pNode);
            m_pScene->addNode(pNode);
        }
        adjust();
    }
}

void Layer::mouseDoubleClickEvent ( QGraphicsSceneMouseEvent * event ) {
    bool ok;
    int iNumber = QInputDialog::getInt(0, QObject::tr("Add neurons to layer"),
                                 QObject::tr("Number of neurons:"), 1, 0, 128*128, 1, &ok);

    if(ok)
        addNodes(iNumber);

    QGraphicsItem::mouseDoubleClickEvent(event);
}

