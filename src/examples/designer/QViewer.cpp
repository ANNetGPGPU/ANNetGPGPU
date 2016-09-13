#include <QViewer.h>
#include <QNode.h>
#include <QEdge.h>
#include <QLayer.h>
#include <QLabel.h>
#include <QScene.h>
#include <iostream>


Viewer::Viewer(QWidget *parent) : QGraphicsView(parent)
{
    m_pScene = new Scene(this);
    m_pScene->setItemIndexMethod(QGraphicsScene::NoIndex);
    setScene(m_pScene);

//    setCacheMode(QGraphicsView::CacheBackground);
    setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
    setRenderHints(QPainter::Antialiasing | QPainter::NonCosmeticDefaultPen);
//    setTransformationAnchor(AnchorUnderMouse);

    //setRubberBandSelectionMode(Qt::ContainsItemShape);
    setDragMode(QGraphicsView::RubberBandDrag);

    setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    ////////////////////////////////////////////////////////////////////////////////////////
    m_pScene->setSceneRect(0, 0, 10000, 10000);
    SetCenter(QPointF(5000.0, 5000.0)); //A modified version of centerOn(), handles special cases

    scale(qreal(0.8), qreal(0.8));
    setMinimumSize(400, 400);
    setWindowTitle(tr("ANNetDesigner"));

    //setBackgroundBrush(QBrush(QColor(250,250,250)));

    m_bStartConnect = false;
    connect(this, SIGNAL(si_selectionChanged()), this, SLOT(sl_waitForDest()) );
}

Viewer::~Viewer() {

}

void Viewer::sl_createConnections() {
    if(m_bStartConnect) {
        m_bStartConnect = false;
        return;
    }

    m_bStartConnect = false;
    m_lNodesSrc.clear();
    // Look for nodes
    foreach(Layer *pLayer, getScene()->layers() ) {
       foreach(Node *pNode, pLayer->nodes() ) {
           if(pNode->isSelected()) {
               m_lNodesSrc << pNode;
               pNode->setSelectedAsGroup(true);
           }
       }
    }
    m_bStartConnect = true;
    update();
}

void Viewer::sl_addNeurons() {
    m_bStartConnect = false;

    QList<Layer*> lRawLayers;
    foreach(Layer *pLayer, getScene()->layers() ) {
       foreach(Node *pNode, pLayer->nodes() ) {
           if(pNode->isSelected())  {
               lRawLayers << pLayer;
           }
       }
    }
    QSet<Layer*> pLayers = QSet<Layer*>::fromList(lRawLayers);

    if(!pLayers.size()) // if nothing selected
        return;         // return

    bool ok;
    int iNumber = QInputDialog::getInt(0, QObject::tr("Add neurons to layers"),
                                 QObject::tr("Number of neurons:"), 1, 0, 128*128, 1, &ok);

    if(ok) {
        QSet<Layer*>::const_iterator i = pLayers.constBegin();
        while (i != pLayers.constEnd()) {
            (*i)->addNodes(iNumber);
            ++i;
        }
    }
}

void Viewer::sl_waitForDest() {
    if(!m_bStartConnect || !m_pScene->selectedItems().size()) {
        m_bStartConnect = false;
        return;
    }

    m_bStartConnect = false;
    m_lNodesDest.clear();
    // Save new selection
    foreach(Layer *pLayer, getScene()->layers() ) {
       bool bInSelection = false;
       for(int i = 0; i < pLayer->nodes().size(); i++) {
           for(int j = 0; j < m_lNodesSrc.size(); j++) {
               if(m_lNodesSrc[j] == pLayer->nodes()[i]) {
                   bInSelection = true;
               }
           }
           if(!bInSelection && pLayer->nodes()[i]->isSelected()) {
               m_lNodesDest << pLayer->nodes()[i];
               pLayer->nodes()[i]->setSelectedAsGroup(false);
           }
       }
    }
    // reset old selection
    foreach(Node *pNode, m_lNodesSrc) {
        pNode->setSelectedAsGroup(false);
    }

    // create edges now
    int i = 0;
    foreach(Node *pSrc, m_lNodesSrc) {
        foreach(Node *pDest, m_lNodesDest) {
            Edge *pEdge = new Edge(pSrc, pDest);
            m_pScene->addEdge(pEdge);
            i++;
        }
    }

    // disconnect to avoid any problems with event handling
    m_lNodesSrc.clear();
    m_lNodesDest.clear();
}

void Viewer::removeSCons() {
    foreach(Layer *pLayer, getScene()->layers() ) {
        foreach(Node *pNode, pLayer->nodes() ) {
            if(pNode->isSelected()) {
            	// remove edges
                foreach(Edge *pEdge, pNode->edgesI() ) {
                    pEdge->sourceNode()->removeEdge(pEdge);
                    pEdge->destNode()->removeEdge(pEdge);
                    m_pScene->removeEdge(pEdge);
                }
                foreach(Edge *pEdge, pNode->edgesO() ) {
                    pEdge->sourceNode()->removeEdge(pEdge);
                    pEdge->destNode()->removeEdge(pEdge);
                    m_pScene->removeEdge(pEdge);
                }
            }
        }
    }
}

void Viewer::sl_removeLayers() {
    m_bStartConnect = false;

    // remove edges
    m_bStartConnect = false;

    QList<Layer*> lRawLayers;
    foreach(Layer *pLayer, getScene()->layers() ) {
       foreach(Node *pNode, pLayer->nodes() ) {
           if(pNode->isSelected())  {
               lRawLayers << pLayer;
           }
       }
    }
    QSet<Layer*> pLayers = QSet<Layer*>::fromList(lRawLayers);

    if(!pLayers.size()) // if nothing selected
        return;         // return

    QSet<Layer*>::const_iterator i = pLayers.constBegin();
    while (i != pLayers.constEnd()) {
        m_pScene->removeLayer(*i);
        ++i;
    }

    m_pScene->adjust();
}

void Viewer::sl_removeNeurons() {
    m_bStartConnect = false;

    // remove nodes
    foreach(Layer *pLayer, getScene()->layers() ) {
       foreach(Node *pNode, pLayer->nodes() ) {
           if(pNode->isSelected())  {
        	   getScene()->removeNode(pNode);
           }
       }
    }
    m_pScene->adjust();

    // remove empty layers
    QList<Layer*> lRawLayers;
    foreach(Layer *pLayer, getScene()->layers() ) {
    	if(!pLayer->nodes().size()) {
    		lRawLayers << pLayer;
    	}
    }

    QSet<Layer*> pLayers = QSet<Layer*>::fromList(lRawLayers);

    if(!pLayers.size()) // if nothing selected
        return;         // return

    QSet<Layer*>::const_iterator i = pLayers.constBegin();
    while (i != pLayers.constEnd()) {
        m_pScene->removeLayer(*i);
        ++i;
    }
}

void Viewer::sl_removeConnections() {
    m_bStartConnect = false;

    // remove edges
    removeSCons();

    // reset selection
    foreach(Layer *pLayer, getScene()->layers() ) {
       foreach(Node *pNode, pLayer->nodes() ) {
		   pNode->setSelected(false);
		   pNode->setSelectedAsGroup(false);
       }
    }
    update();
}

void Viewer::sl_removeAllConnections() {
    m_bStartConnect = false;

    // reset selection
    foreach(Layer *pLayer, getScene()->layers() ) {
       foreach(Node *pNode, pLayer->nodes() ) {
		   pNode->setSelected(false);
		   pNode->setSelectedAsGroup(false);
       }
    }

    // remove edges
    foreach(Layer *pLayer, getScene()->layers() ) {
        foreach(Node *pNode, pLayer->nodes() ) {
        	// remove edges
            foreach(Edge *pEdge, pNode->edgesI() ) {
                pEdge->sourceNode()->removeEdge(pEdge);
                pEdge->destNode()->removeEdge(pEdge);
                m_pScene->removeEdge(pEdge);
            }
            foreach(Edge *pEdge, pNode->edgesO() ) {
                pEdge->sourceNode()->removeEdge(pEdge);
                pEdge->destNode()->removeEdge(pEdge);
                m_pScene->removeEdge(pEdge);
            }
        }
    }
    update();
}

Scene *Viewer::getScene() {
    return m_pScene;
}

/**
  * Zoom the view in and out.
  */
void Viewer::keyPressEvent ( QKeyEvent * event )  {
    //Scale the view ie. do the zoom
    double scaleFactor = 1.15; //How fast we zoom
    if(event->key() == Qt::Key_Plus) {
        //Zoom in
        scale(scaleFactor, scaleFactor);
    } else if(event->key() == Qt::Key_Minus) {
        //Zooming out
        scale(1.0 / scaleFactor, 1.0 / scaleFactor);
    }

    SetCenter(GetCenter() );
}

void Viewer::wheelEvent(QWheelEvent* event) {
    //Get the position of the mouse before scaling, in scene coords
    QPointF pointBeforeScale(mapToScene(event->pos()));

    //Get the original screen centerpoint
    QPointF screenCenter = GetCenter(); //CurrentCenterPoint; //(visRect.center());

    //Scale the view ie. do the zoom
    double scaleFactor = 1.15; //How fast we zoom
    if(event->delta() > 0) {
        //Zoom in
        scale(scaleFactor, scaleFactor);
    } else {
        //Zooming out
        scale(1.0 / scaleFactor, 1.0 / scaleFactor);
    }
    //Get the position after scaling, in scene coords
    QPointF pointAfterScale(mapToScene(event->pos()));

    //Get the offset of how the screen moved
    QPointF offset = pointBeforeScale - pointAfterScale;

    //Adjust to the new center for correct zooming
    QPointF newCenter = screenCenter + offset;
    SetCenter(newCenter);
}

void Viewer::mouseMoveEvent( QMouseEvent * event ) {
    LastPanPoint = event->pos();

    m_pScene->adjust();
    QGraphicsView::mouseMoveEvent(event);
}

void Viewer::mousePressEvent( QMouseEvent * event ) {
    LastPanPoint = event->pos();

    m_pScene->adjust();
    QGraphicsView::mousePressEvent(event);
}

void Viewer::mouseReleaseEvent( QMouseEvent * event ) {
    LastPanPoint = QPoint();

    if(m_bStartConnect && m_pScene->selectedItems().size())
        emit si_selectionChanged();

    m_pScene->adjust();
    QGraphicsView::mouseReleaseEvent(event);
}

/**
  * Sets the current centerpoint.  Also updates the scene's center point.
  * Unlike centerOn, which has no way of getting the floating point center
  * back, SetCenter() stores the center point.  It also handles the special
  * sidebar case.  This function will claim the centerPoint to sceneRec ie.
  * the centerPoint must be within the sceneRec.
  */
//Set the current centerpoint in the
void Viewer::SetCenter(const QPointF& centerPoint) {
    //Get the rectangle of the visible area in scene coords
    QRectF visibleArea = mapToScene(rect()).boundingRect();

    //Get the scene area
    QRectF sceneBounds = sceneRect();

    double boundX = visibleArea.width() / 2.0;
    double boundY = visibleArea.height() / 2.0;
    double boundWidth = sceneBounds.width() - 2.0 * boundX;
    double boundHeight = sceneBounds.height() - 2.0 * boundY;

    //The max boundary that the centerPoint can be to
    QRectF bounds(boundX, boundY, boundWidth, boundHeight);

    if(bounds.contains(centerPoint)) {
        //We are within the bounds
        CurrentCenterPoint = centerPoint;
    } else {
        //We need to clamp or use the center of the screen
        if(visibleArea.contains(sceneBounds)) {
            //Use the center of scene ie. we can see the whole scene
            CurrentCenterPoint = sceneBounds.center();
        } else {

            CurrentCenterPoint = centerPoint;

            //We need to clamp the center. The centerPoint is too large
            if(centerPoint.x() > bounds.x() + bounds.width()) {
                CurrentCenterPoint.setX(bounds.x() + bounds.width());
            } else if(centerPoint.x() < bounds.x()) {
                CurrentCenterPoint.setX(bounds.x());
            }

            if(centerPoint.y() > bounds.y() + bounds.height()) {
                CurrentCenterPoint.setY(bounds.y() + bounds.height());
            } else if(centerPoint.y() < bounds.y()) {
                CurrentCenterPoint.setY(bounds.y());
            }

        }
    }

    //Update the scrollbars
    centerOn(CurrentCenterPoint);
}
