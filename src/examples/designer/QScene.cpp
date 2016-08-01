#include <gui/QScene.h>
#include <gui/QNode.h>
#include <gui/QEdge.h>
#include <gui/QLayer.h>
#include <gui/QLabel.h>
#include <gui/QZLabel.h>
#include <ANMath>
#include <ANContainers>
#include <iostream>
#include <cassert>


Scene::Scene(QObject *parent) : QGraphicsScene(parent)
{
	m_pANNet = new ANN::BPNet;
}

ANN::BPNet *Scene::getANNet(bool bDial) {
	m_pANNet->EraseAll();

	int LayerTypeFlag 	= -1;
	int iSize 			= -1;

	/**
	 * Checke whether anything to do
	 */
	if(!m_lLayers.size() && bDial) {
		QMessageBox msgBox;
		msgBox.setText("There is no network to create.");
		msgBox.exec();

		return NULL;
	}

	/**
	 * Create layers for neural net
	 */
	foreach(Layer *pLayer, m_lLayers) {
		LayerTypeFlag = pLayer->getLabel()->getType();
		iSize = pLayer->nodes().size();

		assert(iSize > 0);	// shouldn't happen

		int iZ = pLayer->getZLabel()->getZLayer();

		if(iZ < 0 && bDial) {
			QMessageBox msgBox;
			msgBox.setText("Z-values must be set for all layers.");
			msgBox.exec();

			return NULL;
		}
		if(LayerTypeFlag < 0 && bDial) {
			QMessageBox msgBox;
			msgBox.setText("Type of layer must be set for all layers.");
			msgBox.exec();

			return NULL;
		}
	}
	
   /**
	* Build connections
	*/
	ANN::ConTable Net;
	Net.NetType 	= ANN::ANNetBP;
	Net.NrOfLayers 	= m_lLayers.size();

	foreach(Layer *pLayer, m_lLayers) {
		Net.SizeOfLayer.push_back(pLayer->nodes().size() );
		Net.ZValOfLayer.push_back(pLayer->getZLabel()->getZLayer() );
		Net.TypeOfLayer.push_back(pLayer->getLabel()->getType() );

		foreach(Node *pNode, pLayer->nodes() ) {
			ANN::NeurDescr neuron;
			neuron.m_iLayerID 		= pLayer->getID();
			neuron.m_iNeurID 		= pNode->getID();
			Net.Neurons.push_back(neuron);

			foreach(Edge *pEdge, pNode->edgesO() ) {
				ANN::ConDescr edge;
				edge.m_iSrcLayerID 	= pEdge->sourceNode()->getLayer()->getID();
				edge.m_iDstLayerID 	= pEdge->destNode()->getLayer()->getID();
				edge.m_iSrcNeurID 	= pEdge->sourceNode()->getID();
				edge.m_iDstNeurID 	= pEdge->destNode()->getID();
				edge.m_fVal 		= ANN::RandFloat(-0.5f, 0.5f);

				Net.NeurCons.push_back(edge);
			}
		}
	}
	// Delete old and create new ANN::BPNet
	if(m_pANNet)
		delete m_pANNet;
	m_pANNet = new ANN::BPNet;
	m_pANNet->CreateNet(Net);

    // Update ANN::BPNet
    emit(si_netChanged(m_pANNet));

	return m_pANNet;
}

void Scene::setANNet(ANN::BPNet &Net) {
	m_pANNet = &Net;
	int iZLayer = 0;

	// Add layers to the scene
	foreach(ANN::AbsLayer *pLayer, m_pANNet->GetLayers()) {
		Layer *pSceneLayer = addLayer(	pLayer->GetNeurons().size(),
										QPointF(5000+ANN::RandFloat(-250, 250), 5000+ANN::RandFloat(-250, 250)),
										"Type not set!");

		pSceneLayer->getLabel()->setType(pLayer->GetFlag());
		pSceneLayer->getZLabel()->setZLayer(iZLayer);

		iZLayer++;
	}

	// Bruteforce free positions to avoid collisions
	foreach(Layer *pSceneOne, m_lLayers) {
		foreach(Layer *pSceneOther, m_lLayers) {
			if(pSceneOne == pSceneOther) {
				continue;
			}
			else {
				while(pSceneOne->collidesWithItem(pSceneOther) ) {
					pSceneOther->shift(ANN::RandFloat(-250, 250), ANN::RandFloat(-250, 250));
				}
			}
		}
	}

	// Setup edges
	foreach(ANN::AbsLayer *pLayer, m_pANNet->GetLayers() ) {
		foreach(ANN::AbsNeuron *pNeuron, pLayer->GetNeurons() ) {
			int iLayerID 	= pLayer->GetID();
			int iNeurID 	= pNeuron->GetID();

			foreach(ANN::Edge *pEdge, pNeuron->GetConsO() ) {
				int iSrcLayerID 	= iLayerID;
				int iDstLayerID 	= pEdge->GetDestination(pNeuron)->GetParent()->GetID();
				int iSrcNeurID 		= iNeurID;
				int iDstNeurID 		= pEdge->GetDestination(pNeuron)->GetID();

				Layer *pSrcLayer 	= m_lLayers.at(iSrcLayerID);
				Layer *pDstLayer 	= m_lLayers.at(iDstLayerID);
				Node *pSrcNode 		= pSrcLayer->nodes().at(iSrcNeurID);
				Node *pDstNode 		= pDstLayer->nodes().at(iDstNeurID);

	            Edge *pNewEdge = new Edge(pSrcNode, pDstNode);
	            addEdge(pNewEdge);
			}
		}
	}
}

void Scene::adjust() {
    foreach (Edge *edge, m_lEdges)
        edge->adjust();
    foreach (Layer *layer, m_lLayers)
        layer->adjust();
}

void Scene::refreshLayerIDs() {
    for(unsigned int i = 0; i < m_lLayers.size(); i++) {
    	Layer *pLayer = m_lLayers.at(i);
    	pLayer->setID(i);
    }
}

Layer* Scene::addLayer(const unsigned int &iNodes, const QPointF &fPos, const QString &sName) {
    Layer *pLayer = new Layer;
    pLayer->setScene(this);

    // Add one node only on the given position
    Node *pNode = new Node;
    pNode->setPos(fPos.x(), fPos.y());
    pLayer->addNode(pNode);
    pNode->setLayer(pLayer);
    addNode(pNode);

    // Add the layer to the scene
    addItem(pLayer);
    addItem(pLayer->addLabel(sName));
    addItem(pLayer->addZLabel(-1));
    m_lLayers << pLayer;
    pLayer->setID(m_lLayers.size()-1);

    // Add the rest of the nodes to the layer
    pLayer->addNodes(iNodes-1);
    pLayer->adjust();

    return pLayer;
}

void Scene::addNode(Node* pNode) {
    m_lNodes << pNode;
    addItem(pNode);
}

void Scene::addEdge(Edge* pEdge) {
    m_lEdges << pEdge;
    addItem(pEdge);
}

void Scene::clearAll() {
	foreach(Layer *pLayer, m_lLayers) {
		removeLayer(pLayer);
	}
}

void Scene::removeEdge(Edge* pDelEdge) {
    removeItem(pDelEdge);

    QList<Edge*> pNewList;
    foreach(Edge *pEdge, m_lEdges) {
        if(pEdge != pDelEdge)
            pNewList << pEdge;
    }
    m_lEdges = pNewList;
}

void Scene::removeNode(Node* pDelNode) {
    removeItem(pDelNode);
    pDelNode->getLayer()->removeNode(pDelNode);

	// remove edges
    foreach(Edge *pEdge, pDelNode->edgesI() ) {
        pEdge->sourceNode()->removeEdge(pEdge);
        pEdge->destNode()->removeEdge(pEdge);
        removeEdge(pEdge);
    }
    foreach(Edge *pEdge, pDelNode->edgesO() ) {
        pEdge->sourceNode()->removeEdge(pEdge);
        pEdge->destNode()->removeEdge(pEdge);
        removeEdge(pEdge);
    }

    QList<Node*> pNewList;
    foreach(Node *pNode, m_lNodes) {
        if(pNode != pDelNode)
            pNewList << pNode;
    }
    m_lNodes = pNewList;
}

void Scene::removeLayer(Layer* pDelLayer) {
	removeItem(pDelLayer->getZLabel());
    removeItem(pDelLayer->getLabel());
    removeItem(pDelLayer);

    foreach(Node *pNode, pDelLayer->nodes()) {
        removeNode(pNode);
    }

    QList<Layer*> pNewList;
    foreach(Layer *pLayer, m_lLayers) {
        if(pLayer != pDelLayer)
            pNewList << pLayer;
    }
    m_lLayers = pNewList;

    /*
     * Refresh the IDs of the layers in the GUI
     */
    refreshLayerIDs();
}

QList<Edge*> Scene::edges() {
    return m_lEdges;
}

QList<Node*> Scene::nodes() {
    return m_lNodes;
}

QList<Layer*> Scene::layers() {
    return m_lLayers;
}
