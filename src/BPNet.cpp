/*
 * Net.cpp
 *
 *  Created on: 30.05.2009
 *      Author: Daniel <dgrat> Frenzel
 */

#include <iostream>
#include <cassert>
#include <algorithm>
#include <omp.h>
//own classes
#include "math/Random.h"
#include "math/Functions.h"
#include "containers/TrainingSet.h"
#include "containers/ConTable.h"
#include "Edge.h"
#include "BPNeuron.h"
#include "BPLayer.h"
#include "BPNet.h"
#include "containers/ConTable.h"

using namespace ANN;


bool smallestFunctor(AbsLayer *i, AbsLayer *j) {
	return ( ((ANN::BPLayer*)i)->GetZLayer() < ((ANN::BPLayer*)j)->GetZLayer() );
}

BPNet::BPNet() {
	m_fTypeFlag 		= ANNetBP;
	SetTransfFunction(&ANN::Functions::fcn_log); 	// TODO not nice
}

BPNet::BPNet(ANN::BPNet *pNet) //: AbsNet(pNet)
{
	assert( pNet != NULL );
	//*this = *GetSubNet( 0, pNet->GetLayers().size()-1 );
	//m_fTypeFlag 	= ANNetBP;

	*this = *pNet;
}

BPNet::~BPNet() {
}

void BPNet::AddLayer(const unsigned int &iSize, const LayerTypeFlag &flType) {
	AbsNet::AddLayer( new BPLayer(iSize, flType) );
}

void BPNet::CreateNet(const ConTable &Net) {
	std::cout<<"Create BPNet"<<std::endl;

	/*
	 * Init
	 */
	unsigned int iDstNeurID 	= 0;
	unsigned int iDstLayerID 	= 0;
	unsigned int iSrcLayerID 	= 0;

	float fEdgeValue			= 0.f;

	AbsLayer *pDstLayer 		= NULL;
	AbsLayer *pSrcLayer 		= NULL;
	AbsNeuron *pDstNeur 		= NULL;
	AbsNeuron *pSrcNeur 		= NULL;

	/*
	 * For all nets necessary: Create Connections (Edges)
	 */
	AbsNet::CreateNet(Net);

	/*
	 * Support z-layers
	 */
	for(unsigned int i = 0; i < m_lLayers.size(); i++) {
		BPLayer *curLayer = ( (BPLayer*)GetLayer(i) );
		curLayer->SetZLayer(Net.ZValOfLayer[i]);
	}

	/*
	 * Only for back propagation networks
	 */
	if(Net.NetType == ANNetBP) {
		for(unsigned int i = 0; i < Net.BiasCons.size(); i++) {
			iDstNeurID 	= Net.BiasCons.at(i).m_iDstNeurID;
			iDstLayerID = Net.BiasCons.at(i).m_iDstLayerID;
			iSrcLayerID = Net.BiasCons.at(i).m_iSrcLayerID;
			if(iDstNeurID < 0 || iDstLayerID < 0 || GetLayers().size() < iDstLayerID || GetLayers().size() < iSrcLayerID) {
				return;
			}
			else {
				fEdgeValue 	= Net.BiasCons.at(i).m_fVal;

				pDstLayer 	= ( (BPLayer*)GetLayer(iDstLayerID) );
				pSrcLayer 	= ( (BPLayer*)GetLayer(iSrcLayerID) );
				pSrcNeur 	= ( (BPLayer*)pSrcLayer)->GetBiasNeuron();
				pDstNeur 	= pDstLayer->GetNeuron(iDstNeurID);

				ANN:Edge *pEdge = new ANN::Edge(pSrcNeur, pDstNeur, fEdgeValue);
				pSrcNeur->AddConO(pEdge);
				pDstNeur->AddConI(pEdge);
				pDstNeur->SetBiasEdge(pEdge);
			}
		}
	}
}

void BPNet::AddLayer(BPLayer *pLayer) {
	AbsNet::AddLayer(pLayer);

	if( ( (BPLayer*)pLayer)->GetFlag() & ANLayerInput ) {
		m_pIPLayer = pLayer;
	}
	else if( ( (BPLayer*)pLayer)->GetFlag() & ANLayerOutput ) {
		m_pOPLayer = pLayer;
	}
	else {
	}
}

/*
 * TODO better use of copy constructors
 */
BPNet *BPNet::GetSubNet(const unsigned int &iStartID, const unsigned int &iStopID) {
	assert( iStopID < GetLayers().size() );
	assert( iStartID >= 0 );

	/*
	 * Return value
	 */
	BPNet *pNet = new BPNet;

	/*
	 * Create layers like in pNet
	 */
	for(unsigned int i = iStartID; i <= iStopID; i++) {
		BPLayer *pLayer = new BPLayer( ( (BPLayer*)GetLayer(i) ) );
		if( i == iStartID && !(( (BPLayer*)GetLayer(i) )->GetFlag() & ANLayerInput) )
			pLayer->AddFlag( ANLayerInput );
		if( i == iStopID && !(( (BPLayer*)GetLayer(i) )->GetFlag() & ANLayerOutput) )
			pLayer->AddFlag( ANLayerOutput );

		pNet->AbsNet::AddLayer( pLayer );
	}

	BPLayer 	*pCurLayer;
	AbsNeuron 	*pCurNeuron;
	Edge 		*pCurEdge;
	for(unsigned int i = iStartID; i <= iStopID; i++) { 	// layers ..
		// NORMAL NEURON
		pCurLayer = ( (BPLayer*)GetLayer(i) );
		for(unsigned int j = 0; j < pCurLayer->GetNeurons().size(); j++) { 		// neurons ..
			pCurNeuron = pCurLayer->GetNeurons().at(j);
			AbsNeuron *pSrcNeuron = ( (BPLayer*)pNet->GetLayer(i) )->GetNeuron(j);
			for(unsigned int k = 0; k < pCurNeuron->GetConsO().size(); k++) { 			// edges ..
				pCurEdge = pCurNeuron->GetConO(k);

				// get iID of the destination neuron of the (next) layer i+1 (j is iID of (the current) layer i)
				int iDestNeurID 	= pCurEdge->GetDestinationID(pCurNeuron);
				int iDestLayerID 	= pCurEdge->GetDestination(pCurNeuron)->GetParent()->GetID();

				// copy edge
				AbsNeuron *pDstNeuron = pNet->GetLayers().at(iDestLayerID)->GetNeuron(iDestNeurID);

				// create edge
				Connect( pSrcNeuron, pDstNeuron,
						pCurEdge->GetValue(),
						pCurEdge->GetMomentum(),
						pCurEdge->GetAdaptationState() );
			}
		}

		// BIAS NEURON
		if( ( (BPLayer*)GetLayer(i) )->GetBiasNeuron() != NULL) {	// importt requirement, else access violation
			pCurLayer 	= ( (BPLayer*)GetLayer(i) );
			pCurNeuron = pCurLayer->GetBiasNeuron();
			BPNeuron *pBiasNeuron 	= ( (BPLayer*)pNet->GetLayer(i) )->GetBiasNeuron();

			for(unsigned int k = 0; k < pCurNeuron->GetConsO().size(); k++) {
				pCurEdge = pCurNeuron->GetConO(k);

				int iDestNeurID 	= pCurEdge->GetDestinationID(pCurNeuron);
				int iDestLayerID 	= pCurEdge->GetDestination(pCurNeuron)->GetParent()->GetID();

				// copy edge
				AbsNeuron *pDstNeuron 	= pNet->GetLayers().at(iDestLayerID)->GetNeuron(iDestNeurID);

				// create edge
				Connect( pBiasNeuron, pDstNeuron,
						pCurEdge->GetValue(),
						pCurEdge->GetMomentum(),
						pCurEdge->GetAdaptationState() );
			}
		}
	}

	// Import further properties
	if( GetTransfFunction() )
		pNet->SetTransfFunction( GetTransfFunction() );
	if( GetTrainingSet() )
		pNet->SetTrainingSet( GetTrainingSet() );
	pNet->SetLearningRate( GetLearningRate() );
	pNet->SetMomentum( GetMomentum() );

	return pNet;
}

void BPNet::PropagateFW() {
	for(unsigned int i = 1; i < m_lLayers.size(); i++) {
		BPLayer *curLayer = ( (BPLayer*)GetLayer(i) );
		#pragma omp parallel for
		for(int j = 0; j < static_cast<int>( curLayer->GetNeurons().size() ); j++) {
			curLayer->GetNeuron(j)->CalcValue();
		}
	}
}


void BPNet::PropagateBW() {
	/*
	 * Calc error delta based on the difference of output from wished result
	 */
	for(int i = m_lLayers.size()-1; i >= 0; i--) {
		BPLayer *curLayer = ( (BPLayer*)GetLayer(i) );
		#pragma omp parallel for
		for(int j = 0; j < static_cast<int>( curLayer->GetNeurons().size() ); j++) {
			curLayer->GetNeuron(j)->AdaptEdges();
		}

		#pragma omp parallel
		if(curLayer->GetBiasNeuron() != NULL) {
			curLayer->GetBiasNeuron()->AdaptEdges();
		}
	}
}

std::vector<float> BPNet::TrainFromData(const unsigned int &iCycles, const float &fTolerance, const bool &bBreak, float &fProgress) {
	bool bZSort = false;
	for(int i = 0; i < m_lLayers.size(); i++) {
		if(((BPLayer*)m_lLayers[i])->GetZLayer() > -1)
			bZSort = true;
		else {
			bZSort = false;
			break;
		}
	}
	if(bZSort) {
		std::sort(m_lLayers.begin(), m_lLayers.end(), smallestFunctor);
		// The IDs of the layers must be set according to Z-Value
		for(int i = 0; i < m_lLayers.size(); i++) {
			m_lLayers.at(i)->SetID(i);
		}
	}

	return AbsNet::TrainFromData(iCycles, fTolerance, bBreak, fProgress);
}

void BPNet::SetLearningRate(const float &fVal)
{
	m_fLearningRate = fVal;
	#pragma omp parallel for
	for(int i = 0; i < static_cast<int>(m_lLayers.size() ); i++) {
		( (BPLayer*)GetLayer(i) )->SetLearningRate(fVal);
	}
}

float BPNet::GetLearningRate() const {
	return m_fLearningRate;
}

void BPNet::SetMomentum(const float &fVal) {
	m_fMomentum = fVal;
	#pragma omp parallel for
	for(int i = 0; i < static_cast<int>(m_lLayers.size() ); i++) {
		( (BPLayer*)GetLayer(i) )->SetMomentum(fVal);
	}
}

float BPNet::GetMomentum() const {
	return m_fMomentum;
}

void BPNet::SetWeightDecay(const float &fVal) {
	m_fWeightDecay = fVal;
	#pragma omp parallel for
	for(int i = 0; i < static_cast<int>(m_lLayers.size() ); i++) {
		( (BPLayer*)GetLayer(i) )->SetWeightDecay(fVal);
	}
}

float BPNet::GetWeightDecay() const {
	return m_fWeightDecay;
}
