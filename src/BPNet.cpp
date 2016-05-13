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

#include "BPNeuron.h"
#include "BPLayer.h"
#include "BPNet.h"
#include "Common.h"
#include "Edge.h"
#include "containers/ConTable.h"
#include "containers/TrainingSet.h"
#include "containers/ConTable.h"
#include "math/Functions.h"
#include "math/Random.h"

using namespace ANN;


template <class Type>
bool smallestFunctor(AbsLayer<Type> *i, AbsLayer<Type> *j) {
	return ( ((ANN::BPLayer<Type>*)i)->GetZLayer() < ((ANN::BPLayer<Type>*)j)->GetZLayer() );
}

template <class Type>
BPNet<Type>::BPNet() {
	this->m_fTypeFlag = ANNetBP;
	this->SetTransfFunction(&ANN::Functions::fcn_log); 	// TODO not nice
}

template <class Type>
BPNet<Type>::BPNet(BPNet<Type> *pNet) //: AbsNet(pNet)
{
	assert( pNet != NULL );
	//*this = *GetSubNet( 0, pNet->GetLayers().size()-1 );
	//m_fTypeFlag 	= ANNetBP;

	*this = *pNet;
}

template <class Type>
BPNet<Type>::~BPNet() {
}

template <class Type>
void BPNet<Type>::AddLayer(const unsigned int &iSize, const LayerTypeFlag &flType) {
	this->AddLayer( new BPLayer<Type>(iSize, flType, -1) );
}

template <class Type>
void BPNet<Type>::CreateNet(const ConTable &Net) {
	std::cout<<"Create BPNet"<<std::endl;

	/*
	 * Init
	 */
	unsigned int iDstNeurID 	= 0;
	unsigned int iDstLayerID 	= 0;
	unsigned int iSrcLayerID 	= 0;

	Type fEdgeValue			= 0.f;

	AbsLayer<Type> *pDstLayer 	= NULL;
	AbsLayer<Type> *pSrcLayer 	= NULL;
	AbsNeuron<Type> *pDstNeur 	= NULL;
	AbsNeuron<Type> *pSrcNeur 	= NULL;

	/*
	 * For all nets necessary: Create Connections (Edges)
	 */
	AbsNet<Type>::CreateNet(Net);

	/*
	 * Support z-layers
	 */
	for(unsigned int i = 0; i < this->m_lLayers.size(); i++) {
		BPLayer<Type> *curLayer = ((BPLayer<Type>*)this->GetLayer(i) );
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
			if(iDstNeurID < 0 || iDstLayerID < 0 || this->GetLayers().size() < iDstLayerID || this->GetLayers().size() < iSrcLayerID) {
				return;
			} else {
				fEdgeValue 	= Net.BiasCons.at(i).m_fVal;

				pDstLayer 	= ((BPLayer<Type>*)this->GetLayer(iDstLayerID) );
				pSrcLayer 	= ((BPLayer<Type>*)this->GetLayer(iSrcLayerID) );
				pSrcNeur 	= ((BPLayer<Type>*)pSrcLayer)->GetBiasNeuron();
				pDstNeur 	= pDstLayer->GetNeuron(iDstNeurID);

				ANN:Edge<Type> *pEdge = new ANN::Edge<Type>(pSrcNeur, pDstNeur, fEdgeValue, 0, true);
				pSrcNeur->AddConO(pEdge);
				pDstNeur->AddConI(pEdge);
				pDstNeur->SetBiasEdge(pEdge);
			}
		}
	}
}

template <class Type>
void BPNet<Type>::AddLayer(BPLayer<Type> *pLayer) {
	AbsNet<Type>::AddLayer(pLayer);

	if( ( (BPLayer<Type>*)pLayer)->GetFlag() & ANLayerInput ) {
		this->m_pIPLayer = pLayer;
	}
	else if( ( (BPLayer<Type>*)pLayer)->GetFlag() & ANLayerOutput ) {
		this->m_pOPLayer = pLayer;
	}
	else {
	}
}

/*
 * TODO better use of copy constructors
 */
template <class Type>
BPNet<Type> *BPNet<Type>::GetSubNet(const unsigned int &iStartID, const unsigned int &iStopID) {
	assert( iStopID < this->GetLayers().size() );
	assert( iStartID >= 0 );

	/*
	 * Return value
	 */
	BPNet<Type> *pNet = new BPNet<Type>;

	/*
	 * Create layers like in pNet
	 */
	for(unsigned int i = iStartID; i <= iStopID; i++) {
		BPLayer<Type> *pLayer = new BPLayer<Type>((BPLayer<Type>*)this->GetLayer(i), -1);
		
		if( i == iStartID && !(( (BPLayer<Type>*)this->GetLayer(i) )->GetFlag() & ANLayerInput) )
			pLayer->AddFlag( ANLayerInput );
		if( i == iStopID && !(( (BPLayer<Type>*)this->GetLayer(i) )->GetFlag() & ANLayerOutput) )
			pLayer->AddFlag( ANLayerOutput );

		pNet->AbsNet<Type>::AddLayer( pLayer );
	}

	BPLayer<Type> *pCurLayer;
	AbsNeuron<Type> *pCurNeuron;
	Edge<Type> *pCurEdge;
	for(unsigned int i = iStartID; i <= iStopID; i++) { 	// layers ..
		// NORMAL NEURON
		pCurLayer = ( (BPLayer<Type>*)this->GetLayer(i) );
		for(unsigned int j = 0; j < pCurLayer->GetNeurons().size(); j++) { 		// neurons ..
			pCurNeuron = pCurLayer->GetNeurons().at(j);
			AbsNeuron<Type> *pSrcNeuron = ( (BPLayer<Type>*)pNet->GetLayer(i) )->GetNeuron(j);
			for(unsigned int k = 0; k < pCurNeuron->GetConsO().size(); k++) { 			// edges ..
				pCurEdge = pCurNeuron->GetConO(k);

				// get iID of the destination neuron of the (next) layer i+1 (j is iID of (the current) layer i)
				int iDestNeurID 	= pCurEdge->GetDestinationID(pCurNeuron);
				int iDestLayerID 	= pCurEdge->GetDestination(pCurNeuron)->GetParent()->GetID();

				// copy edge
				AbsNeuron<Type> *pDstNeuron = pNet->GetLayers().at(iDestLayerID)->GetNeuron(iDestNeurID);

				// create edge
				ANN::Connect( pSrcNeuron, pDstNeuron,
						pCurEdge->GetValue(),
						pCurEdge->GetMomentum(),
						pCurEdge->GetAdaptationState() );
			}
		}

		// BIAS NEURON
		if( ( (BPLayer<Type>*)this->GetLayer(i) )->GetBiasNeuron() != NULL) {	// importt requirement, else access violation
			pCurLayer 	= ( (BPLayer<Type>*)this->GetLayer(i) );
			pCurNeuron = pCurLayer->GetBiasNeuron();
			BPNeuron<Type> *pBiasNeuron 	= ( (BPLayer<Type>*)pNet->GetLayer(i) )->GetBiasNeuron();

			for(unsigned int k = 0; k < pCurNeuron->GetConsO().size(); k++) {
				pCurEdge = pCurNeuron->GetConO(k);

				int iDestNeurID 	= pCurEdge->GetDestinationID(pCurNeuron);
				int iDestLayerID 	= pCurEdge->GetDestination(pCurNeuron)->GetParent()->GetID();

				// copy edge
				AbsNeuron<Type> *pDstNeuron 	= pNet->GetLayers().at(iDestLayerID)->GetNeuron(iDestNeurID);

				// create edge
				ANN::Connect( pBiasNeuron, pDstNeuron,
						pCurEdge->GetValue(),
						pCurEdge->GetMomentum(),
						pCurEdge->GetAdaptationState() );
			}
		}
	}

	// Import further properties
	if(this->GetTransfFunction() ) {
		pNet->SetTransfFunction(this->GetTransfFunction() );
	}
	if(this->GetTrainingSet() ) {
		pNet->SetTrainingSet(this->GetTrainingSet() );
	}
	pNet->SetLearningRate(this->GetLearningRate() );
	pNet->SetMomentum(this->GetMomentum() );

	return pNet;
}

template <class Type>
void BPNet<Type>::PropagateFW() {
	for(unsigned int i = 1; i < this->m_lLayers.size(); i++) {
		BPLayer<Type> *curLayer = ((BPLayer<Type>*)this->GetLayer(i) );
		#pragma omp parallel for
		for(int j = 0; j < static_cast<int>(curLayer->GetNeurons().size() ); j++) {
			curLayer->GetNeuron(j)->CalcValue();
		}
	}
}

template <class Type>
void BPNet<Type>::PropagateBW() {
	/*
	 * Calc error delta based on the difference of output from wished result
	 */
	for(int i = this->m_lLayers.size()-1; i >= 0; i--) {
		BPLayer<Type> *curLayer = ( (BPLayer<Type>*)this->GetLayer(i) );
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

template <class Type>
std::vector<Type> BPNet<Type>::TrainFromData(const unsigned int &iCycles, const Type &fTolerance, const bool &bBreak, Type &fProgress) {
	bool bZSort = false;
	for(int i = 0; i < this->m_lLayers.size(); i++) {
		if(((BPLayer<Type>*)this->m_lLayers[i])->GetZLayer() > -1)
			bZSort = true;
		else {
			bZSort = false;
			break;
		}
	}
	if(bZSort) {
		std::sort(this->m_lLayers.begin(), this->m_lLayers.end(), smallestFunctor<Type>);
		// The IDs of the layers must be set according to Z-Value
		for(int i = 0; i < this->m_lLayers.size(); i++) {
			this->m_lLayers.at(i)->SetID(i);
		}
	}

	return AbsNet<Type>::TrainFromData(iCycles, fTolerance, bBreak, fProgress);
}

template <class Type>
void BPNet<Type>::SetLearningRate(const Type &fVal)
{
	this->m_fLearningRate = fVal;
	#pragma omp parallel for
	for(int i = 0; i < static_cast<int>(this->m_lLayers.size() ); i++) {
		( (BPLayer<Type>*)this->GetLayer(i) )->SetLearningRate(fVal);
	}
}

template <class Type>
Type BPNet<Type>::GetLearningRate() const {
	return this->m_fLearningRate;
}

template <class Type>
void BPNet<Type>::SetMomentum(const Type &fVal) {
	this->m_fMomentum = fVal;
	#pragma omp parallel for
	for(int i = 0; i < static_cast<int>(this->m_lLayers.size() ); i++) {
		( (BPLayer<Type>*)this->GetLayer(i) )->SetMomentum(fVal);
	}
}

template <class Type>
Type BPNet<Type>::GetMomentum() const {
	return this->m_fMomentum;
}

template <class Type>
void BPNet<Type>::SetWeightDecay(const Type &fVal) {
	this->m_fWeightDecay = fVal;
	#pragma omp parallel for
	for(int i = 0; i < static_cast<int>(this->m_lLayers.size() ); i++) {
		( (BPLayer<Type>*)this->GetLayer(i) )->SetWeightDecay(fVal);
	}
}

template <class Type>
Type BPNet<Type>::GetWeightDecay() const {
	return this->m_fWeightDecay;
}


template class BPNet<float>;
template class BPNet<double>;
template class BPNet<long double>;
template class BPNet<short>;
template class BPNet<int>;
template class BPNet<long>;
template class BPNet<long long>;
