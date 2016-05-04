/*
 * SOMNet.cpp
 *
 *  Created on: 26.02.2012
 *      Author: dgrat
 */

#include <vector>
#include <algorithm>
#include <cassert>
#include <limits>
#include <cmath>

#include <omp.h>

#include "Edge.h"

#include "SOMNet.h"
#include "SOMLayer.h"
#include "SOMNeuron.h"

#include "math/Functions.h"
#include "math/Random.h"

#include "containers/TrainingSet.h"
#include "containers/ConTable.h"
#include "containers/Centroid.h"

namespace ANN {

SOMNet::SOMNet() {
	m_pIPLayer 		= NULL;
	m_pOPLayer 		= NULL;
	m_pBMNeuron 		= NULL;

	m_iCycle 		= 0;
	m_fSigma0 		= 0.f;
	m_fSigmaT 		= 0.f;
	m_fLearningRate 	= 0.5f;

	m_iWidthI 		= 0.f;
	m_iHeightI 		= 0.f;
	m_iWidthO 		= 0.f;
	m_iHeightO 		= 0.f;
	
	// Conscience mechanism
	m_fConscienceRate 	= 0.f;

	// mexican hat shaped function for this SOM
	SetDistFunction(&Functions::fcn_gaussian);

	m_fTypeFlag 	= ANNetSOM;
}

SOMNet::SOMNet(AbsNet *pNet) {
	if(pNet == NULL)
		return;

	std::vector<unsigned int> vDimI = ((SOMLayer*)(pNet->GetIPLayer() ))->GetDim();
	std::vector<unsigned int> vDimO = ((SOMLayer*)(pNet->GetOPLayer() ))->GetDim();

	// Copy weights between neurons of the input and output layer
	ANN::F2DArray f2dEdges = pNet->GetOPLayer()->ExpEdgesIn();
	// Copy positions of the neurons in the output layer
	ANN::F2DArray f2dPosistions = pNet->GetOPLayer()->ExpPositions();
	// Create the net finally
	CreateSOM(vDimI, vDimO, f2dEdges, f2dPosistions);
	// Copy training set
	SetTrainingSet(pNet->GetTrainingSet() );

	m_fTypeFlag 	= ANNetSOM;
}

SOMNet::SOMNet(const std::vector<unsigned int> &vDimI, const std::vector<unsigned int> &vDimO) {
  	m_pIPLayer 		= NULL;
	m_pOPLayer 		= NULL;
	m_pBMNeuron 		= NULL;

	m_iCycle 		= 0;
	m_fSigma0 		= 0.f;
	m_fSigmaT 		= 0.f;
	m_fLearningRate 	= 0.5f;

	m_iWidthI 		= 0.f;
	m_iHeightI 		= 0.f;
	m_iWidthO 		= 0.f;
	m_iHeightO 		= 0.f;
	
	// Conscience mechanism
	m_fConscienceRate 	= 0.f;

	// mexican hat shaped function for this SOM
	SetDistFunction(&Functions::fcn_gaussian);

	m_fTypeFlag 	= ANNetSOM;
	
	CreateSOM(vDimI, vDimO);
}

SOMNet::SOMNet(	const unsigned int &iWidthI, const unsigned int &iHeightI,
		const unsigned int &iWidthO, const unsigned int &iHeightO) 
{
	m_pIPLayer 		= NULL;
	m_pOPLayer 		= NULL;
	m_pBMNeuron 		= NULL;

	m_iCycle 		= 0;
	m_fSigma0 		= 0.f;
	m_fSigmaT 		= 0.f;
	m_fLearningRate 	= 0.5f;

	m_iWidthI 		= 0.f;
	m_iHeightI 		= 0.f;
	m_iWidthO 		= 0.f;
	m_iHeightO 		= 0.f;
	
	// Conscience mechanism
	m_fConscienceRate 	= 0.f;

	// mexican hat shaped function for this SOM
	SetDistFunction(&Functions::fcn_gaussian);

	m_fTypeFlag 	= ANNetSOM;
	
	CreateSOM(iWidthI, iHeightI, iWidthO, iHeightO);
}

void SOMNet::AddLayer(AbsLayer *pLayer) {
	AbsNet::AddLayer(pLayer);
}

void SOMNet::AddLayer(const unsigned int &iSize, const LayerTypeFlag &flType) {
	AbsNet::AddLayer( new SOMLayer(iSize, flType) );
}

void SOMNet::CreateNet(const ConTable &Net) {
	std::cout<<"Create SOMNet"<<std::endl;

	/*
	 * For all nets necessary: Create Connections (Edges)
	 */
	AbsNet::CreateNet(Net);

	/*
	 * Set Positions
	 */
	for(unsigned int i = 0; i < Net.Neurons.size(); i++) {
		int iLayerID 	= Net.Neurons.at(i).m_iLayerID;
		int iNeurID 	= Net.Neurons.at(i).m_iNeurID;
		std::vector<float> vPos = Net.Neurons.at(i).m_vPos;
		GetLayer(iLayerID)->GetNeuron(iNeurID)->SetPosition(vPos);
	}
}

SOMNet::~SOMNet() {
	// TODO Auto-generated destructor stub
}

SOMNet *SOMNet::GetNet() {
	/*
	 * Return value
	 */
	SOMNet *pNet = new SOMNet;

	/*
	 * Create layers like in pNet
	 */
	for(unsigned int i = 0; i <= this->GetLayers().size(); i++) {
		SOMLayer *pLayer = new SOMLayer( ( (SOMLayer*)GetLayer(i) ) );
		pNet->AbsNet::AddLayer( pLayer );
	}

	SOMLayer 	*pCurLayer;
	AbsNeuron 	*pCurNeuron;
	Edge 		*pCurEdge;
	for(unsigned int i = 0; i <= this->GetLayers().size(); i++) {
		// NORMAL NEURON
		pCurLayer = ( (SOMLayer*)GetLayer(i) );
		for(unsigned int j = 0; j < pCurLayer->GetNeurons().size(); j++) { 		// neurons ..
			pCurNeuron = pCurLayer->GetNeurons().at(j);
			AbsNeuron *pSrcNeuron = ( (SOMLayer*)pNet->GetLayer(i) )->GetNeuron(j);
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
	}

	// Import further properties
	if( GetTransfFunction() )
		pNet->SetTransfFunction( pNet->GetTransfFunction() );
	if( GetTrainingSet() )
		pNet->SetTrainingSet( this->GetTrainingSet() );
	pNet->SetLearningRate( this->GetLearningRate() );

	return pNet;
}

void SOMNet::FindSigma0() {
	SOMLayer 	*pLayer 	= (SOMLayer*)GetOPLayer();
	SOMNeuron 	*pNeuron 	= (SOMNeuron*)pLayer->GetNeuron(0);
	unsigned int iSize 		= pLayer->GetNeurons().size();

	unsigned int iDim = pNeuron->GetPosition().size();
	std::vector<float> vDimMax(iDim, 0.f);
	std::vector<float> vDimMin(iDim, std::numeric_limits<float>::max() );

	// look in all the nodes
	for(unsigned int i = 0; i < iSize; i++) {
		//std::cout<< "Find m_fSigma0: "<< (float)(i+1)/(float)iSize*100.f <<" %" <<std::endl;

		pNeuron = (SOMNeuron*)pLayer->GetNeuron(i);
		// find the smallest and greatest positions in the network
		for(unsigned int j = 0; j < iDim; j++) {
			// find greatest coordinate
			vDimMin[j] = std::min(vDimMin[j], pNeuron->GetPosition().at(j) );
			vDimMax[j] = std::max(vDimMax[j], pNeuron->GetPosition().at(j) );
		}
	}
	std::sort(vDimMin.begin(), vDimMin.end() );
	std::sort(vDimMax.begin(), vDimMax.end() );

	// save in m_fSigma0
	m_fSigma0 = *(vDimMax.end()-1)+1 - *(vDimMin.begin()+1);
	m_fSigma0 /= 2.f;
}

void SOMNet::CreateSOM(const std::vector<unsigned int> &vDimI, const std::vector<unsigned int> &vDimO) {
	if(m_pIPLayer != NULL || m_pOPLayer != NULL) {
		AbsNet::EraseAll();
	}

	std::cout<< "Create input layer" <<std::endl;
	m_pIPLayer = new SOMLayer(vDimI, ANLayerInput);
	m_pIPLayer->SetID(0);
	AbsNet::AddLayer(m_pIPLayer);

	std::cout<< "Create output layer" <<std::endl;
	m_pOPLayer = new SOMLayer(vDimO, ANLayerOutput);
	m_pOPLayer->SetID(1);
	AbsNet::AddLayer(m_pOPLayer);

	std::cout<< "Connect layer .." <<std::endl;
	((SOMLayer*)m_pIPLayer)->ConnectLayer(m_pOPLayer);

	// find sigma0
	FindSigma0();
}

void SOMNet::CreateSOM(const std::vector<unsigned int> &vDimI, const std::vector<unsigned int> &vDimO,
		const F2DArray &f2dEdgeMat, const F2DArray &f2dNeurPos) {
	if(m_pIPLayer != NULL || m_pOPLayer != NULL) {
		AbsNet::EraseAll();
	}

	std::cout<< "Create input layer" <<std::endl;
	m_pIPLayer = new SOMLayer(vDimI, ANLayerInput);
	m_pIPLayer->SetID(0);
	AbsNet::AddLayer(m_pIPLayer);

	std::cout<< "Create output layer" <<std::endl;
	m_pOPLayer = new SOMLayer(vDimO, ANLayerOutput);
	m_pOPLayer->SetID(1);
	AbsNet::AddLayer(m_pOPLayer);

	std::cout<< "Connect layer .." <<std::endl;
	((SOMLayer*)m_pIPLayer)->ConnectLayer(m_pOPLayer, f2dEdgeMat);

	m_pOPLayer->ImpPositions(f2dNeurPos);

	// find sigma0
	FindSigma0();
}

void SOMNet::CreateSOM(	const unsigned int &iWidthI, const unsigned int &iHeightI,
						const unsigned int &iWidthO, const unsigned int &iHeightO)
{
	if(m_pIPLayer != NULL || m_pOPLayer != NULL) {
		AbsNet::EraseAll();
	}

	m_iWidthI 	= iWidthI;
	m_iHeightI 	= iHeightI;
	m_iWidthO 	= iWidthO;
	m_iHeightO 	= iHeightO;

	std::cout<< "Create input layer" <<std::endl;
	m_pIPLayer = new SOMLayer(iWidthI, iHeightI, ANLayerInput);
	m_pIPLayer->SetID(0);
	AbsNet::AddLayer(m_pIPLayer);

	std::cout<< "Create output layer" <<std::endl;
	m_pOPLayer = new SOMLayer(iWidthO, iHeightO, ANLayerOutput);
	m_pOPLayer->SetID(1);
	AbsNet::AddLayer(m_pOPLayer);

	std::cout<< "Connect layer .." <<std::endl;
	((SOMLayer*)m_pIPLayer)->ConnectLayer(m_pOPLayer);

	// find sigma0
	FindSigma0();
}

void SOMNet::TrainHelper(unsigned int i) {
	assert(i < GetTrainingSet()->GetNrElements() );
	
	SetInput(GetTrainingSet()->GetInput(i) );

	// Present the input vector to each node and determine the BMU
	FindBMNeuron();

	// Calculate the width of the neighborhood for this time step
	m_fSigmaT 		= m_DistFunction->rad_decay(m_fSigma0, m_iCycle, m_fLambda);
	m_fLearningRateT 	= m_DistFunction->lrate_decay(m_fLearningRate, m_iCycle, m_iCycles);

	// Adjust the weight vector of the BMU and its neighbors
	PropagateBW();
}

void SOMNet::Training(const unsigned int &iCycles, const TrainingMode &eMode) {
	assert(iCycles > 0);
	assert(m_fSigma0 > 0.f);
	if(GetTrainingSet() == NULL) {
		std::cout<<"No training set available!"<<std::endl;
		return;
	}
	
	m_iCycles 	= iCycles;
	m_fLambda 	= m_iCycles / log(m_fSigma0);

	int iMin 	= 0;
	int iMax 	= GetTrainingSet()->GetNrElements()-1;
	unsigned int iProgCount = 1;

	std::cout<< "Process the SOM now" <<std::endl;
	for(m_iCycle = 0; m_iCycle < static_cast<unsigned int>(m_iCycles); m_iCycle++) {
		if(m_iCycles >= 10) {
			if(((m_iCycle+1) / (m_iCycles/10)) == iProgCount && (m_iCycle+1) % (m_iCycles/10) == 0) {
				std::cout<<"Current training progress calculated by the CPU is: "<<iProgCount*10.f<<"%/Step="<<m_iCycle+1<<std::endl;
				iProgCount++;
			}
		} else {
			std::cout<<"Current training progress calculated by the CPU is: "<<(float)(m_iCycle+1.f)/(float)m_iCycles*100.f<<"%/Step="<<m_iCycle+1<<std::endl;
		}

		// The input vectors are presented to the network at random
		if(eMode == ANN::ANRandomMode) {
			unsigned int iRandID = RandInt(iMin, iMax);
			TrainHelper(iRandID);
		}
		// The input vectors are presented to the network in serial order
		else if(eMode == ANN::ANSerialMode) {
			for(unsigned int i = 0; i < GetTrainingSet()->GetNrElements(); i++) {
				TrainHelper(i);
			}
		}
	}
}

std::vector<Centroid> SOMNet::GetCentrOInpList() {
	std::vector<Centroid> vCentroids;
	for(unsigned int i = 0; i < GetTrainingSet()->GetNrElements(); i++) {
		vCentroids.push_back(Centroid() );
		SetInput(GetTrainingSet()->GetInput(i) );

		// Present the input vector to each node and determine the BMU
		FindBMNeuron();
		
		vCentroids[i].m_iBMUID 	= m_pBMNeuron->GetID();
		vCentroids[i].m_vCentroid.clear();
		for(unsigned int j = 0; j < m_pBMNeuron->GetConsI().size(); j++) {
			vCentroids[i].m_vCentroid.push_back(m_pBMNeuron->GetConI(j)->GetValue());
		}
		vCentroids[i].m_vInput = GetTrainingSet()->GetInput(i);
		vCentroids[i].m_fEucDist = sqrt(m_pBMNeuron->GetValue() );
	}
	return vCentroids;
}

std::vector<Centroid> SOMNet::GetCentroidList() {
	std::vector<Centroid> vCentroids;
	for(unsigned int i = 0; i < GetTrainingSet()->GetNrElements(); i++) {
		vCentroids.push_back(Centroid() );
		SetInput(GetTrainingSet()->GetInput(i) );

		// Present the input vector to each node and determine the BMU
		FindBMNeuron();
		
		vCentroids[i].m_iBMUID 	= m_pBMNeuron->GetID();
		vCentroids[i].m_vCentroid.clear();
		for(unsigned int j = 0; j < m_pBMNeuron->GetConsI().size(); j++) {
			vCentroids[i].m_vCentroid.push_back(m_pBMNeuron->GetConI(j)->GetValue());
		}
		vCentroids[i].m_vInput = std::vector<float>(0);
		vCentroids[i].m_fEucDist = -1.f;
	}
	// Count the number of centroids
	//std::sort(vCentroids.begin(), vCentroids.end() );
	//vCentroids.erase(std::unique(vCentroids.begin(), vCentroids.end()), vCentroids.end() );
	std::cout<<"Number of clusters found: "<<vCentroids.size()<<std::endl;
	return vCentroids;
}

void SOMNet::PropagateFW() {
	// TODO
}

void SOMNet::PropagateBW() {
	// Run through neurons
	#pragma omp parallel for
	for(int i = 0; i < static_cast<int>(m_pOPLayer->GetNeurons().size() ); i++) {
		// Set some values used below ..
		SOMNeuron *pNeuron 	= (SOMNeuron*)m_pOPLayer->GetNeuron(i);
		float fDist 		= pNeuron->GetDistance2Neur(*m_pBMNeuron);

		//std::cout<<"CPU influence: "<< m_DistFunction->distance(fDist, m_fSigmaT) <<std::endl;
		if(fDist < m_fSigmaT) {
			//calculate by how much weights get adjusted ..
			float fInfluence = m_DistFunction->distance(fDist, m_fSigmaT);
			pNeuron->SetInfluence(fInfluence);
			// .. and adjust them
			pNeuron->AdaptEdges();
		}
		//reduce the learning rate
		pNeuron->SetLearningRate(m_fLearningRateT);
	}
}

void SOMNet::SetLearningRate(const float &fVal) {
	m_fLearningRate = fVal;
	#pragma omp parallel for
	for(int i = 0; i < static_cast<int>(m_lLayers.size() ); i++) {
		( (SOMLayer*)GetLayer(i) )->SetLearningRate(fVal);
	}
}

float SOMNet::GetLearningRate() const {
	return m_fLearningRate;
}

void SOMNet::FindBMNeuron() {
	assert(m_pIPLayer != NULL && m_pOPLayer != NULL);

	float fCurVal 	= 0.f;
	float fSmallest = std::numeric_limits<float>::max();
	float fNrOfNeurons 	= (float)(m_pOPLayer->GetNeurons().size() );

	//#pragma omp parallel for
	for(int i = 0; i < static_cast<int>(m_pOPLayer->GetNeurons().size() ); i++) {
		SOMNeuron *pNeuron = (SOMNeuron*)m_pOPLayer->GetNeuron(i);
		pNeuron->CalcDistance2Inp();
		fCurVal = pNeuron->GetValue();

		// with implementation of conscience mechanism (2nd term)
		float fConscienceBias = 1.f/fNrOfNeurons - pNeuron->GetConscience();
		if(m_fConscienceRate > 0.f) {
			fCurVal -= fConscienceBias;
		}
		// end of implementation of conscience mechanism

		if(fSmallest > fCurVal) {
			fSmallest = fCurVal;
			m_pBMNeuron = pNeuron;
		}
	}

	// implementation of conscience mechanism
	//float fConscience = m_fConscienceRate * (m_pBMNeuron->GetValue() - m_pBMNeuron->GetConscience() ); 	// standard implementation seems to have some problems
	//m_pBMNeuron->AddConscience(fConscience); 																// standard implementation seems to have some problems
	if(m_fConscienceRate > 0.f) {
		#pragma omp parallel for
		for(int i = 0; i < static_cast<int>(m_pOPLayer->GetNeurons().size() ); i++) {
			SOMNeuron *pNeuron = (SOMNeuron*)m_pOPLayer->GetNeuron(i);
			float fConscience = m_fConscienceRate * (pNeuron->GetValue() - pNeuron->GetConscience() );
			pNeuron->SetConscience(fConscience);
		}
	}
	// end of implementation of conscience mechanism
	assert(m_pBMNeuron != NULL);
}

void SOMNet::SetDistFunction (const DistFunction *pFCN) {
	assert(pFCN != NULL);
	this->m_DistFunction = const_cast<DistFunction *>(pFCN);
}

void SOMNet::SetDistFunction (const DistFunction &pFCN) {
	this->m_DistFunction = const_cast<DistFunction *>(&pFCN);
}

DistFunction *SOMNet::GetDistFunction() {
	return (m_DistFunction);
}

void SOMNet::SetConscienceRate(const float &fVal) {
	m_fConscienceRate = fVal;
}
  
float SOMNet::GetConscienceRate() {
	return m_fConscienceRate;
}

}
