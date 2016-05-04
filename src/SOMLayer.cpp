/*
 * SOMLayer.cpp
 *
 *  Created on: 26.02.2012
 *      Author: dgrat
 */

#include <cassert>
#include <omp.h>
#include "SOMLayer.h"
#include "SOMNeuron.h"
#include "Edge.h"


namespace ANN {

SOMLayer::SOMLayer() {

}

SOMLayer::SOMLayer(const SOMLayer *pLayer) {
	int iNumber 			= pLayer->GetNeurons().size();
	LayerTypeFlag fType 	= pLayer->GetFlag();

	Resize(iNumber);
	SetFlag(fType);
}

SOMLayer::SOMLayer(const unsigned int &iSize, LayerTypeFlag fType) {
	Resize(iSize);
	SetFlag(fType);
}

SOMLayer::SOMLayer(const unsigned int &iWidth, const unsigned int &iHeight, LayerTypeFlag fType) {
	Resize(iWidth, iHeight);
	SetFlag(fType);
}

SOMLayer::SOMLayer(const std::vector<unsigned int> &vDim, LayerTypeFlag fType) {
	Resize(vDim);
	SetFlag(fType);
}

SOMLayer::~SOMLayer() {
	// TODO Auto-generated destructor stub
}

void SOMLayer::AddNeurons(const unsigned int &iSize) {
	std::vector<float> vPos(1);
	for(unsigned int x = 0; x < iSize; x++) {
		SOMNeuron *pNeuron = new SOMNeuron(this);
		m_lNeurons.push_back(pNeuron);
		pNeuron->SetID(m_lNeurons.size()-1);

		vPos[0] = m_lNeurons.size()-1;
		pNeuron->SetPosition(vPos);
	}
}

void SOMLayer::Resize(const unsigned int &iSize) {
	EraseAll();

	std::vector<float> vPos(1);
	for(unsigned int x = 0; x < iSize; x++) {
		SOMNeuron *pNeuron = new SOMNeuron(this);
		m_lNeurons.push_back(pNeuron);
		pNeuron->SetID(m_lNeurons.size()-1);

		vPos[0] = m_lNeurons.size()-1;
		pNeuron->SetPosition(vPos);
	}
}

void SOMLayer::Resize(const unsigned int &iWidth, const unsigned int &iHeight) {
	EraseAll();

	// Set m_vDim properly
	m_vDim.clear();
	m_vDim.push_back(iWidth);
	m_vDim.push_back(iHeight);

	std::vector<float> vPos(2);
	for(unsigned int y = 0; y < iHeight; y++) {
		for(unsigned int x = 0; x < iWidth; x++) {
			SOMNeuron *pNeuron = new SOMNeuron(this);
			pNeuron->SetID(y*iWidth + x);
			m_lNeurons.push_back(pNeuron);

			vPos[0] = x; vPos[1] = y;
			pNeuron->SetPosition(vPos);
		}
	}
}

void SOMLayer::Resize(const std::vector<unsigned int> &vDim) {
	EraseAll();

	assert(vDim.size() > 0);

	m_vDim = vDim;

	unsigned int iSize = 1;
	for(unsigned int i = 0; i < vDim.size(); i++) {
		iSize *= vDim[i];
	}
	Resize(iSize);

	/*
	 * TODO implement position handling of the neurons
	 */
}

void SOMLayer::ConnectLayer(AbsLayer *pDestLayer, const bool &bAllowAdapt) {
	AbsNeuron *pSrcNeuron = NULL;

	/*
	 * Vernetze jedes Neuron dieser Schicht mit jedem Neuron in "pDestLayer"
	 */
	for(int i = 0; i < static_cast<int>(m_lNeurons.size() ); i++) {
		std::cout<<"Connect input neuron " << i << " to output layer. Progress: "<<i+1<<"/"<<m_lNeurons.size()<<std::endl;
		pSrcNeuron = m_lNeurons[i];
		if(pSrcNeuron != NULL) {
			Connect(pSrcNeuron, pDestLayer, bAllowAdapt);
		}
	}
}

void SOMLayer::ConnectLayer(AbsLayer *pDestLayer, const F2DArray &f2dEdgeMat, const bool &bAllowAdapt) {
	AbsNeuron *pSrcNeuron = NULL;

	/*
	 * Vernetze jedes Neuron dieser Schicht mit jedem Neuron in "pDestLayer"
	 */
	std::vector<float> fMoms(f2dEdgeMat.GetH(), 0);	// TODO not used by SOMs
	std::vector<float> fVals(f2dEdgeMat.GetH(), 0);

	for(int i = 0; i < static_cast<int>(m_lNeurons.size() ); i++) {
		std::cout<<"Connect input neuron " << i << " to output layer. Progress: "<<i+1<<"/"<<m_lNeurons.size()<<std::endl;
		pSrcNeuron = m_lNeurons[i];

		fVals = f2dEdgeMat.GetSubArrayX(i);
		if(pSrcNeuron != NULL) {
			Connect(pSrcNeuron, pDestLayer, fVals, fMoms, bAllowAdapt);
		}
	}
}

void SOMLayer::SetLearningRate(const float &fVal) {
	#pragma omp parallel for
	for(int j = 0; j < static_cast<int>( m_lNeurons.size() ); j++) {
		((SOMNeuron*)m_lNeurons[j])->SetLearningRate(fVal);
	}
}

std::vector<unsigned int> SOMLayer::GetDim() const {
	return m_vDim;
}

unsigned int SOMLayer::GetDim(const unsigned int &iInd) const {
	return m_vDim.at(iInd);
}

}
