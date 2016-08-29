/*
 * SOMLayer.cpp
 *
 *  Created on: 26.02.2012
 *      Author: dgrat
 */
/*
#include <cassert>
#include <omp.h>
#include "SOMLayer.h"
#include "SOMNeuron.h"
#include "Edge.h"


using namespace ANN;
*/

template <class Type>
SOMLayer<Type>::SOMLayer() {

}

template <class Type>
SOMLayer<Type>::SOMLayer(const SOMLayer<Type> *pLayer) {
	int iNumber 			= pLayer->GetNeurons().size();
	LayerTypeFlag fType 	= pLayer->GetFlag();

	Resize(iNumber);
	this->SetFlag(fType);
}

template <class Type>
SOMLayer<Type>::SOMLayer(const unsigned int &iSize, LayerTypeFlag fType) {
	Resize(iSize);
	this->SetFlag(fType);
}

template <class Type>
SOMLayer<Type>::SOMLayer(const unsigned int &iWidth, const unsigned int &iHeight, LayerTypeFlag fType) {
	Resize(iWidth, iHeight);
	this->SetFlag(fType);
}

template <class Type>
SOMLayer<Type>::SOMLayer(const std::vector<unsigned int> &vDim, LayerTypeFlag fType) {
	Resize(vDim);
	this->SetFlag(fType);
}

template <class Type>
void SOMLayer<Type>::AddNeurons(const unsigned int &iSize) {
	std::vector<Type> vPos(1);
	for(unsigned int x = 0; x < iSize; x++) {
		SOMNeuron<Type> *pNeuron = new SOMNeuron<Type>(this);
		this->m_lNeurons.push_back(pNeuron);
		pNeuron->SetID(this->m_lNeurons.size()-1);

		vPos[0] = this->m_lNeurons.size()-1;
		pNeuron->SetPosition(vPos);
	}
}

template <class Type>
void SOMLayer<Type>::Resize(const unsigned int &iSize) {
	this->EraseAll();

	std::vector<Type> vPos(1);
	for(unsigned int x = 0; x < iSize; x++) {
		SOMNeuron<Type> *pNeuron = new SOMNeuron<Type>(this);
		this->m_lNeurons.push_back(pNeuron);
		pNeuron->SetID(this->m_lNeurons.size()-1);

		vPos[0] = this->m_lNeurons.size()-1;
		pNeuron->SetPosition(vPos);
	}
}

template <class Type>
void SOMLayer<Type>::Resize(const unsigned int &iWidth, const unsigned int &iHeight) {
	this->EraseAll();

	// Set m_vDim properly
	m_vDim.clear();
	m_vDim.push_back(iWidth);
	m_vDim.push_back(iHeight);

	std::vector<Type> vPos(2);
	for(unsigned int y = 0; y < iHeight; y++) {
		for(unsigned int x = 0; x < iWidth; x++) {
			SOMNeuron<Type> *pNeuron = new SOMNeuron<Type>(this);
			pNeuron->SetID(y*iWidth + x);
			this->m_lNeurons.push_back(pNeuron);

			vPos[0] = x; vPos[1] = y;
			pNeuron->SetPosition(vPos);
		}
	}
}

template <class Type>
void SOMLayer<Type>::Resize(const std::vector<unsigned int> &vDim) {
	this->EraseAll();

	assert(vDim.size() > 0);

	m_vDim = vDim;

	unsigned int iSize = 1;
	for(unsigned int i = 0; i < vDim.size(); i++) {
		iSize *= vDim[i];
	}
	Resize(iSize);
}

template <class Type>
void SOMLayer<Type>::ConnectLayer(AbsLayer<Type> *pDestLayer, const bool &bAllowAdapt) {
	AbsNeuron<Type> *pSrcNeuron = NULL;

	/*
	 * Vernetze jedes Neuron dieser Schicht mit jedem Neuron in "pDestLayer"
	 */
	for(int i = 0; i < static_cast<int>(this->m_lNeurons.size() ); i++) {
		std::cout<<"Connect input neuron " << i << " to output layer. Progress: "<<i+1<<"/"<<this->m_lNeurons.size()<<std::endl;
		pSrcNeuron = this->m_lNeurons[i];
		if(pSrcNeuron != NULL) {
			Connect(pSrcNeuron, pDestLayer, bAllowAdapt);
		}
	}
}

template <class Type>
void SOMLayer<Type>::ConnectLayer(AbsLayer<Type> *pDestLayer, const F2DArray<Type> &f2dEdgeMat, const bool &bAllowAdapt) {
	AbsNeuron<Type> *pSrcNeuron = NULL;

	/*
	 * Vernetze jedes Neuron dieser Schicht mit jedem Neuron in "pDestLayer"
	 */
	std::vector<float> fMoms(f2dEdgeMat.GetH(), 0);	// TODO not used by SOMs
	std::vector<float> fVals(f2dEdgeMat.GetH(), 0);

	for(int i = 0; i < static_cast<int>(this->m_lNeurons.size() ); i++) {
		std::cout<<"Connect input neuron " << i << " to output layer. Progress: "<<i+1<<"/"<<this->m_lNeurons.size()<<std::endl;
		pSrcNeuron = this->m_lNeurons[i];

		fVals = f2dEdgeMat.GetSubArrayX(i);
		if(pSrcNeuron != NULL) {
			Connect(pSrcNeuron, pDestLayer, fVals, fMoms, bAllowAdapt);
		}
	}
}

template <class Type>
std::vector<float> SOMLayer<Type>::GetPosition(const unsigned int iNeuronID) {
	AbsNeuron<Type> *pSrcNeuron = this->m_lNeurons[iNeuronID];
	if(pSrcNeuron == NULL) {
		return std::vector<float>();
	}
	return pSrcNeuron->GetPosition();
}

template <class Type>
void SOMLayer<Type>::SetLearningRate(const float &fVal) {
	#pragma omp parallel for
	for(int j = 0; j < static_cast<int>( this->m_lNeurons.size() ); j++) {
		((SOMNeuron<Type>*)this->m_lNeurons[j])->SetLearningRate(fVal);
	}
}

template <class Type>
std::vector<unsigned int> SOMLayer<Type>::GetDim() const {
	return m_vDim;
}

template <class Type>
unsigned int SOMLayer<Type>::GetDim(const unsigned int &iInd) const {
	return m_vDim.at(iInd);
}
