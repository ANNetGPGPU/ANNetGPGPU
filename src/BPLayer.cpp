/*
 * BPLayer.cpp
 *
 *  Created on: 02.06.2009
 *      Author: dgrat
 */
#include <cassert>

#include "AbsNeuron.h"
#include "BPNeuron.h"
#include "BPLayer.h"
#include "Common.h"
#include "Edge.h"
#include "math/Functions.h"

#include "containers/ConTable.h"

using namespace ANN;


template <class Type>
BPLayer<Type>::BPLayer() {
	m_pBiasNeuron = NULL;
	m_iZLayer = -1;
}

template <class Type>
BPLayer<Type>::BPLayer(int iZLayer) {
	m_pBiasNeuron = NULL;
	m_iZLayer = iZLayer;
}

template <class Type>
BPLayer<Type>::BPLayer(const BPLayer *pLayer, int iZLayer) {
	int iNumber 		= pLayer->GetNeurons().size();
	LayerTypeFlag fType 	= pLayer->GetFlag();
	m_pBiasNeuron 		= NULL;

	m_iZLayer = iZLayer;

	this->Resize(iNumber);
	this->SetFlag(fType);
}

template <class Type>
BPLayer<Type>::BPLayer(const unsigned int &iNumber, LayerTypeFlag fType, int iZLayer) {
	this->Resize(iNumber);
	m_pBiasNeuron = NULL;
	this->SetFlag(fType);

	m_iZLayer = iZLayer;
}

template <class Type>
void BPLayer<Type>::SetZLayer(int iZLayer) {
	m_iZLayer = iZLayer;
}

template <class Type>
int BPLayer<Type>::GetZLayer() {
	return m_iZLayer;
}

template <class Type>
BPLayer<Type>::~BPLayer() {
	if(m_pBiasNeuron) {
		delete m_pBiasNeuron;
	}
}

template <class Type>
void BPLayer<Type>::Resize(const unsigned int &iSize) {
	this->EraseAll();
	this->AddNeurons(iSize);
}

template <class Type>
void BPLayer<Type>::AddNeurons(const unsigned int &iSize) {
	for(unsigned int i = 0; i < iSize; i++) {
		AbsNeuron<Type> *pNeuron = new BPNeuron<Type>(this);
		this->m_lNeurons.push_back(pNeuron);
		pNeuron->SetID(this->m_lNeurons.size()-1);
	}
}

template <class Type>
void BPLayer<Type>::ConnectLayer(AbsLayer<Type> *pDestLayer, const bool &bAllowAdapt) {
	AbsNeuron<Type> *pSrcNeuron;

	/*
	 * Vernetze jedes Neuron dieser Schicht mit jedem Neuron in "destLayer"
	 */
	for(unsigned int i = 0; i < this->m_lNeurons.size(); i++) {
		pSrcNeuron = this->m_lNeurons[i];
		ANN::Connect(pSrcNeuron, pDestLayer, bAllowAdapt);
	}

	if(m_pBiasNeuron) {
		ANN::Connect(m_pBiasNeuron, pDestLayer, true);
	}
}

template <class Type>
void BPLayer<Type>::ConnectLayer(
		AbsLayer<Type> *pDestLayer,
		std::vector<std::vector<int> > Connections,
		const bool bAllowAdapt)
{
	AbsNeuron<Type> *pSrcNeuron;

	assert( Connections.size() != this->m_lNeurons.size() );
	for(unsigned int i = 0; i < Connections.size(); i++) {
		std::vector<int> subArray = Connections.at(i);
		pSrcNeuron = this->GetNeuron(i);
		assert(i != pSrcNeuron->GetID() );

		for(unsigned int j = 0; j < subArray.size(); j++) {
			assert( j < pDestLayer->GetNeurons().size() );
			AbsNeuron<Type> *pDestNeuron = pDestLayer->GetNeuron(j);
			assert( j < pDestNeuron->GetID() );
			ANN::Connect(pSrcNeuron, pDestNeuron, bAllowAdapt);
		}
	}

	if(m_pBiasNeuron) {
		ANN::Connect(m_pBiasNeuron, pDestLayer, true);
	}
}

template <class Type>
BPNeuron<Type> *BPLayer<Type>::GetBiasNeuron() const {
	return m_pBiasNeuron;
}

template <class Type>
void BPLayer<Type>::SetFlag(const LayerTypeFlag &fType) {
	this->m_fTypeFlag = fType;
	if( (this->m_fTypeFlag & ANBiasNeuron) && this->m_pBiasNeuron == NULL ) {
		this->m_pBiasNeuron = new BPNeuron<Type>(this);
		this->m_pBiasNeuron->SetValue(1.0f);
	}
}

template <class Type>
void BPLayer<Type>::AddFlag(const LayerTypeFlag &fType) {
	if(!(this->m_fTypeFlag & fType) )
	this->m_fTypeFlag |= fType;
	if( (this->m_fTypeFlag & ANBiasNeuron) && this->m_pBiasNeuron == NULL ) {
		this->m_pBiasNeuron = new BPNeuron<Type>(this);
		this->m_pBiasNeuron->SetValue(1.0f);
	}
}

template <class Type>
void BPLayer<Type>::SetLearningRate(const Type &fVal) {
	#pragma omp parallel for
	for(int j = 0; j < static_cast<int>(this->m_lNeurons.size() ); j++) {
		((BPNeuron<Type>*)this->m_lNeurons[j])->SetLearningRate(fVal);
	}
}

template <class Type>
void BPLayer<Type>::SetMomentum(const Type &fVal) {
	#pragma omp parallel for
	for(int j = 0; j < static_cast<int>(this->m_lNeurons.size() ); j++) {
		((BPNeuron<Type>*)this->m_lNeurons[j])->SetMomentum(fVal);
	}
}

template <class Type>
void BPLayer<Type>::SetWeightDecay(const Type &fVal) {
	#pragma omp parallel for
	for(int j = 0; j < static_cast<int>(this->m_lNeurons.size() ); j++) {
		((BPNeuron<Type>*)this->m_lNeurons[j])->SetWeightDecay(fVal);
	}
}

template <class Type>
void BPLayer<Type>::ExpToFS(BZFILE* bz2out, int iBZ2Error) {
	std::cout<<"Save BPLayer to FS()"<<std::endl;
	AbsLayer<Type>::ExpToFS(bz2out, iBZ2Error);

	unsigned int iNmbOfConnects = 0;
	Type fEdgeValue 	= 0.0f;
	int iDstLayerID 	= -1;
	int iDstNeurID 		= -1;

	bool bHasBias 		= false;
	int iZLayer 		= m_iZLayer;

	(GetBiasNeuron() == NULL) ? bHasBias = false : bHasBias = true;
	BZ2_bzWrite( &iBZ2Error, bz2out, &bHasBias, sizeof(bool) );
	BZ2_bzWrite( &iBZ2Error, bz2out, &iZLayer, sizeof(int) );

	if(bHasBias) {
		AbsNeuron<Type> *pCurNeur = GetBiasNeuron();
		iNmbOfConnects = pCurNeur->GetConsO().size();
		BZ2_bzWrite( &iBZ2Error, bz2out, &iNmbOfConnects, sizeof(int) );
		for(unsigned int k = 0; k < iNmbOfConnects; k++) {
			Edge<Type> *pCurEdge = pCurNeur->GetConO(k);
			iDstLayerID = pCurEdge->GetDestination(pCurNeur)->GetParent()->GetID();
			iDstNeurID = pCurEdge->GetDestinationID(pCurNeur);
			fEdgeValue = pCurEdge->GetValue();
			BZ2_bzWrite( &iBZ2Error, bz2out, &iDstLayerID, sizeof(int) );
			BZ2_bzWrite( &iBZ2Error, bz2out, &iDstNeurID, sizeof(int) );
			BZ2_bzWrite( &iBZ2Error, bz2out, &fEdgeValue, sizeof(Type) );
		}
	}
}

template <class Type>
int BPLayer<Type>::ImpFromFS(BZFILE* bz2in, int iBZ2Error, ConTable &Table) {
	std::cout<<"Load BPLayer from FS()"<<std::endl;
	int iLayerID = AbsLayer<Type>::ImpFromFS(bz2in, iBZ2Error, Table);

	unsigned int iNmbOfConnects = 0;
	Type fEdgeValue = 0.0f;
	int iDstLayerID = -1;
	int iDstNeurID 	= -1;

	bool bHasBias 	= false;
	int iZLayer 	= -1;

	BZ2_bzRead( &iBZ2Error, bz2in, &bHasBias, sizeof(bool) );
	BZ2_bzRead( &iBZ2Error, bz2in, &iZLayer, sizeof(int) );
	Table.ZValOfLayer.push_back(iZLayer);

	if(bHasBias) {
		BZ2_bzRead( &iBZ2Error, bz2in, &iNmbOfConnects, sizeof(int) );
		for(unsigned int j = 0; j < iNmbOfConnects; j++) {
			BZ2_bzRead( &iBZ2Error, bz2in, &iDstLayerID, sizeof(int) );
			BZ2_bzRead( &iBZ2Error, bz2in, &iDstNeurID, sizeof(int) );
			BZ2_bzRead( &iBZ2Error, bz2in, &fEdgeValue, sizeof(Type) );
			ConDescr	cCurCon;
			cCurCon.m_fVal 		= fEdgeValue;
			cCurCon.m_iDstNeurID 	= iDstNeurID;
			cCurCon.m_iSrcLayerID 	= iLayerID;
			cCurCon.m_iDstLayerID 	= iDstLayerID;
			Table.BiasCons.push_back(cCurCon);
		}
	}

	return iLayerID;
}

template <class Type>
F2DArray BPLayer<Type>::ExpBiasEdgesOut() const {
	unsigned int iHeight 	= 1;
	unsigned int iWidth 	= m_pBiasNeuron->GetConsO().size();

	assert(iWidth > 0);

	F2DArray vRes;
	vRes.Alloc(iWidth, iHeight);

	for(int x = 0; x < static_cast<int>(iWidth); x++) {
		vRes[0][x] = m_pBiasNeuron->GetConO(x)->GetValue();
	}
	return vRes;
}

template <class Type>
void BPLayer<Type>::ImpBiasEdgesOut(const F2DArray &mat) const {
	unsigned int iWidth 	= m_pBiasNeuron->GetConsO().size();

	assert(iWidth == mat.GetW() );

	for(int x = 0; x < static_cast<int>(iWidth); x++) {
		//std::cout<<"mat: "<<mat[0][x]<<std::endl;
		m_pBiasNeuron->GetConO(x)->SetValue(mat[0][x]);
		//std::cout<<"val: "<<m_pBiasNeuron->GetConO(x)->GetValue()<<std::endl;
	}
}

template <class Type>
void BPLayer<Type>::ImpMomentumsEdgesIn(const F2DArray &mat) {
	unsigned int iHeight 	= this->m_lNeurons.at(0)->GetConsI().size();
	unsigned int iWidth 	= this->m_lNeurons.size();

	assert(iHeight == mat.GetH() );
	assert(iWidth == mat.GetW() );

	#pragma omp parallel for
	for(int y = 0; y < static_cast<int>(iHeight); y++) {
		for(unsigned int x = 0; x < iWidth; x++) {
			this->m_lNeurons.at(x)->GetConI(y)->SetMomentum(mat[y][x]);
		}
	}
}

template <class Type>
void BPLayer<Type>::ImpMomentumsEdgesOut(const F2DArray &mat) {
	unsigned int iHeight 	= this->m_lNeurons.at(0)->GetConsO().size();
	unsigned int iWidth 	= this->m_lNeurons.size();

	assert(iHeight == mat.GetH() );
	assert(iWidth == mat.GetW() );

	#pragma omp parallel for
	for(int y = 0; y < static_cast<int>(iHeight); y++) {
		for(unsigned int x = 0; x < iWidth; x++) {
			this->m_lNeurons.at(x)->GetConO(y)->SetMomentum(mat[y][x]);
		}
	}
}


template class BPLayer<float>;
template class BPLayer<double>;
template class BPLayer<long double>;
template class BPLayer<short>;
template class BPLayer<int>;
template class BPLayer<long>;
template class BPLayer<long long>;
