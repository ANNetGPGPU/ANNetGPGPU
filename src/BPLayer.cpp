/*
 * BPLayer.cpp
 *
 *  Created on: 02.06.2009
 *      Author: Xerces
 */

#include <cassert>
//own classes
#include "math/Functions.h"
#include "Edge.h"
#include "AbsNeuron.h"
#include "BPNeuron.h"
#include "BPLayer.h"

#include "containers/ConTable.h"

using namespace ANN;


BPLayer::BPLayer(int iZLayer) {
	m_pBiasNeuron = NULL;
	m_iZLayer = iZLayer;
}

BPLayer::BPLayer(const BPLayer *pLayer, int iZLayer) {
	int iNumber 			= pLayer->GetNeurons().size();
	LayerTypeFlag fType 	= pLayer->GetFlag();
	m_pBiasNeuron 			= NULL;

	m_iZLayer = iZLayer;

	Resize(iNumber);
	SetFlag(fType);
}

BPLayer::BPLayer(const unsigned int &iNumber, LayerTypeFlag fType, int iZLayer) {
	Resize(iNumber);
	m_pBiasNeuron = NULL;
	SetFlag(fType);

	m_iZLayer = iZLayer;
}

void BPLayer::SetZLayer(int iZLayer) {
	m_iZLayer = iZLayer;
}

int BPLayer::GetZLayer() {
	return m_iZLayer;
}

BPLayer::~BPLayer() {
	if(m_pBiasNeuron) {
		delete m_pBiasNeuron;
	}
}

void BPLayer::Resize(const unsigned int &iSize) {
	EraseAll();
	AddNeurons(iSize);
}

void BPLayer::AddNeurons(const unsigned int &iSize) {
	for(unsigned int i = 0; i < iSize; i++) {
		AbsNeuron *pNeuron = new BPNeuron(this);
		m_lNeurons.push_back(pNeuron);
		pNeuron->SetID(m_lNeurons.size()-1);
	}
}

void BPLayer::ConnectLayer(AbsLayer *pDestLayer, const bool &bAllowAdapt) {
	AbsNeuron *pSrcNeuron;

	/*
	 * Vernetze jedes Neuron dieser Schicht mit jedem Neuron in "destLayer"
	 */
	for(unsigned int i = 0; i < m_lNeurons.size(); i++) {
		pSrcNeuron = m_lNeurons[i];
		Connect(pSrcNeuron, pDestLayer, bAllowAdapt);
	}

	if(m_pBiasNeuron) {
		Connect(m_pBiasNeuron, pDestLayer, true);
	}
}

void BPLayer::ConnectLayer(
		AbsLayer *pDestLayer,
		std::vector<std::vector<int> > Connections,
		const bool bAllowAdapt)
{
	AbsNeuron *pSrcNeuron;

	assert( Connections.size() != m_lNeurons.size() );
	for(unsigned int i = 0; i < Connections.size(); i++) {
		std::vector<int> subArray = Connections.at(i);
		pSrcNeuron = GetNeuron(i);
		assert(i != pSrcNeuron->GetID() );

		for(unsigned int j = 0; j < subArray.size(); j++) {
			assert( j < pDestLayer->GetNeurons().size() );
			AbsNeuron *pDestNeuron = pDestLayer->GetNeuron(j);
			assert( j < pDestNeuron->GetID() );
			Connect(pSrcNeuron, pDestNeuron, bAllowAdapt);
		}
	}

	if(m_pBiasNeuron) {
		Connect(m_pBiasNeuron, pDestLayer, true);
	}
}

BPNeuron *BPLayer::GetBiasNeuron() const {
	return m_pBiasNeuron;
}

void BPLayer::SetFlag(const LayerTypeFlag &fType) {
	m_fTypeFlag = fType;
	if( (m_fTypeFlag & ANBiasNeuron) && m_pBiasNeuron == NULL ) {
		m_pBiasNeuron = new BPNeuron(this);
		m_pBiasNeuron->SetValue(1.0f);
	}
}

void BPLayer::AddFlag(const LayerTypeFlag &fType) {
	if( !(m_fTypeFlag & fType) )
	m_fTypeFlag |= fType;
	if( (m_fTypeFlag & ANBiasNeuron) && m_pBiasNeuron == NULL ) {
		m_pBiasNeuron = new BPNeuron(this);
		m_pBiasNeuron->SetValue(1.0f);
	}
}

void BPLayer::SetLearningRate(const float &fVal) {
	#pragma omp parallel for
	for(int j = 0; j < static_cast<int>( m_lNeurons.size() ); j++) {
		((BPNeuron*)m_lNeurons[j])->SetLearningRate(fVal);
	}
}

void BPLayer::SetMomentum(const float &fVal) {
	#pragma omp parallel for
	for(int j = 0; j < static_cast<int>( m_lNeurons.size() ); j++) {
		((BPNeuron*)m_lNeurons[j])->SetMomentum(fVal);
	}
}

void BPLayer::SetWeightDecay(const float &fVal) {
	#pragma omp parallel for
	for(int j = 0; j < static_cast<int>( m_lNeurons.size() ); j++) {
		((BPNeuron*)m_lNeurons[j])->SetWeightDecay(fVal);
	}
}

void BPLayer::ExpToFS(BZFILE* bz2out, int iBZ2Error) {
	std::cout<<"Save BPLayer to FS()"<<std::endl;
	AbsLayer::ExpToFS(bz2out, iBZ2Error);

	unsigned int iNmbOfConnects 	= 0;
	float fEdgeValue 	= 0.0f;
	int iDstLayerID 	= -1;
	int iDstNeurID 		= -1;

	bool bHasBias 		= false;
	int iZLayer 		= m_iZLayer;

	(GetBiasNeuron() == NULL) ? bHasBias = false : bHasBias = true;
	BZ2_bzWrite( &iBZ2Error, bz2out, &bHasBias, sizeof(bool) );
	BZ2_bzWrite( &iBZ2Error, bz2out, &iZLayer, sizeof(int) );

	if(bHasBias) {
		AbsNeuron *pCurNeur = GetBiasNeuron();
		iNmbOfConnects = pCurNeur->GetConsO().size();
		BZ2_bzWrite( &iBZ2Error, bz2out, &iNmbOfConnects, sizeof(int) );
		for(unsigned int k = 0; k < iNmbOfConnects; k++) {
			Edge *pCurEdge = pCurNeur->GetConO(k);
			iDstLayerID = pCurEdge->GetDestination(pCurNeur)->GetParent()->GetID();
			iDstNeurID = pCurEdge->GetDestinationID(pCurNeur);
			fEdgeValue = pCurEdge->GetValue();
			BZ2_bzWrite( &iBZ2Error, bz2out, &iDstLayerID, sizeof(int) );
			BZ2_bzWrite( &iBZ2Error, bz2out, &iDstNeurID, sizeof(int) );
			BZ2_bzWrite( &iBZ2Error, bz2out, &fEdgeValue, sizeof(float) );
		}
	}
}

int BPLayer::ImpFromFS(BZFILE* bz2in, int iBZ2Error, ConTable &Table) {
	std::cout<<"Load BPLayer from FS()"<<std::endl;
	int iLayerID = AbsLayer::ImpFromFS(bz2in, iBZ2Error, Table);

	unsigned int iNmbOfConnects 	= 0;
	float fEdgeValue 	= 0.0f;
	int iDstLayerID 	= -1;
	int iDstNeurID 		= -1;

	bool bHasBias 		= false;
	int iZLayer 		= -1;

	BZ2_bzRead( &iBZ2Error, bz2in, &bHasBias, sizeof(bool) );
	BZ2_bzRead( &iBZ2Error, bz2in, &iZLayer, sizeof(int) );
	Table.ZValOfLayer.push_back(iZLayer);

	if(bHasBias) {
		BZ2_bzRead( &iBZ2Error, bz2in, &iNmbOfConnects, sizeof(int) );
		for(unsigned int j = 0; j < iNmbOfConnects; j++) {
			BZ2_bzRead( &iBZ2Error, bz2in, &iDstLayerID, sizeof(int) );
			BZ2_bzRead( &iBZ2Error, bz2in, &iDstNeurID, sizeof(int) );
			BZ2_bzRead( &iBZ2Error, bz2in, &fEdgeValue, sizeof(float) );
			ConDescr	cCurCon;
			cCurCon.m_fVal 			= fEdgeValue;
			cCurCon.m_iDstNeurID 	= iDstNeurID;
			cCurCon.m_iSrcLayerID 	= iLayerID;
			cCurCon.m_iDstLayerID 	= iDstLayerID;
			Table.BiasCons.push_back(cCurCon);
		}
	}

	return iLayerID;
}

F2DArray BPLayer::ExpBiasEdgesOut() const {
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

void BPLayer::ImpBiasEdgesOut(const F2DArray &mat) const {
	unsigned int iWidth 	= m_pBiasNeuron->GetConsO().size();

	assert(iWidth == mat.GetW() );

	for(int x = 0; x < static_cast<int>(iWidth); x++) {
		//std::cout<<"mat: "<<mat[0][x]<<std::endl;
		m_pBiasNeuron->GetConO(x)->SetValue(mat[0][x]);
		//std::cout<<"val: "<<m_pBiasNeuron->GetConO(x)->GetValue()<<std::endl;
	}
}

void BPLayer::ImpMomentumsEdgesIn(const F2DArray &mat) {
	unsigned int iHeight 	= m_lNeurons.at(0)->GetConsI().size();
	unsigned int iWidth 	= m_lNeurons.size();

	assert(iHeight == mat.GetH() );
	assert(iWidth == mat.GetW() );

	#pragma omp parallel for
	for(int y = 0; y < static_cast<int>(iHeight); y++) {
		for(unsigned int x = 0; x < iWidth; x++) {
			m_lNeurons.at(x)->GetConI(y)->SetMomentum(mat[y][x]);
		}
	}
}

void BPLayer::ImpMomentumsEdgesOut(const F2DArray &mat) {
	unsigned int iHeight 	= m_lNeurons.at(0)->GetConsO().size();
	unsigned int iWidth 	= m_lNeurons.size();

	assert(iHeight == mat.GetH() );
	assert(iWidth == mat.GetW() );

	#pragma omp parallel for
	for(int y = 0; y < static_cast<int>(iHeight); y++) {
		for(unsigned int x = 0; x < iWidth; x++) {
			m_lNeurons.at(x)->GetConO(y)->SetMomentum(mat[y][x]);
		}
	}
}

/*
 * AUSGABEOPERATOR
 * OSTREAM
 */
std::ostream& operator << (std::ostream &os, BPLayer &op)
{
	if(op.GetBiasNeuron() != 0)
	os << "Bias neuron: \t" << op.GetBiasNeuron()->GetValue() 	<< std::endl;
    os << "Nr. neurons: \t" << op.GetNeurons().size() 					<< std::endl;
    return os;     // Ref. auf Stream
}

std::ostream& operator << (std::ostream &os, BPLayer *op)
{
	if(op->GetBiasNeuron() != 0)
	os << "Bias neuron: \t" << op->GetBiasNeuron()->GetValue()	<< std::endl;
    os << "Nr. neurons: \t" << op->GetNeurons().size() 					<< std::endl;
    return os;     // Ref. auf Stream
}



