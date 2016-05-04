/*
 * AbsLayer.cpp
 *
 *  Created on: 21.02.2011
 *      Author: dgrat
 */

#include <cassert>
//own classes
#include "math/Functions.h"
#include "Edge.h"
#include "AbsNeuron.h"
#include "AbsLayer.h"

#include "containers/ConTable.h"

using namespace ANN;


AbsLayer::AbsLayer() {

}
/*
AbsLayer::AbsLayer(const unsigned int &iNumber, int iShiftID) {
	Resize(iNumber, iShiftID);
}
*/
AbsLayer::~AbsLayer() {
	EraseAll();
}

void AbsLayer::SetID(const int &iID) {
	m_iID = iID;
}

int AbsLayer::GetID() const {
	return m_iID;
}

const std::vector<AbsNeuron *> &AbsLayer::GetNeurons() const {
	return m_lNeurons;
}

void AbsLayer::EraseAllEdges() {
	for(unsigned int i = 0; i < m_lNeurons.size(); i++) {
		m_lNeurons[i]->EraseAllEdges();
	}
}

void AbsLayer::EraseAll() {
	for(unsigned int i = 0; i < m_lNeurons.size(); i++) {
		m_lNeurons[i]->EraseAllEdges();
		delete m_lNeurons[i];
	}
	m_lNeurons.clear();
}

AbsNeuron *AbsLayer::GetNeuron(const unsigned int &iID) const {
	// quick try
	if(m_lNeurons.at(iID)->GetID() == iID) {
		return m_lNeurons.at(iID);
	}
	// fall back scenario
	else {
		for(unsigned int i = 0; i < m_lNeurons.size(); i++) {
			if(m_lNeurons.at(i)->GetID() == iID)
				return m_lNeurons.at(i);
		}
	}

	return NULL;
}

void AbsLayer::SetNetFunction(const TransfFunction *pFunction) {
	assert( pFunction != 0 );
	#pragma omp parallel for
	for(int j = 0; j < static_cast<int>( m_lNeurons.size() ); j++) {
		m_lNeurons[j]->SetTransfFunction(pFunction);
	}
}

void AbsLayer::SetFlag(const LayerTypeFlag &fType) {
	m_fTypeFlag = fType;
}

void AbsLayer::AddFlag(const LayerTypeFlag &fType) {
	if( !(m_fTypeFlag & fType) )
	m_fTypeFlag |= fType;
}

LayerTypeFlag AbsLayer::GetFlag() const {
	return m_fTypeFlag;
}

void AbsLayer::ExpToFS(BZFILE* bz2out, int iBZ2Error) {
	int iLayerID 				= GetID();
	BZ2_bzWrite( &iBZ2Error, bz2out, &iLayerID, sizeof(int) );

	std::cout<<"Save AbsLayer to FS()"<<std::endl;

	LayerTypeFlag 	fLayerType 	= GetFlag();
	unsigned int iNmbOfNeurons 	= GetNeurons().size();

	BZ2_bzWrite( &iBZ2Error, bz2out, &fLayerType, sizeof(LayerTypeFlag) );	// Type of layer
	BZ2_bzWrite( &iBZ2Error, bz2out, &iNmbOfNeurons, sizeof(int) );			// Number of neuron in this layer (except bias)
	for(unsigned int j = 0; j < iNmbOfNeurons; j++) {
		AbsNeuron *pCurNeur = GetNeuron(j);
		pCurNeur->ExpToFS(bz2out, iBZ2Error);
	}
}

int AbsLayer::ImpFromFS(BZFILE* bz2in, int iBZ2Error, ConTable &Table) {
	int iLayerID 				= -1;
	BZ2_bzRead( &iBZ2Error, bz2in, &iLayerID, sizeof(int) );

	std::cout<<"Load AbsLayer from FS()"<<std::endl;

	LayerTypeFlag 	fLayerType 	= 0;
	unsigned int iNmbOfNeurons 	= 0;

	BZ2_bzRead( &iBZ2Error, bz2in, &fLayerType, sizeof(LayerTypeFlag) );
	BZ2_bzRead( &iBZ2Error, bz2in, &iNmbOfNeurons, sizeof(int) );
	Table.TypeOfLayer.push_back(fLayerType);
	Table.SizeOfLayer.push_back(iNmbOfNeurons);

	for(unsigned int j = 0; j < iNmbOfNeurons; j++) {
		AddNeurons(1);	// Create dummy neuron; more neurons than needed don't disturb, but are necessary if using empty nets

		NeurDescr 	cCurNeur;
		cCurNeur.m_iLayerID = iLayerID;
		Table.Neurons.push_back(cCurNeur);
		AbsNeuron *pCurNeur = GetNeuron(j);
		pCurNeur->ImpFromFS(bz2in, iBZ2Error, Table);
	}

	return iLayerID;
}

/*FRIEND:*/
void SetEdgesToValue(AbsLayer *pSrcLayer, AbsLayer *pDestLayer, const float &fVal, const bool &bAdaptState) {
	AbsNeuron	*pCurNeuron;
	Edge 		*pCurEdge;
	for(unsigned int i = 0; i < pSrcLayer->GetNeurons().size(); i++) {
		pCurNeuron = pSrcLayer->GetNeurons().at(i);
		for(unsigned int j = 0; j < pCurNeuron->GetConsO().size(); j++) {
			pCurEdge = pCurNeuron->GetConO(j);
			// outgoing edge is connected with pDestLayer ..
			if(pCurEdge->GetDestination(pCurNeuron)->GetParent() == pDestLayer) {
				// .. d adapt only these edges
				pCurEdge->SetValue( fVal );
				pCurEdge->SetAdaptationState( bAdaptState );
			}
		}
	}
}

F2DArray AbsLayer::ExpEdgesIn() const {
	unsigned int iHeight 	= m_lNeurons.front()->GetConsI().size();
	unsigned int iWidth 	= m_lNeurons.size();

	assert(iWidth > 0 && iHeight > 0);

	F2DArray vRes;
	vRes.Alloc(iWidth, iHeight);

	#pragma omp parallel for
	for(int y = 0; y < static_cast<int>(iHeight); y++) {
		for(unsigned int x = 0; x < iWidth; x++) {
			vRes[y][x] = m_lNeurons.at(x)->GetConI(y)->GetValue();
		}
	}
	return vRes;
}

F2DArray AbsLayer::ExpEdgesIn(int iStart, int iStop) const {
	unsigned int iWidth 	= iStop-iStart+1;
	unsigned int iHeight 	= m_lNeurons.front()->GetConsI().size();

	assert(iWidth > 0);
	assert(iStart >= 0);
	assert(iStop < m_lNeurons.size() );

	F2DArray vRes;
	vRes.Alloc(iWidth, iHeight);
	
	#pragma omp parallel for
	for(int y = 0; y < static_cast<int>(iHeight); y++) {
		int iC = 0;
		for(unsigned int x = iStart; x <= iStop; x++) {
			vRes[y][iC] = m_lNeurons.at(x)->GetConI(y)->GetValue();
			iC++;
		}
	}
	return vRes;
}

F2DArray AbsLayer::ExpEdgesOut() const {
	unsigned int iHeight 	= m_lNeurons.front()->GetConsO().size();
	unsigned int iWidth 	= m_lNeurons.size();

	assert(iWidth > 0 && iHeight > 0);

	F2DArray vRes;
	vRes.Alloc(iWidth, iHeight);

	#pragma omp parallel for
	for(int y = 0; y < static_cast<int>(iHeight); y++) {
		for(unsigned int x = 0; x < iWidth; x++) {
			vRes[y][x] = m_lNeurons.at(x)->GetConO(y)->GetValue();
		}
	}
	return vRes;
}

F2DArray AbsLayer::ExpEdgesOut(int iStart, int iStop) const {
	unsigned int iWidth 	= iStop-iStart+1;
	unsigned int iHeight 	= m_lNeurons.front()->GetConsO().size();

	assert(iWidth > 0);
	assert(iStart >= 0);
	assert(iStop < m_lNeurons.size() );

	F2DArray vRes;
	vRes.Alloc(iWidth, iHeight);
	
	#pragma omp parallel for
	for(int y = 0; y < static_cast<int>(iHeight); y++) {
		int iC = 0;
		for(unsigned int x = iStart; x <= iStop; x++) {
			vRes[y][iC] = m_lNeurons.at(x)->GetConI(y)->GetValue();
			iC++;
		}
	}
	return vRes;
}

void AbsLayer::ImpEdgesIn(const F2DArray &mat) {
	unsigned int iHeight 	= m_lNeurons.front()->GetConsI().size();
	unsigned int iWidth 	= m_lNeurons.size();

	assert(iHeight == mat.GetH() );
	assert(iWidth == mat.GetW() );

	#pragma omp parallel for
	for(int y = 0; y < static_cast<int>(iHeight); y++) {
		for(unsigned int x = 0; x < iWidth; x++) {
			m_lNeurons.at(x)->GetConI(y)->SetValue(mat[y][x]);
		}
	}
}

void AbsLayer::ImpEdgesIn(const F2DArray &mat, int iStart, int iStop) {
	unsigned int iHeight 	= m_lNeurons.front()->GetConsI().size();
	
	assert(iHeight == mat.GetH() );
	assert(iStop-iStart <= mat.GetW() );

	#pragma omp parallel for
	for(int y = 0; y < static_cast<int>(iHeight); y++) {
		int iC = 0;
		for(unsigned int x = iStart; x <= iStop; x++) {
			m_lNeurons.at(x)->GetConI(y)->SetValue(mat[y][iC]);
			iC++;
		}
	}
}

void AbsLayer::ImpEdgesOut(const F2DArray &mat) {
	unsigned int iHeight 	= m_lNeurons.front()->GetConsO().size();
	unsigned int iWidth 	= m_lNeurons.size();

	assert(iHeight == mat.GetH() );
	assert(iWidth == mat.GetW() );

	#pragma omp parallel for
	for(int y = 0; y < static_cast<int>(iHeight); y++) {
		for(unsigned int x = 0; x < iWidth; x++) {
			m_lNeurons.at(x)->GetConO(y)->SetValue(mat[y][x]);
		}
	}
}

void AbsLayer::ImpEdgesOut(const F2DArray &mat, int iStart, int iStop) {
	unsigned int iHeight 	= m_lNeurons.front()->GetConsO().size();
	
	assert(iHeight == mat.GetH() );
	assert(iStop-iStart <= mat.GetW() );

	#pragma omp parallel for
	for(int y = 0; y < static_cast<int>(iHeight); y++) {
		int iC = 0;
		for(unsigned int x = iStart; x <= iStop; x++) {
			m_lNeurons.at(x)->GetConO(y)->SetValue(mat[y][iC]);
			iC++;
		}
	}
}

F2DArray AbsLayer::ExpPositions() const {
	unsigned int iHeight 	= m_lNeurons.at(0)->GetPosition().size();
	unsigned int iWidth 	= m_lNeurons.size();
	
	assert(iWidth > 0 && iHeight > 0);

	F2DArray vRes;
	vRes.Alloc(iWidth, iHeight);

	#pragma omp parallel for
	for(int y = 0; y < static_cast<int>(iHeight); y++) {
		for(unsigned int x = 0; x < iWidth; x++) {
			vRes[y][x] = m_lNeurons.at(x)->GetPosition().at(y);
		}
	}
	return vRes;
}

F2DArray AbsLayer::ExpPositions(int iStart, int iStop) const {
	unsigned int iHeight 	= m_lNeurons.at(0)->GetPosition().size();
	unsigned int iWidth 	= iStop-iStart+1;

	assert(iStop-iStart <= m_lNeurons.size() );
	assert(iStart >= 0);
	assert(iStop < m_lNeurons.size() );

	F2DArray vRes;
	vRes.Alloc(iWidth, iHeight);

	#pragma omp parallel for
	for(int y = 0; y < static_cast<int>(iHeight); y++) {
		int iC = 0;
		for(int x = iStart; x <= iStop; x++) {
			vRes[y][iC] = m_lNeurons.at(x)->GetPosition().at(y);
			iC++;
		}
	}
	return vRes;
}

void AbsLayer::ImpPositions(const F2DArray &f2dPos) {
	unsigned int iHeight = f2dPos.GetH();
	unsigned int iWidth = f2dPos.GetW();

	assert(iWidth == m_lNeurons.size() );

	#pragma omp parallel for
	for(int x = 0; x < static_cast<int>(iWidth); x++) {
		std::vector<float> vPos(iHeight);
		for(unsigned int y = 0; y < iHeight; y++) {
			vPos[y] = f2dPos.GetValue(x, y);
		}
		m_lNeurons.at(x)->SetPosition(vPos);
	}
}

void AbsLayer::ImpPositions(const F2DArray &f2dPos, int iStart, int iStop) {
	unsigned int iHeight = f2dPos.GetH();
	unsigned int iWidth = f2dPos.GetW();

	assert(iStop-iStart <= m_lNeurons.size() );
	
	int iC = 0;
	#pragma omp parallel for
	for(int x = iStart; x <= static_cast<int>(iStop); x++) {
		std::vector<float> vPos(iHeight);
		for(unsigned int y = 0; y < iHeight; y++) {
			vPos[y] = f2dPos.GetValue(iC, y);
		}
		iC++;
		m_lNeurons.at(x)->SetPosition(vPos);
	}
}

